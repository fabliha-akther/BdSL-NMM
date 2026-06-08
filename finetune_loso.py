"""Fine-tune a pretrained SignNet-V2 checkpoint on the LOSO split.

This script reuses the existing multimodal dataset pipeline, loads the
pretrained random-split checkpoint, and fine-tunes on the LOSO train/val
splits with word-only supervision.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.data import DataConfig, create_data_loaders
from src.models.signet_v2 import SignNetV2, count_parameters


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_sample_list(file_path: Path) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_json(file_path: Path) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(file_path: Path, payload: Dict[str, Any]) -> None:
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SignNet-V2 on LOSO splits")
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default="processed/checkpoints/model_random_split",
        help="Path to pretrained checkpoint directory containing best_model.pth and label_mapping.json",
    )
    parser.add_argument(
        "--normalized_dir",
        type=str,
        default="processed/multimodal_6signers",
        help="Directory containing normalized NPZ files",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--emotion_loss_weight", type=float, default=0.0)
    parser.add_argument("--checkpoint_name", type=str, default="model_finetuned_loso")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def resolve_model_kwargs(pretrained_checkpoint_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    checkpoint_path = pretrained_checkpoint_dir / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    model_kwargs = {
        "num_classes": int(config.get("num_classes", 74)),
        "num_emotions": config.get("num_emotions", 5),
        "body_dim": int(config.get("body_dim", 99)),
        "hand_dim": int(config.get("hand_dim", 63)),
        "face_dim": int(config.get("face_dim", 1434)),
        "d_model": int(config.get("d_model", 256)),
        "num_encoder_layers": int(config.get("num_encoder_layers", 6)),
        "num_heads": int(config.get("num_heads", 8)),
        "d_ff": int(config.get("d_ff", 1024)),
        "dropout": float(config.get("dropout", 0.2)),
        "max_seq_length": int(config.get("max_seq_length", 150)),
        "use_face": True if config.get("use_face") is None else bool(config.get("use_face")),
        "use_hands": True if config.get("use_hands") is None else bool(config.get("use_hands")),
    }

    if model_kwargs["num_emotions"] in (None, 0, "0"):
        model_kwargs["num_emotions"] = 5

    return model_kwargs, checkpoint


def load_model_state(model: nn.Module, checkpoint: Dict[str, Any]) -> None:
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid state dict")

    model_state = model.state_dict()
    filtered_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if missing:
        print(f"   Missing keys ignored: {len(missing)}")
        for key in missing:
            print(f"      MISSING: {key}")
    if unexpected:
        print(f"   Unexpected keys ignored: {len(unexpected)}")
        for key in unexpected:
            print(f"      UNEXPECTED: {key}")


def evaluate(model: nn.Module, loader, device: torch.device, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for batch in loader:
            body_pose = batch["body_pose"].to(device)
            left_hand = batch["left_hand"].to(device)
            right_hand = batch["right_hand"].to(device)
            face = batch["face"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).long().view(-1)

            logits = model(body_pose, left_hand, right_hand, face, attention_mask)
            loss = criterion(logits, labels)

            total_loss += float(loss.item()) * labels.size(0)
            predictions = torch.argmax(logits, dim=1)
            total_correct += int((predictions == labels).sum().item())
            total_samples += int(labels.size(0))

    average_loss = total_loss / total_samples if total_samples else 0.0
    accuracy = total_correct / total_samples if total_samples else 0.0
    return average_loss, accuracy


def train_one_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
    emotion_criterion: nn.Module,
    emotion_loss_weight: float,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in loader:
        body_pose = batch["body_pose"].to(device)
        left_hand = batch["left_hand"].to(device)
        right_hand = batch["right_hand"].to(device)
        face = batch["face"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device).long().view(-1)
        emotion_labels = batch["emotion_label"].to(device).long().view(-1)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, emotion_logits = model.forward_with_aux(
                body_pose, left_hand, right_hand, face, attention_mask
            )
            sign_loss = criterion(logits, labels)
            loss = sign_loss

            if (
                emotion_logits is not None
                and emotion_loss_weight > 0.0
                and int(emotion_labels.min().item()) >= 0
            ):
                emotion_loss = emotion_criterion(emotion_logits, emotion_labels)
                loss = sign_loss + (emotion_loss_weight * emotion_loss)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item()) * labels.size(0)
        total_samples += int(labels.size(0))

    scheduler.step()
    return running_loss / total_samples if total_samples else 0.0


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    base_dir = Path(__file__).resolve().parent
    pretrained_dir = (base_dir / args.pretrained_checkpoint).resolve()
    normalized_dir = (base_dir / args.normalized_dir).resolve()
    processed_dir = (base_dir / "processed").resolve()
    checkpoint_dir = processed_dir / "checkpoints" / args.checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_split_path = processed_dir / "train_samples.txt"
    val_split_path = processed_dir / "val_samples.txt"
    if not train_split_path.exists() or not val_split_path.exists():
        raise FileNotFoundError("LOSO split files are missing from processed/train_samples.txt or processed/val_samples.txt")

    pretrained_label_mapping_path = pretrained_dir / "label_mapping.json"
    if not pretrained_label_mapping_path.exists():
        raise FileNotFoundError(f"Missing label mapping: {pretrained_label_mapping_path}")

    label_mapping = load_json(pretrained_label_mapping_path)
    model_kwargs, checkpoint = resolve_model_kwargs(pretrained_dir)
    model_kwargs["num_classes"] = len(label_mapping["word_to_label"])

    train_samples = load_sample_list(train_split_path)
    val_samples = load_sample_list(val_split_path)

    print(f"\n{'=' * 80}")
    print("🚀 SignNet-V2 LOSO Fine-Tuning")
    print(f"{'=' * 80}")
    print(f"   Pretrained checkpoint: {pretrained_dir}")
    print(f"   Checkpoint directory: {checkpoint_dir}")
    print(f"   Train samples: {len(train_samples)}")
    print(f"   Val samples: {len(val_samples)}")
    print(f"   Normalized dir: {normalized_dir}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Dropout: {args.dropout}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Emotion loss weight: {args.emotion_loss_weight}")

    if model_kwargs.get("num_emotions") is None:
        print("   Emotion head: disabled (word-only fine-tuning)")
    else:
        print(f"   Emotion head classes: {model_kwargs['num_emotions']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    data_config = DataConfig(
        base_dir=str(base_dir),
        processed_dir="processed",
        normalized_dir=str(normalized_dir),
        checkpoint_dir=str(checkpoint_dir),
        max_seq_length=int(model_kwargs.get("max_seq_length", 150)),
        augmentation=False,
        loader_error_mode="permissive",
    )

    train_loader, val_loader, _ = create_data_loaders(
        config=data_config,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=[],
        word_to_label=label_mapping["word_to_label"],
        emotion_to_label=label_mapping.get("emotion_to_label", {}),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_hands=True,
        use_face=True,
    )

    model = SignNetV2(
        num_classes=model_kwargs["num_classes"],
        num_emotions=model_kwargs["num_emotions"],
        body_dim=model_kwargs["body_dim"],
        hand_dim=model_kwargs["hand_dim"],
        face_dim=model_kwargs["face_dim"],
        d_model=model_kwargs["d_model"],
        num_encoder_layers=model_kwargs["num_encoder_layers"],
        num_heads=model_kwargs["num_heads"],
        d_ff=model_kwargs["d_ff"],
        dropout=args.dropout,
        max_seq_length=model_kwargs["max_seq_length"],
        use_face=True,
        use_hands=True,
    )
    load_model_state(model, checkpoint)
    model.to(device)

    params = count_parameters(model)
    print(f"   Model parameters: {params['total']:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    emotion_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.1
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rate": [],
    }

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_state: Optional[Dict[str, Any]] = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            emotion_criterion=emotion_criterion,
            emotion_loss_weight=args.emotion_loss_weight,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            use_amp=(device.type == "cuda"),
        )
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_top1={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "config": {
                    **model_kwargs,
                    "dropout": args.dropout,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "emotion_loss_weight": args.emotion_loss_weight,
                    "checkpoint_name": args.checkpoint_name,
                    "pretrained_checkpoint": str(pretrained_dir),
                    "normalized_dir": str(normalized_dir),
                },
                "history": history,
            }
            torch.save(best_state, checkpoint_dir / "best_model.pth")

    save_json(checkpoint_dir / "label_mapping.json", label_mapping)
    save_json(checkpoint_dir / "training_history.json", history)

    if best_state is None:
        best_state = {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "val_accuracy": best_val_acc,
            "val_loss": best_val_loss,
            "config": {
                **model_kwargs,
                "dropout": args.dropout,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "emotion_loss_weight": args.emotion_loss_weight,
                "checkpoint_name": args.checkpoint_name,
                "pretrained_checkpoint": str(pretrained_dir),
                "normalized_dir": str(normalized_dir),
            },
            "history": history,
        }
        torch.save(best_state, checkpoint_dir / "best_model.pth")

    print("\n================================================================================")
    print("✅ FINE-TUNING COMPLETE")
    print("================================================================================")
    print(f"Output directory: {checkpoint_dir}")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Saved files: best_model.pth, label_mapping.json, training_history.json")


if __name__ == "__main__":
    main()