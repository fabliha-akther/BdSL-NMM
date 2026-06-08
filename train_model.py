"""
Simplified Training Script for SignNet-V2 - BdSL Recognition
Adapted for local directory structure with processed/multimodal/ data
"""

import torch
import torch.nn as nn
from pathlib import Path
import argparse
import json
import numpy as np
import random
import unicodedata
import wandb
import sys
import os
from collections import Counter
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple

# Project imports
from src.models.signet_v2 import SignNetV2, count_parameters
from src.models.baseline_model import SingleStreamBaseline
from src.models.bdsl_spoter_baseline import BdSLSPOTERBaseline
from src.data.preprocessing import DataConfig, create_data_loaders
from src.training.trainer import TrainingConfig, SignNetTrainer
from src.evaluation.evaluator import EvaluationConfig, SignNetEvaluator


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_sample_list(file_path: str):
    """Load sample paths from text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def create_word_mapping(samples):
    """Create word to label and label to word mappings."""
    all_words = sorted(set([s["word"] for s in samples]))
    word_to_label = {word: idx for idx, word in enumerate(all_words)}
    label_to_word = {idx: word for idx, word in enumerate(all_words)}
    return word_to_label, label_to_word


def create_emotion_mapping():
    """Create the fixed 5-way expression mapping used by the thesis."""
    emotions = ["neutral", "happy", "sad", "negation", "question"]
    emotion_to_label = {emotion: idx for idx, emotion in enumerate(emotions)}
    label_to_emotion = {idx: emotion for emotion, idx in emotion_to_label.items()}
    return emotion_to_label, label_to_emotion


def create_stratified_holdout(sample_paths: List[str], val_fraction: float = 0.1, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Create a stratified train/validation split grouped by word."""
    grouped_samples: Dict[str, List[str]] = {}
    for sample_path in sample_paths:
        metadata = parse_metadata(sample_path)
        if metadata is None:
            continue
        grouped_samples.setdefault(metadata["word"], []).append(sample_path)

    rng = random.Random(seed)
    train_split: List[str] = []
    val_split: List[str] = []

    for _, group in grouped_samples.items():
        shuffled_group = group[:]
        rng.shuffle(shuffled_group)
        if len(shuffled_group) <= 1:
            train_split.extend(shuffled_group)
            continue

        val_count = max(1, int(round(len(shuffled_group) * val_fraction)))
        val_count = min(val_count, len(shuffled_group) - 1)
        val_split.extend(shuffled_group[:val_count])
        train_split.extend(shuffled_group[val_count:])

    rng.shuffle(train_split)
    rng.shuffle(val_split)
    return train_split, val_split


def evaluate_s11_model(
    model: torch.nn.Module,
    device: torch.device,
    s11_dir: Path,
    word_to_label: Dict[str, int],
    emotion_to_label: Dict[str, int],
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Evaluate a trained model on the S11 .npz test set."""
    test_files = sorted(s11_dir.glob("*.npz"))
    if not test_files:
        raise FileNotFoundError(f"No .npz files found in {s11_dir}")

    samples = []
    unseen_labels = 0
    for file_path in test_files:
        data = np.load(file_path, allow_pickle=True)
        pose = np.asarray(data["pose_sequence"], dtype=np.float32).reshape(150, -1)
        left_hand = np.asarray(data["left_hand"], dtype=np.float32).reshape(150, -1)
        right_hand = np.asarray(data["right_hand"], dtype=np.float32).reshape(150, -1)
        face = np.asarray(data["face"], dtype=np.float32).reshape(150, -1)
        raw_length = int(np.asarray(data["raw_length"]).reshape(()).item())
        attention_mask = np.zeros(pose.shape[0], dtype=np.float32)
        attention_mask[: min(raw_length, pose.shape[0])] = 1.0

        word = file_path.stem.split("__")[0].strip()
        emotion = file_path.stem.split("__")[-1].strip()
        label = word_to_label.get(word)
        if label is None:
            unseen_labels += 1

        samples.append(
            {
                "body_pose": pose,
                "left_hand": left_hand,
                "right_hand": right_hand,
                "face": face,
                "attention_mask": attention_mask,
                "label": label,
                "emotion_label": emotion_to_label.get(emotion, -1),
                "emotion": emotion,
            }
        )

    correct_top1 = 0
    correct_top5 = 0
    total = len(samples)
    emotion_correct_top1 = 0
    emotion_total = 0
    label_to_emotion = {idx: emotion for emotion, idx in emotion_to_label.items()}
    emotion_confusion = np.zeros((len(emotion_to_label), len(emotion_to_label)), dtype=np.int64)
    emotion_class_counts = Counter()
    emotion_class_correct = Counter()
    expression_available = hasattr(model, "emotion_classifier") and getattr(model, "emotion_classifier", None) is not None

    model.eval()
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            batch = samples[start : start + batch_size]
            body = torch.from_numpy(np.stack([item["body_pose"] for item in batch], axis=0)).to(device)
            left = torch.from_numpy(np.stack([item["left_hand"] for item in batch], axis=0)).to(device)
            right = torch.from_numpy(np.stack([item["right_hand"] for item in batch], axis=0)).to(device)
            face = torch.from_numpy(np.stack([item["face"] for item in batch], axis=0)).to(device)
            mask = torch.from_numpy(
                np.stack([item["attention_mask"] for item in batch], axis=0)
            ).to(device)
            labels = [item["label"] for item in batch]
            emotion_labels = [item["emotion_label"] for item in batch]

            if expression_available:
                logits, emotion_logits = model.forward_with_aux(body, left, right, face, mask)
                emotion_probs = torch.softmax(emotion_logits, dim=1)
                emotion_top1 = torch.argmax(emotion_probs, dim=1).tolist()
            else:
                logits = model(body, left, right, face, mask)
                emotion_top1 = [None] * len(batch)

            probs = torch.softmax(logits, dim=1)
            top1 = torch.argmax(probs, dim=1).tolist()
            top5 = torch.topk(probs, k=min(5, probs.size(1)), dim=1).indices.tolist()

            for label, pred, top5_row, emotion_label, emotion_pred in zip(
                labels, top1, top5, emotion_labels, emotion_top1
            ):
                if label is None:
                    continue
                if pred == label:
                    correct_top1 += 1
                if label in top5_row:
                    correct_top5 += 1
                if emotion_label >= 0 and emotion_pred is not None:
                    emotion_total += 1
                    emotion_name = label_to_emotion.get(emotion_label, str(emotion_label))
                    emotion_class_counts[emotion_name] += 1
                    emotion_confusion[emotion_label, emotion_pred] += 1
                    if emotion_pred == emotion_label:
                        emotion_correct_top1 += 1
                        emotion_class_correct[emotion_name] += 1

    return {
        "total": float(total),
        "correct_top1": float(correct_top1),
        "correct_top5": float(correct_top5),
        "top1_accuracy": correct_top1 / total if total else 0.0,
        "top5_accuracy": correct_top5 / total if total else 0.0,
        "unseen_labels": float(unseen_labels),
        "expression_available": expression_available,
        "expression": {
            "total": float(emotion_total),
            "correct_top1": float(emotion_correct_top1),
            "top1_accuracy": emotion_correct_top1 / emotion_total if emotion_total else 0.0,
            "per_class": {
                emotion: {
                    "correct": float(emotion_class_correct.get(emotion, 0)),
                    "total": float(emotion_class_counts.get(emotion, 0)),
                    "accuracy": (
                        emotion_class_correct.get(emotion, 0) / emotion_class_counts.get(emotion, 1)
                        if emotion_class_counts.get(emotion, 0)
                        else 0.0
                    ),
                }
                for emotion in ["neutral", "happy", "sad", "negation", "question"]
            },
            "confusion_matrix": emotion_confusion.tolist(),
        },
    }


def parse_metadata(filename: str):
    """Parse metadata from filename."""
    parts = filename.replace('.npz', '').split("__")
    if len(parts) != 5:
        return None
    word, signer, session, repetition, emotion = [
        unicodedata.normalize("NFC", part).strip() for part in parts
    ]
    return {
        "word": word,
        "signer": signer,
        "session": session,
        "repetition": repetition,
        "emotion": emotion,
        "grammar": emotion,
        "full_path": filename,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SignNet-V2 for BdSL recognition"
    )
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="processed",
        help="Directory containing train/val/test split files and normalized data",
    )
    parser.add_argument(
        "--normalized_dir",
        type=str,
        default="processed/multimodal_6signers_clean",
        help="Directory containing normalized multimodal .npz files",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--num_emotions",
        type=int,
        default=5,
        help="Number of emotion labels for SignNetV2 (0 disables the emotion head)",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=256,
        help="Transformer hidden dimension for SignNetV2",
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=6,
        help="Number of transformer encoder layers for SignNetV2",
    )
    parser.add_argument(
        "--emotion_loss_weight",
        type=float,
        default=1.0,
        help="Relative weight for the emotion objective",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="retrain_with_emotion",
        help="Checkpoint subdirectory name",
    )
    parser.add_argument(
        "--balance_emotions",
        action="store_true",
        default=False,
        help="Randomly downsample majority emotion classes in the training split",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["signetv2", "baseline", "spoter"],
        default="signetv2",
        help="Model architecture to train",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="loso",
        choices=["loso", "random_no_s11"],
        help="loso: leave S11 out. random_no_s11: random split within S02-S12 only, no S11 anywhere.",
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set random seeds
    set_seeds(args.seed)

    # Setup paths
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir

    if args.base_dir == ".":
        base_path = repo_root
    else:
        base_path = Path(args.base_dir).resolve()

    processed_dir = (base_path / args.processed_dir).resolve()
    normalized_dir = (
        (base_path / args.normalized_dir).resolve()
        if args.normalized_dir is not None
        else (processed_dir / "multimodal").resolve()
    )
    checkpoint_dir = processed_dir / "checkpoints" / args.checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print("🚀 SignNet-V2 Training Pipeline - BdSL Recognition")
    print(f"{'=' * 80}")
    print(f"   Working directory: {base_path.absolute()}")
    print(f"   Checkpoint directory: {checkpoint_dir}")
    print(f"   Configuration:")
    print(f"      - Epochs: {args.epochs}")
    print(f"      - Batch size: {args.batch_size}")
    print(f"      - Learning rate: {args.learning_rate}")
    print(f"      - Mixed precision: {args.use_amp}")
    print(f"      - Emotion loss weight: {args.emotion_loss_weight}")
    print(f"      - Normalized dir: {normalized_dir}")
    print(f"      - Split mode: {args.split_mode}")

    if args.split_mode == "random_no_s11":
        all_files = sorted(
            [str(p) for p in normalized_dir.glob("*.npz") if "_S11_" not in p.stem]
        )
        rng = random.Random(args.seed)
        rng.shuffle(all_files)
        total_files = len(all_files)
        test_count = max(1, int(round(total_files * 0.1)))
        val_count = max(1, int(round(total_files * 0.1)))
        train_count = total_files - test_count - val_count
        if train_count <= 0:
            raise ValueError(
                f"Not enough files for random_no_s11 split: {total_files} files available"
            )
        train_samples = all_files[:train_count]
        val_samples = all_files[train_count : train_count + val_count]
        test_samples = all_files[train_count + val_count :]
        print(f"\n📦 Using random_no_s11 split: {train_count} train, {val_count} val, {test_count} test")
    else:
        train_samples = load_sample_list(processed_dir / "train_samples.txt")
        val_samples = load_sample_list(processed_dir / "val_samples.txt")
        test_samples = load_sample_list(processed_dir / "test_samples.txt")

    print(f"\n📊 Dataset splits:")
    print(f"   Train: {len(train_samples)} samples")
    print(f"   Val: {len(val_samples)} samples")
    print(f"   Test: {len(test_samples)} samples")

    # Parse metadata and create word mapping
    train_metadata = [parse_metadata(s) for s in train_samples]
    val_metadata = [parse_metadata(s) for s in val_samples]
    test_metadata = [parse_metadata(s) for s in test_samples]

    train_metadata = [m for m in train_metadata if m is not None]
    val_metadata = [m for m in val_metadata if m is not None]
    test_metadata = [m for m in test_metadata if m is not None]

    if not val_metadata and train_samples:
        print("\n⚖️  Validation split is empty on disk; creating an internal 90/10 stratified holdout from training samples.")
        train_samples, val_samples = create_stratified_holdout(train_samples, val_fraction=0.1, seed=args.seed)
        train_metadata = [parse_metadata(s) for s in train_samples]
        val_metadata = [parse_metadata(s) for s in val_samples]
        train_metadata = [m for m in train_metadata if m is not None]
        val_metadata = [m for m in val_metadata if m is not None]

    all_metadata = train_metadata + val_metadata + test_metadata

    label_mapping_path = processed_dir / "label_mapping_clean.json"
    if not label_mapping_path.exists():
        alt_mapping_path = repo_root / "label_mapping_clean.json"
        if alt_mapping_path.exists():
            label_mapping_path = alt_mapping_path

    if label_mapping_path.exists():
        print(f"\n📦 Loading clean label mapping from: {label_mapping_path}")
        loaded_mapping = json.loads(label_mapping_path.read_text(encoding="utf-8"))
        word_to_label = loaded_mapping["word_to_label"]
        label_to_word = {int(k): v for k, v in loaded_mapping.get("label_to_word", {}).items()}
        emotion_to_label = loaded_mapping["emotion_to_label"]
        label_to_emotion = {int(k): v for k, v in loaded_mapping.get("label_to_emotion", {}).items()}
        num_classes = len(word_to_label)
        num_emotions = len(set(emotion_to_label.values()))
    else:
        word_to_label, label_to_word = create_word_mapping(all_metadata)
        num_classes = len(word_to_label)
        emotion_to_label, label_to_emotion = create_emotion_mapping()
        num_emotions = len(emotion_to_label)

    print(f"\n📚 Classes: {num_classes} unique Bengali words")
    print(f"📚 Expression classes: {num_emotions} labels")

    expression_counts = Counter(m["emotion"] for m in train_metadata)
    print("\n🧾 Training expression distribution:")
    for emotion in ["neutral", "happy", "sad", "negation", "question"]:
        print(f"   {emotion}: {expression_counts.get(emotion, 0)}")
    missing_expressions = [
        emotion
        for emotion in ["neutral", "happy", "sad", "negation", "question"]
        if expression_counts.get(emotion, 0) == 0
    ]
    if missing_expressions:
        print(f"   Missing expressions in training: {', '.join(missing_expressions)}")
    else:
        print("   All 5 expression classes are present in training.")

    if val_metadata:
        val_signers = Counter(m["signer"] for m in val_metadata)
        print("\n🧪 Validation signer distribution:")
        for signer in sorted(val_signers):
            print(f"   {signer}: {val_signers[signer]}")
        if {"S02", "S10"}.issubset(set(val_signers)):
            print("   Validation includes both S02 and S10.")
        else:
            print("   Warning: validation does not include both S02 and S10.")

    # Save label mapping
    label_mapping = {
        "word_to_label": word_to_label,
        "label_to_word": {str(k): v for k, v in label_to_word.items()},
        "emotion_to_label": emotion_to_label,
        "label_to_emotion": {str(k): v for k, v in label_to_emotion.items()},
    }
    with open(checkpoint_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Initialize WandB
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    
    if wandb_api_key:
        try:
            wandb.login(key=wandb_api_key, relogin=True)
            print("✅ W&B authenticated")
        except Exception as e:
            print(f"⚠️  W&B login warning: {e}")
    
    try:
        wandb.init(
            project="bangla-sign-language-recognition",
            name=f"SignNet-V2_{len(train_samples)}samples_{args.epochs}epochs",
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "num_classes": num_classes,
                "seed": args.seed,
                "use_amp": args.use_amp,
            },
        )
        print("✅ W&B initialized")
    except Exception as e:
        print(f"⚠️  W&B initialization warning: {e}")

    # Create data config
    data_config = DataConfig(
        base_dir=str(base_path),
        processed_dir=args.processed_dir,
        normalized_dir=str(normalized_dir),
        checkpoint_dir=str(checkpoint_dir),
        max_seq_length=150,
        augmentation=False,
    )

    # Avoid Windows multiprocessing spawn interruptions in long training runs.
    loader_workers = 0 if os.name == "nt" else 2

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config=data_config,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        word_to_label=word_to_label,
        emotion_to_label=emotion_to_label,
        batch_size=args.batch_size,
        num_workers=loader_workers,
        use_hands=True,
        use_face=True,
        balance_emotions=args.balance_emotions,
    )

    print(f"\n📦 Data loaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Create training config
    training_config = TrainingConfig(
        num_classes=num_classes,
        body_dim=data_config.body_dim,
        hand_dim=data_config.hand_dim,
        face_dim=data_config.face_dim,
        d_model=args.d_model,
        num_encoder_layers=args.num_encoder_layers,
        num_heads=8,
        d_ff=1024,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.1,
        label_smoothing=args.label_smoothing,
        early_stopping_patience=15,
        gradient_clip_norm=0.5,
        gradient_accumulation_steps=1,
        use_amp=args.use_amp,
        mixup_alpha=0.0,
        warmup_epochs=10,
        emotion_loss_weight=args.emotion_loss_weight,
        checkpoint_dir=str(checkpoint_dir),
    )

    # Initialize model
    if args.model_type == "baseline":
        model = SingleStreamBaseline(
            num_classes=num_classes,
            num_emotions=None,
            body_dim=data_config.body_dim,
            hand_dim=data_config.hand_dim,
            face_dim=data_config.face_dim,
            d_model=training_config.d_model,
            num_encoder_layers=training_config.num_encoder_layers,
            num_heads=training_config.num_heads,
            d_ff=training_config.d_ff,
            dropout=training_config.dropout,
            max_seq_length=data_config.max_seq_length,
            use_face=True,
            use_hands=True,
        )
    elif args.model_type == "spoter":
        model = BdSLSPOTERBaseline(
            num_classes=num_classes,
            face_dim=data_config.face_dim,
            dropout=args.dropout,
            max_seq_length=data_config.max_seq_length,
            num_emotions=num_emotions,
        )
    else:
        model = SignNetV2(
            num_classes=num_classes,
            num_emotions=args.num_emotions,
            body_dim=data_config.body_dim,
            hand_dim=data_config.hand_dim,
            face_dim=data_config.face_dim,
            d_model=training_config.d_model,
            num_encoder_layers=training_config.num_encoder_layers,
            num_heads=training_config.num_heads,
            d_ff=training_config.d_ff,
            dropout=training_config.dropout,
            max_seq_length=data_config.max_seq_length,
            use_face=True,
            use_hands=True,
        )

    print("\n🔧 MODEL CONFIGURATION:")
    print(f"   d_model: {training_config.d_model}")
    print(f"   num_encoder_layers: {training_config.num_encoder_layers}")
    print(f"   d_ff: {training_config.d_ff}")
    print(f"   dropout: {training_config.dropout}")
    print(f"   emotion_loss_weight: {training_config.emotion_loss_weight}")

    # Count parameters
    params = count_parameters(model)
    print(f"\n🧠 Model: {model.__class__.__name__}")
    print(f"   Total parameters: {params['total']:,}")
    print(f"   Trainable parameters: {params['trainable']:,}")
    print(f"   Model size: {params['total'] * 4 / 1024**2:.2f} MB")

    # Watch model with WandB
    try:
        wandb.watch(model, log_freq=100)
        wandb.log({"model/total_parameters": params['total']})
    except:
        pass

    # Setup trainer
    trainer = SignNetTrainer(
        config=training_config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    # Test forward pass
    print(f"\n🧪 Testing forward pass...")
    test_input = torch.randn(2, data_config.max_seq_length, data_config.body_dim).to(device)
    test_left = torch.randn(2, data_config.max_seq_length, data_config.hand_dim).to(device)
    test_right = torch.randn(2, data_config.max_seq_length, data_config.hand_dim).to(device)
    test_face = torch.randn(2, data_config.max_seq_length, 478 * 3).to(device)
    test_mask = torch.ones(2, data_config.max_seq_length).to(device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.cuda.amp.autocast():
                test_output = model(test_input, test_left, test_right, test_face, test_mask)
        else:
            test_output = model(test_input, test_left, test_right, test_face, test_mask)

    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {test_output.shape}")
    print(f"   ✅ Forward pass successful!")

    # Train model
    print(f"\n{'=' * 80}")
    print("🏋️  STARTING TRAINING")
    print(f"{'=' * 80}\n")
    
    history = trainer.train(start_epoch=0)

    # Save training history
    with open(checkpoint_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Reload the best checkpoint before evaluation.
    checkpoint_to_load = checkpoint_dir / "best_model.pth"
    if not checkpoint_to_load.exists():
        checkpoint_to_load = checkpoint_dir / "latest_checkpoint.pth"
    trainer.load_checkpoint(checkpoint_to_load)

    results = None
    # Evaluate best model on the held-out test split when one exists.
    if len(test_samples) > 0:
        print(f"\n{'=' * 80}")
        print("📊 EVALUATING BEST MODEL")
        print(f"{'=' * 80}\n")

        eval_config = EvaluationConfig(
            checkpoint_dir=str(checkpoint_dir), 
            num_classes=num_classes
        )

        evaluator = SignNetEvaluator(
            model=model,
            test_loader=test_loader,
            device=device,
            label_to_word=label_to_word,
            config=eval_config,
        )

        results = evaluator.evaluate()
        evaluator.print_results(results)
        evaluator.save_results(results)
        evaluator.generate_visualizations(results)
    else:
        print(f"\n{'=' * 80}")
        print("📊 EVALUATING BEST MODEL")
        print(f"{'=' * 80}\n")
        print("   Skipping held-out test evaluation because test_samples.txt is empty.")

    # Evaluate on the unseen S11 signer set.
    print(f"\n{'=' * 80}")
    print("📊 EVALUATING ON S11 (UNSEEN SIGNER)")
    print(f"{'=' * 80}\n")

    s11_dir = processed_dir / "s11_full"
    s11_results = evaluate_s11_model(
        model=model,
        device=device,
        s11_dir=s11_dir,
        word_to_label=word_to_label,
        emotion_to_label=emotion_to_label,
        batch_size=args.batch_size,
    )

    print("Word Recognition:")
    print(f"   Total samples: {int(s11_results['total'])}")
    print(f"   Top-1 Accuracy: {s11_results['top1_accuracy']:.4f} ({s11_results['top1_accuracy'] * 100:.2f}%)")
    print(f"   Top-5 Accuracy: {s11_results['top5_accuracy']:.4f} ({s11_results['top5_accuracy'] * 100:.2f}%)")
    print(f"   Unseen labels in S11: {int(s11_results['unseen_labels'])}")

    expression_results = s11_results["expression"]
    print("\nExpression Recognition:")
    if s11_results.get("expression_available", False):
        print(f"   Total samples: {int(expression_results['total'])}")
        print(f"   Top-1 Accuracy: {expression_results['top1_accuracy']:.4f} ({expression_results['top1_accuracy'] * 100:.2f}%)")
        for emotion in ["neutral", "happy", "sad", "negation", "question"]:
            stats = expression_results["per_class"][emotion]
            print(
                f"   {emotion}: {stats['accuracy']:.4f} ({stats['accuracy'] * 100:.2f}%) "
                f"[{int(stats['correct'])}/{int(stats['total'])}]"
            )
        print("   Confusion matrix (rows=true, cols=pred):")
        for row in expression_results["confusion_matrix"]:
            print(f"      {row}")
    else:
        print("   Skipped (model has no emotion head).")

    with open(checkpoint_dir / "s11_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(s11_results, f, indent=2)

    # Save final model
    torch.save(model.state_dict(), checkpoint_dir / "final_model.pth")

    # Log final metrics to WandB
    try:
        log_payload = {
            "s11/top1_accuracy": s11_results["top1_accuracy"],
            "s11/top5_accuracy": s11_results["top5_accuracy"],
            "s11/expression_top1_accuracy": expression_results["top1_accuracy"],
        }
        if results is not None:
            log_payload.update({
                "test/accuracy": results["test_accuracy"],
                "test/precision": results["test_precision"],
                "test/recall": results["test_recall"],
                "test/f1_score": results["test_f1"],
                "test/top5_accuracy": results["top_5_accuracy"],
            })
        wandb.log(log_payload)
        wandb.finish()
    except:
        pass

    # Print final summary
    print(f"\n{'=' * 80}")
    print("🎉 TRAINING COMPLETE - FINAL SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n📁 Output Directory: {checkpoint_dir}")
    if results is not None:
        print(f"\n📊 Test Results:")
        print(f"   Top-1 Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy'] * 100:.2f}%)")
        print(f"   Top-5 Accuracy: {results['top_5_accuracy']:.4f} ({results['top_5_accuracy'] * 100:.2f}%)")
        print(f"   Precision: {results['test_precision']:.4f}")
        print(f"   Recall: {results['test_recall']:.4f}")
        print(f"   F1-Score: {results['test_f1']:.4f}")
    else:
        print(f"\n📊 Test Results:")
        print("   Skipped (empty local test split)")
    print(f"\n📊 S11 Results:")
    print(
        f"   Top-1 Accuracy: {s11_results['top1_accuracy']:.4f} ({s11_results['top1_accuracy'] * 100:.2f}%)"
    )
    print(
        f"   Top-5 Accuracy: {s11_results['top5_accuracy']:.4f} ({s11_results['top5_accuracy'] * 100:.2f}%)"
    )
    print(f"   Expression Top-1 Accuracy: {expression_results['top1_accuracy']:.4f} ({expression_results['top1_accuracy'] * 100:.2f}%)")
    print(f"\n🧠 Model Information:")
    print(f"   Architecture: SignNet-V2")
    print(f"   Total Parameters: {params['total']:,}")
    print(f"   Model Size: {params['total'] * 4 / 1024**2:.2f} MB")
    print(f"   Best Val Accuracy: {trainer.best_val_acc:.4f} ({trainer.best_val_acc * 100:.2f}%)")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
