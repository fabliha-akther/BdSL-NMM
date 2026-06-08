from __future__ import annotations

import argparse
import json
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT / "code" / "BdSL-SignNet"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import DataConfig, SignLanguageDataset
from src.models.signet_v2 import SignNetV2

DEFAULT_CHECKPOINT_DIR = Path(
    r"C:/Users/T2430397/Documents/bangla-sign-language-recognition-main/new model/BdSL-Enhanced-SignNet/processed/checkpoints/retrain_with_emotion"
)
DEFAULT_TEST_DIR = Path(
    r"C:/Users/T2430397/Downloads/BdSL-Thesis/code/BdSL-SignNet/processed/multimodal_6signers_clean"
)
DEFAULT_LABEL_MAPPING = Path(
    r"C:/Users/T2430397/Downloads/BdSL-Thesis/code/BdSL-SignNet/processed/label_mapping_clean.json"
)
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_FINAL_MODEL_DIR = ROOT / "models" / "checkpoints" / "final_model"
DEFAULT_BATCH_SIZE = 16
TARGET_FACE_DIM = 1434
EMOTION_NAMES = ["neutral", "happy", "sad", "negation", "question"]


def normalize_word(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_checkpoint_state(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def infer_num_classes_from_state_dict(state_dict: Dict[str, Any]) -> int:
    candidate_keys = [
        "classifier.classifier.7.weight",
        "classifier.weight",
        "fc.weight",
    ]
    for key in candidate_keys:
        tensor = state_dict.get(key)
        if tensor is not None and hasattr(tensor, "shape") and len(tensor.shape) >= 1:
            return int(tensor.shape[0])
    # Fallback for baseline: find output layer = smallest classifier output dim
    candidate_sizes = []
    for key in state_dict.keys():
        if 'classifier' in key and key.endswith('.weight'):
            shape = state_dict[key].shape
            if len(shape) == 2 and shape[0] < 200:
                candidate_sizes.append(shape[0])
    if candidate_sizes:
        return min(candidate_sizes)
    raise KeyError("Could not infer num_classes from checkpoint state_dict")


def load_label_mappings(mapping_path: Path) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]]:
    mapping = load_json(mapping_path)

    if "word_to_label" in mapping:
        word_to_label = {normalize_word(str(key)): int(value) for key, value in mapping["word_to_label"].items()}
    elif "label_to_word" in mapping:
        word_to_label = {
            normalize_word(str(value)): int(key)
            for key, value in mapping["label_to_word"].items()
        }
    else:
        raise KeyError("label_mapping.json does not contain word mappings")

    if "label_to_word" in mapping:
        label_to_word = {int(key): normalize_word(str(value)) for key, value in mapping["label_to_word"].items()}
    else:
        label_to_word = {index: word for word, index in word_to_label.items()}

    if "emotion_to_label" in mapping:
        emotion_to_label = {normalize_word(str(key)): int(value) for key, value in mapping["emotion_to_label"].items()}
    else:
        emotion_to_label = {emotion: index for index, emotion in enumerate(EMOTION_NAMES)}

    if "label_to_emotion" in mapping:
        label_to_emotion = {
            int(key): normalize_word(str(value)) for key, value in mapping["label_to_emotion"].items()
        }
    else:
        label_to_emotion = {index: emotion for emotion, index in emotion_to_label.items()}

    return word_to_label, label_to_word, emotion_to_label, label_to_emotion


def build_model(num_classes: int, num_emotions: int) -> SignNetV2:
    return SignNetV2(
        num_classes=num_classes,
        num_emotions=num_emotions,
        body_dim=99,
        hand_dim=63,
        face_dim=1434,
        d_model=256,
        num_encoder_layers=6,
        num_heads=8,
        d_ff=1024,
        dropout=0.2,
        max_seq_length=150,
        use_face=True,
        use_hands=True,
    )


def load_model_checkpoint(model: SignNetV2, state_dict: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    model_state = model.state_dict()
    filtered_state: Dict[str, Any] = {}
    skipped_keys: List[str] = []

    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            filtered_state[key] = value
        else:
            skipped_keys.append(key)

    merged_state = dict(model_state)
    merged_state.update(filtered_state)
    model.load_state_dict(merged_state, strict=True)

    missing_keys = [key for key in model_state.keys() if key not in filtered_state]
    unexpected_keys = skipped_keys

    if skipped_keys:
        preview = skipped_keys[:3]
        suffix = "..." if len(skipped_keys) > 3 else ""
        print(
            f"[eval] Skipped {len(skipped_keys)} checkpoint keys due to mismatch/absence: {preview}{suffix}"
        )

    return {
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
        "checkpoint_keys": list(state_dict.keys()) if isinstance(state_dict, dict) else [],
    }


def patch_mean_pooling(model: SignNetV2) -> None:
    def _mean_pool_face_features(
        self: SignNetV2,
        face_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if face_features is None:
            return None

        if attention_mask is None:
            return face_features.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).to(dtype=face_features.dtype)
        pooled = (face_features * mask).sum(dim=1)
        normalizer = mask.sum(dim=1).clamp_min(1.0)
        return pooled / normalizer

    model._pool_face_features = MethodType(_mean_pool_face_features, model)


def build_dataset(test_dir: Path, normalized_dir: Path, word_to_label: Dict[str, int], emotion_to_label: Dict[str, int]) -> SignLanguageDataset:
    sample_paths = sorted(
        str(path)
        for path in test_dir.glob("*.npz")
        if "__S11__" in path.stem
    )
    if not sample_paths:
        raise FileNotFoundError(f"No S11 .npz files found in {test_dir}")

    safe_word_to_label = defaultdict(lambda: -1)
    safe_word_to_label.update(word_to_label)

    config = DataConfig(
        processed_dir=str(ROOT / "code" / "BdSL-SignNet" / "processed"),
        normalized_dir=str(normalized_dir),
        checkpoint_dir=str(normalized_dir),
        max_seq_length=150,
        min_seq_length=10,
        target_fps=30,
        body_dim=99,
        hand_dim=63,
        face_dim=1434,
        augmentation=False,
        loader_error_mode="strict",
    )

    return SignLanguageDataset(
        sample_paths=sample_paths,
        word_to_label=safe_word_to_label,
        normalized_dir=str(normalized_dir),
        config=config,
        augment=False,
        mode="test",
        use_hands=True,
        use_face=True,
        emotion_to_label=emotion_to_label,
    )


def evaluate_model(
    model: SignNetV2,
    dataloader: DataLoader,
    device: torch.device,
    label_to_word: Dict[int, str],
    label_to_emotion: Dict[int, str],
) -> Dict[str, Any]:
    model.eval()

    total_samples = 0
    load_errors = 0
    unseen_labels = 0
    word_correct_top1 = 0
    word_correct_top5 = 0
    emotion_correct_top1 = 0
    emotion_total = 0

    word_class_counts: Counter[int] = Counter()
    word_class_correct: Counter[int] = Counter()
    emotion_class_counts: Counter[str] = Counter()
    emotion_class_correct: Counter[str] = Counter()
    emotion_confusion = np.zeros((len(EMOTION_NAMES), len(EMOTION_NAMES)), dtype=np.int64)
    has_emotion_head = False

    with torch.inference_mode():
        for batch in dataloader:
            body_pose = batch["body_pose"].to(device)
            left_hand = batch["left_hand"].to(device) if "left_hand" in batch else None
            right_hand = batch["right_hand"].to(device) if "right_hand" in batch else None
            face = batch["face"].to(device) if "face" in batch else None
            if face is not None and face.size(-1) < TARGET_FACE_DIM:
                pad_width = TARGET_FACE_DIM - face.size(-1)
                face = torch.nn.functional.pad(face, (0, pad_width))
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).long().view(-1)
            emotion_labels = batch["emotion_label"].to(device).long().view(-1)

            load_failed = batch["load_failed"].to(device).long().view(-1)
            load_errors += int(load_failed.sum().item())

            logits, emotion_logits = model.forward_with_aux(
                body_pose,
                left_hand,
                right_hand,
                face,
                attention_mask,
            )

            if emotion_logits is not None:
                has_emotion_head = True

            probs = torch.softmax(logits, dim=1)
            top1 = torch.argmax(probs, dim=1)
            top5 = torch.topk(probs, k=min(5, probs.size(1)), dim=1).indices
            emotion_top1 = None
            if emotion_logits is not None:
                emotion_probs = torch.softmax(emotion_logits, dim=1)
                emotion_top1 = torch.argmax(emotion_probs, dim=1)

            batch_size = labels.size(0)
            total_samples += batch_size

            for index in range(batch_size):
                label = int(labels[index].item())
                emotion_label = int(emotion_labels[index].item())
                pred = int(top1[index].item())
                top5_row = top5[index].tolist()

                if label not in label_to_word:
                    unseen_labels += 1
                    continue

                word_class_counts[label] += 1
                if pred == label:
                    word_correct_top1 += 1
                    word_class_correct[label] += 1
                if label in top5_row:
                    word_correct_top5 += 1

                if emotion_logits is not None and emotion_label >= 0:
                    emotion_pred = int(emotion_top1[index].item())
                    emotion_total += 1
                    emotion_name = label_to_emotion.get(emotion_label, str(emotion_label))
                    emotion_pred_name = label_to_emotion.get(emotion_pred, str(emotion_pred))
                    emotion_class_counts[emotion_name] += 1
                    emotion_confusion[emotion_label, emotion_pred] += 1
                    if emotion_pred == emotion_label:
                        emotion_correct_top1 += 1
                        emotion_class_correct[emotion_name] += 1

    word_per_class = {
        label_to_word[label]: {
            "correct": int(word_class_correct.get(label, 0)),
            "total": int(word_class_counts.get(label, 0)),
            "accuracy": (
                word_class_correct.get(label, 0) / word_class_counts.get(label, 1)
                if word_class_counts.get(label, 0)
                else 0.0
            ),
        }
        for label in sorted(label_to_word)
    }

    expression_per_class = {
        emotion: {
            "correct": int(emotion_class_correct.get(emotion, 0)),
            "total": int(emotion_class_counts.get(emotion, 0)),
            "accuracy": (
                emotion_class_correct.get(emotion, 0) / emotion_class_counts.get(emotion, 1)
                if emotion_class_counts.get(emotion, 0)
                else 0.0
            ),
        }
        for emotion in EMOTION_NAMES
    }

    word_summary = {
        "total": int(total_samples),
        "errors": int(load_errors),
        "unseen_labels": int(unseen_labels),
        "word": {
            "correct_top1": int(word_correct_top1),
            "correct_top5": int(word_correct_top5),
            "top1_accuracy": word_correct_top1 / total_samples if total_samples else 0.0,
            "top5_accuracy": word_correct_top5 / total_samples if total_samples else 0.0,
        },
        "word_per_class": word_per_class,
        "expression": {
            "total": int(emotion_total),
            "correct_top1": int(emotion_correct_top1),
            "top1_accuracy": emotion_correct_top1 / emotion_total if emotion_total else 0.0,
            "per_class": expression_per_class,
            "confusion_matrix": emotion_confusion.tolist(),
        },
    }

    expression_summary = {
        "word_top1": word_summary["word"]["top1_accuracy"],
        "word_top5": word_summary["word"]["top5_accuracy"],
        "expr_top1": word_summary["expression"]["top1_accuracy"],
        "per_class": expression_per_class,
        "confusion_matrix": emotion_confusion.tolist(),
        "total": int(emotion_total),
        "errors": int(load_errors),
    }

    return {
        "word_summary": word_summary,
        "expression_summary": expression_summary,
        "has_emotion_head": has_emotion_head,
    }


def save_results(results: Dict[str, Any], results_root: Path, checkpoint_name: str, model_type: str) -> None:
    word_dir = results_root / "word_recognition"
    expr_dir = results_root / "expression_recognition"
    word_dir.mkdir(parents=True, exist_ok=True)
    expr_dir.mkdir(parents=True, exist_ok=True)

    artifact_name = f"{checkpoint_name}_s11_{model_type}.json"
    word_path = word_dir / artifact_name
    expr_path = expr_dir / artifact_name

    with word_path.open("w", encoding="utf-8") as handle:
        json.dump(results["word_summary"], handle, ensure_ascii=False, indent=2)

    with expr_path.open("w", encoding="utf-8") as handle:
        json.dump(results["expression_summary"], handle, ensure_ascii=False, indent=2)

    print(f"✅ Saved word results to {word_path}")
    print(f"✅ Saved expression results to {expr_path}")


def save_checkpoint_evaluation(
    results: Dict[str, Any],
    checkpoint_dir: Path,
    model_type: str,
) -> None:
    payload = {
        "model_type": model_type,
        "word_summary": results.get("word_summary", {}),
        "expression_summary": results.get("expression_summary", {}),
        "has_emotion_head": bool(results.get("has_emotion_head", False)),
    }
    eval_path = checkpoint_dir / "s11_evaluation.json"
    with eval_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"✅ Saved checkpoint evaluation to {eval_path}")


def copy_final_artifacts(checkpoint_dir: Path, final_model_dir: Path) -> None:
    final_model_dir.mkdir(parents=True, exist_ok=True)
    src_model = checkpoint_dir / "best_model.pth"
    src_mapping = checkpoint_dir / "label_mapping.json"
    if not src_mapping.exists():
        src_mapping = ROOT / "code" / "BdSL-SignNet" / "processed" / "label_mapping_clean.json"
    dst_model = final_model_dir / "best_model.pth"
    dst_mapping = final_model_dir / "label_mapping.json"

    if src_model.resolve() != dst_model.resolve():
        shutil.copy2(src_model, dst_model)
    if src_mapping.resolve() != dst_mapping.resolve():
        shutil.copy2(src_mapping, dst_mapping)

    print(f"✅ Copied best_model.pth and label_mapping.json to {final_model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the final BdSL model on S11")
    parser.add_argument("--checkpoint_dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        '--model_type',
        type=str,
        default='signetv2',
        choices=['signetv2', 'baseline', 'spoter'],
        help='Model architecture to evaluate'
    )
    parser.add_argument("--test_dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--final_model_dir", type=Path, default=DEFAULT_FINAL_MODEL_DIR)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    test_dir = args.test_dir
    results_root = args.results_root
    final_model_dir = args.final_model_dir

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoint_path = checkpoint_dir / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_state = load_checkpoint_state(checkpoint_path, device)
    state_dict = checkpoint_state.get(
        "model_state_dict",
        checkpoint_state.get("state_dict", checkpoint_state),
    )

    # Read architecture config saved during training
    ckpt_config = checkpoint_state.get("config", {})
    if isinstance(ckpt_config, dict):
        d_model = ckpt_config.get("d_model", None)
        num_encoder_layers = ckpt_config.get("num_encoder_layers", None)
        d_ff = ckpt_config.get("d_ff", None)
        dropout = ckpt_config.get("dropout", 0.2)
        face_dim = ckpt_config.get("face_dim", 1434)
    else:
        d_model = None

    # If config not found, infer d_model directly from weight shapes
    if d_model is None:
        probe_key = "body_encoder.input_projection.0.weight"
        if probe_key in state_dict:
            d_model = state_dict[probe_key].shape[0]
        else:
            d_model = 256  # hard fallback
        num_encoder_layers = 6
        d_ff = max(512, d_model * 4)
        dropout = 0.2
        face_dim = 1434

    print(f"[eval] Loaded config: d_model={d_model}, num_encoder_layers={num_encoder_layers}, d_ff={d_ff}")

    num_classes = infer_num_classes_from_state_dict(state_dict)
    cls_key = "classifier.classifier.7.weight"
    if cls_key in state_dict:
        num_classes_from_ckpt = state_dict[cls_key].shape[0]
        print(f"[eval] Checkpoint num_classes={num_classes_from_ckpt}")
        num_classes = num_classes_from_ckpt
    elif args.model_type == 'spoter' and "classifier.7.weight" in state_dict:
        num_classes_from_ckpt = state_dict["classifier.7.weight"].shape[0]
        print(f"[eval] Checkpoint num_classes={num_classes_from_ckpt}")
        num_classes = num_classes_from_ckpt
    word_to_label, label_to_word, emotion_to_label, label_to_emotion = load_label_mappings(DEFAULT_LABEL_MAPPING)
    # Enforce explicit 5-way emotion mapping and remove any 'affective' entries
    emotion_to_label = {"neutral": 0, "happy": 1, "sad": 2, "negation": 3, "question": 4}
    label_to_emotion = {0: "neutral", 1: "happy", 2: "sad", 3: "negation", 4: "question"}

    num_classes = len(word_to_label)
    num_emotions = 5

    dataset = build_dataset(test_dir, test_dir, word_to_label, emotion_to_label)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.model_type == 'baseline':
        from src.models.baseline_model import SingleStreamBaseline
        model = SingleStreamBaseline(
            num_classes=num_classes,
            body_dim=99,
            hand_dim=63,
            face_dim=face_dim,
            d_model=d_model,
            num_encoder_layers=num_encoder_layers,
            num_heads=8,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_length=150,
            use_face=True,
            use_hands=True,
        )
    elif args.model_type == 'spoter':
        from src.models.bdsl_spoter_baseline import BdSLSPOTERBaseline
        model = BdSLSPOTERBaseline(
            num_classes=num_classes,
            face_dim=face_dim,
        )
        print("Using model: BdSLSPOTERBaseline")
    else:
        model = SignNetV2(
            num_classes=num_classes,
            num_emotions=num_emotions,
            d_model=d_model,
            num_encoder_layers=num_encoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            face_dim=face_dim,
            body_dim=99,
            hand_dim=63,
            max_seq_length=150,
            use_face=True,
            use_hands=True,
        )
    load_info = load_model_checkpoint(model, state_dict, device)
    model.to(device)

    print(f"Using model: {model.__class__.__name__}")

    checkpoint_keys = load_info.get("checkpoint_keys", [])
    print("Using 8D z-score emotion head (no mean pooling compatibility fallback).")

    if load_info["missing_keys"]:
        print(f"Missing keys: {load_info['missing_keys']}")
    if load_info["unexpected_keys"]:
        print(f"Unexpected keys: {load_info['unexpected_keys']}")

    results = evaluate_model(model, dataloader, device, label_to_word, label_to_emotion)

    word_summary = results["word_summary"]
    expr_summary = results["expression_summary"]
    has_emotion_head = results.get("has_emotion_head", False)
    print("\nFinal S11 results")
    print(f"Word top-1: {word_summary['word']['top1_accuracy']:.4f}")
    print(f"Word top-5: {word_summary['word']['top5_accuracy']:.4f}")
    if has_emotion_head:
        print(f"Expression top-1: {word_summary['expression']['top1_accuracy']:.4f}")
        for emotion in EMOTION_NAMES:
            print(f"{emotion}: {word_summary['expression']['per_class'][emotion]['accuracy']:.4f}")
    else:
        print("Expression top-1: N/A (no emotion head)")

    save_results(results, results_root, checkpoint_dir.name, args.model_type)
    save_checkpoint_evaluation(results, checkpoint_dir, args.model_type)
    copy_final_artifacts(checkpoint_dir, final_model_dir)


if __name__ == "__main__":
    main()
