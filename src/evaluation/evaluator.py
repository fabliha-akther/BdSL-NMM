"""
Evaluation Suite for Sign Language Recognition Models
======================================================

Comprehensive evaluation including:
- Standard metrics (accuracy, precision, recall, F1)
- Per-class analysis
- Confusion matrix visualization
- Per-signer analysis
- Confidence intervals
- Comparative analysis with baseline

Author: BDSL Recognition Team
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from collections import defaultdict
import json


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    checkpoint_dir: str = "Data/processed/new_model/checkpoints"
    output_dir: str = "Data/processed/new_model/evaluation"
    num_classes: int = 72
    confidence_level: float = 0.95


class SignNetEvaluator:
    """Comprehensive evaluator for SignNet models."""

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        label_to_word: Dict[int, str],
        config: EvaluationConfig = None,
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained model
            test_loader: Test data loader
            device: Device for inference
            label_to_word: Label to word mapping
            config: Evaluation configuration
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.label_to_word = label_to_word
        self.config = config or EvaluationConfig()

        self.output_dir = Path(self.config.checkpoint_dir) / "evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Evaluation results
        self.all_predictions = []
        self.all_labels = []
        self.all_words = []
        self.all_signers = []
        self.all_grammars = []
        self.all_probs = []

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on test set.

        Returns:
            Dictionary containing all evaluation results
        """
        self.model.eval()

        all_predictions = []
        all_labels = []
        all_words = []
        all_signers = []
        all_probs = []

        print("🔍 Evaluating on test set...")

        for batch in self.test_loader:
            # Unpack batch
            body_pose = batch["body_pose"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device).long().view(-1)

            left_hand = batch.get("left_hand")
            right_hand = batch.get("right_hand")
            face = batch.get("face")

            if left_hand is not None:
                left_hand = left_hand.to(self.device)
            if right_hand is not None:
                right_hand = right_hand.to(self.device)
            if face is not None:
                face = face.to(self.device)

            # Forward pass
            logits = self.model(body_pose, left_hand, right_hand, face, attention_mask)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_words.extend(batch["word"])
            all_signers.extend(batch["signer"])
            all_probs.extend(probs.cpu().numpy())

        self.all_predictions = np.array(all_predictions)
        self.all_labels = np.array(all_labels)
        self.all_words = all_words
        self.all_signers = all_signers
        self.all_probs = np.array(all_probs)

        print(f"✅ Test evaluation complete: {len(all_predictions)} samples")

        # Calculate all metrics
        results = {}
        results.update(self._calculate_overall_metrics())
        results.update(self._calculate_top_k_accuracy())
        results.update(self._calculate_per_signer_metrics())
        results.update(self._calculate_per_class_metrics())
        results.update(self._calculate_confidence_intervals())

        return results

    def _calculate_overall_metrics(self) -> Dict[str, float]:
        """Calculate overall classification metrics."""
        metrics = {}

        metrics["test_accuracy"] = accuracy_score(self.all_labels, self.all_predictions)
        metrics["test_precision"] = precision_score(
            self.all_labels, self.all_predictions, average="macro", zero_division=0
        )
        metrics["test_recall"] = recall_score(
            self.all_labels, self.all_predictions, average="macro", zero_division=0
        )
        metrics["test_f1"] = f1_score(
            self.all_labels, self.all_predictions, average="macro", zero_division=0
        )

        # Per-class precision, recall, F1
        metrics["test_precision_weighted"] = precision_score(
            self.all_labels, self.all_predictions, average="weighted", zero_division=0
        )
        metrics["test_recall_weighted"] = recall_score(
            self.all_labels, self.all_predictions, average="weighted", zero_division=0
        )
        metrics["test_f1_weighted"] = f1_score(
            self.all_labels, self.all_predictions, average="weighted", zero_division=0
        )

        return metrics

    def _calculate_top_k_accuracy(self) -> Dict[str, float]:
        """Calculate top-k accuracy."""
        metrics = {}

        for k in [1, 3, 5, 10]:
            top_k_preds = np.argsort(self.all_probs, axis=1)[:, -k:]
            correct = sum(
                1
                for label, preds in zip(self.all_labels, top_k_preds)
                if label in preds
            )
            metrics[f"top_{k}_accuracy"] = correct / len(self.all_labels)

        return metrics

    def _calculate_per_signer_metrics(self) -> Dict[str, Any]:
        """Calculate metrics per signer."""
        signer_metrics = {}

        signers = set(self.all_signers)

        for signer in signers:
            indices = [i for i, s in enumerate(self.all_signers) if s == signer]

            if indices:
                labels = [self.all_labels[i] for i in indices]
                preds = [self.all_predictions[i] for i in indices]

                signer_metrics[f"signer_{signer}_accuracy"] = accuracy_score(
                    labels, preds
                )
                signer_metrics[f"signer_{signer}_count"] = len(indices)

        return signer_metrics

    def _calculate_per_class_metrics(self) -> Dict[str, Any]:
        """Calculate per-class metrics."""
        class_metrics = {}

        # Count correct predictions per class
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        for true_label, pred_label in zip(self.all_labels, self.all_predictions):
            class_total[true_label] += 1
            if true_label == pred_label:
                class_correct[true_label] += 1

        # Calculate accuracy per class
        for label_id in range(self.config.num_classes):
            if class_total[label_id] > 0:
                class_metrics[f"class_{label_id}_accuracy"] = (
                    class_correct[label_id] / class_total[label_id]
                )
                class_metrics[f"class_{label_id}_word"] = self.label_to_word.get(
                    label_id, f"unknown_{label_id}"
                )
                class_metrics[f"class_{label_id}_correct"] = class_correct[label_id]
                class_metrics[f"class_{label_id}_total"] = class_total[label_id]
            else:
                class_metrics[f"class_{label_id}_accuracy"] = 0.0

        # Create DataFrame for analysis
        class_df = pd.DataFrame(
            [
                {
                    "label_id": i,
                    "word": self.label_to_word.get(i, f"unknown_{i}"),
                    "accuracy": class_metrics.get(f"class_{i}_accuracy", 0.0),
                    "correct": class_metrics.get(f"class_{i}_correct", 0),
                    "total": class_metrics.get(f"class_{i}_total", 0),
                }
                for i in range(self.config.num_classes)
            ]
        )

        class_metrics["class_df"] = class_df
        class_metrics["worst_classes"] = class_df.nsmallest(5, "accuracy")
        class_metrics["best_classes"] = class_df.nlargest(5, "accuracy")
        class_metrics["mean_class_accuracy"] = class_df["accuracy"].mean()
        class_metrics["std_class_accuracy"] = class_df["accuracy"].std()

        return class_metrics

    def _calculate_confidence_intervals(self) -> Dict[str, float]:
        """Calculate confidence intervals for metrics."""
        n = len(self.all_labels)
        z = 1.96  # 95% confidence level

        # Accuracy CI
        acc = accuracy_score(self.all_labels, self.all_predictions)
        std = np.sqrt(acc * (1 - acc) / n)

        ci_metrics = {
            "accuracy": acc,
            "accuracy_ci_lower": max(0.0, acc - z * std),
            "accuracy_ci_upper": min(1.0, acc + z * std),
            "num_samples": n,
        }

        return ci_metrics

    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results."""
        print("\n" + "=" * 70)
        print("📊 EVALUATION RESULTS")
        print("=" * 70)

        print(f"\n🎯 Overall Metrics:")
        print(
            f"   Top-1 Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy'] * 100:.2f}%)"
        )
        print(
            f"   Top-3 Accuracy: {results['top_3_accuracy']:.4f} ({results['top_3_accuracy'] * 100:.2f}%)"
        )
        print(
            f"   Top-5 Accuracy: {results['top_5_accuracy']:.4f} ({results['top_5_accuracy'] * 100:.2f}%)"
        )
        print(
            f"   Top-10 Accuracy: {results['top_10_accuracy']:.4f} ({results['top_10_accuracy'] * 100:.2f}%)"
        )
        print(f"   Precision (macro): {results['test_precision']:.4f}")
        print(f"   Recall (macro): {results['test_recall']:.4f}")
        print(f"   F1-Score (macro): {results['test_f1']:.4f}")

        print(f"\n📈 Confidence Interval (95%):")
        print(
            f"   Accuracy: {results['accuracy']:.4f} [{results['accuracy_ci_lower']:.4f}, {results['accuracy_ci_upper']:.4f}]"
        )
        print(f"   Samples: {results['num_samples']}")

        print(f"\n👥 Per-Signer Performance:")
        for key, value in results.items():
            if key.startswith("signer_") and key.endswith("_accuracy"):
                signer = key.split("_")[1]
                count_key = f"signer_{signer}_count"
                count = results.get(count_key, 0)
                print(
                    f"   {signer}: {value:.4f} ({value * 100:.2f}%) - {count} samples"
                )

        print(f"\n📚 Per-Class Statistics:")
        print(f"   Mean class accuracy: {results['mean_class_accuracy']:.4f}")
        print(f"   Std class accuracy: {results['std_class_accuracy']:.4f}")

        print(f"\n❌ Worst 5 performing classes:")
        worst = results["worst_classes"]
        for _, row in worst.iterrows():
            print(
                f"   {row['word']}: {row['accuracy']:.4f} ({row['correct']}/{row['total']})"
            )

        print(f"\n✅ Best 5 performing classes:")
        best = results["best_classes"]
        for _, row in best.iterrows():
            print(
                f"   {row['word']}: {row['accuracy']:.4f} ({row['correct']}/{row['total']})"
            )

        print("=" * 70)

    def save_results(
        self, results: Dict[str, Any], filename: str = "evaluation_results.json"
    ):
        """Save evaluation results to JSON."""
        # Remove non-serializable items
        serializable = {
            k: v
            for k, v in results.items()
            if not isinstance(v, (pd.DataFrame, nn.Module))
        }

        # Convert class_df to dict
        if "class_df" in results:
            serializable["class_df"] = results["class_df"].to_dict("records")

        with open(self.output_dir / filename, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        print(f"✅ Results saved to {self.output_dir / filename}")

    def generate_visualizations(self, results: Dict[str, Any]):
        """Generate evaluation visualizations."""
        plt.style.use("default")

        # 1. Confusion Matrix
        self._plot_confusion_matrix()

        # 2. Per-signer accuracy
        self._plot_signer_accuracy(results)

        # 3. Per-class accuracy distribution
        self._plot_class_accuracy_distribution(results)

        # 4. Top-k accuracy
        self._plot_top_k_accuracy(results)

        # 5. Training curves comparison
        self._plot_training_curves_comparison(results)

        print(f"✅ Visualizations saved to {self.output_dir}")

    def _plot_confusion_matrix(self):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(self.all_labels, self.all_predictions)

        # Full confusion matrix
        fig, ax = plt.subplots(figsize=(24, 20))

        short_labels = [
            self.label_to_word.get(i, f"C{i}")[:10]
            for i in range(self.config.num_classes)
        ]

        sns.heatmap(
            cm,
            annot=False,
            fmt="d",
            cmap="Blues",
            xticklabels=short_labels,
            yticklabels=short_labels,
            ax=ax,
            cbar_kws={"label": "Count"},
        )

        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("Confusion Matrix (72 Classes)", fontsize=16, fontweight="bold")
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        # Normalized confusion matrix
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        fig, ax = plt.subplots(figsize=(24, 20))

        sns.heatmap(
            cm_normalized,
            annot=False,
            fmt=".2f",
            cmap="Blues",
            xticklabels=short_labels,
            yticklabels=short_labels,
            ax=ax,
            cbar_kws={"label": "Normalized Count"},
            vmin=0,
            vmax=1,
        )

        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("Normalized Confusion Matrix", fontsize=16, fontweight="bold")
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "confusion_matrix_normalized.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # Log to WandB
        wandb.log(
            {
                "eval/confusion_matrix": wandb.Image(
                    str(self.output_dir / "confusion_matrix.png")
                ),
                "eval/confusion_matrix_normalized": wandb.Image(
                    str(self.output_dir / "confusion_matrix_normalized.png")
                ),
            }
        )

    def _plot_signer_accuracy(self, results: Dict[str, Any]):
        """Plot per-signer accuracy."""
        signers = [
            k.split("_")[1]
            for k in results.keys()
            if k.startswith("signer_") and k.endswith("_accuracy")
        ]

        accuracies = [results[f"signer_{s}_accuracy"] * 100 for s in signers]
        counts = [results.get(f"signer_{s}_count", 0) for s in signers]

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]
        bars = ax.bar(signers, accuracies, color=colors[: len(signers)], alpha=0.8)

        ax.set_xlabel("Signer", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("Per-Signer Test Accuracy", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}%\n(n={count})",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "per_signer_accuracy.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        wandb.log(
            {
                "eval/per_signer_accuracy": wandb.Image(
                    str(self.output_dir / "per_signer_accuracy.png")
                )
            }
        )

    def _plot_class_accuracy_distribution(self, results: Dict[str, Any]):
        """Plot per-class accuracy distribution."""
        class_df = results["class_df"]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Bar plot of all class accuracies
        axes[0].bar(
            range(len(class_df)),
            class_df["accuracy"] * 100,
            color="steelblue",
            alpha=0.7,
        )
        axes[0].axhline(
            y=results["test_accuracy"] * 100,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Overall: {results['test_accuracy'] * 100:.1f}%",
        )
        axes[0].set_xlabel("Class ID", fontsize=11)
        axes[0].set_ylabel("Accuracy (%)", fontsize=11)
        axes[0].set_title(
            "Per-Class Accuracy (72 Classes)", fontsize=13, fontweight="bold"
        )
        axes[0].set_ylim(0, 100)
        axes[0].legend()
        axes[0].grid(axis="y", alpha=0.3)

        # Histogram of class accuracies
        axes[1].hist(
            class_df["accuracy"] * 100,
            bins=20,
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
        )
        axes[1].axvline(
            x=results["test_accuracy"] * 100,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Overall: {results['test_accuracy'] * 100:.1f}%",
        )
        axes[1].set_xlabel("Class Accuracy (%)", fontsize=11)
        axes[1].set_ylabel("Number of Classes", fontsize=11)
        axes[1].set_title(
            "Distribution of Class Accuracies", fontsize=13, fontweight="bold"
        )
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "per_class_accuracy.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        wandb.log(
            {
                "eval/per_class_accuracy": wandb.Image(
                    str(self.output_dir / "per_class_accuracy.png")
                )
            }
        )

    def _plot_top_k_accuracy(self, results: Dict[str, Any]):
        """Plot top-k accuracy."""
        k_values = [1, 3, 5, 10]
        accuracies = [results[f"top_{k}_accuracy"] * 100 for k in k_values]

        fig, ax = plt.subplots(figsize=(8, 5))

        bars = ax.bar(
            [f"Top-{k}" for k in k_values],
            accuracies,
            color=["#3498db", "#2ecc71", "#f39c12", "#9b59b6"],
            alpha=0.8,
        )

        ax.set_xlabel("Metric", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("Top-K Accuracy Analysis", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)

        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{acc:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
            )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "top_k_accuracy.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        wandb.log(
            {
                "eval/top_k_accuracy": wandb.Image(
                    str(self.output_dir / "top_k_accuracy.png")
                )
            }
        )

    def _plot_training_curves_comparison(self, results: Dict[str, Any]):
        """Plot training curves comparison with baseline."""
        # This would be called if historical training data is available
        pass


class ComparativeAnalyzer:
    """Compare results between SignNet-V2 and baseline models."""

    def __init__(
        self,
        signet_results: Dict[str, Any],
        baseline_results: Dict[str, Any],
        output_dir: Path,
    ):
        """
        Initialize comparator.

        Args:
            signet_results: SignNet-V2 evaluation results
            baseline_results: Baseline model results
            output_dir: Output directory for comparison plots
        """
        self.signet_results = signet_results
        self.baseline_results = baseline_results
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_comparison_report(self) -> Dict[str, Any]:
        """Generate comparative analysis report."""
        comparison = {}

        # Key metrics comparison
        key_metrics = [
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "top_5_accuracy",
            "mean_class_accuracy",
        ]

        for metric in key_metrics:
            signet_value = self.signet_results.get(metric, 0)
            baseline_value = self.baseline_results.get(metric, 0)

            comparison[f"{metric}_signet"] = signet_value
            comparison[f"{metric}_baseline"] = baseline_value
            comparison[f"{metric}_improvement"] = signet_value - baseline_value
            comparison[f"{metric}_improvement_pct"] = (
                (signet_value - baseline_value) / baseline_value * 100
                if baseline_value > 0
                else 0
            )

        # Per-signer comparison
        signers = set()
        for key in self.signet_results.keys():
            if key.startswith("signer_") and key.endswith("_accuracy"):
                signer = key.split("_")[1]
                signers.add(signer)

        comparison["signer_comparison"] = {}
        for signer in signers:
            signet_acc = self.signet_results.get(f"signer_{signer}_accuracy", 0)
            baseline_acc = self.baseline_results.get(f"signer_{signer}_accuracy", 0)

            comparison["signer_comparison"][signer] = {
                "signet": signet_acc,
                "baseline": baseline_acc,
                "improvement": signet_acc - baseline_acc,
            }

        self._plot_comparison(comparison)
        self._print_comparison_report(comparison)

        return comparison

    def _plot_comparison(self, comparison: Dict[str, Any]):
        """Plot comparison between models."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Overall metrics comparison
        metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1"]
        signet_values = [comparison[f"{m}_signet"] * 100 for m in metrics]
        baseline_values = [comparison[f"{m}_baseline"] * 100 for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        axes[0, 0].bar(
            x - width / 2,
            signet_values,
            width,
            label="SignNet-V2",
            color="#2ecc71",
            alpha=0.8,
        )
        axes[0, 0].bar(
            x + width / 2,
            baseline_values,
            width,
            label="Baseline",
            color="#e74c3c",
            alpha=0.8,
        )
        axes[0, 0].set_xlabel("Metric", fontsize=11)
        axes[0, 0].set_ylabel("Score (%)", fontsize=11)
        axes[0, 0].set_title(
            "Overall Metrics Comparison", fontsize=13, fontweight="bold"
        )
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(
            ["Accuracy", "Precision", "Recall", "F1"], fontsize=10
        )
        axes[0, 0].legend()
        axes[0, 0].grid(axis="y", alpha=0.3)
        axes[0, 0].set_ylim(0, 100)

        # 2. Top-K accuracy comparison
        k_metrics = [
            "top_1_accuracy",
            "top_3_accuracy",
            "top_5_accuracy",
            "top_10_accuracy",
        ]
        signet_k = [comparison[f"{m}_signet"] * 100 for m in k_metrics]
        baseline_k = [comparison[f"{m}_baseline"] * 100 for m in k_metrics]

        x = np.arange(len(k_metrics))
        axes[0, 1].bar(
            x - width / 2,
            signet_k,
            width,
            label="SignNet-V2",
            color="#2ecc71",
            alpha=0.8,
        )
        axes[0, 1].bar(
            x + width / 2,
            baseline_k,
            width,
            label="Baseline",
            color="#e74c3c",
            alpha=0.8,
        )
        axes[0, 1].set_xlabel("K", fontsize=11)
        axes[0, 1].set_ylabel("Accuracy (%)", fontsize=11)
        axes[0, 1].set_title(
            "Top-K Accuracy Comparison", fontsize=13, fontweight="bold"
        )
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(["Top-1", "Top-3", "Top-5", "Top-10"], fontsize=10)
        axes[0, 1].legend()
        axes[0, 1].grid(axis="y", alpha=0.3)
        axes[0, 1].set_ylim(0, 100)

        # 3. Per-signer comparison
        signers = list(comparison["signer_comparison"].keys())
        signet_signers = [
            comparison["signer_comparison"][s]["signet"] * 100 for s in signers
        ]
        baseline_signers = [
            comparison["signer_comparison"][s]["baseline"] * 100 for s in signers
        ]

        x = np.arange(len(signers))
        axes[1, 0].bar(
            x - width / 2,
            signet_signers,
            width,
            label="SignNet-V2",
            color="#2ecc71",
            alpha=0.8,
        )
        axes[1, 0].bar(
            x + width / 2,
            baseline_signers,
            width,
            label="Baseline",
            color="#e74c3c",
            alpha=0.8,
        )
        axes[1, 0].set_xlabel("Signer", fontsize=11)
        axes[1, 0].set_ylabel("Accuracy (%)", fontsize=11)
        axes[1, 0].set_title(
            "Per-Signer Accuracy Comparison", fontsize=13, fontweight="bold"
        )
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(signers, fontsize=10)
        axes[1, 0].legend()
        axes[1, 0].grid(axis="y", alpha=0.3)
        axes[1, 0].set_ylim(0, 100)

        # 4. Improvement summary
        improvements = [
            comparison["test_accuracy_improvement_pct"],
            comparison["test_f1_improvement_pct"],
            comparison["top_5_accuracy_improvement_pct"],
            comparison["mean_class_accuracy_improvement_pct"],
        ]
        improvement_labels = ["Accuracy", "F1-Score", "Top-5", "Mean Class"]

        colors = ["#2ecc71" if i >= 0 else "#e74c3c" for i in improvements]
        axes[1, 1].bar(improvement_labels, improvements, color=colors, alpha=0.8)
        axes[1, 1].set_xlabel("Metric", fontsize=11)
        axes[1, 1].set_ylabel("Improvement (%)", fontsize=11)
        axes[1, 1].set_title(
            "SignNet-V2 Improvement over Baseline", fontsize=13, fontweight="bold"
        )
        axes[1, 1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        axes[1, 1].grid(axis="y", alpha=0.3)

        for i, (label, imp) in enumerate(zip(improvement_labels, improvements)):
            axes[1, 1].text(
                i,
                imp + (1 if imp >= 0 else -2),
                f"{imp:.1f}%",
                ha="center",
                va="bottom" if imp >= 0 else "top",
                fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "model_comparison.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        wandb.log(
            {
                "eval/model_comparison": wandb.Image(
                    str(self.output_dir / "model_comparison.png")
                )
            }
        )

    def _print_comparison_report(self, comparison: Dict[str, Any]):
        """Print comparison report."""
        print("\n" + "=" * 70)
        print("📊 COMPARATIVE ANALYSIS: SignNet-V2 vs Baseline")
        print("=" * 70)

        print(f"\n🎯 Overall Performance:")
        print(
            f"   Accuracy:    {comparison['test_accuracy_signet'] * 100:.2f}% vs {comparison['test_accuracy_baseline'] * 100:.2f}% (+{comparison['test_accuracy_improvement_pct']:.2f}%)"
        )
        print(
            f"   Precision:   {comparison['test_precision_signet'] * 100:.2f}% vs {comparison['test_precision_baseline'] * 100:.2f}% (+{comparison['test_precision_improvement_pct']:.2f}%)"
        )
        print(
            f"   Recall:      {comparison['test_recall_signet'] * 100:.2f}% vs {comparison['test_recall_baseline'] * 100:.2f}% (+{comparison['test_recall_improvement_pct']:.2f}%)"
        )
        print(
            f"   F1-Score:    {comparison['test_f1_signet'] * 100:.2f}% vs {comparison['test_f1_baseline'] * 100:.2f}% (+{comparison['test_f1_improvement_pct']:.2f}%)"
        )

        print(f"\n📈 Top-K Accuracy:")
        print(
            f"   Top-1: {comparison['top_1_accuracy_signet'] * 100:.2f}% vs {comparison['top_1_accuracy_baseline'] * 100:.2f}%"
        )
        print(
            f"   Top-3: {comparison['top_3_accuracy_signet'] * 100:.2f}% vs {comparison['top_3_accuracy_baseline'] * 100:.2f}%"
        )
        print(
            f"   Top-5: {comparison['top_5_accuracy_signet'] * 100:.2f}% vs {comparison['top_5_accuracy_baseline'] * 100:.2f}%"
        )

        print(f"\n👥 Per-Signer Performance:")
        for signer, data in comparison["signer_comparison"].items():
            print(
                f"   {signer}: {data['signet'] * 100:.2f}% vs {data['baseline'] * 100:.2f}% (+{data['improvement'] * 100:.2f}%)"
            )

        print("\n" + "=" * 70)

        # Log to WandB
        wandb.log(
            {
                "comparison/accuracy_improvement": comparison[
                    "test_accuracy_improvement_pct"
                ],
                "comparison/f1_improvement": comparison["test_f1_improvement_pct"],
                "comparison/top5_improvement": comparison[
                    "top_5_accuracy_improvement_pct"
                ],
            }
        )


if __name__ == "__main__":
    print("✅ Evaluation suite loaded")
    print("\nEvaluation components:")
    print("   - SignNetEvaluator: Comprehensive model evaluation")
    print("   - ComparativeAnalyzer: Baseline comparison")
    print("   - Metrics: Accuracy, Precision, Recall, F1, Top-K")
    print("   - Visualizations: Confusion matrix, per-signer, per-class")
