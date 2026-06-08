"""
Training Pipeline for SignNet-V2
=================================

Advanced training pipeline with:
- Mixed precision training
- Learning rate scheduling (OneCycleLR with warmup)
- Gradient clipping and accumulation
- Early stopping
- Model checkpointing
- WandB integration
- Mixup augmentation

Author: BDSL Recognition Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
from collections import Counter
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass, field
from tqdm import tqdm
import wandb
import time
import math


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Model
    num_classes: int = 72
    body_dim: int = 99
    hand_dim: int = 63
    face_dim: int = 1404
    d_model: int = 128
    num_encoder_layers: int = 4
    num_heads: int = 8
    d_ff: int = 512
    dropout: float = 0.2
    warmup_epochs: int = 10

    # Training
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    early_stopping_patience: int = 25
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Mixed precision
    use_amp: bool = True

    # Augmentation
    mixup_alpha: float = 0.2
    emotion_loss_weight: float = 1.0
    emotion_class_weights: Optional[List[float]] = None

    # Paths
    base_dir: str = "/home/raco/Repos/bangla-sign-language-recognition"
    checkpoint_dir: str = "Data/processed/new_model/checkpoints"

    # Logging
    log_interval: int = 10
    save_interval: int = 5


class Lookahead(Optimizer):
    """
    Lookahead Optimizer wrapper.

    Based on: "Lookahead Optimizer: k steps forward, 1 step back"
    (Zhang et al., 2019)
    """

    def __init__(self, optimizer: Optimizer, la_steps: int = 5, alpha: float = 0.5):
        """
        Initialize Lookahead optimizer.

        Args:
            optimizer: Inner optimizer
            la_steps: Number of lookahead steps
            alpha: Lookahead alpha parameter
        """
        self.optimizer = optimizer
        self.la_steps = la_steps
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self._la_step_count = 0

        # Initialize slow weights
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    param_state = self.optimizer.state[p]
                    param_state["slow_param"] = p.data.clone()

    def __getattr__(self, name):
        """Delegate attribute access to underlying optimizer."""
        return getattr(self.optimizer, name)

    def step(self, closure=None):
        """Perform optimization step."""
        # Mark the wrapper as having executed an optimizer step so LR schedulers
        # can see the call order correctly when wrapped by Lookahead.
        self._opt_called = True
        loss = self.optimizer.step(closure)
        self.optimizer._opt_called = True

        self._la_step_count += 1

        if self._la_step_count % self.la_steps == 0:
            self._lookahead()

        return loss

    def _lookahead(self):
        """Perform lookahead update."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                param_state = self.optimizer.state[p]
                if "slow_param" not in param_state:
                    continue

                slow_p = param_state["slow_param"]
                # Update slow weights: slow_p = slow_p + alpha * (fast_p - slow_p)
                slow_p.add_(p.data - slow_p, alpha=self.alpha)
                # Update fast weights to match slow weights
                p.data.copy_(slow_p)

    def zero_grad(self, *args, **kwargs):
        """Zero gradients."""
        self.optimizer.zero_grad(*args, **kwargs)


class Mixup:
    """Mixup augmentation for sign language data."""

    def __init__(self, alpha: float = 0.2):
        """
        Initialize mixup.

        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha

    def mixup_data(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, float]:
        """
        Apply mixup to a batch.

        Args:
            x: Input features (body_pose, optional hands, optional face)
            y: Labels

        Returns:
            Tuple of mixed inputs, labels, and lambda value
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = x[0].size(0)
        index = torch.randperm(batch_size, device=x[0].device, dtype=torch.long)

        mixed_x = []
        for xi in x:
            if xi is not None:
                mixed_x.append(lam * xi + (1 - lam) * xi[index])
            else:
                mixed_x.append(None)

        return mixed_x, index, lam

    def mixup_criterion(
        self,
        criterion: nn.Module,
        logits: torch.Tensor,
        y: torch.Tensor,
        index: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """
        Compute mixup loss.

        Args:
            criterion: Loss function
            logits: Model predictions
            y: Original labels
            index: Permuted indices
            lam: Mixup lambda

        Returns:
            Mixup loss
        """
        return lam * criterion(logits, y) + (1 - lam) * criterion(logits, y[index])


class SignNetTrainer:
    """Trainer for SignNet-V2 model."""

    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        checkpoint_dir: Path,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            model: SignNet-V2 model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            checkpoint_dir: Directory for saving checkpoints
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.has_validation = len(self.val_loader.dataset) > 0

        # Move model to device
        self.model = self.model.to(self.device)

        # Mixed precision scaler (use new API)
        self.scaler = (
            torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")
            if config.use_amp
            else None
        )

        # Optimizer: use a single optimizer for all trainable model parameters.
        self.sign_params = [p for p in model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            self.sign_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Lookahead optimizer
        self.optimizer = Lookahead(self.optimizer, la_steps=5, alpha=0.5)

        # Learning rate scheduler (linear warmup over the requested warmup window)
        self.steps_per_epoch = max(
            1, math.ceil(len(train_loader) / config.gradient_accumulation_steps)
        )
        warmup_fraction = min(
            max(config.warmup_epochs / max(1, config.epochs), 0.01), 0.5
        )
        # Bind the scheduler to the inner AdamW optimizer so PyTorch's step
        # tracking matches the real optimizer step executed through Lookahead.
        self.scheduler = OneCycleLR(
            self.optimizer.optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=self.steps_per_epoch,
            pct_start=warmup_fraction,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=25.0,
        )

        self.emotion_scheduler = None

        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        # Emotion loss: prefer explicit class weights if provided in config.
        # Safer default: do not pass a weight tensor so the loss works for
        # any number of emotion classes. If users supply `emotion_class_weights`
        # in `config`, they'll still be honored.
        if config.emotion_class_weights:
            emotion_weight_tensor = torch.tensor(
                config.emotion_class_weights, dtype=torch.float32, device=self.device
            )
            self.emotion_criterion = nn.CrossEntropyLoss(
                label_smoothing=0.1,
                weight=emotion_weight_tensor,
            )
        else:
            # No explicit weights provided — use unweighted CrossEntropyLoss
            # which is compatible with any `num_emotions` value.
            self.emotion_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Mixup
        self.mixup = Mixup(alpha=config.mixup_alpha)

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.no_improve_count = 0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "emotion_train_acc": [],
            "emotion_val_acc": [],
            "emotion_train_per_class": [],
            "emotion_val_per_class": [],
            "learning_rate": [],
            "optimizer_steps": [],
            "train_loader_failures": [],
            "val_loader_failures": [],
        }
        self.last_epoch_diagnostics: Dict[str, Any] = {}
        self.history_path = self.checkpoint_dir / "training_history.json"

    def _emotion_class_name(self, emotion_index: int) -> str:
        emotion_names = ["neutral", "happy", "sad", "negation", "question"]
        if 0 <= emotion_index < len(emotion_names):
            return emotion_names[emotion_index]
        return str(emotion_index)

    def _emotion_per_class_accuracy(
        self, class_correct: Counter, class_total: Counter
    ) -> Dict[str, float]:
        return {
            emotion: (
                class_correct.get(emotion, 0) / class_total.get(emotion, 0)
                if class_total.get(emotion, 0)
                else 0.0
            )
            for emotion in ["neutral", "happy", "sad", "negation", "question"]
        }

    def _save_training_history(self) -> None:
        with open(self.history_path, "w", encoding="utf-8") as handle:
            json.dump(self.history, handle, ensure_ascii=False, indent=2)

    def train_epoch(self) -> Tuple[float, float, Dict[str, Any]]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        emotion_total = 0
        emotion_correct = 0
        emotion_class_correct: Counter = Counter()
        emotion_class_total: Counter = Counter()

        accumulation_steps = self.config.gradient_accumulation_steps
        optimizer_steps = 0
        loader_failures = 0
        expected_optimizer_steps = self.steps_per_epoch
        self.optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {self.current_epoch + 1}", leave=False
        )

        for batch_idx, batch in enumerate(progress_bar):
            body_pose = batch["body_pose"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device).long().view(-1)
            emotion_labels = batch.get("emotion_label")
            if emotion_labels is not None:
                emotion_labels = emotion_labels.to(self.device).long().view(-1)

            if "load_failed" in batch:
                loader_failures += int(batch["load_failed"].view(-1).sum().item())

            left_hand = batch.get("left_hand")
            right_hand = batch.get("right_hand")
            face = batch.get("face")

            if left_hand is not None:
                left_hand = left_hand.to(self.device)
            if right_hand is not None:
                right_hand = right_hand.to(self.device)
            if face is not None:
                face = face.to(self.device)

            use_aux_emotion = (
                hasattr(self.model, "emotion_classifier")
                and getattr(self.model, "emotion_classifier") is not None
                and emotion_labels is not None
                and int(emotion_labels.min().item()) >= 0
            )

            apply_mixup = (
                self.config.mixup_alpha > 0
                and np.random.random() < 0.5
                and not use_aux_emotion
            )

            if apply_mixup:
                x_tuple = (body_pose, left_hand, right_hand, face)
                x_mixed, index, lam = self.mixup.mixup_data(x_tuple, labels)
                body_pose, left_hand, right_hand, face = x_mixed

            batch_loss = None

            if use_aux_emotion:
                if self.scaler is not None:
                    with torch.amp.autocast(device_type=self.device.type, enabled=True):
                        logits, emotion_logits = self.model.forward_with_aux(
                            body_pose,
                            left_hand,
                            right_hand,
                            face,
                            attention_mask,
                        )
                        if apply_mixup:
                            sign_loss = self.mixup.mixup_criterion(
                                self.criterion, logits, labels, index, lam
                            )
                        else:
                            sign_loss = self.criterion(logits, labels)

                        emotion_loss = self.emotion_criterion(
                            emotion_logits.float(), emotion_labels
                        )

                    batch_loss = sign_loss + (self.config.emotion_loss_weight * emotion_loss)
                    if not torch.isfinite(batch_loss):
                        raise RuntimeError(
                            f"Non-finite training loss at batch {batch_idx + 1}"
                        )

                    scaled_sign_loss = sign_loss / accumulation_steps
                    scaled_emotion_loss = (
                        self.config.emotion_loss_weight * emotion_loss / accumulation_steps
                    )
                    self.scaler.scale(scaled_sign_loss).backward(retain_graph=True)
                    self.scaler.scale(scaled_emotion_loss).backward()
                    should_step = (batch_idx + 1) % accumulation_steps == 0 or (
                        batch_idx + 1
                    ) == len(self.train_loader)
                    if should_step:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.sign_params, self.config.gradient_clip_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        optimizer_steps += 1
                else:
                    logits, emotion_logits = self.model.forward_with_aux(
                        body_pose,
                        left_hand,
                        right_hand,
                        face,
                        attention_mask,
                    )
                    if apply_mixup:
                        sign_loss = self.mixup.mixup_criterion(
                            self.criterion, logits, labels, index, lam
                        )
                    else:
                        sign_loss = self.criterion(logits, labels)
                    emotion_loss = self.emotion_criterion(emotion_logits, emotion_labels)

                    batch_loss = sign_loss + (self.config.emotion_loss_weight * emotion_loss)
                    if not torch.isfinite(batch_loss):
                        raise RuntimeError(
                            f"Non-finite training loss at batch {batch_idx + 1}"
                        )

                    sign_loss = sign_loss / accumulation_steps
                    emotion_loss = (
                        self.config.emotion_loss_weight * emotion_loss / accumulation_steps
                    )
                    sign_loss.backward(retain_graph=True)
                    emotion_loss.backward()

                    should_step = (batch_idx + 1) % accumulation_steps == 0 or (
                        batch_idx + 1
                    ) == len(self.train_loader)
                    if should_step:
                        torch.nn.utils.clip_grad_norm_(
                            self.sign_params, self.config.gradient_clip_norm
                        )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        optimizer_steps += 1
            else:
                if self.scaler is not None:
                    with torch.amp.autocast(device_type=self.device.type, enabled=True):
                        logits = self.model(
                            body_pose, left_hand, right_hand, face, attention_mask
                        )
                        if apply_mixup:
                            sign_loss = self.mixup.mixup_criterion(
                                self.criterion, logits, labels, index, lam
                            )
                        else:
                            sign_loss = self.criterion(logits, labels)

                    if not torch.isfinite(sign_loss):
                        raise RuntimeError(
                            f"Non-finite training loss at batch {batch_idx + 1}"
                        )

                    batch_loss = sign_loss
                    scaled_loss = sign_loss / accumulation_steps
                    self.scaler.scale(scaled_loss).backward()
                    should_step = (batch_idx + 1) % accumulation_steps == 0 or (
                        batch_idx + 1
                    ) == len(self.train_loader)
                    if should_step:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.gradient_clip_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        optimizer_steps += 1
                else:
                    logits = self.model(
                        body_pose, left_hand, right_hand, face, attention_mask
                    )
                    if apply_mixup:
                        sign_loss = self.mixup.mixup_criterion(
                            self.criterion, logits, labels, index, lam
                        )
                    else:
                        sign_loss = self.criterion(logits, labels)

                    if not torch.isfinite(sign_loss):
                        raise RuntimeError(
                            f"Non-finite training loss at batch {batch_idx + 1}"
                        )

                    batch_loss = sign_loss
                    scaled_loss = sign_loss / accumulation_steps
                    scaled_loss.backward()

                    should_step = (batch_idx + 1) % accumulation_steps == 0 or (
                        batch_idx + 1
                    ) == len(self.train_loader)
                    if should_step:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.gradient_clip_norm
                        )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        optimizer_steps += 1

            if use_aux_emotion:
                with torch.no_grad():
                    emotion_predictions = torch.argmax(emotion_logits, dim=1)
                    emotion_correct += (emotion_predictions == emotion_labels).sum().item()
                    emotion_total += len(emotion_labels)
                    for label_value, pred_value in zip(
                        emotion_labels.cpu().tolist(), emotion_predictions.cpu().tolist()
                    ):
                        emotion_name = self._emotion_class_name(int(label_value))
                        emotion_class_total[emotion_name] += 1
                        if int(pred_value) == int(label_value):
                            emotion_class_correct[emotion_name] += 1

            with torch.no_grad():
                predictions = torch.argmax(logits, dim=1)
                if apply_mixup:
                    correct = (predictions == labels).sum().item() * lam + (
                        predictions == labels[index]
                    ).sum().item() * (1 - lam)
                else:
                    correct = (predictions == labels).sum().item()

                total_loss += batch_loss.item() * len(labels)
                total_correct += correct
                total_samples += len(labels)

            progress_bar.set_postfix(
                {
                    "loss": f"{batch_loss.item():.4f}",
                    "acc": f"{correct / len(labels):.4f}",
                }
            )

            if (batch_idx + 1) % 5 == 0:
                try:
                    wandb.log(
                        {
                            "train/batch_loss": batch_loss.item(),
                            "train/batch_accuracy": correct / len(labels),
                            "train/batch": batch_idx + 1,
                            "train/optimizer_steps_so_far": optimizer_steps,
                        }
                    )
                except Exception:
                    pass

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        emotion_accuracy = emotion_correct / emotion_total if emotion_total else 0.0

        diagnostics = {
            "optimizer_steps": optimizer_steps,
            "expected_optimizer_steps": expected_optimizer_steps,
            "loader_failures": loader_failures,
            "emotion_accuracy": emotion_accuracy,
            "emotion_total": emotion_total,
            "emotion_correct": emotion_correct,
            "emotion_per_class_accuracy": self._emotion_per_class_accuracy(
                emotion_class_correct, emotion_class_total
            ),
        }
        self.last_epoch_diagnostics = diagnostics

        return avg_loss, accuracy, diagnostics

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, Dict[str, Any]]:
        """Validate the model."""
        if not self.has_validation:
            return 0.0, 0.0, {
                "loader_failures": 0,
                "skipped": True,
                "emotion_accuracy": 0.0,
                "emotion_total": 0,
                "emotion_correct": 0,
                "emotion_per_class_accuracy": {
                    "neutral": 0.0,
                    "happy": 0.0,
                    "sad": 0.0,
                    "negation": 0.0,
                    "question": 0.0,
                },
            }

        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        loader_failures = 0
        emotion_total = 0
        emotion_correct = 0
        emotion_class_correct: Counter = Counter()
        emotion_class_total: Counter = Counter()

        all_predictions = []
        all_labels = []
        _ = all_predictions, all_labels

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            # Unpack batch
            body_pose = batch["body_pose"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device).long().view(-1)
            emotion_labels = batch.get("emotion_label")
            if emotion_labels is not None:
                emotion_labels = emotion_labels.to(self.device).long().view(-1)

            if "load_failed" in batch:
                loader_failures += int(batch["load_failed"].view(-1).sum().item())

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
            if self.scaler is not None:
                with torch.amp.autocast(device_type=self.device.type, enabled=True):
                    logits, emotion_logits = self.model.forward_with_aux(
                        body_pose, left_hand, right_hand, face, attention_mask
                    )
                    loss = self.criterion(logits, labels)
            else:
                logits, emotion_logits = self.model.forward_with_aux(
                    body_pose, left_hand, right_hand, face, attention_mask
                )
                loss = self.criterion(logits, labels)

            # Metrics
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (
                emotion_logits is not None
                and emotion_labels is not None
                and int(emotion_labels.min().item()) >= 0
            ):
                emotion_predictions = torch.argmax(emotion_logits, dim=1)
                emotion_correct += (emotion_predictions == emotion_labels).sum().item()
                emotion_total += len(emotion_labels)
                for label_value, pred_value in zip(
                    emotion_labels.cpu().tolist(), emotion_predictions.cpu().tolist()
                ):
                    emotion_name = self._emotion_class_name(int(label_value))
                    emotion_class_total[emotion_name] += 1
                    if int(pred_value) == int(label_value):
                        emotion_class_correct[emotion_name] += 1

            total_loss += loss.item() * len(labels)
            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        emotion_accuracy = emotion_correct / emotion_total if emotion_total else 0.0

        diagnostics = {
            "loader_failures": loader_failures,
            "emotion_accuracy": emotion_accuracy,
            "emotion_total": emotion_total,
            "emotion_correct": emotion_correct,
            "emotion_per_class_accuracy": self._emotion_per_class_accuracy(
                emotion_class_correct, emotion_class_total
            ),
        }

        return avg_loss, accuracy, diagnostics

    def save_checkpoint(self, is_best: bool = False, is_intermediate: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_accuracy": self.best_val_acc,
            "val_loss": self.history["val_loss"][-1]
            if self.history["val_loss"]
            else float("inf"),
            "config": self.config.__dict__,
            "history": self.history,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        # Save best model
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pth")
            print(f"   ✨ Best model saved! Val Acc: {self.best_val_acc:.4f}")

        # Save intermediate checkpoint
        if is_intermediate:
            torch.save(
                checkpoint,
                self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pth",
            )

        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / "latest_checkpoint.pth")

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """Load checkpoint and return starting epoch."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint.get("val_accuracy", 0.0)
        self.history = checkpoint.get("history", self.history)

        print(f"✅ Loaded checkpoint from epoch {self.current_epoch}")
        print(f"   Best val accuracy: {self.best_val_acc:.4f}")

        return self.current_epoch

    def train(self, start_epoch: int = 0, max_epochs: Optional[int] = None) -> Dict:
        """
        Complete training loop.

        Args:
            start_epoch: Starting epoch
            max_epochs: Maximum epochs (uses config if None)

        Returns:
            Training history dictionary
        """
        if max_epochs is None:
            max_epochs = self.config.epochs

        print(f"\n🚀 Starting training for {max_epochs} epochs")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Learning rate: {self.config.learning_rate}")
        print(f"   Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"   Expected optimizer steps/epoch: {self.steps_per_epoch}")
        print(f"   Mixed precision: {self.config.use_amp}")
        print(f"   Checkpoint directory: {self.checkpoint_dir}")

        train_malformed = getattr(self.train_loader.dataset, "malformed_metadata_count", 0)
        val_malformed = getattr(self.val_loader.dataset, "malformed_metadata_count", 0)
        if train_malformed or val_malformed:
            print(
                f"   Malformed sample names skipped -> train: {train_malformed}, val: {val_malformed}"
            )

        for epoch in range(start_epoch, max_epochs):
            self.current_epoch = epoch

            # Train
            train_loss, train_acc, train_diag = self.train_epoch()

            # Validate
            if self.has_validation:
                val_loss, val_acc, val_diag = self.validate()
            else:
                val_loss, val_acc = float("nan"), float("nan")
                val_diag = {"loader_failures": 0, "skipped": True}

            # Get learning rate
            current_lr = self.scheduler.get_last_lr()[0]

            # Log to history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["emotion_train_acc"].append(train_diag.get("emotion_accuracy", 0.0))
            self.history["emotion_val_acc"].append(val_diag.get("emotion_accuracy", 0.0))
            self.history["emotion_train_per_class"].append(
                train_diag.get("emotion_per_class_accuracy", {})
            )
            self.history["emotion_val_per_class"].append(
                val_diag.get("emotion_per_class_accuracy", {})
            )
            self.history["learning_rate"].append(current_lr)
            self.history["optimizer_steps"].append(train_diag["optimizer_steps"])
            self.history["train_loader_failures"].append(train_diag["loader_failures"])
            self.history["val_loader_failures"].append(val_diag["loader_failures"])

            # Log to WandB in real-time
            try:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train/loss": train_loss,
                        "train/accuracy": train_acc,
                        "train/emotion_accuracy": train_diag.get("emotion_accuracy", 0.0),
                        "val/loss": val_loss if self.has_validation else float("nan"),
                        "val/accuracy": val_acc if self.has_validation else float("nan"),
                        "val/emotion_accuracy": val_diag.get("emotion_accuracy", 0.0),
                        "learning_rate": current_lr,
                        "train/optimizer_steps": train_diag["optimizer_steps"],
                        "train/expected_optimizer_steps": train_diag[
                            "expected_optimizer_steps"
                        ],
                        "train/loader_failures": train_diag["loader_failures"],
                        "val/loader_failures": val_diag["loader_failures"],
                    }
                )
            except Exception:
                pass  # W&B logging not available

            # Print epoch summary
            print(f"\n📊 Epoch {epoch + 1}/{max_epochs} Summary:")
            print(
                f"   Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f} ({train_acc * 100:.2f}%)"
            )
            print(
                f"   Emotion Train - Acc: {train_diag.get('emotion_accuracy', 0.0):.4f} ({train_diag.get('emotion_accuracy', 0.0) * 100:.2f}%)"
            )
            if self.has_validation:
                print(
                    f"   Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f} ({val_acc * 100:.2f}%)"
                )
                print(
                    f"   Emotion Val   - Acc: {val_diag.get('emotion_accuracy', 0.0):.4f} ({val_diag.get('emotion_accuracy', 0.0) * 100:.2f}%)"
                )
            else:
                print("   Val   - skipped (no validation split)")
            print(f"   LR: {current_lr:.6f}")
            print(
                "   Optimizer steps: "
                f"{train_diag['optimizer_steps']}/{train_diag['expected_optimizer_steps']}"
            )
            print(
                "   Loader failures: "
                f"train={train_diag['loader_failures']}, val={val_diag['loader_failures']}"
            )
            if (epoch + 1) % 10 == 0:
                print("   Emotion per-class accuracy (train):")
                for emotion_name, emotion_acc in train_diag.get("emotion_per_class_accuracy", {}).items():
                    print(f"      {emotion_name}: {emotion_acc * 100:.2f}%")
                if self.has_validation:
                    print("   Emotion per-class accuracy (val):")
                    for emotion_name, emotion_acc in val_diag.get("emotion_per_class_accuracy", {}).items():
                        print(f"      {emotion_name}: {emotion_acc * 100:.2f}%")

            # Save best model
            if self.has_validation:
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                    self.no_improve_count = 0
                else:
                    self.no_improve_count += 1
            else:
                is_best = train_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = train_acc

            # Save checkpoint
            self.save_checkpoint(
                is_best=is_best,
                is_intermediate=(epoch + 1) % self.config.save_interval == 0,
            )
            self._save_training_history()

            # Early stopping
            if self.has_validation and self.no_improve_count >= self.config.early_stopping_patience:
                print(
                    f"\n⏹️  Early stopping triggered after {self.config.early_stopping_patience} epochs without improvement"
                )
                print(
                    f"   Best validation accuracy: {self.best_val_acc:.4f} ({self.best_val_acc * 100:.2f}%)"
                )
                break

        print(f"\n✅ Training complete!")
        print(f"   Total epochs: {self.current_epoch + 1}")
        if self.has_validation:
            print(
                f"   Best validation accuracy: {self.best_val_acc:.4f} ({self.best_val_acc * 100:.2f}%)"
            )
        else:
            print(
                f"   Best training accuracy (validation skipped): {self.best_val_acc:.4f} ({self.best_val_acc * 100:.2f}%)"
            )

        return self.history


def setup_training(
    config: TrainingConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> SignNetTrainer:
    """
    Setup training components.

    Args:
        config: Training configuration
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on

    Returns:
        Configured SignNetTrainer
    """
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = SignNetTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    return trainer


if __name__ == "__main__":
    # Test training setup
    from src.models.signet_v2 import SignNetV2
    from src.data.preprocessing import DataConfig, create_data_loaders

    print("✅ Training components imported")
    print("\nTesting training configuration...")

    config = TrainingConfig()
    print(f"   Epochs: {config.epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Mixed precision: {config.use_amp}")
    print(f"   Mixup alpha: {config.mixup_alpha}")

    print("\n✅ Training pipeline setup complete")
