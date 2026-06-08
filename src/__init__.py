"""
SignNet-V2: Enhanced Multi-Stream Spatiotemporal Transformer for Sign Language Recognition

This package contains the complete implementation of SignNet-V2, including:
- Multi-stream architecture (body, hands, face)
- Advanced data preprocessing and augmentation
- Training pipeline with mixed precision, Lookahead optimizer, Mixup
- Comprehensive evaluation suite
"""

__version__ = "2.0.0"
__author__ = "BDSL Recognition Team"

from .models.signet_v2 import SignNetV2
from .data.preprocessing import (
    DataConfig,
    SignLanguageDataset,
    PoseNormalizer,
    Augmentor,
)
from .training.trainer import TrainingConfig, SignNetTrainer
from .evaluation.evaluator import EvaluationConfig, SignNetEvaluator

__all__ = [
    "SignNetV2",
    "DataConfig",
    "SignLanguageDataset",
    "PoseNormalizer",
    "Augmentor",
    "TrainingConfig",
    "SignNetTrainer",
    "EvaluationConfig",
    "SignNetEvaluator",
]
