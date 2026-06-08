"""
Data Preprocessing and Augmentation Pipeline for Sign Language Recognition
===========================================================================

Comprehensive data processing for Bengali Sign Language video/pose data.
Includes:
- MediaPipe landmark extraction (body, hands, face)
- Multi-stream normalization
- Temporal augmentation
- Spatial augmentation
- Semantic augmentation

Author: BDSL Recognition Team
"""

import unicodedata
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import random


# Landmark indices for MediaPipe
class LandmarkIndices:
    """Indices for MediaPipe pose, hand, and face landmarks."""

    # Pose landmarks (33 total)
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

    # Pose connections for visualization
    POSE_CONNECTIONS = [
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (11, 23),
        (12, 24),
        (23, 24),
        (23, 25),
        (24, 26),
        (25, 27),
        (26, 28),
        (27, 29),
        (28, 30),
        (29, 31),
        (30, 32),
    ]

    POSE_FLIP_PAIRS = [
        (1, 4),
        (2, 5),
        (3, 6),
        (7, 8),
        (9, 10),
        (11, 12),
        (13, 14),
        (15, 16),
        (17, 18),
        (19, 20),
        (21, 22),
        (23, 24),
        (25, 26),
        (27, 28),
        (29, 30),
        (31, 32),
    ]

    # Hand landmarks (21 per hand)
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    # Face landmarks (468 total) - simplified regions
    FACE_OUTLINE = list(range(0, 17))
    LEFT_EYEBROW = list(range(17, 22))
    RIGHT_EYEBROW = list(range(22, 27))
    NOSE = list(range(27, 36))
    LEFT_EYE = list(range(36, 42))
    RIGHT_EYE = list(range(42, 48))
    LIPS_OUTER = list(range(48, 60))
    LIPS_INNER = list(range(60, 68))


@dataclass
class DataConfig:
    """Configuration for data processing."""

    # Dataset paths
    base_dir: str = "/home/raco/Repos/bangla-sign-language-recognition"
    processed_dir: str = "Data/processed/new_model"
    normalized_dir: str = "Data/processed/new_model/normalized"
    checkpoint_dir: str = "Data/processed/new_model/checkpoints"

    # Sequence parameters
    max_seq_length: int = 150
    min_seq_length: int = 10
    target_fps: int = 30

    # Feature dimensions
    body_dim: int = 99  # 33 landmarks * 3 coordinates
    hand_dim: int = 63  # 21 landmarks * 3 coordinates
    face_dim: int = 1434  # 478 landmarks * 3 coordinates

    # Augmentation
    augmentation: bool = True
    temporal_scale_range: Tuple[float, float] = (0.8, 1.2)
    random_crop_range: Tuple[float, float] = (0.8, 1.0)
    frame_dropout_range: Tuple[float, float] = (0.05, 0.10)
    noise_std: float = 0.01
    rotation_range: float = 15  # degrees
    scale_range: Tuple[float, float] = (0.9, 1.1)
    horizontal_flip_prob: float = 0.5

    # Normalization
    use_shoulder_normalization: bool = True
    use_hand_normalization: bool = True

    # Loader behavior
    loader_error_mode: str = "permissive"  # "permissive" or "strict"
    max_logged_loader_failures: int = 20


class PoseNormalizer:
    """Normalize pose landmarks using shoulder-based reference."""

    def __init__(self, min_shoulder_scale: float = 0.1):
        self.min_shoulder_scale = min_shoulder_scale

    def normalize(self, pose_sequence: np.ndarray) -> np.ndarray:
        """
        Normalize pose sequence using shoulder-based centering and scaling.

        Args:
            pose_sequence: Array of shape (num_frames, num_landmarks, 3)

        Returns:
            Normalized pose sequence
        """
        # Calculate shoulder center
        left_shoulder = pose_sequence[:, LandmarkIndices.LEFT_SHOULDER, :3]
        right_shoulder = pose_sequence[:, LandmarkIndices.RIGHT_SHOULDER, :3]
        shoulder_center = (left_shoulder + right_shoulder) / 2.0

        # Calculate shoulder width for scaling
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=-1)
        valid = np.isfinite(shoulder_width) & (shoulder_width > 0)

        scale = (
            float(shoulder_width[valid].mean())
            if np.any(valid)
            else self.min_shoulder_scale
        )
        scale = max(scale, self.min_shoulder_scale)

        # Center and scale
        centered = pose_sequence - shoulder_center[:, None, :]
        normalized = centered / scale

        return normalized

    def normalize_with_reference(
        self,
        pose_sequence: np.ndarray,
        reference_center: np.ndarray,
        reference_scale: float,
    ) -> np.ndarray:
        """
        Normalize using provided reference values (for test-time consistency).
        """
        centered = pose_sequence - reference_center[None, None, :]
        normalized = centered / reference_scale
        return normalized


class HandNormalizer:
    """Normalize hand landmarks using wrist-based reference."""

    def __init__(self, min_scale: float = 0.1):
        self.min_scale = min_scale

    def normalize(self, hand_sequence: np.ndarray) -> np.ndarray:
        """
        Normalize hand sequence using wrist-based reference.

        Args:
            hand_sequence: Array of shape (num_frames, 21, 3)

        Returns:
            Normalized hand sequence
        """
        wrist = hand_sequence[:, LandmarkIndices.WRIST, :3]

        # Calculate scale based on hand spread
        palm_points = hand_sequence[:, [0, 5, 9, 13, 17], :3]
        palm_center = palm_points.mean(axis=1)
        palm_spread = np.linalg.norm(
            palm_points - palm_center[:, None, :], axis=-1
        ).mean(axis=1)

        valid = np.isfinite(palm_spread) & (palm_spread > self.min_scale)
        scale = float(palm_spread[valid].mean()) if np.any(valid) else 1.0
        scale = max(scale, self.min_scale)

        # Center at wrist and scale
        centered = hand_sequence - wrist[:, None, :]
        normalized = centered / scale

        return normalized


class TemporalAligner:
    """Align sequences to target length using various strategies."""

    def __init__(self, target_length: int, min_length: int = 10):
        self.target_length = target_length
        self.min_length = min_length

    def pad_or_crop(self, sequence: np.ndarray) -> np.ndarray:
        """
        Pad or crop sequence to target length using centered approach.

        Args:
            sequence: Input sequence of shape (seq_len, ...)

        Returns:
            Aligned sequence of shape (target_length, ...)
        """
        seq_len = sequence.shape[0]

        if seq_len == self.target_length:
            return sequence

        if seq_len > self.target_length:
            # Crop centered
            start = max(0, (seq_len - self.target_length) // 2)
            return sequence[start : start + self.target_length]

        # Pad with zeros
        pad_length = self.target_length - seq_len
        pad_shape = (pad_length,) + sequence.shape[1:]
        padding = np.zeros(pad_shape, dtype=sequence.dtype)
        return np.concatenate([sequence, padding], axis=0)

    def resample(self, sequence: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Resample sequence to different length.

        Args:
            sequence: Input sequence
            scale_factor: Multiplier for sequence length

        Returns:
            Resampled sequence
        """
        new_length = max(self.min_length, int(len(sequence) * scale_factor))
        new_length = min(new_length, self.target_length)

        # Linear interpolation
        indices = np.linspace(0, len(sequence) - 1, new_length)
        resampled = []
        for idx in indices:
            lower = int(np.floor(idx))
            upper = min(lower + 1, len(sequence) - 1)
            weight = idx - lower
            resampled.append((1 - weight) * sequence[lower] + weight * sequence[upper])

        return np.array(resampled)


class Augmentor:
    """Comprehensive augmentation for sign language data."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.temporal_aligner = TemporalAligner(
            config.max_seq_length, config.min_seq_length
        )

    def augment(
        self,
        body_pose: np.ndarray,
        left_hand: Optional[np.ndarray] = None,
        right_hand: Optional[np.ndarray] = None,
        face: Optional[np.ndarray] = None,
        attention_mask: Optional[np.ndarray] = None,
        apply_augmentation: bool = True,
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        Apply augmentation to all input streams.

        Args:
            body_pose: Body pose sequence (seq_len, 33, 3)
            left_hand: Left hand sequence (seq_len, 21, 3) or None
            right_hand: Right hand sequence (seq_len, 21, 3) or None
            face: Face sequence (seq_len, 468, 3) or None
            apply_augmentation: Whether to apply augmentation

        Returns:
            Tuple of augmented sequences
        """
        if not apply_augmentation or not self.config.augmentation:
            # Just align without augmentation
            body_pose = self.temporal_aligner.pad_or_crop(body_pose)
            if left_hand is not None:
                left_hand = self.temporal_aligner.pad_or_crop(left_hand)
            if right_hand is not None:
                right_hand = self.temporal_aligner.pad_or_crop(right_hand)
            if face is not None:
                face = self.temporal_aligner.pad_or_crop(face)
            if attention_mask is not None:
                attention_mask = self.temporal_aligner.pad_or_crop(attention_mask)
            return body_pose, left_hand, right_hand, face, attention_mask

        body_pose, left_hand, right_hand, face, attention_mask = self._apply_speed_jitter(
            body_pose, left_hand, right_hand, face, attention_mask
        )
        body_pose, left_hand, right_hand, face, attention_mask = self._apply_random_crop(
            body_pose, left_hand, right_hand, face, attention_mask
        )
        body_pose, left_hand, right_hand, face, attention_mask = self._apply_frame_dropout(
            body_pose, left_hand, right_hand, face, attention_mask
        )

        if random.random() < self.config.horizontal_flip_prob:
            body_pose, left_hand, right_hand, face, attention_mask = self._apply_horizontal_flip(
                body_pose, left_hand, right_hand, face, attention_mask
            )

        if random.random() < 0.4:
            body_pose = self._add_gaussian_noise(body_pose)
            if left_hand is not None:
                left_hand = self._add_gaussian_noise(left_hand)
            if right_hand is not None:
                right_hand = self._add_gaussian_noise(right_hand)
            if face is not None:
                face = self._add_gaussian_noise(face)

        if random.random() < 0.3:
            scale = random.uniform(*self.config.scale_range)
            body_pose = self._scale(body_pose, scale)
            if left_hand is not None:
                left_hand = self._scale(left_hand, scale)
            if right_hand is not None:
                right_hand = self._scale(right_hand, scale)
            if face is not None:
                face = self._scale(face, scale)

        # Align to target length
        body_pose = self.temporal_aligner.pad_or_crop(body_pose)
        if left_hand is not None:
            left_hand = self.temporal_aligner.pad_or_crop(left_hand)
        if right_hand is not None:
            right_hand = self.temporal_aligner.pad_or_crop(right_hand)
        if face is not None:
            face = self.temporal_aligner.pad_or_crop(face)
        if attention_mask is not None:
            attention_mask = self.temporal_aligner.pad_or_crop(attention_mask)

        return body_pose, left_hand, right_hand, face, attention_mask

    def _apply_speed_jitter(
        self,
        body_pose: np.ndarray,
        left_hand: Optional[np.ndarray],
        right_hand: Optional[np.ndarray],
        face: Optional[np.ndarray],
        attention_mask: Optional[np.ndarray],
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        if random.random() >= 0.5:
            return body_pose, left_hand, right_hand, face, attention_mask

        scale = random.uniform(*self.config.temporal_scale_range)
        body_pose = self._temporal_scale(body_pose, scale)
        if left_hand is not None:
            left_hand = self._temporal_scale(left_hand, scale)
        if right_hand is not None:
            right_hand = self._temporal_scale(right_hand, scale)
        if face is not None:
            face = self._temporal_scale(face, scale)
        if attention_mask is not None:
            attention_mask = self._temporal_scale(attention_mask, scale)
        return body_pose, left_hand, right_hand, face, attention_mask

    def _apply_random_crop(
        self,
        body_pose: np.ndarray,
        left_hand: Optional[np.ndarray],
        right_hand: Optional[np.ndarray],
        face: Optional[np.ndarray],
        attention_mask: Optional[np.ndarray],
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        seq_len = body_pose.shape[0]
        if seq_len <= 1:
            return body_pose, left_hand, right_hand, face, attention_mask

        min_crop = max(1, int(np.ceil(seq_len * self.config.random_crop_range[0])))
        max_crop = max(min_crop, int(np.floor(seq_len * self.config.random_crop_range[1])))
        crop_len = random.randint(min_crop, max_crop)

        if crop_len >= seq_len:
            return body_pose, left_hand, right_hand, face, attention_mask

        start = random.randint(0, seq_len - crop_len)
        end = start + crop_len

        body_pose = body_pose[start:end]
        if left_hand is not None:
            left_hand = left_hand[start:end]
        if right_hand is not None:
            right_hand = right_hand[start:end]
        if face is not None:
            face = face[start:end]
        if attention_mask is not None:
            attention_mask = attention_mask[start:end]

        return body_pose, left_hand, right_hand, face, attention_mask

    def _apply_frame_dropout(
        self,
        body_pose: np.ndarray,
        left_hand: Optional[np.ndarray],
        right_hand: Optional[np.ndarray],
        face: Optional[np.ndarray],
        attention_mask: Optional[np.ndarray],
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        seq_len = body_pose.shape[0]
        if seq_len <= 1:
            return body_pose, left_hand, right_hand, face, attention_mask

        dropout_ratio = random.uniform(*self.config.frame_dropout_range)
        drop_count = max(1, int(round(seq_len * dropout_ratio)))
        drop_count = min(drop_count, seq_len - 1)
        drop_indices = np.random.choice(seq_len, size=drop_count, replace=False)

        body_pose = body_pose.copy()
        body_pose[drop_indices] = 0.0
        if left_hand is not None:
            left_hand = left_hand.copy()
            left_hand[drop_indices] = 0.0
        if right_hand is not None:
            right_hand = right_hand.copy()
            right_hand[drop_indices] = 0.0
        if face is not None:
            face = face.copy()
            face[drop_indices] = 0.0
        if attention_mask is not None:
            attention_mask = attention_mask.copy()
            attention_mask[drop_indices] = 0.0

        return body_pose, left_hand, right_hand, face, attention_mask

    def _apply_horizontal_flip(
        self,
        body_pose: np.ndarray,
        left_hand: Optional[np.ndarray],
        right_hand: Optional[np.ndarray],
        face: Optional[np.ndarray],
        attention_mask: Optional[np.ndarray],
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        body_pose = body_pose.copy()
        body_pose[:, :, 0] *= -1.0
        for left_idx, right_idx in LandmarkIndices.POSE_FLIP_PAIRS:
            body_pose[:, [left_idx, right_idx], :] = body_pose[:, [right_idx, left_idx], :]

        if left_hand is not None and right_hand is not None:
            flipped_left = right_hand.copy()
            flipped_left[:, :, 0] *= -1.0
            flipped_right = left_hand.copy()
            flipped_right[:, :, 0] *= -1.0
            left_hand, right_hand = flipped_left, flipped_right
        elif left_hand is not None:
            left_hand = left_hand.copy()
            left_hand[:, :, 0] *= -1.0
        elif right_hand is not None:
            right_hand = right_hand.copy()
            right_hand[:, :, 0] *= -1.0

        if face is not None:
            face = face.copy()
            face[:, :, 0] *= -1.0

        return body_pose, left_hand, right_hand, face, attention_mask

    def _temporal_scale(self, sequence: np.ndarray, scale: float) -> np.ndarray:
        """Apply temporal scaling."""
        return self.temporal_aligner.resample(sequence, scale)

    def _add_gaussian_noise(self, sequence: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to sequence."""
        noise = np.random.normal(0, self.config.noise_std, sequence.shape)
        return sequence + noise.astype(sequence.dtype)

    def _rotate_2d(self, sequence: np.ndarray, angle: float) -> np.ndarray:
        """Apply 2D rotation in XY plane."""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotated = sequence.copy()

        # Rotate x and y coordinates (indices 0 and 1)
        x = sequence[:, :, 0] - 0.5
        y = sequence[:, :, 1] - 0.5
        rotated[:, :, 0] = cos_a * x - sin_a * y + 0.5
        rotated[:, :, 1] = sin_a * x + cos_a * y + 0.5

        return rotated

    def _scale(self, sequence: np.ndarray, scale: float) -> np.ndarray:
        """Apply uniform scaling."""
        return sequence * scale


class SignLanguageDataset(Dataset):
    """Dataset for sign language recognition."""

    def __init__(
        self,
        sample_paths: List[str],
        word_to_label: Dict[str, int],
        normalized_dir: str,
        config: DataConfig,
        augment: bool = False,
        mode: str = "train",
        use_hands: bool = True,
        use_face: bool = True,
        emotion_to_label: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize dataset.

        Args:
            sample_paths: List of paths to video files
            word_to_label: Mapping from words to label indices
            emotion_to_label: Optional mapping from emotion names to label indices
            normalized_dir: Directory containing preprocessed .npz files
            config: Data configuration
            augment: Whether to apply augmentation
            mode: 'train', 'val', or 'test'
            use_hands: Include hand landmarks
            use_face: Include face landmarks
        """
        self.sample_paths = sample_paths
        self.word_to_label = word_to_label
        self.emotion_to_label = emotion_to_label or {}
        self.normalized_dir = Path(normalized_dir)
        self.config = config
        self.augment = augment and mode == "train"
        self.mode = mode
        self.use_hands = use_hands
        self.use_face = use_face
        self.loader_error_mode = config.loader_error_mode.lower().strip()
        if self.loader_error_mode not in {"permissive", "strict"}:
            raise ValueError(
                "DataConfig.loader_error_mode must be either 'permissive' or 'strict'"
            )
        self.max_logged_loader_failures = max(1, int(config.max_logged_loader_failures))
        self._npz_index: Dict[str, Path] = {}

        if self.normalized_dir.exists():
            for candidate in self.normalized_dir.rglob("*.npz"):
                self._npz_index.setdefault(candidate.name, candidate)
                normalized_candidate_name = unicodedata.normalize("NFC", candidate.name)
                if normalized_candidate_name != candidate.name:
                    self._npz_index.setdefault(normalized_candidate_name, candidate)

        self.augmentor = Augmentor(config)
        self.pose_normalizer = PoseNormalizer()
        self.hand_normalizer = HandNormalizer()

        # Parse metadata for all samples
        self.metadata_list = []
        self.malformed_metadata_paths: List[str] = []
        for sample_path in sample_paths:
            parsed = self._parse_metadata(sample_path)
            if parsed is None:
                self.malformed_metadata_paths.append(sample_path)
            else:
                self.metadata_list.append(parsed)

        self.malformed_metadata_count = len(self.malformed_metadata_paths)
        self.load_failure_count = 0
        self.failed_sample_paths: List[str] = []

        if self.malformed_metadata_count > 0:
            if self.loader_error_mode == "strict":
                preview = self.malformed_metadata_paths[:5]
                raise ValueError(
                    "Malformed sample names found in strict mode "
                    f"({self.malformed_metadata_count} total). Example(s): {preview}"
                )
            print(
                f"⚠️  [{self.mode}] Skipping {self.malformed_metadata_count} malformed sample names"
            )

    def __len__(self) -> int:
        return len(self.metadata_list)

    def balance_emotion_classes(
        self,
        emotion_to_label: Dict[str, int],
        seed: int = 42,
    ) -> Dict[str, int]:
        """Undersample majority emotion classes for training only.

        Keeps all samples from the minority classes negation and question,
        and downsamples larger classes to the second-largest class count.
        """
        if not emotion_to_label:
            return {}

        minority_labels = {
            emotion_to_label[emotion]
            for emotion in ("negation", "question")
            if emotion in emotion_to_label
        }

        class_samples: Dict[int, List[str]] = {}
        for sample_path in self.sample_paths:
            metadata = self._parse_metadata(sample_path)
            if metadata is None:
                continue
            label = emotion_to_label.get(metadata["emotion"], -1)
            if label < 0:
                continue
            class_samples.setdefault(label, []).append(sample_path)

        if not class_samples:
            return {}

        counts = sorted((len(paths) for paths in class_samples.values()), reverse=True)
        threshold = counts[1] if len(counts) > 1 else counts[0]

        rng = random.Random(seed)
        balanced_items: List[Tuple[str, Dict]] = []
        for label, paths in class_samples.items():
            if label in minority_labels or len(paths) <= threshold:
                selected = paths
            else:
                selected = rng.sample(paths, threshold)

            for path in selected:
                metadata = self._parse_metadata(path)
                if metadata is not None:
                    balanced_items.append((path, metadata))

        balanced_items.sort(key=lambda item: item[0])
        self.sample_paths = [path for path, _ in balanced_items]
        self.metadata_list = [metadata for _, metadata in balanced_items]

        balanced_counts: Dict[str, int] = {}
        for metadata in self.metadata_list:
            emotion = metadata.get("emotion", "")
            balanced_counts[emotion] = balanced_counts.get(emotion, 0) + 1

        return balanced_counts

    def _parse_metadata(self, video_path: str) -> Optional[Dict]:
        """Parse metadata from video filename."""
        path = Path(video_path)
        filename = path.stem
        parts = filename.split("__")

        if len(parts) != 5:
            return None

        word, signer, session, repetition, emotion = [
            unicodedata.normalize("NFC", p).strip() for p in parts
        ]

        return {
            "word": word,
            "signer": signer,
            "session": session,
            "repetition": repetition,
            "emotion": emotion,
            "grammar": emotion,
            "full_path": video_path,
        }

    def _normalize_npz_filename(self, filename: str) -> str:
        """Normalize NPZ filename strings for lookup."""
        filename = unicodedata.normalize("NFC", filename).strip()
        filename = filename.replace(" __", "__").replace("__ ", "__")
        return filename

    def _get_npz_path(self, metadata: Dict) -> Path:
        """Get path to preprocessed .npz file."""
        # Use the exact filename from sample list to avoid mismatches caused by
        # normalization of metadata fields (e.g., trimmed whitespace in tokens).
        filename = Path(metadata["full_path"]).name
        filename = self._normalize_npz_filename(filename)
        direct_path = self.normalized_dir / filename
        if direct_path.exists():
            return direct_path

        cached_path = self._npz_index.get(filename)
        if cached_path is not None and cached_path.exists():
            return cached_path

        normalized_filename = unicodedata.normalize("NFC", filename)
        if normalized_filename != filename:
            cached_normalized_path = self._npz_index.get(normalized_filename)
            if cached_normalized_path is not None and cached_normalized_path.exists():
                return cached_normalized_path

        if self.normalized_dir.exists():
            matches = list(self.normalized_dir.rglob(filename))
            if matches:
                resolved_path = matches[0]
                self._npz_index[filename] = resolved_path
                return resolved_path
            if normalized_filename != filename:
                matches = list(self.normalized_dir.rglob(normalized_filename))
                if matches:
                    resolved_path = matches[0]
                    self._npz_index[normalized_filename] = resolved_path
                    return resolved_path

        return direct_path

    def _load_raw_pose(
        self, metadata: Dict
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Load raw pose data from .npz file.

        Returns:
            body_pose: (seq_len, 33, 3)
            left_hand: (seq_len, 21, 3) or None
            right_hand: (seq_len, 21, 3) or None
            face: (seq_len, 478, 3) or None
            raw_length: Original sequence length
        """
        npz_path = self._get_npz_path(metadata)

        if not npz_path.exists():
            raise FileNotFoundError(f"Missing .npz: {npz_path}")

        data = np.load(npz_path)
        keys = list(data.keys())
        has_validity_masks = {
            "pose_valid",
            "left_hand_valid",
            "right_hand_valid",
            "face_valid",
        }.issubset(keys)

        # Handle separate keys format (pose, hand_left, hand_right, face)
        if "pose" in keys:
            body_pose = data["pose"]
            if self.use_hands:
                if "left_hand" in keys:
                    left_hand = data["left_hand"]
                elif "hand_left" in keys:
                    left_hand = data["hand_left"]
                else:
                    left_hand = None

                if "right_hand" in keys:
                    right_hand = data["right_hand"]
                elif "hand_right" in keys:
                    right_hand = data["hand_right"]
                else:
                    right_hand = None
            else:
                left_hand = None
                right_hand = None
            raw_length = int(np.asarray(data.get("raw_length", body_pose.shape[0])).reshape(()).item())
            face = data["face"] if self.use_face and "face" in keys else None

            if has_validity_masks:
                pose_valid = np.asarray(data["pose_valid"], dtype=bool)
                left_hand_valid = np.asarray(data["left_hand_valid"], dtype=bool)
                right_hand_valid = np.asarray(data["right_hand_valid"], dtype=bool)
                face_valid = np.asarray(data["face_valid"], dtype=bool)
            else:
                pose_valid = np.zeros(body_pose.shape[0], dtype=bool)
                pose_valid[:raw_length] = True
                left_hand_valid = np.zeros(body_pose.shape[0], dtype=bool)
                left_hand_valid[:raw_length] = left_hand is not None
                right_hand_valid = np.zeros(body_pose.shape[0], dtype=bool)
                right_hand_valid[:raw_length] = right_hand is not None
                face_valid = np.zeros(body_pose.shape[0], dtype=bool)
                face_valid[:raw_length] = face is not None

            return (
                body_pose,
                left_hand,
                right_hand,
                face,
                raw_length,
                pose_valid,
                left_hand_valid,
                right_hand_valid,
                face_valid,
            )

        # Handle pose_sequence format
        if "pose_sequence" in keys:
            pose_sequence = data["pose_sequence"]
        else:
            pose_sequence = data[keys[0]]

        # Reshape if needed
        if pose_sequence.ndim == 2:
            # Already flattened: (seq_len, dim)
            if pose_sequence.shape[1] == 99:
                # Body only: (seq_len, 99) -> (seq_len, 33, 3)
                body_flat = pose_sequence
                body_pose = body_flat.reshape(-1, 33, 3)
                left_hand = None
                right_hand = None
            elif pose_sequence.shape[1] == 108:
                # Body with extra zeros: (seq_len, 108) -> take first 99 cols -> (seq_len, 33, 3)
                body_flat = pose_sequence[:, :99]
                body_pose = body_flat.reshape(-1, 33, 3)
                left_hand = None
                right_hand = None
            else:
                raise ValueError(f"Unexpected flattened shape: {pose_sequence.shape}")
        elif pose_sequence.ndim == 3:
            # Multi-stream data
            if pose_sequence.shape[1] == 33:
                body_pose = pose_sequence
                left_hand = None
                right_hand = None
            elif pose_sequence.shape[1] == 75:
                # body + hands
                body_pose = pose_sequence[:, :33, :]
                if self.use_hands:
                    left_hand = pose_sequence[:, 33:54, :]
                    right_hand = pose_sequence[:, 54:75, :]
                else:
                    left_hand = None
                    right_hand = None
            elif pose_sequence.shape[1] == 36:
                # body + extra (take first 33)
                body_pose = pose_sequence[:, :33, :]
                left_hand = None
                right_hand = None
            else:
                # Assume body only
                body_pose = pose_sequence
                left_hand = None
                right_hand = None
        else:
            raise ValueError(f"Unexpected pose shape: {pose_sequence.shape}")

        raw_length = int(np.asarray(data.get("raw_length", pose_sequence.shape[0])).reshape(()).item())
        face = None

        if has_validity_masks:
            pose_valid = np.asarray(data["pose_valid"], dtype=bool)
            left_hand_valid = np.asarray(data["left_hand_valid"], dtype=bool)
            right_hand_valid = np.asarray(data["right_hand_valid"], dtype=bool)
            face_valid = np.asarray(data["face_valid"], dtype=bool)
        else:
            pose_valid = np.zeros(pose_sequence.shape[0], dtype=bool)
            pose_valid[:raw_length] = True
            left_hand_valid = np.zeros(pose_sequence.shape[0], dtype=bool)
            left_hand_valid[:raw_length] = left_hand is not None
            right_hand_valid = np.zeros(pose_sequence.shape[0], dtype=bool)
            right_hand_valid[:raw_length] = right_hand is not None
            face_valid = np.zeros(pose_sequence.shape[0], dtype=bool)

        return (
            body_pose,
            left_hand,
            right_hand,
            face,
            raw_length,
            pose_valid,
            left_hand_valid,
            right_hand_valid,
            face_valid,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        metadata = self.metadata_list[idx]
        label = self.word_to_label[metadata["word"]]
        emotion_label = self.emotion_to_label.get(metadata.get("emotion", ""), -1)

        load_failed = 0
        load_error = ""

        try:
            (
                body_pose,
                left_hand,
                right_hand,
                face,
                raw_length,
                pose_valid,
                left_hand_valid,
                right_hand_valid,
                face_valid,
            ) = self._load_raw_pose(metadata)
        except Exception as e:
            if self.loader_error_mode == "strict":
                raise RuntimeError(
                    f"Failed to load sample in strict mode: {metadata['full_path']} ({e})"
                ) from e

            load_failed = 1
            load_error = str(e)
            self.load_failure_count += 1
            self.failed_sample_paths.append(metadata["full_path"])

            if self.load_failure_count <= self.max_logged_loader_failures:
                print(f"⚠️  Error loading {metadata['full_path']}: {e}")
            elif self.load_failure_count == self.max_logged_loader_failures + 1:
                print(
                    f"⚠️  Additional load failures suppressed after {self.max_logged_loader_failures} logs"
                )

            # Return zeros on error
            body_pose = np.zeros((self.config.max_seq_length, 33, 3), dtype=np.float32)
            left_hand = (
                np.zeros((self.config.max_seq_length, 21, 3), dtype=np.float32)
                if self.use_hands
                else None
            )
            right_hand = (
                np.zeros((self.config.max_seq_length, 21, 3), dtype=np.float32)
                if self.use_hands
                else None
            )
            face = (
                np.zeros((self.config.max_seq_length, 478, 3), dtype=np.float32)
                if self.use_face
                else None
            )
            raw_length = 0
            pose_valid = np.zeros(self.config.max_seq_length, dtype=bool)
            left_hand_valid = np.zeros(self.config.max_seq_length, dtype=bool)
            right_hand_valid = np.zeros(self.config.max_seq_length, dtype=bool)
            face_valid = np.zeros(self.config.max_seq_length, dtype=bool)

        # Apply normalization
        body_pose = self.pose_normalizer.normalize(body_pose)
        if left_hand is not None:
            left_hand = self.hand_normalizer.normalize(left_hand)
        if right_hand is not None:
            right_hand = self.hand_normalizer.normalize(right_hand)

        def _align_mask(mask: np.ndarray, target_length: int) -> np.ndarray:
            aligned = np.asarray(mask, dtype=bool).reshape(-1)
            if aligned.shape[0] > target_length:
                return aligned[:target_length]
            if aligned.shape[0] < target_length:
                padding = np.zeros(target_length - aligned.shape[0], dtype=bool)
                return np.concatenate([aligned, padding], axis=0)
            return aligned

        target_length = body_pose.shape[0]
        pose_valid = _align_mask(pose_valid, target_length)
        left_hand_valid = _align_mask(left_hand_valid, target_length)
        right_hand_valid = _align_mask(right_hand_valid, target_length)
        face_valid = _align_mask(face_valid, target_length)

        # A frame is valid whenever pose landmarks exist.
        # Hand visibility is already preserved separately in the hand masks,
        # so we do not discard pose frames just because one or both hands are missing.
        frame_valid = pose_valid
        attention_mask = frame_valid.astype(np.float32)

        # Apply augmentation
        body_pose, left_hand, right_hand, face, attention_mask = self.augmentor.augment(
            body_pose,
            left_hand,
            right_hand,
            face,
            attention_mask,
            self.augment,
        )

        # Flatten landmarks to feature vectors
        body_features = body_pose.reshape(body_pose.shape[0], -1)  # (seq_len, 99)

        if left_hand is not None:
            left_hand = left_hand.reshape(left_hand.shape[0], -1)  # (seq_len, 63)
        if right_hand is not None:
            right_hand = right_hand.reshape(right_hand.shape[0], -1)  # (seq_len, 63)
        if face is not None:
            face = face.reshape(face.shape[0], -1)  # (seq_len, 1434)
        elif self.use_face:
            face = np.zeros((self.config.max_seq_length, 478 * 3), dtype=np.float32)

        # Create attention mask
        seq_length = min(raw_length, self.config.max_seq_length)
        if attention_mask is None:
            attention_mask = np.zeros(self.config.max_seq_length, dtype=np.float32)
            attention_mask[:seq_length] = 1
        else:
            attention_mask = np.asarray(attention_mask, dtype=np.float32)

        # Convert to tensors
        result = {
            "body_pose": torch.FloatTensor(body_features),
            "label": torch.tensor(label, dtype=torch.long),
            "emotion_label": torch.tensor(emotion_label, dtype=torch.long),
            "attention_mask": torch.FloatTensor(attention_mask),
            "seq_length": torch.tensor(seq_length, dtype=torch.long),
            "word": metadata["word"],
            "signer": metadata["signer"],
            "emotion": metadata["emotion"],
            "grammar": metadata["grammar"],
            "sample_path": metadata["full_path"],
            "load_failed": torch.tensor(load_failed, dtype=torch.long),
            "load_error": load_error,
        }

        if self.use_hands:
            result["left_hand"] = (
                torch.FloatTensor(left_hand) if left_hand is not None 
                else torch.zeros((self.config.max_seq_length, 63), dtype=torch.float32)
            )
            result["right_hand"] = (
                torch.FloatTensor(right_hand) if right_hand is not None 
                else torch.zeros((self.config.max_seq_length, 63), dtype=torch.float32)
            )

        if self.use_face:
            result["face"] = (
                torch.FloatTensor(face)
                if face is not None
                else torch.zeros((self.config.max_seq_length, 478 * 3), dtype=torch.float32)
            )

        return result


def create_data_loaders(
    config: DataConfig,
    train_samples: List[str],
    val_samples: List[str],
    test_samples: List[str],
    word_to_label: Dict[str, int],
    emotion_to_label: Optional[Dict[str, int]] = None,
    batch_size: int = 16,
    num_workers: int = 2,
    use_hands: bool = True,
    use_face: bool = True,
    balance_emotions: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        config: Data configuration
        train_samples: List of training sample paths
        val_samples: List of validation sample paths
        test_samples: List of test sample paths
        word_to_label: Word to label mapping
        emotion_to_label: Optional emotion to label mapping
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_hands: Include hand landmarks
        use_face: Include face landmarks

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    pin_memory = torch.cuda.is_available()

    # Create datasets
    train_dataset = SignLanguageDataset(
        sample_paths=train_samples,
        word_to_label=word_to_label,
        emotion_to_label=emotion_to_label,
        normalized_dir=config.normalized_dir,
        config=config,
        augment=config.augmentation,
        mode="train",
        use_hands=use_hands,
        use_face=use_face,
    )

    if balance_emotions and emotion_to_label:
        balanced_counts = train_dataset.balance_emotion_classes(
            emotion_to_label=emotion_to_label,
            seed=42,
        )
        print("\n⚖️  Balanced emotion distribution:")
        for emotion, label in sorted(emotion_to_label.items(), key=lambda item: item[1]):
            print(f"   {emotion}: {balanced_counts.get(emotion, 0)}")

    val_dataset = SignLanguageDataset(
        sample_paths=val_samples,
        word_to_label=word_to_label,
        emotion_to_label=emotion_to_label,
        normalized_dir=config.normalized_dir,
        config=config,
        augment=False,
        mode="val",
        use_hands=use_hands,
        use_face=use_face,
    )

    test_dataset = SignLanguageDataset(
        sample_paths=test_samples,
        word_to_label=word_to_label,
        emotion_to_label=emotion_to_label,
        normalized_dir=config.normalized_dir,
        config=config,
        augment=False,
        mode="test",
        use_hands=use_hands,
        use_face=use_face,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data pipeline
    config = DataConfig()
    print("✅ DataConfig created")
    print(
        f"   Body dim: {config.body_dim}, Hand dim: {config.hand_dim}, Face dim: {config.face_dim}"
    )

    # Test normalizer
    normalizer = PoseNormalizer()
    dummy_pose = np.random.randn(100, 33, 3).astype(np.float32)
    normalized = normalizer.normalize(dummy_pose)
    print(f"✅ PoseNormalizer test: {dummy_pose.shape} -> {normalized.shape}")

    # Test augmentor
    augmentor = Augmentor(config)
    augmented = augmentor.augment(dummy_pose, apply_augmentation=False)
    print(f"✅ Augmentor test: {augmented[0].shape}")

    print("\n✅ All data pipeline components working correctly")
