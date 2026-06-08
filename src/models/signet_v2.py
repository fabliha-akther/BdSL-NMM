"""
Enhanced Sign Language Recognition Model (SignNet-V2)
=====================================================

Multi-stream spatiotemporal transformer architecture for Bengali Sign Language recognition.
Improvements over baseline BDSLW_SPOTER:
1. Multi-stream input (body + hands + face)
2. Hierarchical temporal modeling
3. Spatial attention mechanisms
4. Advanced augmentation pipeline
5. Mixed precision training support

Author: BDSL Recognition Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, LayerNorm, Dropout
import math
from typing import Optional, Dict


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    def __init__(self, d_model: int, max_seq_length: int = 300, dropout: float = 0.1):
        super().__init__()
        self.dropout = Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(
            max_seq_length + 2, d_model
        )  # +2 for class token and potential padding
        position = torch.arange(0, max_seq_length + 2, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_length+2, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TemporalConvBlock(nn.Module):
    """1D Temporal convolution block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = Dropout(dropout)

        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Conv1d(in_channels, out_channels, 1, stride)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temporal convolution.

        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Output tensor of shape (batch, channels, seq_len)
        """
        residual = self.residual(x)
        x = self.conv(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x + residual


class SpatialAttentionBlock(nn.Module):
    """Spatial attention mechanism for landmark sequences."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = LayerNorm(d_model)
        self.dropout = Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply spatial attention to input features.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Attended features of same shape
        """
        attended, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm(x + self.dropout(attended))
        return x


class StreamSpecificEncoder(nn.Module):
    """Encoder for a specific input stream (body, left_hand, right_hand, face)."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), Dropout(dropout * 0.5)
        )

        # Temporal convolutional layers
        self.temporal_convs = nn.ModuleList(
            [
                TemporalConvBlock(d_model, d_model, kernel_size=3, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        # Spatial attention
        self.spatial_attention = SpatialAttentionBlock(d_model, num_heads, dropout)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, seq_length: int) -> torch.Tensor:
        """Encode stream-specific features.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            seq_length: Actual sequence length (before padding)

        Returns:
            Encoded features of shape (batch, seq_len, d_model)
        """
        # Project input
        x = self.input_projection(x)

        # Apply temporal convolutions
        for conv in self.temporal_convs:
            x_conv = x.transpose(1, 2)  # (batch, d_model, seq_len)
            x_conv = conv(x_conv)
            x = x_conv.transpose(1, 2)  # (batch, seq_len, d_model)

        # Apply spatial attention
        x = self.spatial_attention(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        return x



class EmotionFaceEncoder(nn.Module):
    """Dedicated face encoder for emotion recognition."""

    def __init__(self, input_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x: torch.Tensor, seq_length: Optional[int] = None) -> torch.Tensor:
        x = self.projection(x)
        x = self.transformer(x)
        return x


class EmotionHeadPoseEncoder(nn.Module):
    """Dedicated head-pose encoder for expression recognition."""

    def __init__(self, input_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = self.transformer(x)
        return x


class CrossStreamFusion(nn.Module):
    """Multi-stream fusion using cross-attention."""

    def __init__(
        self,
        d_model: int,
        num_streams: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_streams = num_streams

        # Cross-attention for each stream
        self.cross_attentions = nn.ModuleList(
            [
                MultiheadAttention(
                    d_model, num_heads, dropout=dropout, batch_first=True
                )
                for _ in range(num_streams)
            ]
        )

        # Fusion projections
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_model * num_streams, d_model),
            nn.LayerNorm(d_model),
            Dropout(dropout),
        )

        self.norm = LayerNorm(d_model)

    def forward(self, stream_features: list, stream_lengths: list) -> torch.Tensor:
        """Fuse features from multiple streams using cross-attention.

        Args:
            stream_features: List of tensors, each (batch, seq_len, d_model)
            stream_lengths: List of actual sequence lengths for each stream

        Returns:
            Fused features of shape (batch, seq_len, d_model)
        """
        batch_size = stream_features[0].size(0)
        max_len = max(f.size(1) for f in stream_features)

        # Pad all streams to same length
        padded_streams = []
        for features, length in zip(stream_features, stream_lengths):
            if features.size(1) < max_len:
                padding = torch.zeros(
                    batch_size,
                    max_len - features.size(1),
                    self.d_model,
                    device=features.device,
                    dtype=features.dtype,
                )
                padded = torch.cat([features, padding], dim=1)
            else:
                padded = features
            padded_streams.append(padded)

        # Stack streams as additional "time" dimension: (batch, num_streams * seq_len, d_model)
        # Or concat along feature dimension
        concat_features = torch.cat(
            padded_streams, dim=-1
        )  # (batch, max_len, d_model * num_streams)

        # Project to single stream
        fused = self.fusion_proj(concat_features)

        # Apply cross-attention between streams
        attended = []
        for i, stream in enumerate(padded_streams):
            attn_output, _ = self.cross_attentions[i](stream, fused, fused)
            attended.append(attn_output)

        # Combine attended features
        combined = torch.mean(torch.stack(attended), dim=0)  # Average across streams
        combined = self.norm(combined + stream_features[0])  # Residual connection

        return combined


class HierarchicalTemporalEncoder(nn.Module):
    """Hierarchical temporal encoder using multi-scale temporal windows."""

    def __init__(
        self,
        d_model: int,
        num_scales: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_scales = num_scales

        # Multi-scale temporal attention
        self.temporal_attentions = nn.ModuleList(
            [
                MultiheadAttention(
                    d_model, num_heads, dropout=dropout, batch_first=True
                )
                for _ in range(num_scales)
            ]
        )

        # Temporal pooling for each scale
        self.temporal_pools = nn.ModuleList(
            [
                nn.AvgPool1d(kernel_size=2**s, stride=2**s) if s > 0 else nn.Identity()
                for s in range(num_scales)
            ]
        )

        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(d_model * num_scales, d_model),
            nn.LayerNorm(d_model),
            Dropout(dropout),
        )

        self.norm = LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply hierarchical temporal encoding.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Temporally encoded features
        """
        batch_size, seq_len, d_model = x.shape

        # Apply multi-scale processing
        scale_features = []
        for i, (attention, pool) in enumerate(
            zip(self.temporal_attentions, self.temporal_pools)
        ):
            if i == 0:
                scale_x = x
            else:
                # Pool and interpolate back
                x_pooled = x.transpose(1, 2)  # (batch, d_model, seq_len)
                x_pooled = pool(x_pooled)
                scale_len = x_pooled.size(2)
                scale_x = x_pooled.transpose(1, 2)  # (batch, scale_len, d_model)

                # Interpolate to original length
                if scale_len != seq_len:
                    scale_x = F.interpolate(
                        scale_x.transpose(1, 2),
                        size=seq_len,
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)

            # Apply temporal attention
            attended, _ = attention(scale_x, scale_x, scale_x)
            scale_features.append(attended)

        # Concatenate and fuse
        concat = torch.cat(
            scale_features, dim=-1
        )  # (batch, seq_len, d_model * num_scales)
        fused = self.scale_fusion(concat)

        return self.norm(x + fused)


class GlobalTemporalEncoder(nn.Module):
    """Transformer encoder for global temporal modeling."""

    def __init__(
        self,
        d_model: int,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply transformer encoding to temporal sequence.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            key_padding_mask: Optional mask for padding (True = padding)

        Returns:
            Encoded features of same shape
        """
        encoded = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        return self.norm(encoded)


class ClassificationHead(nn.Module):
    """Multi-layer classification head with residual connections."""

    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.3):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify input features.

        Args:
            x: Input tensor of shape (batch, d_model)

        Returns:
            Class logits of shape (batch, num_classes)
        """
        return self.classifier(x)


class SignNetV2(nn.Module):
    """
    SignNet-V2: Enhanced Multi-Stream Spatiotemporal Transformer for Sign Language Recognition

    Architecture:
    1. Multi-stream input processing (body: 33 landmarks, left_hand: 21, right_hand: 21, face: 468)
    2. Stream-specific encoders with temporal convolutions and spatial attention
    3. Cross-stream fusion using attention mechanisms
    4. Hierarchical temporal encoder for multi-scale modeling
    5. Global transformer encoder for sequence understanding
    6. Classification head for sign prediction

    Key improvements over baseline SPOTER:
    - Multi-stream architecture captures hand and facial expressions
    - Hierarchical temporal modeling handles variable-length sequences
    - Cross-stream attention learns inter-stream relationships
    - Enhanced regularization through dropout and label smoothing
    """

    def __init__(
        self,
        num_classes: int = 72,
        num_emotions: Optional[int] = None,
        body_dim: int = 99,  # 33 landmarks * 3 coords
        hand_dim: int = 63,  # 21 landmarks * 3 coords
        face_dim: int = 1404,  # 468 landmarks * 3 coords
        d_model: int = 128,
        num_encoder_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.2,
        max_seq_length: int = 150,
        use_face: bool = True,
        use_hands: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_emotions = num_emotions
        self.d_model = d_model
        self.use_face = use_face
        self.use_hands = use_hands

        # Calculate input dimensions based on enabled streams
        self.body_dim = body_dim
        self.hand_dim = hand_dim
        self.face_dim = face_dim

        # Stream-specific encoders
        self.body_encoder = StreamSpecificEncoder(
            input_dim=body_dim,
            d_model=d_model,
            num_layers=2,
            num_heads=num_heads // 2,
            dropout=dropout,
        )

        if use_hands:
            self.left_hand_encoder = StreamSpecificEncoder(
                input_dim=hand_dim,
                d_model=d_model,
                num_layers=2,
                num_heads=num_heads // 4,
                dropout=dropout,
            )

            self.right_hand_encoder = StreamSpecificEncoder(
                input_dim=hand_dim,
                d_model=d_model,
                num_layers=2,
                num_heads=num_heads // 4,
                dropout=dropout,
            )

        if use_face:
            self.face_encoder = StreamSpecificEncoder(
                input_dim=face_dim,
                d_model=d_model,
                num_layers=2,
                num_heads=num_heads // 2,
                dropout=dropout,
            )

        # Cross-stream fusion
        num_streams = 1 + (2 if use_hands else 0) + (1 if use_face else 0)
        self.cross_stream_fusion = CrossStreamFusion(
            d_model=d_model,
            num_streams=num_streams,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Hierarchical temporal encoder
        self.hierarchical_encoder = HierarchicalTemporalEncoder(
            d_model=d_model, num_scales=3, num_heads=num_heads, dropout=dropout
        )

        # Global temporal encoder
        self.global_encoder = GlobalTemporalEncoder(
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Class token for classification
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)

        # Classification head
        self.classifier = ClassificationHead(
            d_model=d_model, num_classes=num_classes, dropout=dropout
        )

        self.emotion_classifier = None
        self.emotion_projection = None
        self.emotion_attention_pool = None
        self.emotion_query = None
        self.emotion_head_pose_encoder = None
        self.emotion_temporal_pool = None
        if num_emotions is not None and num_emotions > 0:
            # Use a compact hybrid expression branch built from head motion and lip motion.
            self.emotion_head_pose_encoder = EmotionHeadPoseEncoder(
                input_dim=8,
                d_model=64,
                dropout=dropout,
            )
            self.emotion_temporal_pool = nn.Sequential(
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.emotion_classifier = nn.Linear(64, num_emotions)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode_backbone(
        self,
        body_pose: torch.Tensor,
        left_hand: Optional[torch.Tensor] = None,
        right_hand: Optional[torch.Tensor] = None,
        face: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_face_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = body_pose.size(0)

        # Get sequence length from body pose
        seq_length = body_pose.size(1)

        # Encode each stream
        stream_features = []
        stream_lengths = []

        # Body pose encoding
        body_features = self.body_encoder(body_pose, seq_length)
        stream_features.append(body_features)
        stream_lengths.append(seq_length)

        # Hand encoding
        if self.use_hands and left_hand is not None:
            left_features = self.left_hand_encoder(left_hand, seq_length)
            stream_features.append(left_features)
            stream_lengths.append(seq_length)

        if self.use_hands and right_hand is not None:
            right_features = self.right_hand_encoder(right_hand, seq_length)
            stream_features.append(right_features)
            stream_lengths.append(seq_length)

        # Face encoding
        if self.use_face and face is not None:
            face_features = self.face_encoder(face, seq_length)
            stream_features.append(face_features)
            stream_lengths.append(seq_length)

        # Cross-stream fusion
        fused_features = self.cross_stream_fusion(stream_features, stream_lengths)

        # Add class token
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, fused_features], dim=1)

        # Update attention mask to include class token
        if attention_mask is not None:
            class_mask = torch.ones(batch_size, 1, device=attention_mask.device)
            full_mask = torch.cat([class_mask, attention_mask], dim=1)
            # Convert to transformer format: True = padding
            src_key_padding_mask = full_mask == 0
        else:
            src_key_padding_mask = None

        # Hierarchical temporal encoding
        x = self.hierarchical_encoder(x)

        # Global temporal encoding
        x = self.global_encoder(x, src_key_padding_mask)

        class_representation = x[:, 0]
        if return_face_features:
            return class_representation, face_features if 'face_features' in locals() else None
        return class_representation

    def _pool_face_features(
        self, face_features: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if face_features is None:
            return None
        B = face_features.size(0)
        query = self.emotion_query.expand(B, -1, -1)
        key_padding_mask = (
            (attention_mask == 0)
            if attention_mask is not None
            else None
        )
        pooled, _ = self.emotion_attention_pool(
            query,
            face_features,
            face_features,
            key_padding_mask=key_padding_mask
        )
        return pooled.squeeze(1)

    def forward(
        self,
        body_pose: torch.Tensor,
        left_hand: Optional[torch.Tensor] = None,
        right_hand: Optional[torch.Tensor] = None,
        face: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of SignNet-V2.

        Args:
            body_pose: Body pose landmarks (batch, seq_len, body_dim)
            left_hand: Left hand landmarks (batch, seq_len, hand_dim) or None
            right_hand: Right hand landmarks (batch, seq_len, hand_dim) or None
            face: Face landmarks (batch, seq_len, face_dim) or None
            attention_mask: Attention mask for padding (batch, seq_len)

        Returns:
            Class logits (batch, num_classes)
        """
        class_representation = self._encode_backbone(
            body_pose, left_hand, right_hand, face, attention_mask
        )

        # Classification
        logits = self.classifier(class_representation)

        return logits

    def forward_with_aux(
        self,
        body_pose: torch.Tensor,
        left_hand: Optional[torch.Tensor] = None,
        right_hand: Optional[torch.Tensor] = None,
        face: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass returning sign logits and optional emotion logits."""
        class_representation, face_features = self._encode_backbone(
            body_pose,
            left_hand,
            right_hand,
            face,
            attention_mask,
            return_face_features=True,
        )
        sign_logits = self.classifier(class_representation)
        emotion_logits = None
        if self.emotion_classifier is not None and self.emotion_head_pose_encoder is not None and face is not None:

            def z_norm_signal(sig, mask=None):
                if mask is not None:
                    mf = mask.float()
                    n = mf.sum(dim=1, keepdim=True).clamp(min=1.0)
                    mu = (sig * mf).sum(dim=1, keepdim=True) / n
                    var = ((sig - mu).pow(2) * mf).sum(dim=1, keepdim=True) / n
                else:
                    mu = sig.mean(dim=1, keepdim=True)
                    var = sig.var(dim=1, keepdim=True, unbiased=False)
                return (sig - mu) / (var.sqrt() + 1e-6)

            pose_xyz = body_pose.contiguous().reshape(
                body_pose.size(0), body_pose.size(1), 33, 3
            )
            assert pose_xyz.shape[-2:] == (33, 3)

            nose      = pose_xyz[:, :, 0, :]
            left_ear  = pose_xyz[:, :, 7, :]
            right_ear = pose_xyz[:, :, 8, :]

            raw_head_tilt = left_ear[:, :, 1] - right_ear[:, :, 1]
            raw_head_nod  = nose[:, :, 1] - 0.5 * (
                left_ear[:, :, 1] + right_ear[:, :, 1]
            )

            face_xyz = face.contiguous().reshape(
                face.size(0), face.size(1), -1, 3
            )
            n_lm = face_xyz.size(2)

            lc_idx, rc_idx = min(61, n_lm - 1), min(291, n_lm - 1)
            el_idx, er_idx = min(159, n_lm - 1), min(386, n_lm - 1)
            bl_idx, br_idx = min(70, n_lm - 1), min(300, n_lm - 1)

            lip_left_y = face_xyz[:, :, lc_idx, 1]
            lip_right_y = face_xyz[:, :, rc_idx, 1]
            lip_smile = lip_right_y - lip_left_y

            eye_left_y  = face_xyz[:, :, el_idx, 1]
            eye_right_y = face_xyz[:, :, er_idx, 1]
            brow_left_y = face_xyz[:, :, bl_idx, 1]
            brow_right_y = face_xyz[:, :, br_idx, 1]
            raw_eyebrow = (
                brow_left_y + brow_right_y
            ) * 0.5 - (eye_left_y + eye_right_y) * 0.5

            amask = attention_mask

            z_head_tilt     = z_norm_signal(raw_head_tilt, amask)
            z_head_nod      = z_norm_signal(raw_head_nod, amask)
            z_eyebrow_raise = z_norm_signal(raw_eyebrow, amask)
            z_lip_smile     = z_norm_signal(lip_smile, amask)
            z_lip_smile     = torch.sign(z_lip_smile) * z_lip_smile.abs().clamp(min=1e-8).pow(0.5)

            abs_head_tilt   = z_head_tilt.abs()
            abs_lip_smile   = z_lip_smile.abs()
            vel_head_tilt   = torch.zeros_like(z_head_tilt)
            vel_head_tilt[:, 1:] = z_head_tilt[:, 1:] - z_head_tilt[:, :-1]
            vel_lip_smile   = torch.zeros_like(z_lip_smile)
            vel_lip_smile[:, 1:] = z_lip_smile[:, 1:] - z_lip_smile[:, :-1]

            features = torch.stack(
                [z_head_tilt, z_head_nod, z_eyebrow_raise, z_lip_smile,
                 abs_head_tilt, abs_lip_smile, vel_head_tilt, vel_lip_smile],
                dim=-1
            )

            encoded = self.emotion_head_pose_encoder(features)

            if attention_mask is not None:
                mf = attention_mask.unsqueeze(-1).float()
                mean_pool = (encoded * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
                masked_max = encoded + (1.0 - mf) * (-1e9)
                max_pool  = masked_max.max(dim=1).values
            else:
                mean_pool = encoded.mean(dim=1)
                max_pool = encoded.max(dim=1).values

            pooled         = torch.cat([mean_pool, max_pool], dim=-1)
            pooled         = self.emotion_temporal_pool(pooled)
            emotion_logits = self.emotion_classifier(pooled)
        return sign_logits, emotion_logits

    def get_embedding(
        self,
        body_pose: torch.Tensor,
        left_hand: Optional[torch.Tensor] = None,
        right_hand: Optional[torch.Tensor] = None,
        face: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get feature embedding for downstream tasks."""
        return self._encode_backbone(body_pose, left_hand, right_hand, face, attention_mask)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    # Test model
    model = SignNetV2(
        num_classes=72,
        body_dim=99,
        hand_dim=63,
        face_dim=1404,
        d_model=128,
        num_encoder_layers=4,
        num_heads=8,
        d_ff=512,
        dropout=0.2,
        use_face=True,
        use_hands=True,
    )

    params = count_parameters(model)
    print(
        f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable"
    )
    print(f"Model size: {params['total'] * 4 / 1024**2:.2f} MB")

    # Test forward pass
    batch_size = 4
    seq_length = 150

    body_pose = torch.randn(batch_size, seq_length, 99)
    left_hand = torch.randn(batch_size, seq_length, 63)
    right_hand = torch.randn(batch_size, seq_length, 63)
    face = torch.randn(batch_size, seq_length, 1404)
    mask = torch.ones(batch_size, seq_length)

    with torch.no_grad():
        logits = model(body_pose, left_hand, right_hand, face, mask)

    print(
        f"Input shapes: body={body_pose.shape}, hands=({left_hand.shape}, {right_hand.shape}), face={face.shape}"
    )
    print(f"Output shape: {logits.shape}")
    print(f"Expected: ({batch_size}, 72)")
