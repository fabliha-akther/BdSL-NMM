"""BdSL-SPOTER baseline model adapted from the original SPOTER architecture."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal pose sequences."""

    def __init__(self, d_model: int, max_seq_length: int = 150):
        super().__init__()

        pe = torch.zeros(max_seq_length + 1, d_model)
        position = torch.arange(0, max_seq_length + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class BdSLSPOTERBaseline(nn.Module):
    """SPOTER baseline with a multi-stream input adapter for BdSL data."""

    def __init__(
        self,
        num_classes: int = 74,
        face_dim: int = 1434,
        dropout: Optional[float] = None,
        max_seq_length: int = 150,
        **kwargs: Any,
    ):
        super().__init__()

        kwargs.pop("num_emotions", None)

        self.d_model = 108
        self.max_seq_length = max_seq_length
        self.dropout = 0.15 if dropout is None else dropout
        self.body_dim = 99
        self.hand_dim = 63
        self.face_dim = face_dim

        self.input_projection = nn.Sequential(
            nn.Linear(self.body_dim + 2 * self.hand_dim + self.face_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )

        self.positional_encoding = PositionalEncoding(self.d_model, max_seq_length)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=9,
                dim_feedforward=512,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=8,
        )

        self.class_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.5),
            nn.Linear(self.d_model // 2, num_classes),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _resolve_inputs(
        self,
        body_pose: Optional[torch.Tensor],
        left_hand: Optional[torch.Tensor],
        right_hand: Optional[torch.Tensor],
        face: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        reference = next(
            (tensor for tensor in (body_pose, left_hand, right_hand, face, attention_mask) if tensor is not None),
            None,
        )
        if reference is None:
            raise ValueError("At least one input tensor or attention_mask must be provided.")

        if reference.dim() == 2:
            batch_size, seq_len = reference.shape
            device = reference.device
            dtype = reference.dtype
        elif reference.dim() == 3:
            batch_size, seq_len, _ = reference.shape
            device = reference.device
            dtype = reference.dtype
        else:
            raise ValueError("Inputs must be rank-2 masks or rank-3 feature tensors.")

        def ensure_stream(
            tensor: Optional[torch.Tensor], feature_dim: int, name: str
        ) -> torch.Tensor:
            if tensor is None:
                return torch.zeros(batch_size, seq_len, feature_dim, device=device, dtype=dtype)
            if tensor.dim() != 3:
                raise ValueError(f"{name} must have shape [batch, seq_len, features].")
            if tensor.size(0) != batch_size or tensor.size(1) != seq_len:
                raise ValueError(f"{name} must match the batch and sequence dimensions of the other inputs.")
            return tensor

        if attention_mask is None:
            resolved_mask = None
        else:
            if attention_mask.dim() != 2:
                raise ValueError("attention_mask must have shape [batch, seq_len].")
            if attention_mask.size(0) != batch_size or attention_mask.size(1) != seq_len:
                raise ValueError("attention_mask must match the batch and sequence dimensions of the other inputs.")
            resolved_mask = attention_mask.to(device=device)

        return (
            ensure_stream(body_pose, self.body_dim, "body_pose"),
            ensure_stream(left_hand, self.hand_dim, "left_hand"),
            ensure_stream(right_hand, self.hand_dim, "right_hand"),
            ensure_stream(face, self.face_dim, "face"),
            resolved_mask,
        )

    def forward(
        self,
        body_pose: Optional[torch.Tensor],
        left_hand: Optional[torch.Tensor] = None,
        right_hand: Optional[torch.Tensor] = None,
        face: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        body_pose, left_hand, right_hand, face, attention_mask = self._resolve_inputs(
            body_pose, left_hand, right_hand, face, attention_mask
        )

        x = torch.cat([body_pose, left_hand, right_hand, face], dim=-1)
        x = self.input_projection(x)

        batch_size, _, _ = x.shape
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, x], dim=1)

        x = self.positional_encoding(x)

        if attention_mask is not None:
            class_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
            full_mask = torch.cat([class_mask, attention_mask], dim=1)
            transformer_mask = full_mask == 0
            encoded = self.transformer_encoder(x, src_key_padding_mask=transformer_mask)
        else:
            encoded = self.transformer_encoder(x)

        logits = self.classifier(encoded[:, 0])
        return logits

    def forward_with_aux(
        self,
        body_pose: Optional[torch.Tensor],
        left_hand: Optional[torch.Tensor] = None,
        right_hand: Optional[torch.Tensor] = None,
        face: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        logits = self.forward(body_pose, left_hand, right_hand, face, attention_mask)
        return logits, None


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}