"""Baseline single-stream model for word recognition."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SingleStreamBaseline(nn.Module):
    """Simple baseline that keeps the existing SignNet-style forward signature.

    The model uses only body pose features for word classification and ignores
    the optional emotion head path.
    """

    def __init__(
        self,
        num_classes: int,
        body_dim: int,
        hand_dim: int,
        face_dim: int,
        d_model: int,
        num_encoder_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        max_seq_length: int = 150,
        use_face: bool = True,
        use_hands: bool = True,
        num_emotions: Optional[int] = None,
    ):
        super().__init__()
        from .signet_v2 import SignNetV2

        self.model = SignNetV2(
            num_classes=num_classes,
            num_emotions=None,
            body_dim=body_dim,
            hand_dim=hand_dim,
            face_dim=face_dim,
            d_model=d_model,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_length=max_seq_length,
            use_face=use_face,
            use_hands=use_hands,
        )

    def forward(
        self,
        body_pose: torch.Tensor,
        left_hand: Optional[torch.Tensor] = None,
        right_hand: Optional[torch.Tensor] = None,
        face: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(body_pose, left_hand, right_hand, face, attention_mask)

    def forward_with_aux(
        self,
        body_pose: torch.Tensor,
        left_hand=None,
        right_hand=None,
        face=None,
        attention_mask=None,
    ):
        """
        Matches SignNetV2.forward_with_aux() signature.
        Baseline has no emotion head, so emotion logits are always None.
        """
        logits = self.forward(
            body_pose, left_hand, right_hand, face, attention_mask
        )
        return logits, None
