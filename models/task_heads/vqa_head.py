"""
VQA classification head.

Takes the [CLS] token from the encoder and produces per-answer logits.
Supports both closed-set classification (VQA v2, 3129 answers) and
open-ended generation via a language model head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gdt.config import ModelConfig


class VQAHead(nn.Module):
    """
    Two-layer MLP classification head for VQA.

    @param cfg         - model configuration
    @param num_answers - size of the answer vocabulary (default: 3129)
    """

    def __init__(self, cfg: ModelConfig, num_answers: int = 3129) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, num_answers),
        )

    def forward(self, cls_token: Tensor) -> Tensor:
        """
        @param cls_token - [B, D]
        @returns logits [B, num_answers]
        """
        return self.classifier(cls_token)

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Soft-label cross-entropy (VQA v2 uses per-answer soft scores).

        @param logits  - [B, num_answers]
        @param targets - [B, num_answers] soft scores in [0, 1]
        @returns scalar loss
        """
        log_probs = F.log_softmax(logits, dim=-1)
        # Normalise targets to form a valid distribution
        norm_targets = targets / (targets.sum(dim=-1, keepdim=True).clamp(min=1e-6))
        return -(norm_targets * log_probs).sum(dim=-1).mean()

    @torch.no_grad()
    def predict(self, cls_token: Tensor) -> Tensor:
        """
        @param cls_token - [B, D]
        @returns predicted answer indices [B]
        """
        return self.forward(cls_token).argmax(dim=-1)
