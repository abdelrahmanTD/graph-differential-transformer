"""
Graph-Differential Transformer block.

Block(x) = FFN(Norm(Attn_diff_graph(x) + x)) + x

Uses pre-norm (LayerNorm before sub-layer) for training stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gdt.config import ModelConfig
from gdt.models.differential_attention import DifferentialMultiHeadAttention


class FeedForward(nn.Module):
    """
    Two-layer FFN with GELU activation.
    hidden_dim → ffn_dim → hidden_dim
    @param cfg - model configuration
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_dim, cfg.ffn_dim)
        self.fc2 = nn.Linear(cfg.ffn_dim, cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class GDTBlock(nn.Module):
    """
    Single Graph-Differential Transformer block.

    @param cfg - model configuration
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = DifferentialMultiHeadAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        self.ffn = FeedForward(cfg)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        @param x              - [B, T, D]
        @param edge_index     - [2, E]
        @param edge_type      - [E]
        @param attention_mask - optional additive mask [B, 1, T, T]
        @returns FloatTensor [B, T, D]
        """
        # Pre-norm self-attention with residual
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, edge_index, edge_type, attention_mask)
        # Pre-norm FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x
