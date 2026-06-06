"""
Differential Multi-Head Attention with graph-biased logits.

Implements §7.2 of the GDT report:

  Attn1 = softmax((Q K^T / sqrt(d)) + B1) V
  Attn2 = softmax((Q K^T / sqrt(d)) + B2) V
  Attn_diff = alpha * Attn1 - (1 - alpha) * Attn2

Graph structure is injected as an additive per-edge-type bias on the
attention logits before each softmax, following §7.3:

  logits_ij += edge_bias_table[edge_type_ij]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gdt.config import ModelConfig


class DifferentialMultiHeadAttention(nn.Module):
    """
    Multi-head attention using the differential mechanism described in
    arXiv:2410.05258, augmented with graph-biased logits from §7.3.

    @param cfg - model configuration (hidden_dim, num_heads,
                 num_edge_types, alpha, dropout)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.hidden_dim % cfg.num_heads == 0, (
            "hidden_dim must be divisible by num_heads"
        )
        self.hidden_dim = cfg.hidden_dim
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_dim // cfg.num_heads
        self.alpha = nn.Parameter(torch.tensor(cfg.alpha))

        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

        # B1, B2: per-head learnable scalars broadcast over sequence pairs
        self.B1 = nn.Parameter(torch.zeros(cfg.num_heads))
        self.B2 = nn.Parameter(torch.zeros(cfg.num_heads))

        # Edge-type bias table: shape [num_edge_types, num_heads]
        self.edge_bias_table = nn.Embedding(cfg.num_edge_types, cfg.num_heads)
        nn.init.zeros_(self.edge_bias_table.weight)

        self.dropout = nn.Dropout(cfg.dropout)
        self.scale = math.sqrt(self.head_dim)

    def _split_heads(self, x: Tensor) -> Tensor:
        """[B, T, D] → [B, H, T, d]"""
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """[B, H, T, d] → [B, T, D]"""
        B, H, T, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * d)

    def _apply_graph_bias(
        self,
        logits: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> Tensor:
        """
        Add learnable per-edge-type biases to attention logits in-place.

        @param logits     - [B, H, T, T]
        @param edge_index - [2, E] source/target token indices
        @param edge_type  - [E] integer edge type per edge
        @returns logits with graph bias added [B, H, T, T]
        """
        if edge_index.shape[1] == 0:
            return logits

        src, tgt = edge_index[0], edge_index[1]

        # Clamp indices to valid token range
        T = logits.shape[-1]
        valid = (src < T) & (tgt < T)
        src, tgt, edge_type = src[valid], tgt[valid], edge_type[valid]

        # bias: [E, H]
        bias = self.edge_bias_table(edge_type)
        # reshape for broadcasting: [E, H] → [H, E] → add to [B, H, T, T]
        logits[:, :, src, tgt] = (
            logits[:, :, src, tgt] + bias.T.unsqueeze(0)
        )
        return logits

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        @param query          - [B, Tq, D]
        @param key            - [B, Tk, D]
        @param value          - [B, Tk, D]
        @param edge_index     - [2, E]
        @param edge_type      - [E]
        @param attention_mask - optional additive mask [B, 1, Tq, Tk]
        @returns FloatTensor [B, Tq, D]
        """
        Q = self._split_heads(self.q_proj(query))   # [B, H, Tq, d]
        K = self._split_heads(self.k_proj(key))     # [B, H, Tk, d]
        V = self._split_heads(self.v_proj(value))   # [B, H, Tk, d]

        base_logits = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # [B, H, Tq, Tk]

        # Differential mechanism: two bias scalars per head, broadcast
        # B1/B2 shape [H] → [1, H, 1, 1]
        b1 = self.B1.view(1, self.num_heads, 1, 1)
        b2 = self.B2.view(1, self.num_heads, 1, 1)

        logits1 = base_logits + b1
        logits2 = base_logits + b2

        # Graph bias injection (same bias applied to both maps)
        logits1 = self._apply_graph_bias(logits1, edge_index, edge_type)
        logits2 = self._apply_graph_bias(logits2, edge_index, edge_type)

        if attention_mask is not None:
            logits1 = logits1 + attention_mask
            logits2 = logits2 + attention_mask

        attn1 = F.softmax(logits1, dim=-1)  # [B, H, Tq, Tk]
        attn2 = F.softmax(logits2, dim=-1)

        # α is a learnable scalar; clamp to (0, 1) for stability
        alpha = torch.sigmoid(self.alpha)
        attn_diff = alpha * attn1 - (1.0 - alpha) * attn2

        # Clamp to prevent extreme negatives from destabilising training
        attn_diff = attn_diff.clamp(min=0.0)

        attn_diff = self.dropout(attn_diff)
        out = torch.matmul(attn_diff, V)   # [B, H, Tq, d]
        out = self._merge_heads(out)        # [B, Tq, D]
        return self.out_proj(out)
