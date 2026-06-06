"""
Baseline encoder: same architecture as GDTEncoder but uses standard
nn.MultiheadAttention (no differential mechanism, no graph bias).

Used for ablation comparison to isolate GDT contributions.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gdt.config import ModelConfig
from gdt.models.embeddings import ModalityFusion, TextEmbedding, VisionPatchEmbedding


class BaselineBlock(nn.Module):
    """
    Standard Transformer encoder block (pre-norm) with no differential
    attention and no graph bias.
    @param cfg - model configuration
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = nn.MultiheadAttention(
            cfg.hidden_dim, cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        self.fc1 = nn.Linear(cfg.hidden_dim, cfg.ffn_dim)
        self.fc2 = nn.Linear(cfg.ffn_dim, cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        @param x                - [B, T, D]
        @param key_padding_mask - [B, T] True=padding to ignore
        @returns FloatTensor [B, T, D]
        """
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed, key_padding_mask=key_padding_mask
        )
        x = x + attn_out
        x = x + self.fc2(self.dropout(F.gelu(self.fc1(self.norm2(x)))))
        return x


class BaselineEncoder(nn.Module):
    """
    Baseline multimodal encoder: identical embeddings and depth to
    GDTEncoder but with standard attention and no graph structure.

    @param cfg - model configuration
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.text_embed = TextEmbedding(cfg)
        self.vision_embed = VisionPatchEmbedding(cfg)
        self.fusion = ModalityFusion(cfg)
        self.blocks = nn.ModuleList(
            [BaselineBlock(cfg) for _ in range(cfg.num_layers)]
        )
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        images: Tensor,
        text_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        @param images         - [B, 3, H, W]
        @param text_ids       - [B, Lt]
        @param attention_mask - optional padding mask [B, Lt], True=keep
        @returns FloatTensor [B, 1+Lt+Lv, D]
        """
        text_tokens = self.text_embed(text_ids)
        vision_tokens = self.vision_embed(images)
        fused, _ = self.fusion(text_tokens, vision_tokens)

        pad_mask = None
        if attention_mask is not None:
            B = attention_mask.shape[0]
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=attention_mask.device)
            vis_mask = torch.ones(
                B, vision_tokens.shape[1], dtype=torch.bool, device=attention_mask.device
            )
            full_mask = torch.cat([cls_mask, attention_mask, vis_mask], dim=1)
            # MultiheadAttention expects True = positions to IGNORE
            pad_mask = ~full_mask

        x = fused
        for block in self.blocks:
            x = block(x, pad_mask)

        return self.norm(x)

    def get_cls_token(self, encoder_out: Tensor) -> Tensor:
        """Extract [CLS] token [B, T, D] → [B, D]."""
        return encoder_out[:, 0, :]
