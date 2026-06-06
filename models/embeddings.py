"""
Text + vision patch embeddings with modality-type encoding.

Text tokens  → type id 0
Image patches → type id 1
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from gdt.config import ModelConfig


class TextEmbedding(nn.Module):
    """
    Token embedding + learnable positional embedding + modality type
    embedding for the text stream.
    @param cfg - model configuration
    """

    MODALITY_ID = 0

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.type_embed = nn.Embedding(2, cfg.hidden_dim)
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        @param token_ids - LongTensor [B, L]
        @returns FloatTensor [B, L, D]
        """
        B, L = token_ids.shape
        device = token_ids.device
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        modality = torch.full((B, L), self.MODALITY_ID, device=device)

        x = (
            self.token_embed(token_ids)
            + self.pos_embed(positions)
            + self.type_embed(modality)
        )
        return self.dropout(self.norm(x))


class VisionPatchEmbedding(nn.Module):
    """
    ViT-style patch embedding: image → flattened patches → linear projection,
    plus 2-D sine-cosine positional encoding and modality-type embedding.
    @param cfg - model configuration
    """

    MODALITY_ID = 1

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.patch_size = cfg.patch_size
        self.num_patches_h = cfg.img_size // cfg.patch_size
        self.num_patches_w = cfg.img_size // cfg.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Single conv replaces manual patch extraction + linear projection.
        self.proj = nn.Conv2d(
            3,
            cfg.hidden_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        self.type_embed = nn.Embedding(2, cfg.hidden_dim)
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)

        pos = self._build_sincos_pos(
            self.num_patches_h, self.num_patches_w, cfg.hidden_dim
        )
        self.register_buffer("pos_embed", pos)

    @staticmethod
    def _build_sincos_pos(h: int, w: int, d: int) -> Tensor:
        """Build 2-D sine-cosine positional encoding [h*w, d]."""
        assert d % 4 == 0, "hidden_dim must be divisible by 4 for 2-D sincos"
        y_pos = torch.arange(h, dtype=torch.float32)
        x_pos = torch.arange(w, dtype=torch.float32)
        gy, gx = torch.meshgrid(y_pos, x_pos, indexing="ij")

        dim_half = d // 4
        omega = torch.arange(dim_half, dtype=torch.float32) / dim_half
        omega = 1.0 / (10000.0 ** omega)

        sin_y = torch.einsum("n,d->nd", gy.flatten(), omega)
        cos_y = torch.cos(sin_y)
        sin_y = torch.sin(sin_y)
        sin_x = torch.einsum("n,d->nd", gx.flatten(), omega)
        cos_x = torch.cos(sin_x)
        sin_x = torch.sin(sin_x)

        return torch.cat([sin_y, cos_y, sin_x, cos_x], dim=-1)  # [h*w, d]

    def forward(self, images: Tensor) -> Tensor:
        """
        @param images - FloatTensor [B, 3, H, W]
        @returns FloatTensor [B, num_patches, D]
        """
        B = images.shape[0]
        x = self.proj(images)        # [B, D, h, w]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, D]

        device = x.device
        modality = torch.full(
            (B, self.num_patches), self.MODALITY_ID, device=device
        )
        x = x + self.pos_embed.unsqueeze(0) + self.type_embed(modality)
        return self.dropout(self.norm(x))


class ModalityFusion(nn.Module):
    """
    Concatenate text and vision tokens into a single sequence, prepending
    a learnable [CLS] token.  Returns the unified representation and a
    boolean mask marking vision token positions.
    @param cfg - model configuration
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self, text_tokens: Tensor, vision_tokens: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        @param text_tokens   - [B, Lt, D]
        @param vision_tokens - [B, Lv, D]
        @returns
            fused      [B, 1+Lt+Lv, D]
            is_vision  [B, 1+Lt+Lv]  bool mask, True for vision positions
        """
        B = text_tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        fused = torch.cat([cls, text_tokens, vision_tokens], dim=1)

        Lt = text_tokens.shape[1]
        Lv = vision_tokens.shape[1]
        device = text_tokens.device
        is_vision = torch.cat(
            [
                torch.zeros(B, 1 + Lt, dtype=torch.bool, device=device),
                torch.ones(B, Lv, dtype=torch.bool, device=device),
            ],
            dim=1,
        )
        return fused, is_vision
