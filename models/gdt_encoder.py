"""
GDT Encoder: stacks L GDTBlocks over fused multimodal token sequences.

Accepts raw images and text token ids, returns final token representations
[B, 1+Lt+Lv, D] where 1 is the [CLS] position.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from gdt.config import ModelConfig
from gdt.models.embeddings import ModalityFusion, TextEmbedding, VisionPatchEmbedding
from gdt.models.gdt_block import GDTBlock
from gdt.models.graph_constructor import GraphConstructor


class GDTEncoder(nn.Module):
    """
    Full GDT encoder for multimodal vision-language tasks.

    @param cfg - model configuration
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.text_embed = TextEmbedding(cfg)
        self.vision_embed = VisionPatchEmbedding(cfg)
        self.fusion = ModalityFusion(cfg)
        self.blocks = nn.ModuleList(
            [GDTBlock(cfg) for _ in range(cfg.num_layers)]
        )
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.graph_constructor = GraphConstructor(
            hidden_dim=cfg.hidden_dim,
            num_edge_types=cfg.num_edge_types,
        )

    @property
    def num_patches(self) -> int:
        return (self.cfg.img_size // self.cfg.patch_size) ** 2

    @property
    def num_patches_h(self) -> int:
        return self.cfg.img_size // self.cfg.patch_size

    def forward(
        self,
        images: Tensor,
        text_ids: Tensor,
        scene_annotations: Optional[list[dict]] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        @param images            - [B, 3, H, W]
        @param text_ids          - [B, Lt] LongTensor
        @param scene_annotations - optional list of VG-style annotation dicts
        @param attention_mask    - optional padding mask [B, Lt], True=keep
        @returns FloatTensor [B, 1+Lt+Lv, D]
        """
        text_tokens = self.text_embed(text_ids)      # [B, Lt, D]
        vision_tokens = self.vision_embed(images)    # [B, Lv, D]
        fused, is_vision = self.fusion(text_tokens, vision_tokens)

        # Build combined attention mask for padding
        attn_bias = None
        if attention_mask is not None:
            # attention_mask covers text tokens only; prepend True for [CLS]
            B = attention_mask.shape[0]
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=attention_mask.device)
            vision_mask = torch.ones(
                B, vision_tokens.shape[1], dtype=torch.bool, device=attention_mask.device
            )
            full_mask = torch.cat([cls_mask, attention_mask, vision_mask], dim=1)
            # Additive mask: 0 for keep, -inf for ignore
            additive = torch.zeros_like(full_mask, dtype=fused.dtype)
            additive[~full_mask] = float("-inf")
            attn_bias = additive.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]

        # Build token graph
        text_len = text_ids.shape[1]
        edge_index, edge_type = self.graph_constructor(
            fused,
            text_len=text_len,
            num_patches_h=self.num_patches_h,
            num_patches_w=self.num_patches_h,
            scene_annotations=scene_annotations,
        )

        x = fused
        for block in self.blocks:
            x = block(x, edge_index, edge_type, attn_bias)

        return self.norm(x)  # [B, T, D]

    def get_cls_token(self, encoder_out: Tensor) -> Tensor:
        """Extract the [CLS] token from encoder output [B, T, D] → [B, D]."""
        return encoder_out[:, 0, :]
