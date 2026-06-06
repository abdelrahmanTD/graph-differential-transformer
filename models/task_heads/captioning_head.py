"""
Image captioning head (Run B).

Wraps GDTDecoder and provides:
  - Teacher-forced training with cross-entropy loss
  - Beam-search inference returning token-id sequences
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gdt.config import ModelConfig
from gdt.models.gdt_decoder import GDTDecoder


class CaptioningHead(nn.Module):
    """
    Caption generation head using the autoregressive GDTDecoder.

    @param cfg    - model configuration
    @param bos_id - begin-of-sequence token id (default: 101 = [CLS] in BERT)
    @param eos_id - end-of-sequence token id   (default: 102 = [SEP] in BERT)
    """

    def __init__(
        self,
        cfg: ModelConfig,
        bos_id: int = 101,
        eos_id: int = 102,
    ) -> None:
        super().__init__()
        self.decoder = GDTDecoder(cfg)
        self.bos_id = bos_id
        self.eos_id = eos_id

    def forward(
        self,
        encoder_out: Tensor,
        caption_ids: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Teacher-forced forward pass.

        @param encoder_out - [B, Tm, D] from GDTEncoder
        @param caption_ids - [B, Lt] ground-truth ids (input, shifted right)
        @param padding_mask - [B, Lt] True=padding
        @returns logits [B, Lt, vocab_size]
        """
        return self.decoder(caption_ids, encoder_out, padding_mask)

    def compute_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        ignore_index: int = 0,
    ) -> Tensor:
        """
        Token-level cross-entropy loss.

        @param logits       - [B, Lt, V]
        @param targets      - [B, Lt] target token ids
        @param ignore_index - token id to ignore in loss (typically padding)
        @returns scalar loss
        """
        B, Lt, V = logits.shape
        return F.cross_entropy(
            logits.reshape(B * Lt, V),
            targets.reshape(B * Lt),
            ignore_index=ignore_index,
        )

    @torch.no_grad()
    def generate(
        self,
        encoder_out: Tensor,
        max_new_tokens: int = 50,
        beam_size: int = 5,
    ) -> list[list[int]]:
        """
        Beam-search generation.

        @param encoder_out   - [B, Tm, D]
        @param max_new_tokens - maximum tokens to generate
        @param beam_size     - beam width
        @returns list of token-id lists (one per batch item, BOS stripped)
        """
        return self.decoder.generate(
            encoder_out,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            max_new_tokens=max_new_tokens,
            beam_size=beam_size,
        )
