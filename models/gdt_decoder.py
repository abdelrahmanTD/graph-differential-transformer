"""
Autoregressive captioning decoder for Run B.

Architecture: standard cross-attention transformer decoder that attends
to encoder output (multimodal tokens) with causal self-attention mask.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gdt.config import ModelConfig


class DecoderBlock(nn.Module):
    """
    One decoder block: causal self-attn → cross-attn → FFN, all pre-norm.
    @param cfg - model configuration
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            cfg.hidden_dim, cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            cfg.hidden_dim, cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm3 = nn.LayerNorm(cfg.hidden_dim)
        self.fc1 = nn.Linear(cfg.hidden_dim, cfg.ffn_dim)
        self.fc2 = nn.Linear(cfg.ffn_dim, cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> Tensor:
        """Upper-triangular additive mask for causal self-attention."""
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )
        return mask

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        @param tgt                   - [B, Lt, D] decoder input
        @param memory                - [B, Tm, D] encoder output
        @param tgt_key_padding_mask  - [B, Lt] True=padding
        @returns FloatTensor [B, Lt, D]
        """
        causal = self._causal_mask(tgt.shape[1], tgt.device)

        # Causal self-attention
        normed = self.norm1(tgt)
        sa_out, _ = self.self_attn(
            normed, normed, normed,
            attn_mask=causal,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = tgt + sa_out

        # Cross-attention over encoder memory
        normed = self.norm2(tgt)
        ca_out, _ = self.cross_attn(normed, memory, memory)
        tgt = tgt + ca_out

        # FFN
        normed = self.norm3(tgt)
        tgt = tgt + self.fc2(self.dropout(F.gelu(self.fc1(normed))))
        return tgt


class GDTDecoder(nn.Module):
    """
    Autoregressive decoder that cross-attends to GDTEncoder output.

    @param cfg - model configuration
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.blocks = nn.ModuleList(
            [DecoderBlock(cfg) for _ in range(cfg.num_layers // 2)]
        )
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        # Weight tying: share embedding and lm_head weights
        self.lm_head.weight = self.token_embed.weight

    def forward(
        self,
        tgt_ids: Tensor,
        encoder_out: Tensor,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Teacher-forced forward pass.

        @param tgt_ids               - [B, Lt] LongTensor (shifted right)
        @param encoder_out           - [B, Tm, D]
        @param tgt_key_padding_mask  - [B, Lt] True=padding
        @returns logits [B, Lt, vocab_size]
        """
        B, Lt = tgt_ids.shape
        device = tgt_ids.device
        positions = torch.arange(Lt, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(tgt_ids) + self.pos_embed(positions)

        for block in self.blocks:
            x = block(x, encoder_out, tgt_key_padding_mask)

        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate(
        self,
        encoder_out: Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = 50,
        beam_size: int = 5,
    ) -> list[list[int]]:
        """
        Beam-search caption generation.

        @param encoder_out   - [B, Tm, D]
        @param bos_id        - begin-of-sequence token id
        @param eos_id        - end-of-sequence token id
        @param max_new_tokens - maximum tokens to generate
        @param beam_size     - beam width
        @returns list of token-id lists, one per batch item
        """
        B = encoder_out.shape[0]
        device = encoder_out.device
        results = []

        for b in range(B):
            mem = encoder_out[b:b+1]   # [1, Tm, D]
            # Beam: list of (score, tokens)
            beams = [(0.0, [bos_id])]
            completed = []

            for _ in range(max_new_tokens):
                candidates = []
                for score, tokens in beams:
                    if tokens[-1] == eos_id:
                        completed.append((score, tokens))
                        continue
                    ids = torch.tensor([tokens], device=device)
                    logits = self.forward(ids, mem)[:, -1, :]   # [1, V]
                    log_probs = F.log_softmax(logits, dim=-1)[0]
                    top_lp, top_ids = log_probs.topk(beam_size)
                    for lp, tid in zip(top_lp.tolist(), top_ids.tolist()):
                        candidates.append((score + lp, tokens + [tid]))

                if not candidates:
                    break
                beams = sorted(candidates, key=lambda x: x[0], reverse=True)
                beams = beams[:beam_size]

            all_beams = completed + beams
            best_score, best_tokens = max(all_beams, key=lambda x: x[0])
            results.append(best_tokens[1:])  # strip BOS

        return results
