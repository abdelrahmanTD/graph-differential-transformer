"""
Initialize GDTEncoder weights from pretrained BERT-base + ViT-base/16.

Strategy:
  - Text embeddings      ← BERT-base word/position/type embeddings
  - Transformer blocks   ← BERT-base encoder layers (shared for text+vision)
  - Vision patch proj    ← ViT-base/16 patch_embed.proj

This gives the encoder meaningful language + vision representations from day one,
dramatically reducing the number of training steps needed for useful outputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gdt.models.gdt_encoder import GDTEncoder


def _copy(dst: nn.Parameter | None, src: torch.Tensor) -> None:
    """Copy src into dst, truncating or padding rows if sizes differ."""
    if dst is None:
        return
    d, s = dst.data, src.detach().float()
    if d.shape == s.shape:
        d.copy_(s)
    elif d.dim() == 2 and s.dim() == 2:
        rows = min(d.shape[0], s.shape[0])
        cols = min(d.shape[1], s.shape[1])
        d[:rows, :cols].copy_(s[:rows, :cols])
    elif d.dim() == 1 and s.dim() == 1:
        n = min(d.shape[0], s.shape[0])
        d[:n].copy_(s[:n])


def _load_bert_into_encoder(encoder: GDTEncoder, bert_model_name: str) -> None:
    from transformers import BertModel

    print(f"  Loading BERT weights from '{bert_model_name}'...")
    bert = BertModel.from_pretrained(bert_model_name)
    be = bert.embeddings

    # Text embeddings
    te = encoder.text_embed
    _copy(te.token_embed.weight, be.word_embeddings.weight)
    _copy(te.pos_embed.weight, be.position_embeddings.weight)
    _copy(te.type_embed.weight, be.token_type_embeddings.weight)
    _copy(te.norm.weight, be.LayerNorm.weight)
    _copy(te.norm.bias, be.LayerNorm.bias)

    # Transformer blocks
    num_layers = min(len(encoder.blocks), len(bert.encoder.layer))
    for i in range(num_layers):
        gdt_blk = encoder.blocks[i]
        bert_blk = bert.encoder.layer[i]
        bsa = bert_blk.attention.self

        # Attention projections (q/k/v have no bias in GDT)
        _copy(gdt_blk.attn.q_proj.weight, bsa.query.weight)
        _copy(gdt_blk.attn.k_proj.weight, bsa.key.weight)
        _copy(gdt_blk.attn.v_proj.weight, bsa.value.weight)
        _copy(gdt_blk.attn.out_proj.weight, bert_blk.attention.output.dense.weight)
        _copy(gdt_blk.attn.out_proj.bias, bert_blk.attention.output.dense.bias)

        # Post-attention LayerNorm (norm1)
        _copy(gdt_blk.norm1.weight, bert_blk.attention.output.LayerNorm.weight)
        _copy(gdt_blk.norm1.bias, bert_blk.attention.output.LayerNorm.bias)

        # FFN
        _copy(gdt_blk.ffn.fc1.weight, bert_blk.intermediate.dense.weight)
        _copy(gdt_blk.ffn.fc1.bias, bert_blk.intermediate.dense.bias)
        _copy(gdt_blk.ffn.fc2.weight, bert_blk.output.dense.weight)
        _copy(gdt_blk.ffn.fc2.bias, bert_blk.output.dense.bias)

        # Post-FFN LayerNorm (norm2)
        _copy(gdt_blk.norm2.weight, bert_blk.output.LayerNorm.weight)
        _copy(gdt_blk.norm2.bias, bert_blk.output.LayerNorm.bias)

    # Final encoder norm from BERT pooler LayerNorm (fallback: identity)
    # BERT doesn't have a final encoder norm, so leave encoder.norm as default.

    del bert
    print("  BERT weights loaded.")


def _load_vit_patch_proj(encoder: GDTEncoder, vit_model_name: str) -> None:
    import timm

    print(f"  Loading ViT patch projection from '{vit_model_name}'...")
    vit = timm.create_model(vit_model_name, pretrained=True)
    _copy(encoder.vision_embed.proj.weight, vit.patch_embed.proj.weight)
    _copy(encoder.vision_embed.proj.bias, vit.patch_embed.proj.bias)
    del vit
    print("  ViT patch projection loaded.")


def load_pretrained_weights(
    encoder: GDTEncoder,
    bert_model_name: str = "bert-base-uncased",
    vit_model_name: str = "vit_base_patch16_224",
) -> None:
    """
    Initialize encoder in-place from BERT-base and ViT-base/16 pretrained weights.

    Safe to call before or after moving the model to a device.
    """
    print("Initializing GDTEncoder from pretrained weights...")
    _load_bert_into_encoder(encoder, bert_model_name)
    _load_vit_patch_proj(encoder, vit_model_name)
    print("Pretrained initialization complete.")
