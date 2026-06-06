"""
Unit tests for TextEmbedding, VisionPatchEmbedding, and ModalityFusion.
"""

import torch
import pytest

from gdt.config import ModelConfig
from gdt.models.embeddings import TextEmbedding, VisionPatchEmbedding, ModalityFusion


@pytest.fixture
def small_cfg():
    return ModelConfig(
        hidden_dim=64,
        num_heads=4,
        patch_size=8,
        img_size=32,
        vocab_size=256,
        max_seq_len=64,
        dropout=0.0,
    )


class TestTextEmbedding:
    def test_output_shape(self, small_cfg):
        emb = TextEmbedding(small_cfg)
        ids = torch.randint(0, small_cfg.vocab_size, (2, 10))
        out = emb(ids)
        assert out.shape == (2, 10, small_cfg.hidden_dim)

    def test_modality_id_is_zero(self):
        assert TextEmbedding.MODALITY_ID == 0

    def test_different_sequences_different_outputs(self, small_cfg):
        emb = TextEmbedding(small_cfg)
        a = torch.randint(0, small_cfg.vocab_size, (2, 8))
        b = torch.randint(0, small_cfg.vocab_size, (2, 8))
        out_a = emb(a)
        out_b = emb(b)
        # With random inputs they are almost certainly different
        assert not torch.allclose(out_a, out_b)


class TestVisionPatchEmbedding:
    def test_output_shape(self, small_cfg):
        emb = VisionPatchEmbedding(small_cfg)
        imgs = torch.randn(2, 3, small_cfg.img_size, small_cfg.img_size)
        out = emb(imgs)
        num_patches = (small_cfg.img_size // small_cfg.patch_size) ** 2
        assert out.shape == (2, num_patches, small_cfg.hidden_dim)

    def test_modality_id_is_one(self):
        assert VisionPatchEmbedding.MODALITY_ID == 1

    def test_pos_embed_registered_buffer(self, small_cfg):
        emb = VisionPatchEmbedding(small_cfg)
        assert hasattr(emb, "pos_embed")
        num_patches = (small_cfg.img_size // small_cfg.patch_size) ** 2
        assert emb.pos_embed.shape == (num_patches, small_cfg.hidden_dim)

    def test_sincos_pos_no_nans(self, small_cfg):
        emb = VisionPatchEmbedding(small_cfg)
        assert not torch.isnan(emb.pos_embed).any()


class TestModalityFusion:
    def test_fused_shape(self, small_cfg):
        fuse = ModalityFusion(small_cfg)
        B, Lt, Lv = 2, 5, 4
        text = torch.randn(B, Lt, small_cfg.hidden_dim)
        vision = torch.randn(B, Lv, small_cfg.hidden_dim)
        fused, mask = fuse(text, vision)
        # [CLS] + text + vision
        assert fused.shape == (B, 1 + Lt + Lv, small_cfg.hidden_dim)

    def test_is_vision_mask(self, small_cfg):
        fuse = ModalityFusion(small_cfg)
        B, Lt, Lv = 2, 3, 4
        text = torch.randn(B, Lt, small_cfg.hidden_dim)
        vision = torch.randn(B, Lv, small_cfg.hidden_dim)
        _, mask = fuse(text, vision)
        # First 1+Lt positions are NOT vision; last Lv are vision
        assert mask.shape == (B, 1 + Lt + Lv)
        assert not mask[:, :1 + Lt].any()
        assert mask[:, 1 + Lt:].all()

    def test_cls_token_learned(self, small_cfg):
        """CLS token should have trainable parameters."""
        fuse = ModalityFusion(small_cfg)
        assert fuse.cls_token.requires_grad
        assert fuse.cls_token.shape == (1, 1, small_cfg.hidden_dim)
