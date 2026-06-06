"""
Unit tests for GDTBlock and full GDTEncoder.
"""

import torch
import pytest

from gdt.config import ModelConfig
from gdt.models.gdt_block import GDTBlock, FeedForward
from gdt.models.gdt_encoder import GDTEncoder


@pytest.fixture
def small_cfg():
    """Minimal config for fast tests."""
    return ModelConfig(
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        ffn_dim=128,
        patch_size=16,
        img_size=32,
        max_seq_len=64,
        vocab_size=256,
        num_edge_types=8,
        dropout=0.0,
    )


def make_empty_graph():
    return (
        torch.zeros(2, 0, dtype=torch.long),
        torch.zeros(0, dtype=torch.long),
    )


class TestFeedForward:
    def test_output_shape(self, small_cfg):
        ffn = FeedForward(small_cfg)
        x = torch.randn(2, 10, small_cfg.hidden_dim)
        out = ffn(x)
        assert out.shape == x.shape

    def test_gradient_flows(self, small_cfg):
        ffn = FeedForward(small_cfg)
        x = torch.randn(2, 10, small_cfg.hidden_dim, requires_grad=True)
        ffn(x).sum().backward()
        assert x.grad is not None


class TestGDTBlock:
    def test_output_shape_preserved(self, small_cfg):
        block = GDTBlock(small_cfg)
        B, T = 2, 12
        x = torch.randn(B, T, small_cfg.hidden_dim)
        ei, et = make_empty_graph()
        out = block(x, ei, et)
        assert out.shape == (B, T, small_cfg.hidden_dim)

    def test_residual_connection(self, small_cfg):
        """Output should not equal input (residual + attn/ffn applied)."""
        block = GDTBlock(small_cfg)
        x = torch.randn(2, 8, small_cfg.hidden_dim)
        ei, et = make_empty_graph()
        out = block(x, ei, et)
        assert not torch.allclose(out, x)

    def test_gradient_flows(self, small_cfg):
        block = GDTBlock(small_cfg)
        x = torch.randn(2, 8, small_cfg.hidden_dim, requires_grad=True)
        ei, et = make_empty_graph()
        block(x, ei, et).sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_with_graph_edges(self, small_cfg):
        """Block with graph edges should still produce correct output shape."""
        block = GDTBlock(small_cfg)
        B, T = 2, 8
        x = torch.randn(B, T, small_cfg.hidden_dim)
        ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        et = torch.tensor([1, 1, 2], dtype=torch.long)
        out = block(x, ei, et)
        assert out.shape == (B, T, small_cfg.hidden_dim)

    def test_with_attention_mask(self, small_cfg):
        """Causal mask should not break forward pass."""
        block = GDTBlock(small_cfg)
        B, T = 2, 6
        x = torch.randn(B, T, small_cfg.hidden_dim)
        mask = torch.triu(
            torch.full((1, 1, T, T), float("-inf")), diagonal=1
        )
        ei, et = make_empty_graph()
        out = block(x, ei, et, attention_mask=mask)
        assert out.shape == (B, T, small_cfg.hidden_dim)


class TestGDTEncoder:
    def test_output_shape(self, small_cfg):
        encoder = GDTEncoder(small_cfg)
        B = 2
        Lt = 4
        img_h = small_cfg.img_size // small_cfg.patch_size
        Lv = img_h * img_h

        images = torch.randn(B, 3, small_cfg.img_size, small_cfg.img_size)
        text_ids = torch.randint(0, small_cfg.vocab_size, (B, Lt))
        out = encoder(images, text_ids)
        # Shape: [B, 1 (CLS) + Lt + Lv, D]
        assert out.shape == (B, 1 + Lt + Lv, small_cfg.hidden_dim)

    def test_cls_token_extraction(self, small_cfg):
        encoder = GDTEncoder(small_cfg)
        images = torch.randn(2, 3, small_cfg.img_size, small_cfg.img_size)
        text_ids = torch.randint(0, small_cfg.vocab_size, (2, 4))
        enc_out = encoder(images, text_ids)
        cls = encoder.get_cls_token(enc_out)
        assert cls.shape == (2, small_cfg.hidden_dim)

    def test_gradient_flows_through_encoder(self, small_cfg):
        encoder = GDTEncoder(small_cfg)
        images = torch.randn(2, 3, small_cfg.img_size, small_cfg.img_size, requires_grad=True)
        text_ids = torch.randint(0, small_cfg.vocab_size, (2, 4))
        out = encoder(images, text_ids)
        out.sum().backward()
        assert images.grad is not None

    def test_attention_mask_applied_to_encoder(self, small_cfg):
        """Padding mask should change encoder output."""
        encoder = GDTEncoder(small_cfg)
        images = torch.randn(2, 3, small_cfg.img_size, small_cfg.img_size)
        text_ids = torch.randint(0, small_cfg.vocab_size, (2, 4))

        mask = torch.ones(2, 4, dtype=torch.bool)
        mask[0, 2:] = False  # mask last 2 tokens for sample 0

        out_no_mask = encoder(images, text_ids)
        out_with_mask = encoder(images, text_ids, attention_mask=mask)
        assert not torch.allclose(out_no_mask, out_with_mask, atol=1e-5)
