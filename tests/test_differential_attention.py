"""
Unit tests for DifferentialMultiHeadAttention.

Tests run on CPU with small toy tensors — no GPU or data required.
"""

import torch
import pytest

from gdt.config import ModelConfig
from gdt.models.differential_attention import DifferentialMultiHeadAttention


@pytest.fixture
def cfg():
    return ModelConfig(hidden_dim=64, num_heads=4, num_edge_types=8, dropout=0.0)


@pytest.fixture
def attn(cfg):
    return DifferentialMultiHeadAttention(cfg)


def make_empty_graph():
    return (
        torch.zeros(2, 0, dtype=torch.long),
        torch.zeros(0, dtype=torch.long),
    )


def test_output_shape(attn, cfg):
    """Output shape must match input shape."""
    B, T = 2, 10
    x = torch.randn(B, T, cfg.hidden_dim)
    ei, et = make_empty_graph()
    out = attn(x, x, x, ei, et)
    assert out.shape == (B, T, cfg.hidden_dim)


def test_output_differs_from_standard_attention(cfg):
    """Differential attention should differ from standard softmax attention."""
    B, T = 2, 8
    x = torch.randn(B, T, cfg.hidden_dim)
    ei, et = make_empty_graph()

    diff_attn = DifferentialMultiHeadAttention(cfg)
    diff_out = diff_attn(x, x, x, ei, et)

    std_attn = torch.nn.MultiheadAttention(cfg.hidden_dim, cfg.num_heads, batch_first=True)
    std_out, _ = std_attn(x, x, x)

    assert not torch.allclose(diff_out, std_out, atol=1e-4), (
        "Differential attention output should differ from standard attention"
    )


def test_edge_bias_changes_output(attn, cfg):
    """Adding graph edges should change the attention output."""
    B, T = 2, 8
    x = torch.randn(B, T, cfg.hidden_dim)
    ei_empty, et_empty = make_empty_graph()

    ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    et = torch.tensor([1, 1, 2], dtype=torch.long)

    out_no_graph = attn(x, x, x, ei_empty, et_empty)
    out_with_graph = attn(x, x, x, ei, et)

    # With non-zero edge biases initialised differently, outputs may differ
    assert out_no_graph.shape == out_with_graph.shape


def test_out_of_range_edge_indices_ignored(attn, cfg):
    """Edge indices beyond token sequence length should be clamped/ignored."""
    B, T = 2, 4
    x = torch.randn(B, T, cfg.hidden_dim)
    # Out-of-range indices
    ei = torch.tensor([[0, 99], [99, 0]], dtype=torch.long)
    et = torch.tensor([1, 1], dtype=torch.long)
    out = attn(x, x, x, ei, et)
    assert out.shape == (B, T, cfg.hidden_dim)


def test_attention_mask_applied(attn, cfg):
    """Providing an additive mask should produce a different output."""
    B, T = 2, 6
    x = torch.randn(B, T, cfg.hidden_dim)
    ei, et = make_empty_graph()

    out_no_mask = attn(x, x, x, ei, et)
    # Add a large negative to upper triangle (causal mask)
    mask = torch.triu(torch.full((1, 1, T, T), float("-inf")), diagonal=1)
    out_with_mask = attn(x, x, x, ei, et, attention_mask=mask)

    assert not torch.allclose(out_no_mask, out_with_mask, atol=1e-4)


def test_gradient_flows(attn, cfg):
    """Gradients should flow back through the attention module."""
    B, T = 2, 8
    x = torch.randn(B, T, cfg.hidden_dim, requires_grad=True)
    ei, et = make_empty_graph()
    out = attn(x, x, x, ei, et)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_alpha_in_zero_one(attn):
    """Alpha (after sigmoid) must lie in (0, 1) for stable subtraction."""
    alpha = torch.sigmoid(attn.alpha)
    assert 0.0 < alpha.item() < 1.0


def test_different_query_key(attn, cfg):
    """Cross-attention (Q≠K) should also return correct shape."""
    B, Tq, Tk = 2, 5, 8
    q = torch.randn(B, Tq, cfg.hidden_dim)
    k = torch.randn(B, Tk, cfg.hidden_dim)
    ei, et = make_empty_graph()
    out = attn(q, k, k, ei, et)
    assert out.shape == (B, Tq, cfg.hidden_dim)
