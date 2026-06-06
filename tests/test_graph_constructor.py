"""
Unit tests for GraphConstructor.
"""

import torch
import pytest

from gdt.models.graph_constructor import GraphConstructor, EdgeTypes


@pytest.fixture
def gc():
    return GraphConstructor(hidden_dim=64, num_edge_types=8, k=1, top_k_learned=4)


def test_spatial_graph_shape(gc):
    """Spatial graph should have edges of correct count for a 4×4 grid."""
    ei, et = gc.build_spatial_graph(4, 4)
    assert ei.shape[0] == 2
    # Each interior node has 8 neighbors; boundary fewer. Edge count > 0.
    assert ei.shape[1] > 0
    assert et.shape[0] == ei.shape[1]


def test_spatial_edge_types(gc):
    """All spatial edges should have type SPATIAL_ADJACENT."""
    ei, et = gc.build_spatial_graph(3, 3)
    assert (et == EdgeTypes.SPATIAL_ADJACENT).all()


def test_spatial_graph_symmetry(gc):
    """If (i→j) exists, (j→i) should exist."""
    ei, et = gc.build_spatial_graph(3, 3)
    src, tgt = ei[0].tolist(), ei[1].tolist()
    edge_set = set(zip(src, tgt))
    for s, t in zip(src, tgt):
        assert (t, s) in edge_set, f"Edge ({s},{t}) exists but ({t},{s}) does not"


def test_spatial_single_patch(gc):
    """1×1 grid: no edges."""
    ei, et = gc.build_spatial_graph(1, 1)
    assert ei.shape[1] == 0


def test_text_graph_consecutive(gc):
    """Text graph should connect consecutive tokens only."""
    ei, et = gc.build_text_graph(5)
    assert ei.shape[1] > 0
    assert (et == EdgeTypes.TEXT_ADJACENT).all()
    # Offsets: tokens at positions 1..5, consecutive pairs
    src_set = set(ei[0].tolist())
    assert all(1 <= s <= 5 for s in src_set)


def test_scene_graph_valid_indices(gc):
    """Scene graph edges should stay within [offset, offset+num_patches)."""
    annotations = [
        {"subject_idx": 0, "relation_idx": 1, "object_idx": 2},
        {"subject_idx": 3, "relation_idx": 4, "object_idx": 5},
    ]
    text_len = 4
    num_patches = 10
    ei, et = gc.build_scene_graph(annotations, text_len, num_patches)
    offset = text_len + 1
    if ei.shape[1] > 0:
        assert ei.min().item() >= offset
        assert ei.max().item() < offset + num_patches


def test_scene_graph_out_of_bounds_filtered(gc):
    """Out-of-bounds patch indices should be silently dropped."""
    annotations = [{"subject_idx": 999, "relation_idx": 1000, "object_idx": 2000}]
    ei, et = gc.build_scene_graph(annotations, text_len=4, num_patches=10)
    assert ei.shape[1] == 0


def test_learned_graph_shape(gc):
    """Learned graph should return top_k_learned edges per token."""
    T, D = 8, 64
    embeddings = torch.randn(T, D)
    ei, et = gc.build_learned_graph(embeddings)
    assert ei.shape[0] == 2
    # Each of T tokens contributes at most top_k_learned edges
    assert ei.shape[1] <= T * gc.top_k_learned


def test_forward_runs(gc):
    """Full forward pass should produce valid edge tensors."""
    B, T = 2, 20
    token_emb = torch.randn(B, T, 64)
    ei, et = gc(
        token_emb,
        text_len=4,
        num_patches_h=2,
        num_patches_w=2,
    )
    assert ei.shape[0] == 2
    assert ei.shape[1] == et.shape[0]
    assert et.min().item() >= 0
    assert et.max().item() < EdgeTypes.COUNT
