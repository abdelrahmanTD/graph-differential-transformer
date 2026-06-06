"""
Graph construction over multimodal token sequences.

Edge-type integer codes:
  0 = no-edge / default
  1 = spatial adjacent (patch neighbors within k hops)
  2 = spatial same-region
  3 = scene-graph subject
  4 = scene-graph relation
  5 = scene-graph object
  6 = text adjacent (consecutive tokens)
  7 = text far (non-adjacent text pair)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class EdgeTypes:
    NO_EDGE = 0
    SPATIAL_ADJACENT = 1
    SPATIAL_SAME_REGION = 2
    SG_SUBJECT = 3
    SG_RELATION = 4
    SG_OBJECT = 5
    TEXT_ADJACENT = 6
    TEXT_FAR = 7
    COUNT = 8


class GraphConstructor(nn.Module):
    """
    Builds token-level graphs for GDT blocks.

    Supports three graph types:
      - Spatial graph  (vision patches on a 2-D grid)
      - Scene graph    (subject-relation-object triplets from annotations)
      - Learned graph  (edge predictor MLP)

    @param hidden_dim   - token embedding dimension
    @param num_edge_types - number of discrete edge types
    @param k            - spatial k-hop neighbourhood radius
    @param top_k_learned - number of learned edges to keep per token
    """

    def __init__(
        self,
        hidden_dim: int,
        num_edge_types: int = EdgeTypes.COUNT,
        k: int = 1,
        top_k_learned: int = 8,
    ) -> None:
        super().__init__()
        self.k = k
        self.top_k_learned = top_k_learned

        # Lightweight edge predictor: [D, D] → scalar
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def build_spatial_graph(
        self, num_patches_h: int, num_patches_w: int
    ) -> tuple[Tensor, Tensor]:
        """
        Build a spatial adjacency graph over a 2-D patch grid.

        @param num_patches_h - number of patch rows
        @param num_patches_w - number of patch columns
        @returns (edge_index [2, E], edge_type [E])
        """
        k = self.k
        rows, cols, types = [], [], []

        for r in range(num_patches_h):
            for c in range(num_patches_w):
                node_i = r * num_patches_w + c
                for dr in range(-k, k + 1):
                    for dc in range(-k, k + 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < num_patches_h and 0 <= nc < num_patches_w:
                            node_j = nr * num_patches_w + nc
                            rows.append(node_i)
                            cols.append(node_j)
                            types.append(EdgeTypes.SPATIAL_ADJACENT)

        if not rows:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)
        else:
            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_type = torch.tensor(types, dtype=torch.long)

        return edge_index, edge_type

    def build_scene_graph(
        self,
        annotations: list[dict],
        text_len: int,
        num_patches: int,
    ) -> tuple[Tensor, Tensor]:
        """
        Build a scene graph from Visual Genome-style annotation dicts.

        Each annotation dict should have:
          { "subject_idx": int, "relation_idx": int, "object_idx": int }
        where indices are relative to the vision token sequence (patch indices).
        Patch indices are offset by text_len+1 (for [CLS]) in the fused sequence.

        @param annotations - list of triplet dicts
        @param text_len    - number of text tokens
        @param num_patches - number of vision patch tokens
        @returns (edge_index [2, E], edge_type [E])
        """
        offset = text_len + 1  # +1 for [CLS]
        rows, cols, types = [], [], []

        for ann in annotations:
            s = ann.get("subject_idx", 0) + offset
            r = ann.get("relation_idx", 0) + offset
            o = ann.get("object_idx", 0) + offset

            if not all(
                0 <= idx - offset < num_patches for idx in [s, r, o]
            ):
                continue

            rows += [s, r, o]
            cols += [r, o, s]
            types += [
                EdgeTypes.SG_SUBJECT,
                EdgeTypes.SG_RELATION,
                EdgeTypes.SG_OBJECT,
            ]

        if not rows:
            return (
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )

        return (
            torch.tensor([rows, cols], dtype=torch.long),
            torch.tensor(types, dtype=torch.long),
        )

    def build_text_graph(self, text_len: int) -> tuple[Tensor, Tensor]:
        """
        Build a text adjacency graph (consecutive token edges).

        @param text_len - number of text tokens (excluding [CLS])
        @returns (edge_index [2, E], edge_type [E])
        """
        offset = 1  # +1 for [CLS]
        rows, cols, types = [], [], []
        for i in range(text_len - 1):
            ni = i + offset
            nj = i + 1 + offset
            rows += [ni, nj]
            cols += [nj, ni]
            types += [EdgeTypes.TEXT_ADJACENT, EdgeTypes.TEXT_ADJACENT]

        if not rows:
            return (
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            )

        return (
            torch.tensor([rows, cols], dtype=torch.long),
            torch.tensor(types, dtype=torch.long),
        )

    def build_learned_graph(
        self,
        token_embeddings: Tensor,
        offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """
        Predict edges from token embeddings using a lightweight MLP.
        Keeps the top-k edges per token by predicted logit.

        @param token_embeddings - [T, D] for a single sample
        @param offset           - index offset for returned edge_index
        @returns (edge_index [2, E], edge_type [E])
        """
        T, D = token_embeddings.shape
        # Pairwise concatenation [T*T, 2D]
        xi = token_embeddings.unsqueeze(1).expand(-1, T, -1)
        xj = token_embeddings.unsqueeze(0).expand(T, -1, -1)
        pairs = torch.cat([xi, xj], dim=-1).reshape(T * T, 2 * D)

        logits = self.edge_predictor(pairs).reshape(T, T)
        diag_mask = torch.eye(T, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(diag_mask, float("-inf"))

        k = min(self.top_k_learned, T - 1)
        _, top_idx = logits.topk(k, dim=-1)  # [T, k]

        src = torch.arange(T, device=logits.device).unsqueeze(1).expand(-1, k)
        rows = (src + offset).flatten()
        cols = (top_idx + offset).flatten()
        edge_type = torch.full(
            (rows.shape[0],), EdgeTypes.SPATIAL_ADJACENT, dtype=torch.long
        )
        return torch.stack([rows, cols]), edge_type

    def forward(
        self,
        token_embeddings: Tensor,
        text_len: int,
        num_patches_h: int,
        num_patches_w: int,
        scene_annotations: Optional[list[dict]] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Compose all graph types into a single edge_index and edge_type.

        @param token_embeddings  - [B, T, D] — used for learned graph
        @param text_len          - number of text tokens per sample
        @param num_patches_h     - patch grid height
        @param num_patches_w     - patch grid width
        @param scene_annotations - optional list of annotation dicts per batch
        @returns (edge_index [2, E], edge_type [E])  on the same device
        """
        device = token_embeddings.device
        num_patches = num_patches_h * num_patches_w

        ei_list, et_list = [], []

        # Spatial graph (patch offset: text_len+1)
        sp_ei, sp_et = self.build_spatial_graph(num_patches_h, num_patches_w)
        patch_offset = text_len + 1
        sp_ei = sp_ei + patch_offset
        ei_list.append(sp_ei.to(device))
        et_list.append(sp_et.to(device))

        # Text adjacency graph
        tx_ei, tx_et = self.build_text_graph(text_len)
        ei_list.append(tx_ei.to(device))
        et_list.append(tx_et.to(device))

        # Scene graph (first sample only; can be extended per-batch)
        if scene_annotations:
            sg_ei, sg_et = self.build_scene_graph(
                scene_annotations, text_len, num_patches
            )
            ei_list.append(sg_ei.to(device))
            et_list.append(sg_et.to(device))

        # Learned graph on first sample's embeddings (shared across batch)
        with torch.no_grad():
            lg_ei, lg_et = self.build_learned_graph(token_embeddings[0])
        ei_list.append(lg_ei.to(device))
        et_list.append(lg_et.to(device))

        edge_index = torch.cat(ei_list, dim=1)
        edge_type = torch.cat(et_list, dim=0)
        return edge_index, edge_type
