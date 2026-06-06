"""
Multi-task loss functions for GDT pre-training and fine-tuning.

Losses:
  - InfoNCELoss        : image-text contrastive alignment (Phase 1)
  - MLMLoss            : masked language modeling        (Phase 1)
  - MIMLoss            : masked image modeling           (Phase 1)
  - GraphLoss          : edge classification + triplet   (Phase 2)
  - MultiTaskLoss      : weighted combination

References:
  [1] Differential Transformer  arXiv:2410.05258
  [2] Transformers are GNNs     arXiv:2506.22084
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class InfoNCELoss(nn.Module):
    """
    CLIP-style image-text contrastive loss using cosine similarity.

    @param temperature - softmax temperature (default: 0.07)
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, image_feats: Tensor, text_feats: Tensor) -> Tensor:
        """
        @param image_feats - [B, D]  L2-normalised image representations
        @param text_feats  - [B, D]  L2-normalised text representations
        @returns scalar contrastive loss
        """
        image_feats = F.normalize(image_feats, dim=-1)
        text_feats = F.normalize(text_feats, dim=-1)

        logits = (image_feats @ text_feats.T) / self.temperature.exp().clamp(min=1e-4)
        labels = torch.arange(image_feats.shape[0], device=image_feats.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2.0


class MLMLoss(nn.Module):
    """
    Masked Language Modeling loss.

    Expects a linear projection head to map hidden states to vocab logits.
    Masking (15%) is applied externally before calling forward.

    @param hidden_dim - encoder hidden dimension
    @param vocab_size - target vocabulary size
    """

    def __init__(self, hidden_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        hidden_states: Tensor,
        masked_positions: Tensor,
        target_ids: Tensor,
    ) -> Tensor:
        """
        @param hidden_states    - [B, T, D]
        @param masked_positions - [B, T] bool mask; True = masked token
        @param target_ids       - [B, T] original token ids
        @returns scalar cross-entropy loss on masked positions
        """
        # Extract only masked positions for efficiency
        masked_hs = hidden_states[masked_positions]   # [N_masked, D]
        masked_tgt = target_ids[masked_positions]     # [N_masked]

        if masked_hs.shape[0] == 0:
            return hidden_states.sum() * 0.0

        logits = self.lm_head(masked_hs)              # [N_masked, V]
        return F.cross_entropy(logits, masked_tgt)


class MIMLoss(nn.Module):
    """
    Masked Image Modeling loss: reconstruct pixel features of masked patches.

    @param hidden_dim - encoder hidden dimension
    @param patch_dim  - number of dimensions per patch (e.g. 3*16*16 = 768)
    """

    def __init__(self, hidden_dim: int, patch_dim: int) -> None:
        super().__init__()
        self.decoder_head = nn.Linear(hidden_dim, patch_dim)

    def forward(
        self,
        hidden_states: Tensor,
        masked_positions: Tensor,
        target_patches: Tensor,
    ) -> Tensor:
        """
        @param hidden_states    - [B, Lv, D]  vision token representations
        @param masked_positions - [B, Lv] bool mask; True = masked patch
        @param target_patches   - [B, Lv, patch_dim] normalised pixel values
        @returns scalar MSE reconstruction loss
        """
        masked_hs = hidden_states[masked_positions]
        masked_tgt = target_patches[masked_positions]

        if masked_hs.shape[0] == 0:
            return hidden_states.sum() * 0.0

        pred = self.decoder_head(masked_hs)
        return F.mse_loss(pred, masked_tgt)


class GraphLoss(nn.Module):
    """
    Graph-supervised objectives for Phase 2:
      1. Edge classification: predict edge type for a given token pair.
      2. Triplet prediction: given subject embedding, predict relation+object.

    @param hidden_dim     - encoder hidden dimension
    @param num_edge_types - number of discrete edge type classes
    """

    def __init__(self, hidden_dim: int, num_edge_types: int = 8) -> None:
        super().__init__()
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_edge_types),
        )
        self.triplet_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        hidden_states: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        triplet_subjects: Tensor | None = None,
        triplet_objects: Tensor | None = None,
    ) -> Tensor:
        """
        @param hidden_states     - [B, T, D]
        @param edge_index        - [2, E]
        @param edge_type         - [E] ground-truth edge type labels
        @param triplet_subjects  - [K, D] subject token embeddings
        @param triplet_objects   - [K, D] target object token embeddings
        @returns scalar combined graph loss
        """
        loss = torch.tensor(0.0, device=hidden_states.device)

        # Edge classification loss
        if edge_index.shape[1] > 0:
            src, tgt = edge_index[0], edge_index[1]
            T = hidden_states.shape[1]
            valid = (src < T) & (tgt < T)
            src, tgt, etype = src[valid], tgt[valid], edge_type[valid]

            # Use first batch item's hidden states for efficiency
            hs = hidden_states[0]
            pairs = torch.cat([hs[src], hs[tgt]], dim=-1)   # [E, 2D]
            edge_logits = self.edge_classifier(pairs)         # [E, num_types]
            loss = loss + F.cross_entropy(edge_logits, etype.to(loss.device))

        # Triplet prediction: cosine similarity between predicted and actual object
        if triplet_subjects is not None and triplet_objects is not None:
            pred_obj = self.triplet_head(triplet_subjects)
            pred_obj = F.normalize(pred_obj, dim=-1)
            obj = F.normalize(triplet_objects.to(loss.device), dim=-1)
            cosine = (pred_obj * obj).sum(dim=-1)
            loss = loss + (1.0 - cosine).mean()

        return loss


class MultiTaskLoss(nn.Module):
    """
    Weighted combination of all pre-training losses.

    @param hidden_dim    - encoder hidden dimension
    @param vocab_size    - text vocabulary size
    @param patch_dim     - pixels per patch (3 * patch_size^2)
    @param num_edge_types - number of edge type classes
    @param w_align       - weight for contrastive alignment loss
    @param w_mlm         - weight for MLM loss
    @param w_mim         - weight for MIM loss
    @param w_graph       - weight for graph loss
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        patch_dim: int,
        num_edge_types: int = 8,
        w_align: float = 1.0,
        w_mlm: float = 0.5,
        w_mim: float = 0.5,
        w_graph: float = 0.3,
    ) -> None:
        super().__init__()
        self.w_align = w_align
        self.w_mlm = w_mlm
        self.w_mim = w_mim
        self.w_graph = w_graph

        self.align_loss = InfoNCELoss()
        self.mlm_loss = MLMLoss(hidden_dim, vocab_size)
        self.mim_loss = MIMLoss(hidden_dim, patch_dim)
        self.graph_loss = GraphLoss(hidden_dim, num_edge_types)

    def forward(
        self,
        image_feats: Tensor,
        text_feats: Tensor,
        text_hidden: Tensor,
        vision_hidden: Tensor,
        mlm_mask: Tensor,
        mlm_targets: Tensor,
        mim_mask: Tensor,
        mim_targets: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        all_hidden: Tensor,
    ) -> dict[str, Tensor]:
        """
        Compute and return all sub-losses plus the weighted total.

        @returns dict with keys: total, align, mlm, mim, graph
        """
        l_align = self.align_loss(image_feats, text_feats)
        l_mlm = self.mlm_loss(text_hidden, mlm_mask, mlm_targets)
        l_mim = self.mim_loss(vision_hidden, mim_mask, mim_targets)
        l_graph = self.graph_loss(all_hidden, edge_index, edge_type)

        total = (
            self.w_align * l_align
            + self.w_mlm * l_mlm
            + self.w_mim * l_mim
            + self.w_graph * l_graph
        )
        return {
            "total": total,
            "align": l_align,
            "mlm": l_mlm,
            "mim": l_mim,
            "graph": l_graph,
        }
