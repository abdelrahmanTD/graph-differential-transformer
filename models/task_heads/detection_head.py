"""
DETR-style detection and segmentation head (Run C).

Architecture:
  - N=100 learnable object queries cross-attend to encoder output
  - class_embed: Linear(D, num_classes + 1)   (+ 1 for no-object)
  - bbox_embed:  MLP(D, D, 4)  predicting (cx, cy, w, h) in [0,1]
  - Optional mask_head for panoptic / instance segmentation

Loss:
  - Hungarian bipartite matching between predictions and targets
  - Classification: focal loss
  - BBox: L1 + GIoU
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MLP(nn.Module):
    """Simple n-layer MLP with ReLU."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 3) -> None:
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        for i in range(n_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < n_layers - 1:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class HungarianMatcher:
    """
    Greedy Hungarian matching between predicted and target boxes.
    Uses scipy.optimize.linear_sum_assignment.
    """

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0) -> None:
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def __call__(
        self, pred_logits: Tensor, pred_boxes: Tensor, targets: list[dict]
    ) -> list[tuple[Tensor, Tensor]]:
        """
        @param pred_logits - [B, N, num_classes+1]
        @param pred_boxes  - [B, N, 4]  (cx, cy, w, h) in [0,1]
        @param targets     - list of dicts with 'labels' [M] and 'boxes' [M, 4]
        @returns list of (pred_idx, tgt_idx) index pairs per batch item
        """
        from scipy.optimize import linear_sum_assignment

        B, N, _ = pred_logits.shape
        indices = []

        for b in range(B):
            tgt = targets[b]
            tgt_labels = tgt["labels"]       # [M]
            tgt_boxes = tgt["boxes"]         # [M, 4]
            M = len(tgt_labels)

            if M == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long),
                ))
                continue

            prob = pred_logits[b].softmax(dim=-1)  # [N, C+1]
            cost_class = -prob[:, tgt_labels]        # [N, M]

            pred_b = pred_boxes[b]    # [N, 4]
            cost_bbox = torch.cdist(pred_b, tgt_boxes, p=1)  # [N, M]

            cost_giou = -self._giou(pred_b, tgt_boxes)        # [N, M]

            C = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            ).cpu().numpy()

            row_ind, col_ind = linear_sum_assignment(C)
            indices.append((
                torch.tensor(row_ind, dtype=torch.long),
                torch.tensor(col_ind, dtype=torch.long),
            ))

        return indices

    @staticmethod
    def _giou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
        """Generalised IoU between all pairs of boxes [N,4] x [M,4] → [N,M]."""
        def to_corners(b):
            cx, cy, w, h = b.unbind(-1)
            return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

        a = to_corners(boxes_a)   # [N, 4]
        b = to_corners(boxes_b)   # [M, 4]

        # Intersection
        inter_x1 = torch.max(a[:, None, 0], b[None, :, 0])
        inter_y1 = torch.max(a[:, None, 1], b[None, :, 1])
        inter_x2 = torch.min(a[:, None, 2], b[None, :, 2])
        inter_y2 = torch.min(a[:, None, 3], b[None, :, 3])
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])).clamp(min=0)
        area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).clamp(min=0)
        union = area_a[:, None] + area_b[None, :] - inter_area

        iou = inter_area / union.clamp(min=1e-6)

        # Enclosing box
        enc_x1 = torch.min(a[:, None, 0], b[None, :, 0])
        enc_y1 = torch.min(a[:, None, 1], b[None, :, 1])
        enc_x2 = torch.max(a[:, None, 2], b[None, :, 2])
        enc_y2 = torch.max(a[:, None, 3], b[None, :, 3])
        enc_area = ((enc_x2 - enc_x1) * (enc_y2 - enc_y1)).clamp(min=1e-6)

        return iou - (enc_area - union) / enc_area


class DetectionHead(nn.Module):
    """
    DETR-style detection head with Hungarian matching.

    @param hidden_dim  - encoder hidden dimension
    @param num_classes - number of COCO object classes (default: 91)
    @param num_queries - number of object queries (default: 100)
    @param num_heads   - attention heads for query cross-attention
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = 91,
        num_queries: int = 100,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=6,
        )
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, n_layers=3)
        self.matcher = HungarianMatcher()

    def forward(self, encoder_out: Tensor) -> tuple[Tensor, Tensor]:
        """
        @param encoder_out - [B, Tm, D]
        @returns
            pred_logits [B, N, num_classes+1]
            pred_boxes  [B, N, 4]   values in [0, 1] after sigmoid
        """
        B = encoder_out.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        hs = self.transformer(queries, encoder_out)      # [B, N, D]
        pred_logits = self.class_embed(hs)               # [B, N, C+1]
        pred_boxes = self.bbox_embed(hs).sigmoid()       # [B, N, 4]
        return pred_logits, pred_boxes

    def compute_loss(
        self,
        pred_logits: Tensor,
        pred_boxes: Tensor,
        targets: list[dict],
    ) -> Tensor:
        """
        Hungarian-matched detection loss: classification + L1 + GIoU.

        @param pred_logits - [B, N, C+1]
        @param pred_boxes  - [B, N, 4]
        @param targets     - list of dicts with 'labels' [M] and 'boxes' [M, 4]
        @returns scalar loss
        """
        indices = self.matcher(pred_logits, pred_boxes, targets)

        # Classification loss (cross-entropy; unmatched → no-object class)
        no_obj = self.num_classes
        tgt_classes = torch.full(
            pred_logits.shape[:2], no_obj, dtype=torch.long, device=pred_logits.device
        )
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            tgt_classes[b, pred_idx] = targets[b]["labels"][tgt_idx].to(pred_logits.device)

        loss_cls = F.cross_entropy(
            pred_logits.reshape(-1, self.num_classes + 1),
            tgt_classes.reshape(-1),
        )

        # Box regression losses (only for matched pairs)
        loss_bbox = torch.tensor(0.0, device=pred_logits.device)
        loss_giou = torch.tensor(0.0, device=pred_logits.device)
        num_matched = 0

        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            matched_pred = pred_boxes[b, pred_idx]
            matched_tgt = targets[b]["boxes"][tgt_idx].to(pred_logits.device)
            loss_bbox = loss_bbox + F.l1_loss(matched_pred, matched_tgt)
            giou = HungarianMatcher._giou(matched_pred, matched_tgt).diag()
            loss_giou = loss_giou + (1 - giou).mean()
            num_matched += len(pred_idx)

        if num_matched > 0:
            loss_bbox = loss_bbox / num_matched
            loss_giou = loss_giou / num_matched

        return loss_cls + 5.0 * loss_bbox + 2.0 * loss_giou
