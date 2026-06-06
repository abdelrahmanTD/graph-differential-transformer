"""
Evaluation metrics for all three GDT runs.

  - vqa_accuracy       : exact-match VQA accuracy
  - cider_score        : CIDEr via pycocoevalcap
  - bleu_score         : BLEU-4 via pycocoevalcap
  - coco_ap            : COCO detection AP via pycocotools
  - relative_gain      : (gdt - baseline) / baseline * 100
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def vqa_accuracy(
    preds: Tensor,
    targets: Tensor,
    answer_vocab: Optional[list[str]] = None,
) -> float:
    """
    VQA exact-match accuracy.

    @param preds        - [N] predicted answer indices
    @param targets      - [N] ground-truth answer indices
    @param answer_vocab - optional list of answer strings (unused here)
    @returns accuracy in [0, 1]
    """
    correct = (preds == targets).sum().item()
    return correct / len(preds)


def vqa_soft_accuracy(
    logits: Tensor,
    soft_targets: Tensor,
) -> float:
    """
    VQA soft-score accuracy (per VQA v2 evaluation protocol).

    For each example, accuracy = min(count_predicted / 3, 1).

    @param logits       - [N, A] raw logits
    @param soft_targets - [N, A] soft scores (already in [0, 1])
    @returns mean soft accuracy
    """
    preds = logits.argmax(dim=-1)       # [N]
    scores = soft_targets[torch.arange(len(preds)), preds]  # [N]
    return scores.mean().item()


def cider_score(
    hypotheses: dict[int, list[str]],
    references: dict[int, list[str]],
) -> float:
    """
    Compute CIDEr score using pycocoevalcap.

    @param hypotheses - {image_id: [caption_string]}
    @param references - {image_id: [ref1, ref2, ...]}
    @returns CIDEr score (typically 0-200+ range)
    """
    try:
        from pycocoevalcap.cider.cider import Cider
        scorer = Cider()
        score, _ = scorer.compute_score(references, hypotheses)
        return float(score)
    except ImportError:
        # Fallback: return 0.0 with a warning
        print("Warning: pycocoevalcap not installed; CIDEr returns 0.0")
        return 0.0


def bleu_score(
    hypotheses: dict[int, list[str]],
    references: dict[int, list[str]],
    n: int = 4,
) -> float:
    """
    Compute BLEU-n score using pycocoevalcap.

    @param hypotheses - {image_id: [caption_string]}
    @param references - {image_id: [ref1, ref2, ...]}
    @param n          - max n-gram order (default: 4)
    @returns BLEU score in [0, 1]
    """
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        scorer = Bleu(n)
        scores, _ = scorer.compute_score(references, hypotheses)
        return float(scores[n - 1])
    except ImportError:
        print("Warning: pycocoevalcap not installed; BLEU returns 0.0")
        return 0.0


def spice_score(
    hypotheses: dict[int, list[str]],
    references: dict[int, list[str]],
) -> float:
    """
    Compute SPICE score using pycocoevalcap.
    Note: SPICE requires Java; falls back to 0.0 if unavailable.

    @param hypotheses - {image_id: [caption_string]}
    @param references - {image_id: [ref1, ref2, ...]}
    @returns SPICE F-score
    """
    try:
        from pycocoevalcap.spice.spice import Spice
        scorer = Spice()
        score, _ = scorer.compute_score(references, hypotheses)
        return float(score)
    except Exception:
        print("Warning: SPICE unavailable; returns 0.0")
        return 0.0


def coco_ap(
    detections: list[dict],
    annotation_path: str,
    iou_type: str = "bbox",
) -> float:
    """
    Compute COCO Average Precision using pycocotools.

    @param detections      - list of {image_id, category_id, bbox, score}
    @param annotation_path - path to COCO instances annotation JSON
    @param iou_type        - 'bbox' or 'segm'
    @returns AP @ IoU=0.50:0.95
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt = COCO(annotation_path)
        coco_dt = coco_gt.loadRes(detections)
        evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return float(evaluator.stats[0])  # AP @ IoU=0.50:0.95
    except ImportError:
        print("Warning: pycocotools not installed; COCO AP returns 0.0")
        return 0.0


def relative_gain(gdt_score: float, baseline_score: float) -> float:
    """
    Relative improvement of GDT over baseline, as a percentage.

    Gain = (GDT - Baseline) / Baseline × 100

    @param gdt_score      - GDT primary metric value
    @param baseline_score - Baseline primary metric value
    @returns gain in percent (positive = improvement)
    """
    if baseline_score == 0.0:
        return float("inf")
    return (gdt_score - baseline_score) / baseline_score * 100.0


def print_results_table(
    run_name: str,
    primary_metric: str,
    results: dict[str, tuple[float, float]],
) -> None:
    """
    Print a formatted results table comparing GDT to baseline.

    @param run_name      - e.g. "Run A: VQA-Primary"
    @param primary_metric - name of the primary KPI
    @param results       - {metric_name: (baseline_score, gdt_score)}
    """
    print(f"\n{'='*65}")
    print(f"  {run_name}")
    print(f"{'='*65}")
    print(f"  {'Metric':<30} {'Baseline':>10} {'GDT':>10} {'Gain %':>10}")
    print(f"  {'-'*60}")
    for metric, (base, gdt) in results.items():
        gain = relative_gain(gdt, base)
        marker = " *" if metric == primary_metric else ""
        print(f"  {metric:<30} {base:>10.4f} {gdt:>10.4f} {gain:>+10.1f}{marker}")
    print(f"{'='*65}")
    print(f"  * Primary KPI (target: +20% relative)\n")
