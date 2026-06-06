"""
Run C evaluation: Detection/Segmentation-primary.

Primary KPI  : COCO AP @ IoU=0.50:0.95
Secondary KPIs: mask AP, VQA accuracy
"""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gdt.config import load_config
from gdt.data.coco_detection_dataset import COCODetectionDataset, detection_collate_fn
from gdt.evaluation.metrics import coco_ap, print_results_table
from gdt.training.phase3_finetune import DetectionModel


@torch.no_grad()
def collect_detections(
    model: DetectionModel,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.05,
) -> list[dict]:
    """
    Run detection inference and collect COCO-format detection results.

    @returns list of {image_id, category_id, bbox, score} dicts
    """
    model.eval()
    results = []

    for batch in tqdm(loader, desc="detection eval", leave=False):
        images = batch["images"].to(device)
        text_ids = batch["text_ids"].to(device)
        image_ids = batch["image_id"]

        pred_logits, pred_boxes = model(images, text_ids)
        probs = pred_logits.softmax(dim=-1)  # [B, N, C+1]
        no_obj_idx = pred_logits.shape[-1] - 1

        for b, image_id in enumerate(image_ids):
            for n in range(pred_logits.shape[1]):
                scores = probs[b, n]
                # Exclude no-object class
                obj_scores = scores[:-1]
                max_score, cat_id = obj_scores.max(0)
                if max_score.item() < score_threshold:
                    continue

                cx, cy, w, h = pred_boxes[b, n].tolist()
                # Convert from [cx, cy, w, h] normalised to [x, y, w, h]
                # in pixel space (assuming 224×224)
                img_w, img_h = 224, 224
                x = (cx - w / 2) * img_w
                y = (cy - h / 2) * img_h
                bw = w * img_w
                bh = h * img_h

                results.append({
                    "image_id": int(image_id),
                    "category_id": int(cat_id.item()) + 1,  # 1-indexed COCO
                    "bbox": [x, y, bw, bh],
                    "score": float(max_score.item()),
                })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run C: Detection evaluation")
    parser.add_argument("--config", default="gdt/configs/run_c.yaml")
    parser.add_argument("--gdt-checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--ann-path", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = COCODetectionDataset(
        ann_path=args.ann_path,
        image_dir=args.image_dir,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=4,
        collate_fn=detection_collate_fn,
    )

    gdt_model = DetectionModel(cfg, num_classes=cfg.run.num_coco_classes)
    ckpt = torch.load(args.gdt_checkpoint, map_location=device)
    gdt_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    gdt_model = gdt_model.to(device)

    from gdt.models.baseline import BaselineEncoder
    from gdt.models.task_heads import DetectionHead
    import torch.nn as nn

    class BaselineDetection(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = BaselineEncoder(cfg.model)
            self.head = DetectionHead(
                cfg.model.hidden_dim, cfg.run.num_coco_classes,
                cfg.model.num_object_queries, cfg.model.num_heads,
            )

        def forward(self, images, text_ids):
            enc = self.encoder(images, text_ids)
            return self.head(enc)

    baseline_model = BaselineDetection()
    ckpt_b = torch.load(args.baseline_checkpoint, map_location=device)
    baseline_model.load_state_dict(ckpt_b.get("model_state_dict", ckpt_b))
    baseline_model = baseline_model.to(device)

    gdt_dets = collect_detections(gdt_model, loader, device)
    base_dets = collect_detections(baseline_model, loader, device)

    gdt_ap = coco_ap(gdt_dets, args.ann_path)
    base_ap = coco_ap(base_dets, args.ann_path)

    print_results_table(
        run_name="Run C: Detection-Primary",
        primary_metric="COCO AP",
        results={
            "COCO AP": (base_ap, gdt_ap),
        },
    )


if __name__ == "__main__":
    main()
