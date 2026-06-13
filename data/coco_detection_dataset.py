"""COCO detection dataset and collate function for GDT Run C."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


def detection_collate_fn(batch: list[dict]) -> dict:
    """Collate detection samples — stacks tensors, keeps targets as a list."""
    images = torch.stack([s["images"] for s in batch])
    text_ids = torch.stack([s["text_ids"] for s in batch])
    targets = [s["targets"] for s in batch]
    image_ids = [s.get("image_id", -1) for s in batch]
    return {
        "images": images,
        "text_ids": text_ids,
        "targets": targets,
        "image_id": image_ids,
    }
