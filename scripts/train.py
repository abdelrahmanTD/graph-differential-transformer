"""
Main training entry point.

Usage:
  python -m gdt.scripts.train --phase 1 --config gdt/configs/base.yaml
  python -m gdt.scripts.train --phase 2 --config gdt/configs/base.yaml \\
      --resume checkpoints/phase1/best.pt
  python -m gdt.scripts.train --phase 3 --run a --config gdt/configs/run_a.yaml \\
      --resume checkpoints/phase2/best.pt
"""

from __future__ import annotations

import argparse
import sys

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GDT training")
    p.add_argument("--phase", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--config", default="gdt/configs/base.yaml")
    p.add_argument("--run", choices=["a", "b", "c"], default="a",
                   help="Phase 3 run (a=VQA, b=captioning, c=detection)")
    p.add_argument("--resume", default=None, help="Checkpoint to resume from")
    p.add_argument("--data-root", default="data")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from gdt.config import load_config
    cfg = load_config(args.config)

    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.epochs:
        cfg.run.epochs = args.epochs
    cfg.run.data_root = args.data_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Phase: {args.phase}  Config: {args.config}")

    if args.phase == 1:
        _run_phase1(cfg, args)
    elif args.phase == 2:
        _run_phase2(cfg, args)
    else:
        _run_phase3(cfg, args)


def _dummy_loader(cfg, batch_size: int = 4):
    """
    Return a minimal DataLoader with synthetic data for smoke-testing
    when real datasets are unavailable.
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    B = 8
    Lv = (cfg.model.img_size // cfg.model.patch_size) ** 2
    Lt = 16
    images = torch.randn(B, 3, cfg.model.img_size, cfg.model.img_size)
    text_ids = torch.randint(0, cfg.model.vocab_size, (B, Lt))
    ds = TensorDataset(images, text_ids)

    class WrapDataset(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            imgs, tids = self.ds[i]
            return {"images": imgs, "text_ids": tids}

    return DataLoader(WrapDataset(ds), batch_size=batch_size)


def _run_phase1(cfg, args) -> None:
    from gdt.training.phase1_pretrain import run_phase1

    print("Starting Phase 1 pre-training...")
    train_loader = _dummy_loader(cfg, cfg.training.batch_size)
    run_phase1(cfg, train_loader, resume_from=args.resume)


def _run_phase2(cfg, args) -> None:
    from gdt.training.phase2_graph import run_phase2

    print("Starting Phase 2 graph integration...")
    train_loader = _dummy_loader(cfg, cfg.training.batch_size)
    run_phase2(cfg, train_loader, phase1_checkpoint=args.resume)


def _run_phase3(cfg, args) -> None:
    from gdt.training.phase3_finetune import run_a_vqa, run_b_captioning, run_c_detection
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    B, Lt = 8, 16
    img = torch.randn(B, 3, cfg.model.img_size, cfg.model.img_size)
    tid = torch.randint(0, cfg.model.vocab_size, (B, Lt))

    if args.run == "a":
        print("Starting Run A (VQA)...")
        scores = torch.rand(B, cfg.run.num_vqa_answers)

        class VQADs(torch.utils.data.Dataset):
            def __len__(self): return B
            def __getitem__(self, i):
                return {
                    "images": img[i], "text_ids": tid[i],
                    "answer_scores": scores[i],
                }

        loader = DataLoader(VQADs(), batch_size=4)
        run_a_vqa(cfg, loader, phase2_checkpoint=args.resume)

    elif args.run == "b":
        print("Starting Run B (Captioning)...")
        cap_in = torch.randint(0, cfg.model.vocab_size, (B, 20))
        cap_out = torch.randint(0, cfg.model.vocab_size, (B, 20))

        class CapDs(torch.utils.data.Dataset):
            def __len__(self): return B
            def __getitem__(self, i):
                return {
                    "images": img[i], "text_ids": tid[i],
                    "caption_ids_in": cap_in[i], "caption_ids_out": cap_out[i],
                }

        loader = DataLoader(CapDs(), batch_size=4)
        run_b_captioning(cfg, loader, phase2_checkpoint=args.resume)

    else:
        print("Starting Run C (Detection)...")
        from gdt.data.coco_detection_dataset import detection_collate_fn

        class DetDs(torch.utils.data.Dataset):
            def __len__(self): return B
            def __getitem__(self, i):
                return {
                    "images": img[i], "text_ids": tid[i],
                    "targets": {
                        "labels": torch.randint(0, cfg.run.num_coco_classes, (3,)),
                        "boxes": torch.rand(3, 4),
                    },
                    "image_id": i,
                }

        loader = DataLoader(
            DetDs(), batch_size=4, collate_fn=detection_collate_fn
        )
        run_c_detection(cfg, loader, phase2_checkpoint=args.resume)


if __name__ == "__main__":
    main()
