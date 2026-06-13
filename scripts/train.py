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
import os
import sys
import types

# Bootstrap: register the project root as the "gdt" namespace package so that
# all internal `from gdt.X import Y` imports resolve correctly when running
# this script directly (mirrors the same trick used in app.py).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if "gdt" not in sys.modules:
    _gdt_mod = types.ModuleType("gdt")
    _gdt_mod.__path__ = [_PROJECT_ROOT]  # type: ignore[assignment]
    _gdt_mod.__package__ = "gdt"
    _gdt_mod.__file__ = os.path.join(_PROJECT_ROOT, "__init__.py")
    sys.modules["gdt"] = _gdt_mod

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


def _vqa_loader(cfg, args):
    """Return a DataLoader for VQA training. Uses real data when available, dummy otherwise."""
    import os
    from torch.utils.data import DataLoader

    data_root = cfg.run.data_root
    ann_path = os.path.join(data_root, cfg.run.vqa_ann_path)
    q_path = os.path.join(data_root, cfg.run.vqa_q_path)
    image_dir = os.path.join(data_root, "coco", "train2014")

    if os.path.exists(ann_path) and os.path.exists(q_path) and os.path.isdir(image_dir):
        print(f"  Loading real VQA v2 data from {data_root}")
        from gdt.data.vqa_dataset import VQADataset, build_bert_tokenizer
        tokenizer = build_bert_tokenizer(cfg.model.vocab_size)
        dataset = VQADataset(
            ann_path=ann_path,
            q_path=q_path,
            image_dir=image_dir,
            tokenizer=tokenizer,
            num_answers=cfg.run.num_vqa_answers,
            split="train",
        )
        return DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    print("  VQA data not found — using synthetic dummy data for smoke-testing.")
    print(f"  To train for real, download VQA v2 + COCO to: {data_root}")
    B, Lt = 8, 16
    img = torch.randn(B, 3, cfg.model.img_size, cfg.model.img_size)
    tid = torch.randint(0, cfg.model.vocab_size, (B, Lt))
    scores = torch.rand(B, cfg.run.num_vqa_answers)

    class _DummyVQA(torch.utils.data.Dataset):
        def __len__(self): return B
        def __getitem__(self, i):
            return {"images": img[i], "text_ids": tid[i], "answer_scores": scores[i]}

    return torch.utils.data.DataLoader(_DummyVQA(), batch_size=4)


def _run_phase3(cfg, args) -> None:
    from gdt.training.phase3_finetune import run_a_vqa, run_b_captioning, run_c_detection
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    B, Lt = 8, 16
    img = torch.randn(B, 3, cfg.model.img_size, cfg.model.img_size)
    tid = torch.randint(0, cfg.model.vocab_size, (B, Lt))

    if args.run == "a":
        print("Starting Run A (VQA)...")
        loader = _vqa_loader(cfg, args)
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
