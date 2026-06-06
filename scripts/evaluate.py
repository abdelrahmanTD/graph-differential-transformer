"""
Evaluation entry point.

Usage:
  python -m gdt.scripts.evaluate --run a \\
      --gdt-checkpoint checkpoints/run_a/best.pt \\
      --baseline-checkpoint checkpoints/baseline_a/best.pt \\
      --ann-path data/vqa/v2_mscoco_val2014_annotations.json \\
      --q-path   data/vqa/v2_OpenEnded_mscoco_val2014_questions.json \\
      --image-dir data/coco/val2014
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GDT evaluation")
    p.add_argument("--run", choices=["a", "b", "c"], required=True,
                   help="a=VQA, b=captioning, c=detection")
    p.add_argument("--config", default=None, help="Path to run YAML config")
    p.add_argument("--gdt-checkpoint", required=True)
    p.add_argument("--baseline-checkpoint", required=True)
    p.add_argument("--ann-path", required=True)
    p.add_argument("--q-path", default=None, help="Questions JSON (Run A only)")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--beam-size", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    sys.argv = [sys.argv[0]]   # clear so sub-parsers don't conflict

    if args.run == "a":
        config = args.config or "gdt/configs/run_a.yaml"
        from gdt.evaluation.run_a_vqa import main as run_a
        import sys
        sys.argv = [
            "run_a",
            "--config", config,
            "--gdt-checkpoint", args.gdt_checkpoint,
            "--baseline-checkpoint", args.baseline_checkpoint,
            "--ann-path", args.ann_path,
            "--q-path", args.q_path or "",
            "--image-dir", args.image_dir,
            "--batch-size", str(args.batch_size),
        ]
        run_a()

    elif args.run == "b":
        config = args.config or "gdt/configs/run_b.yaml"
        from gdt.evaluation.run_b_captioning import main as run_b
        sys.argv = [
            "run_b",
            "--config", config,
            "--gdt-checkpoint", args.gdt_checkpoint,
            "--baseline-checkpoint", args.baseline_checkpoint,
            "--ann-path", args.ann_path,
            "--image-dir", args.image_dir,
            "--batch-size", str(args.batch_size),
            "--beam-size", str(args.beam_size),
        ]
        run_b()

    else:
        config = args.config or "gdt/configs/run_c.yaml"
        from gdt.evaluation.run_c_detection import main as run_c
        sys.argv = [
            "run_c",
            "--config", config,
            "--gdt-checkpoint", args.gdt_checkpoint,
            "--baseline-checkpoint", args.baseline_checkpoint,
            "--ann-path", args.ann_path,
            "--image-dir", args.image_dir,
            "--batch-size", str(args.batch_size),
        ]
        run_c()


if __name__ == "__main__":
    main()
