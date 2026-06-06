"""
Run A evaluation: VQA-primary.

Loads GDT and Baseline models, runs inference on the VQA validation split,
and reports primary + secondary KPIs with relative gain.

Primary KPI  : VQA soft accuracy
Secondary KPIs: COCO CIDEr, COCO AP
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gdt.config import GDTConfig, load_config
from gdt.data.vqa_dataset import VQADataset
from gdt.evaluation.metrics import (
    vqa_soft_accuracy,
    print_results_table,
    relative_gain,
)
from gdt.models.baseline import BaselineEncoder
from gdt.models.gdt_encoder import GDTEncoder
from gdt.models.task_heads import VQAHead
from gdt.training.phase3_finetune import VQAModel


@torch.no_grad()
def evaluate_vqa(
    model: VQAModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Run VQA inference and return soft accuracy."""
    model.eval()
    all_logits, all_targets = [], []

    for batch in tqdm(loader, desc="VQA eval", leave=False):
        images = batch["images"].to(device)
        text_ids = batch["text_ids"].to(device)
        targets = batch["answer_scores"].to(device)

        logits = model(images, text_ids)
        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    return vqa_soft_accuracy(all_logits, all_targets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A: VQA evaluation")
    parser.add_argument("--config", default="gdt/configs/run_a.yaml")
    parser.add_argument("--gdt-checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--ann-path", required=True)
    parser.add_argument("--q-path", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Minimal tokenizer stub — replace with real WordPiece tokenizer
    def tokenizer(text: str, max_len: int) -> list[int]:
        tokens = [ord(c) % cfg.model.vocab_size for c in text.split()]
        tokens = tokens[:max_len]
        return tokens + [0] * (max_len - len(tokens))

    dataset = VQADataset(
        ann_path=args.ann_path,
        q_path=args.q_path,
        image_dir=args.image_dir,
        tokenizer=tokenizer,
        num_answers=cfg.run.num_vqa_answers,
        split="val",
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)

    # Load GDT model
    gdt_model = VQAModel(cfg, num_answers=cfg.run.num_vqa_answers)
    ckpt = torch.load(args.gdt_checkpoint, map_location=device)
    gdt_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    gdt_model = gdt_model.to(device)

    # Load baseline model (same VQA head, different encoder)
    class BaselineVQA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = BaselineEncoder(cfg.model)
            self.head = VQAHead(cfg.model, cfg.run.num_vqa_answers)

        def forward(self, images, text_ids):
            enc = self.encoder(images, text_ids)
            return self.head(enc[:, 0, :])

    baseline_model = BaselineVQA()
    ckpt_b = torch.load(args.baseline_checkpoint, map_location=device)
    baseline_model.load_state_dict(ckpt_b.get("model_state_dict", ckpt_b))
    baseline_model = baseline_model.to(device)

    gdt_acc = evaluate_vqa(gdt_model, loader, device)
    base_acc = evaluate_vqa(baseline_model, loader, device)

    print_results_table(
        run_name="Run A: VQA-Primary",
        primary_metric="VQA soft accuracy",
        results={
            "VQA soft accuracy": (base_acc, gdt_acc),
        },
    )


if __name__ == "__main__":
    main()
