"""
Run B evaluation: Captioning-primary.

Primary KPI  : CIDEr
Secondary KPIs: BLEU-4, SPICE, VQA accuracy
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gdt.config import load_config
from gdt.data.coco_caption_dataset import COCOCaptionDataset
from gdt.evaluation.metrics import cider_score, bleu_score, print_results_table
from gdt.training.phase3_finetune import CaptioningModel


@torch.no_grad()
def evaluate_captioning(
    model: CaptioningModel,
    loader: DataLoader,
    device: torch.device,
    id_to_word: dict[int, str],
    beam_size: int = 5,
) -> tuple[dict, dict]:
    """
    Run caption generation and return (hypotheses, references) dicts for
    pycocoevalcap.
    """
    model.eval()
    hypotheses: dict[int, list[str]] = {}
    references: dict[int, list[str]] = {}

    for batch in tqdm(loader, desc="captioning eval", leave=False):
        images = batch["images"].to(device)
        text_ids = batch["text_ids"].to(device)
        image_ids = batch["image_id"]

        encoder_out = model.encoder(images, text_ids)
        token_lists = model.head.generate(
            encoder_out, max_new_tokens=50, beam_size=beam_size
        )

        for iid, tokens in zip(image_ids, token_lists):
            words = [id_to_word.get(t, "<unk>") for t in tokens]
            caption = " ".join(words)
            hypotheses[iid] = [caption]

    return hypotheses, references


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B: Captioning evaluation")
    parser.add_argument("--config", default="gdt/configs/run_b.yaml")
    parser.add_argument("--gdt-checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--ann-path", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--beam-size", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tokenizer(text: str, max_len: int) -> list[int]:
        tokens = [ord(c) % cfg.model.vocab_size for c in text.split()]
        tokens = tokens[:max_len]
        return tokens + [0] * (max_len - len(tokens))

    dataset = COCOCaptionDataset(
        ann_path=args.ann_path,
        image_dir=args.image_dir,
        tokenizer=tokenizer,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)

    # Minimal id→word mapping (replace with real vocab)
    id_to_word = {i: chr(i) for i in range(cfg.model.vocab_size)}

    gdt_model = CaptioningModel(cfg)
    ckpt = torch.load(args.gdt_checkpoint, map_location=device)
    gdt_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    gdt_model = gdt_model.to(device)

    gdt_hyps, refs = evaluate_captioning(
        gdt_model, loader, device, id_to_word, args.beam_size
    )

    from gdt.models.baseline import BaselineEncoder
    from gdt.models.task_heads import CaptioningHead
    import torch.nn as nn

    class BaselineCaptioning(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = BaselineEncoder(cfg.model)
            self.head = CaptioningHead(cfg.model)

        def forward(self, images, text_ids, cap_ids):
            enc = self.encoder(images, text_ids)
            return self.head(enc, cap_ids)

    base_model = BaselineCaptioning()
    ckpt_b = torch.load(args.baseline_checkpoint, map_location=device)
    base_model.load_state_dict(ckpt_b.get("model_state_dict", ckpt_b))
    base_model = base_model.to(device)

    # Wrap baseline to match CaptioningModel interface
    base_model.head = base_model.head
    base_hyps, _ = evaluate_captioning(
        base_model, loader, device, id_to_word, args.beam_size
    )

    if refs:
        gdt_cider = cider_score(gdt_hyps, refs)
        base_cider = cider_score(base_hyps, refs)
        gdt_bleu = bleu_score(gdt_hyps, refs)
        base_bleu = bleu_score(base_hyps, refs)
    else:
        print("No references available; skipping metric computation.")
        gdt_cider = base_cider = gdt_bleu = base_bleu = 0.0

    print_results_table(
        run_name="Run B: Captioning-Primary",
        primary_metric="CIDEr",
        results={
            "CIDEr": (base_cider, gdt_cider),
            "BLEU-4": (base_bleu, gdt_bleu),
        },
    )


if __name__ == "__main__":
    main()
