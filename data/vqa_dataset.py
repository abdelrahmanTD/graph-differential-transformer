"""VQA v2 dataset loader for the GDT VQA fine-tuning pipeline (Phase 3, Run A)."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Callable

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

_PREPROCESS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

_MAX_Q_LEN = 16
_SOFT_SCORE_DENOM = 3  # VQA v2 soft-score formula: min(count / 3, 1.0)


class VQADataset(Dataset):
    """
    VQA v2 dataset.

    Each sample returns:
        images        FloatTensor [3, 224, 224]
        text_ids      LongTensor  [16]            tokenized question
        answer_scores FloatTensor [num_answers]   soft VQA v2 scores
        image_id      int
    """

    def __init__(
        self,
        ann_path: str,
        q_path: str,
        image_dir: str,
        tokenizer: Callable[[str, int], list[int]],
        num_answers: int = 3129,
        split: str = "train",
        vocab_path: str | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.num_answers = num_answers
        self.split = split

        with open(ann_path) as f:
            anns = {a["question_id"]: a for a in json.load(f)["annotations"]}
        with open(q_path) as f:
            qs = {q["question_id"]: q for q in json.load(f)["questions"]}

        self.samples = [
            (qid, qs[qid]["image_id"], qs[qid]["question"], anns[qid]["answers"])
            for qid in qs
            if qid in anns
        ]

        # Build or load answer vocabulary
        if vocab_path is None:
            vocab_path = os.path.join(os.path.dirname(ann_path), "answer_vocab.json")
        self.vocab_path = vocab_path

        if split == "train" or not os.path.exists(vocab_path):
            self.answer2idx = self._build_vocab(vocab_path)
        else:
            with open(vocab_path) as f:
                self.answer2idx = json.load(f)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        qid, image_id, question, answers = self.samples[idx]

        # Load and preprocess image
        img_file = self._find_image(image_id)
        image = _PREPROCESS(Image.open(img_file).convert("RGB"))

        # Tokenize question
        text_ids = torch.tensor(
            self.tokenizer(question, _MAX_Q_LEN), dtype=torch.long
        )

        # Build soft-score vector
        answer_scores = self._soft_scores(answers)

        return {
            "images": image,
            "text_ids": text_ids,
            "answer_scores": answer_scores,
            "image_id": image_id,
        }

    # ------------------------------------------------------------------
    def _build_vocab(self, vocab_path: str) -> dict[str, int]:
        counts: Counter = Counter()
        for _, _, _, answers in self.samples:
            for a in answers:
                counts[a["answer"].lower().strip()] += 1

        top_answers = [ans for ans, _ in counts.most_common(self.num_answers)]
        vocab = {ans: i for i, ans in enumerate(top_answers)}

        os.makedirs(os.path.dirname(vocab_path) or ".", exist_ok=True)
        with open(vocab_path, "w") as f:
            json.dump(vocab, f)

        return vocab

    def _soft_scores(self, answers: list[dict]) -> torch.Tensor:
        scores = torch.zeros(self.num_answers, dtype=torch.float32)
        counts: Counter = Counter()
        for a in answers:
            counts[a["answer"].lower().strip()] += 1
        for ans, count in counts.items():
            idx = self.answer2idx.get(ans)
            if idx is not None:
                scores[idx] = min(count / _SOFT_SCORE_DENOM, 1.0)
        return scores

    def _find_image(self, image_id: int) -> Path:
        # COCO images are named COCO_<split>2014_<id:012d>.jpg
        for pattern in [
            f"COCO_train2014_{image_id:012d}.jpg",
            f"COCO_val2014_{image_id:012d}.jpg",
            f"{image_id:012d}.jpg",
            f"{image_id}.jpg",
        ]:
            p = self.image_dir / pattern
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Image {image_id} not found in {self.image_dir}. "
            "Ensure COCO images are downloaded and placed correctly."
        )


def build_bert_tokenizer(vocab_size: int = 30522) -> Callable[[str, int], list[int]]:
    """Return a BertTokenizer-backed callable matching the VQADataset tokenizer interface."""
    from transformers import BertTokenizer

    tok = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(text: str, max_len: int) -> list[int]:
        enc = tok(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        return enc["input_ids"]

    return tokenize
