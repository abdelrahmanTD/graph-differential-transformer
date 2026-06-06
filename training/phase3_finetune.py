"""
Phase 3: Task-Specific Fine-Tuning

Three runs, each fine-tuning the shared GDT backbone with a task head:
  Run A (VQA-primary)       : VQAHead, lr=1e-5, 10-20 epochs
  Run B (Captioning-primary): CaptioningHead, lr=5e-5, 10-15 epochs
  Run C (Detection-primary) : DetectionHead, lr=1e-5, 12-24 epochs

Each run initialises from the Phase 2 checkpoint.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from gdt.config import GDTConfig
from gdt.models.gdt_encoder import GDTEncoder
from gdt.models.task_heads import VQAHead, CaptioningHead, DetectionHead
from gdt.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Combined model wrappers
# ---------------------------------------------------------------------------

class VQAModel(nn.Module):
    """GDTEncoder + VQAHead."""

    def __init__(self, cfg: GDTConfig, num_answers: int) -> None:
        super().__init__()
        self.encoder = GDTEncoder(cfg.model)
        self.head = VQAHead(cfg.model, num_answers)

    def forward(self, images: Tensor, text_ids: Tensor) -> Tensor:
        enc = self.encoder(images, text_ids)
        return self.head(enc[:, 0, :])  # [B, num_answers]


class CaptioningModel(nn.Module):
    """GDTEncoder + CaptioningHead."""

    def __init__(self, cfg: GDTConfig) -> None:
        super().__init__()
        self.encoder = GDTEncoder(cfg.model)
        self.head = CaptioningHead(cfg.model)

    def forward(
        self, images: Tensor, text_ids: Tensor, caption_ids: Tensor
    ) -> Tensor:
        enc = self.encoder(images, text_ids)
        return self.head(enc, caption_ids)  # [B, Lt, V]


class DetectionModel(nn.Module):
    """GDTEncoder + DetectionHead."""

    def __init__(self, cfg: GDTConfig, num_classes: int = 91) -> None:
        super().__init__()
        self.encoder = GDTEncoder(cfg.model)
        self.head = DetectionHead(
            hidden_dim=cfg.model.hidden_dim,
            num_classes=num_classes,
            num_queries=cfg.model.num_object_queries,
            num_heads=cfg.model.num_heads,
        )

    def forward(self, images: Tensor, text_ids: Tensor):
        enc = self.encoder(images, text_ids)
        return self.head(enc)  # (pred_logits, pred_boxes)


# ---------------------------------------------------------------------------
# Loss wrappers
# ---------------------------------------------------------------------------

def _vqa_loss(model: VQAModel, batch: dict) -> dict[str, Tensor]:
    logits = model(batch["images"], batch["text_ids"])
    targets = batch["answer_scores"]   # [B, num_answers] soft scores
    loss = model.head.compute_loss(logits, targets)
    return {"total": loss}


def _captioning_loss(model: CaptioningModel, batch: dict) -> dict[str, Tensor]:
    logits = model(batch["images"], batch["text_ids"], batch["caption_ids_in"])
    targets = batch["caption_ids_out"]   # [B, Lt]
    loss = model.head.compute_loss(logits, targets)
    return {"total": loss}


def _detection_loss(model: DetectionModel, batch: dict) -> dict[str, Tensor]:
    pred_logits, pred_boxes = model(batch["images"], batch["text_ids"])
    loss = model.head.compute_loss(pred_logits, pred_boxes, batch["targets"])
    return {"total": loss}


# ---------------------------------------------------------------------------
# Run launchers
# ---------------------------------------------------------------------------

def _load_encoder_weights(model_with_encoder: nn.Module, checkpoint_path: str) -> None:
    """Load Phase 2 encoder weights into a model that has a .encoder attribute."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    # State keys may be prefixed with "0." from nn.ModuleList in phase2
    encoder_state = {}
    for k, v in state.items():
        if k.startswith("0."):
            encoder_state[k[2:]] = v
        else:
            encoder_state[k] = v
    missing, unexpected = model_with_encoder.encoder.load_state_dict(
        encoder_state, strict=False
    )
    if missing:
        print(f"  Missing keys when loading encoder: {missing[:5]}")


def run_a_vqa(
    cfg: GDTConfig,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    phase2_checkpoint: Optional[str] = None,
    num_answers: int = 3129,
) -> VQAModel:
    """
    Run A: VQA-primary fine-tuning.

    @param cfg                - GDTConfig (run.primary_task should be 'vqa')
    @param train_loader       - DataLoader with {images, text_ids, answer_scores}
    @param val_loader         - optional validation DataLoader
    @param phase2_checkpoint  - path to Phase 2 checkpoint
    @param num_answers        - answer vocabulary size
    @returns fine-tuned VQAModel
    """
    cfg.training.lr = 1e-5
    model = VQAModel(cfg, num_answers)

    if phase2_checkpoint:
        _load_encoder_weights(model, phase2_checkpoint)
        print(f"Loaded Phase 2 encoder from {phase2_checkpoint}")

    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        compute_loss=_vqa_loss,
        log_dir=f"{cfg.run.checkpoint_dir}/run_a/logs",
    )
    trainer.fit()
    return model


def run_b_captioning(
    cfg: GDTConfig,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    phase2_checkpoint: Optional[str] = None,
) -> CaptioningModel:
    """
    Run B: Captioning-primary fine-tuning.

    @param cfg                - GDTConfig (run.primary_task should be 'captioning')
    @param train_loader       - DataLoader with {images, text_ids, caption_ids_in,
                               caption_ids_out}
    @param val_loader         - optional validation DataLoader
    @param phase2_checkpoint  - path to Phase 2 checkpoint
    @returns fine-tuned CaptioningModel
    """
    cfg.training.lr = 5e-5
    model = CaptioningModel(cfg)

    if phase2_checkpoint:
        _load_encoder_weights(model, phase2_checkpoint)
        print(f"Loaded Phase 2 encoder from {phase2_checkpoint}")

    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        compute_loss=_captioning_loss,
        log_dir=f"{cfg.run.checkpoint_dir}/run_b/logs",
    )
    trainer.fit()
    return model


def run_c_detection(
    cfg: GDTConfig,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    phase2_checkpoint: Optional[str] = None,
    num_classes: int = 91,
) -> DetectionModel:
    """
    Run C: Detection/segmentation-primary fine-tuning.

    @param cfg                - GDTConfig (run.primary_task should be 'detection')
    @param train_loader       - DataLoader with {images, text_ids, targets}
    @param val_loader         - optional validation DataLoader
    @param phase2_checkpoint  - path to Phase 2 checkpoint
    @param num_classes        - number of COCO detection classes
    @returns fine-tuned DetectionModel
    """
    cfg.training.lr = 1e-5
    model = DetectionModel(cfg, num_classes=num_classes)

    if phase2_checkpoint:
        _load_encoder_weights(model, phase2_checkpoint)
        print(f"Loaded Phase 2 encoder from {phase2_checkpoint}")

    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        compute_loss=_detection_loss,
        log_dir=f"{cfg.run.checkpoint_dir}/run_c/logs",
    )
    trainer.fit()
    return model
