"""
Phase 1: Multimodal Pre-training

Objectives (§3.1 of the GDT report):
  1. Image-text contrastive alignment (InfoNCE)
  2. Masked language modeling (MLM, 15% mask rate)
  3. Masked image modeling (MIM, 75% mask rate)
  4. (Optional) Region-phrase matching from Visual Genome

Optimizer: AdamW, lr=1e-4, cosine schedule, 10k warmup steps.
"""

from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from gdt.config import GDTConfig
from gdt.models.gdt_encoder import GDTEncoder
from gdt.training.losses import MultiTaskLoss
from gdt.training.trainer import Trainer


def _apply_mlm_mask(
    token_ids: Tensor, mask_ratio: float = 0.15, mask_token_id: int = 103
) -> tuple[Tensor, Tensor]:
    """
    Apply BERT-style MLM masking.
    Returns (masked_ids, bool_mask) where True = masked position.
    """
    mask = torch.rand_like(token_ids, dtype=torch.float) < mask_ratio
    masked_ids = token_ids.clone()
    masked_ids[mask] = mask_token_id
    return masked_ids, mask


def _apply_mim_mask(
    vision_tokens: Tensor, mask_ratio: float = 0.75
) -> tuple[Tensor, Tensor]:
    """
    Randomly zero out a fraction of vision tokens.
    Returns (masked_vision, bool_mask) where True = masked patch.
    """
    B, L, D = vision_tokens.shape
    mask = torch.rand(B, L, device=vision_tokens.device) < mask_ratio
    masked = vision_tokens.clone()
    masked[mask] = 0.0
    return masked, mask


def build_phase1_loss_fn(cfg: GDTConfig, patch_dim: int) -> MultiTaskLoss:
    """
    Construct the multi-task loss for Phase 1.
    @param cfg       - GDTConfig
    @param patch_dim - 3 * patch_size^2
    @returns MultiTaskLoss module
    """
    return MultiTaskLoss(
        hidden_dim=cfg.model.hidden_dim,
        vocab_size=cfg.model.vocab_size,
        patch_dim=patch_dim,
        num_edge_types=cfg.model.num_edge_types,
        w_align=cfg.training.w_align,
        w_mlm=cfg.training.w_mlm,
        w_mim=cfg.training.w_mim,
        w_graph=0.0,  # graph loss introduced in Phase 2
    )


def make_compute_loss(loss_fn: MultiTaskLoss, cfg: GDTConfig):
    """
    Return a compute_loss callable suitable for Trainer.

    Expected batch keys:
      images    [B, 3, H, W]
      text_ids  [B, Lt]
    """
    mask_ratio_mlm = cfg.training.mlm_mask_ratio
    mask_ratio_mim = cfg.training.mim_mask_ratio

    def compute_loss(model: GDTEncoder, batch: dict) -> dict[str, Tensor]:
        images = batch["images"]
        text_ids = batch["text_ids"]

        # Apply masking
        masked_text, mlm_mask = _apply_mlm_mask(text_ids, mask_ratio_mlm)

        # Encode
        encoder_out = model(images, masked_text)  # [B, T, D]

        # Split encoder output into text and vision parts
        Lt = text_ids.shape[1]
        Lv = (cfg.model.img_size // cfg.model.patch_size) ** 2

        cls_token = encoder_out[:, 0, :]
        text_hidden = encoder_out[:, 1 : 1 + Lt, :]
        vision_hidden = encoder_out[:, 1 + Lt :, :]

        # Apply MIM mask to vision hidden states (post-encoding)
        vision_tokens_clean = model.vision_embed(images)  # [B, Lv, D]
        mim_targets = vision_tokens_clean.detach()
        _, mim_mask = _apply_mim_mask(vision_tokens_clean, mask_ratio_mim)

        # Global representations for contrastive loss
        image_feats = vision_hidden.mean(dim=1)
        text_feats = text_hidden.mean(dim=1)

        # Empty graph for Phase 1 (graph loss weight is 0)
        device = images.device
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_type = torch.zeros(0, dtype=torch.long, device=device)

        return loss_fn(
            image_feats=image_feats,
            text_feats=text_feats,
            text_hidden=text_hidden,
            vision_hidden=vision_hidden,
            mlm_mask=mlm_mask,
            mlm_targets=text_ids,
            mim_mask=mim_mask,
            mim_targets=mim_targets,
            edge_index=edge_index,
            edge_type=edge_type,
            all_hidden=encoder_out,
        )

    return compute_loss


def run_phase1(
    cfg: GDTConfig,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    resume_from: Optional[str] = None,
) -> GDTEncoder:
    """
    Run Phase 1 pre-training.

    @param cfg          - GDTConfig
    @param train_loader - DataLoader yielding {images, text_ids} batches
    @param val_loader   - optional validation DataLoader
    @param resume_from  - optional path to checkpoint to resume from
    @returns trained GDTEncoder
    """
    patch_dim = 3 * cfg.model.patch_size ** 2
    model = GDTEncoder(cfg.model)
    loss_fn = build_phase1_loss_fn(cfg, patch_dim).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Combine model + loss parameters for joint optimisation
    import torch.nn as nn
    combined = nn.ModuleList([model, loss_fn])

    trainer = Trainer(
        model=combined,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        compute_loss=lambda m, b: make_compute_loss(loss_fn, cfg)(m[0], b),
        log_dir=f"{cfg.run.checkpoint_dir}/phase1/logs",
    )

    if resume_from:
        trainer.load_checkpoint(resume_from)

    trainer.fit()
    return model
