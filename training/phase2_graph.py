"""
Phase 2: Graph Structure Integration

Continues from Phase 1 checkpoint. Introduces:
  - GraphLoss (edge classification + triplet prediction)
  - Graph-biased attention (already in GDTBlock; this phase fine-tunes it)

Objectives (§3.2 of the GDT report):
  1. Continue MLM + MIM + alignment at reduced weight
  2. Edge-type classification from token embeddings
  3. Subject-relation-object triplet prediction

LR: 5e-5 (lower than Phase 1); Epochs: 20-40.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from gdt.config import GDTConfig
from gdt.models.gdt_encoder import GDTEncoder
from gdt.training.losses import MultiTaskLoss
from gdt.training.phase1_pretrain import _apply_mlm_mask, _apply_mim_mask
from gdt.training.trainer import Trainer


def make_phase2_compute_loss(loss_fn: MultiTaskLoss, cfg: GDTConfig):
    """
    Phase 2 compute_loss that uses graph annotations from the batch.

    Expected additional batch keys (optional):
      edge_index   [2, E]
      edge_type    [E]
    """
    mask_ratio_mlm = cfg.training.mlm_mask_ratio
    mask_ratio_mim = cfg.training.mim_mask_ratio

    def compute_loss(model: GDTEncoder, batch: dict) -> dict[str, Tensor]:
        images = batch["images"]
        text_ids = batch["text_ids"]

        masked_text, mlm_mask = _apply_mlm_mask(text_ids, mask_ratio_mlm)
        encoder_out = model(images, masked_text)

        Lt = text_ids.shape[1]
        text_hidden = encoder_out[:, 1 : 1 + Lt, :]
        vision_hidden = encoder_out[:, 1 + Lt :, :]

        vision_tokens_clean = model.vision_embed(images)
        mim_targets = vision_tokens_clean.detach()
        _, mim_mask = _apply_mim_mask(vision_tokens_clean, mask_ratio_mim)

        image_feats = vision_hidden.mean(dim=1)
        text_feats = text_hidden.mean(dim=1)

        device = images.device
        edge_index = batch.get(
            "edge_index", torch.zeros(2, 0, dtype=torch.long, device=device)
        )
        edge_type = batch.get(
            "edge_type", torch.zeros(0, dtype=torch.long, device=device)
        )

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


def run_phase2(
    cfg: GDTConfig,
    train_loader: DataLoader,
    phase1_checkpoint: Optional[str] = None,
    val_loader: Optional[DataLoader] = None,
) -> GDTEncoder:
    """
    Run Phase 2 graph structure integration.

    @param cfg               - GDTConfig
    @param train_loader      - DataLoader with graph-annotated batches
    @param phase1_checkpoint - path to Phase 1 best.pt
    @param val_loader        - optional validation loader
    @returns GDTEncoder with graph-aware attention trained
    """
    patch_dim = 3 * cfg.model.patch_size ** 2
    model = GDTEncoder(cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = MultiTaskLoss(
        hidden_dim=cfg.model.hidden_dim,
        vocab_size=cfg.model.vocab_size,
        patch_dim=patch_dim,
        num_edge_types=cfg.model.num_edge_types,
        w_align=cfg.training.w_align * 0.5,
        w_mlm=cfg.training.w_mlm * 0.5,
        w_mim=cfg.training.w_mim * 0.5,
        w_graph=cfg.training.w_graph,
    ).to(device)

    # Override LR to Phase 2 value
    cfg.training.lr = 5e-5

    trainer = Trainer(
        model=nn.ModuleList([model, loss_fn]),
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        compute_loss=lambda m, b: make_phase2_compute_loss(loss_fn, cfg)(m[0], b),
        log_dir=f"{cfg.run.checkpoint_dir}/phase2/logs",
    )

    if phase1_checkpoint:
        ckpt = torch.load(phase1_checkpoint, map_location=device)
        model_state = ckpt.get("model_state_dict", ckpt)
        # Load only encoder weights (ignore loss_fn keys)
        filtered = {
            k.replace("0.", ""): v
            for k, v in model_state.items()
            if k.startswith("0.")
        }
        model.load_state_dict(filtered, strict=False)
        print(f"Loaded Phase 1 checkpoint from {phase1_checkpoint}")

    trainer.fit()
    return model
