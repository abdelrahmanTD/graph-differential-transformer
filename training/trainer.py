"""
General-purpose training loop for GDT.

Features:
  - AdamW with cosine LR schedule + linear warmup
  - Gradient clipping
  - Checkpoint save/load
  - TensorBoard logging
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gdt.config import GDTConfig


def _cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Cosine LR schedule with linear warmup.
    Decays to min_lr_ratio × peak_lr at num_training_steps.
    """
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """
    Generic trainer wrapping a model + optimizer + scheduler.

    @param model          - nn.Module to train
    @param cfg            - GDTConfig
    @param train_loader   - training DataLoader
    @param val_loader     - validation DataLoader (optional)
    @param compute_loss   - callable(model, batch) → dict with 'total' key
    @param compute_metric - callable(model, val_loader) → float (optional)
    @param log_dir        - TensorBoard log directory
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: GDTConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        compute_loss: Optional[Callable] = None,
        compute_metric: Optional[Callable] = None,
        log_dir: str = "runs/gdt",
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.compute_loss = compute_loss
        self.compute_metric = compute_metric

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * cfg.run.epochs

        self.scheduler = _cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=cfg.training.warmup_steps,
            num_training_steps=total_steps,
        )

        self.global_step = 0
        self.best_metric = float("-inf")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def train_epoch(self) -> dict[str, float]:
        """Run one full training epoch. Returns average loss dict."""
        self.model.train()
        total_losses: dict[str, float] = {}
        n_batches = 0

        for batch in tqdm(self.train_loader, desc="train", leave=False):
            batch = self._to_device(batch)
            self.optimizer.zero_grad()

            loss_dict = self.compute_loss(self.model, batch)
            loss = loss_dict["total"]
            loss.backward()

            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.training.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()

            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0.0) + v.item()

            self.writer.add_scalar("train/loss", loss.item(), self.global_step)
            self.writer.add_scalar(
                "train/lr",
                self.scheduler.get_last_lr()[0],
                self.global_step,
            )
            self.global_step += 1
            n_batches += 1

        return {k: v / n_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation and return the primary metric."""
        if self.val_loader is None or self.compute_metric is None:
            return 0.0
        self.model.eval()
        metric = self.compute_metric(self.model, self.val_loader)
        self.writer.add_scalar("val/metric", metric, self.global_step)
        return metric

    def fit(self, epochs: Optional[int] = None) -> None:
        """
        Full training loop.
        @param epochs - overrides cfg.run.epochs if provided
        """
        num_epochs = epochs or self.cfg.run.epochs
        ckpt_dir = self.cfg.run.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        for epoch in range(num_epochs):
            loss_dict = self.train_epoch()
            metric = self.evaluate()

            print(
                f"Epoch {epoch+1}/{num_epochs}  "
                + "  ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
                + f"  val_metric={metric:.4f}"
            )

            if metric >= self.best_metric:
                self.best_metric = metric
                self.save_checkpoint(
                    os.path.join(ckpt_dir, "best.pt"), epoch, metric
                )

            self.save_checkpoint(
                os.path.join(ckpt_dir, f"epoch_{epoch+1:03d}.pt"), epoch, metric
            )

        self.writer.close()

    def save_checkpoint(self, path: str, epoch: int, metric: float) -> None:
        """
        @param path   - file path to write
        @param epoch  - current epoch number
        @param metric - current validation metric
        """
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "metric": metric,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> dict[str, Any]:
        """
        Load a checkpoint and restore model + optimizer + scheduler.
        @param path - path to checkpoint file
        @returns checkpoint dict
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.best_metric = ckpt.get("metric", float("-inf"))
        return ckpt

    def _to_device(self, batch: Any) -> Any:
        """Recursively move a nested dict/list/tensor to self.device."""
        if isinstance(batch, Tensor):
            return batch.to(self.device)
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            moved = [self._to_device(x) for x in batch]
            return type(batch)(moved)
        return batch
