from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    ffn_dim: int = 3072          # 4 × hidden_dim
    patch_size: int = 16
    img_size: int = 224
    max_seq_len: int = 512
    vocab_size: int = 30522      # WordPiece / BERT default
    num_edge_types: int = 8
    alpha: float = 0.5           # differential attention weighting
    dropout: float = 0.1
    num_object_queries: int = 100  # DETR-style detection


@dataclass
class TrainingConfig:
    lr: float = 1e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    batch_size: int = 256
    warmup_steps: int = 10_000
    max_grad_norm: float = 1.0
    # multi-task loss weights
    w_align: float = 1.0
    w_mlm: float = 0.5
    w_mim: float = 0.5
    w_graph: float = 0.3
    # infonce temperature
    temperature: float = 0.07
    # masking ratios
    mlm_mask_ratio: float = 0.15
    mim_mask_ratio: float = 0.75


@dataclass
class RunConfig:
    primary_task: str = "vqa"        # "vqa" | "captioning" | "detection"
    epochs: int = 15
    lr: float = 1e-5
    checkpoint_dir: str = "checkpoints"
    data_root: str = "data"
    # dataset paths (relative to data_root)
    vqa_ann_path: str = "vqa/v2_mscoco_train2014_annotations.json"
    vqa_q_path: str = "vqa/v2_OpenEnded_mscoco_train2014_questions.json"
    coco_ann_path: str = "coco/annotations"
    vg_ann_path: str = "visual_genome"
    num_vqa_answers: int = 3129
    num_coco_classes: int = 91
    beam_size: int = 5


@dataclass
class GDTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    run: RunConfig = field(default_factory=RunConfig)


def load_config(yaml_path: str, overrides: Optional[dict] = None) -> GDTConfig:
    """
    Load a GDTConfig from a YAML file, merging optional overrides.
    @param yaml_path - path to the YAML config file
    @param overrides - dict of flat key=value overrides
    @returns GDTConfig populated from the file
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    cfg = GDTConfig()

    if "model" in raw:
        for k, v in raw["model"].items():
            if hasattr(cfg.model, k):
                setattr(cfg.model, k, v)

    if "training" in raw:
        for k, v in raw["training"].items():
            if hasattr(cfg.training, k):
                setattr(cfg.training, k, v)

    if "run" in raw:
        for k, v in raw["run"].items():
            if hasattr(cfg.run, k):
                setattr(cfg.run, k, v)

    if overrides:
        for key, val in overrides.items():
            parts = key.split(".")
            if len(parts) == 2:
                section, attr = parts
                obj = getattr(cfg, section, None)
                if obj is not None and hasattr(obj, attr):
                    setattr(obj, attr, val)

    return cfg


def save_config(cfg: GDTConfig, path: str) -> None:
    """Serialize a GDTConfig to YAML."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "model": cfg.model.__dict__,
        "training": cfg.training.__dict__,
        "run": cfg.run.__dict__,
    }
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
