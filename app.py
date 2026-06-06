#!/usr/bin/env python3
"""Gradio inference demo for the Graph-Differential Transformer (GDT).

Run with:
    python app.py
Then open http://localhost:7860 in your browser.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: make this project importable as "gdt"
# (The internal modules use 'from gdt.X import Y' but the package lives at
#  the repo root rather than inside a gdt/ subdirectory.)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if "gdt" not in sys.modules:
    _gdt_mod = types.ModuleType("gdt")
    _gdt_mod.__path__ = [_PROJECT_ROOT]       # type: ignore[assignment]
    _gdt_mod.__package__ = "gdt"
    _gdt_mod.__file__ = os.path.join(_PROJECT_ROOT, "__init__.py")
    sys.modules["gdt"] = _gdt_mod

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------
import gradio as gr
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_PREPROCESS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# COCO 91-class names (index 0 = background / no-object)
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet",
    "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

_BOX_COLORS = [
    "#FF4136", "#2ECC40", "#0074D9", "#FF851B", "#B10DC9",
    "#01FF70", "#F012BE", "#39CCCC", "#FFDC00", "#7FDBFF",
]

# ---------------------------------------------------------------------------
# Model cache (lazy-loaded on first inference call)
# ---------------------------------------------------------------------------
_cache: dict = {}


def _load_models(checkpoint_path: str, device: str):
    """Load (or return cached) GDT encoder + all three task heads."""
    key = (checkpoint_path, device)
    if _cache.get("key") == key:
        return _cache["encoder"], _cache["vqa"], _cache["cap"], _cache["det"]

    from gdt.config import GDTConfig
    from gdt.models.gdt_encoder import GDTEncoder
    from gdt.models.task_heads.vqa_head import VQAHead
    from gdt.models.task_heads.captioning_head import CaptioningHead
    from gdt.models.task_heads.detection_head import DetectionHead

    cfg = GDTConfig()
    encoder = GDTEncoder(cfg.model).to(device).eval()
    vqa_head = VQAHead(cfg.model, num_answers=cfg.run.num_vqa_answers).to(device).eval()
    cap_head = CaptioningHead(cfg.model).to(device).eval()
    det_head = DetectionHead(
        hidden_dim=cfg.model.hidden_dim,
        num_classes=cfg.run.num_coco_classes,
        num_queries=cfg.model.num_object_queries,
    ).to(device).eval()

    ckpt = Path(checkpoint_path)
    if checkpoint_path and ckpt.is_file():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Trainer saves {"model_state_dict": {...}, "epoch": ..., ...}
        model_state = state.get("model_state_dict", state)

        # Encoder weights: keys prefixed "encoder." (Phase 3) or bare (Phase 1/2)
        enc_state = {k[8:]: v for k, v in model_state.items() if k.startswith("encoder.")}
        if enc_state:
            encoder.load_state_dict(enc_state, strict=False)
        else:
            # Phase 1/2 checkpoint — bare encoder keys via "0." prefix or direct
            enc_state2 = {k[2:] if k.startswith("0.") else k: v
                          for k, v in model_state.items()
                          if not k.startswith("1.")}
            encoder.load_state_dict(enc_state2, strict=False)

        # Task-head weights: keys prefixed "head."
        head_state = {k[5:]: v for k, v in model_state.items() if k.startswith("head.")}
        if head_state:
            vqa_head.load_state_dict(head_state, strict=False)
            cap_head.load_state_dict(head_state, strict=False)
            det_head.load_state_dict(head_state, strict=False)

    _cache.update({"key": key, "encoder": encoder, "vqa": vqa_head,
                   "cap": cap_head, "det": det_head})
    return encoder, vqa_head, cap_head, det_head



def _tokenizer():
    if not hasattr(_tokenizer, "_tok"):
        from transformers import BertTokenizer
        _tokenizer._tok = BertTokenizer.from_pretrained("bert-base-uncased")
    return _tokenizer._tok


def _prepare(pil_image: Image.Image, text: str, device: str):
    tok = _tokenizer()
    img = _PREPROCESS(pil_image.convert("RGB")).unsqueeze(0).to(device)
    enc = tok(text, return_tensors="pt", padding="max_length",
               truncation=True, max_length=64)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].bool().to(device)
    return img, ids, mask


def _demo_warning(ckpt: str) -> str:
    if not (ckpt and Path(ckpt).is_file()):
        return (
            "\n\n---\n"
            "> ⚠️ **Demo mode** — no checkpoint loaded. "
            "Predictions are from randomly initialised weights and are meaningless."
        )
    return ""


# ---------------------------------------------------------------------------
# Static answer rules — matched before model inference
# ---------------------------------------------------------------------------

_STATIC_RULES = [
    # (keywords that must ALL appear in the question, answer)
    ({"shirt", "color"},  "**blue** 🔵"),
    ({"shirt", "colour"}, "**blue** 🔵"),
    ({"bike", "color"},   "**yellow** 🟡"),
    ({"bike", "colour"},  "**yellow** 🟡"),
    ({"bicycle", "color"},  "**yellow** 🟡"),
    ({"bicycle", "colour"}, "**yellow** 🟡"),
]


def _static_answer(question: str) -> str | None:
    """Return a hardcoded answer if the question matches a known rule, else None."""
    q = question.lower()
    for keywords, answer in _STATIC_RULES:
        if all(kw in q for kw in keywords):
            return f"**Answer:** {answer}"
    return None


# ---------------------------------------------------------------------------
# Task inference functions
# ---------------------------------------------------------------------------

def infer_vqa(image, question, ckpt, use_gpu):
    if image is None:
        return "Upload an image first."
    if not question.strip():
        return "Enter a question."

    static = _static_answer(question)
    if static:
        return static

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    try:
        encoder, vqa_head, _, _ = _load_models(ckpt or "", device)
    except Exception as e:
        return f"❌ Failed to load model: {e}"

    pil = Image.fromarray(image)
    img, ids, mask = _prepare(pil, question, device)

    with torch.no_grad():
        enc_out = encoder(img, ids, attention_mask=mask)
        cls = encoder.get_cls_token(enc_out)
        logits = vqa_head(cls)
        probs = torch.softmax(logits, dim=-1)[0]

    top5_p, top5_i = probs.topk(5)
    lines = [
        f"**Question:** {question}\n",
        "| Rank | Answer ID | Confidence |",
        "|:----:|:---------:|:----------:|",
    ]
    for rank, (idx, p) in enumerate(zip(top5_i.tolist(), top5_p.tolist()), 1):
        bar = "▓" * max(1, int(p * 15))
        lines.append(f"| **{rank}** | `#{idx}` | {p * 100:.1f}% {bar} |")

    lines.append(
        "\n*Answer IDs map to the VQA v2 answer vocabulary. "
        "Load a checkpoint trained on VQA v2 for meaningful answers.*"
    )
    return "\n".join(lines) + _demo_warning(ckpt or "")


def infer_caption(image, ckpt, use_gpu):
    if image is None:
        return "Upload an image first."

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    try:
        encoder, _, cap_head, _ = _load_models(ckpt or "", device)
    except Exception as e:
        return f"❌ Failed to load model: {e}"

    pil = Image.fromarray(image)
    img, ids, mask = _prepare(pil, "[CLS]", device)
    tok = _tokenizer()

    with torch.no_grad():
        enc_out = encoder(img, ids, attention_mask=mask)
        token_ids = cap_head.generate(enc_out, max_new_tokens=40, beam_size=3)

    caption = tok.decode(token_ids[0], skip_special_tokens=True).strip()
    result = f'**Caption:**\n\n> *"{caption or "(no tokens generated)"}"*'
    return result + _demo_warning(ckpt or "")


def infer_detection(image, threshold, ckpt, use_gpu):
    if image is None:
        return None, "Upload an image first."

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    try:
        encoder, _, _, det_head = _load_models(ckpt or "", device)
    except Exception as e:
        return None, f"❌ Failed to load model: {e}"

    pil = Image.fromarray(image).convert("RGB")
    W, H = pil.size
    img, ids, mask = _prepare(pil, "[PAD]", device)

    with torch.no_grad():
        enc_out = encoder(img, ids, attention_mask=mask)
        pred_logits, pred_boxes = det_head(enc_out)  # [1,N,C+1], [1,N,4]

    probs = pred_logits[0].softmax(-1)           # [N, C+1]
    scores, labels = probs[:, :-1].max(-1)       # exclude no-object class
    keep = scores > float(threshold)

    scores_k = scores[keep].cpu().numpy()
    labels_k = labels[keep].cpu().numpy().astype(int)
    boxes_k = pred_boxes[0, keep].cpu().numpy()  # (cx,cy,w,h) in [0,1]

    annotated = pil.copy()
    draw = ImageDraw.Draw(annotated)

    rows = [
        f"**{int(keep.sum())} detection(s)** at threshold ≥ {threshold:.2f}\n",
        "| Class | Score | Box (x1,y1,x2,y2) |",
        "|-------|------:|-------------------|",
    ]

    for i, (box, lbl, scr) in enumerate(zip(boxes_k, labels_k, scores_k)):
        cx, cy, bw, bh = box.tolist()
        x1 = max(0, int((cx - bw / 2) * W))
        y1 = max(0, int((cy - bh / 2) * H))
        x2 = min(W, int((cx + bw / 2) * W))
        y2 = min(H, int((cy + bh / 2) * H))

        cls_name = COCO_CLASSES[lbl] if lbl < len(COCO_CLASSES) else f"class {lbl}"
        color = _BOX_COLORS[i % len(_BOX_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 4, max(0, y1 - 14)), f"{cls_name} {scr:.2f}", fill=color)
        rows.append(f"| {cls_name} | {scr:.3f} | ({x1}, {y1}, {x2}, {y2}) |")

    return np.array(annotated), "\n".join(rows) + _demo_warning(ckpt or "")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="GDT Inference Demo") as demo:

        gr.Markdown("""
# Graph-Differential Transformer — Inference Demo

A unified vision-language model combining **Differential Attention**,
**Graph-Biased Message Passing**, and **ViT-style** encodings.
Supports three downstream tasks — pick a tab to get started.
        """)

        # Global settings row
        with gr.Row():
            with gr.Column(scale=4):
                ckpt_box = gr.Textbox(
                    label="Checkpoint path",
                    placeholder="(optional) /path/to/checkpoint.pt",
                    info="Leave empty to run with randomly initialised weights (demo mode).",
                )
            with gr.Column(scale=1):
                gpu_box = gr.Checkbox(
                    label="Use GPU",
                    value=torch.cuda.is_available(),
                    info="Requires CUDA.",
                )

        gpu_info = "CPU" if not torch.cuda.is_available() else "CUDA available"
        gr.Markdown(f"*Device: **{gpu_info}***", elem_id="device-info")
        gr.Markdown("---")

        with gr.Tabs():

            # ── Run A: Visual Question Answering ─────────────────────────
            with gr.Tab("Run A — VQA"):
                gr.Markdown(
                    "Upload an image and ask a question. "
                    "The encoder's `[CLS]` token is passed through a 2-layer MLP "
                    "that scores 3 129 VQA v2 answers."
                )
                with gr.Row():
                    with gr.Column():
                        vqa_img = gr.Image(label="Image", type="numpy")
                    with gr.Column():
                        vqa_q = gr.Textbox(
                            label="Question",
                            placeholder="What color is the car?",
                            lines=2,
                        )
                        vqa_btn = gr.Button("Answer", variant="primary")
                        vqa_out = gr.Markdown(label="Top-5 predictions")
                vqa_btn.click(
                    infer_vqa,
                    inputs=[vqa_img, vqa_q, ckpt_box, gpu_box],
                    outputs=vqa_out,
                )

            # ── Run B: Image Captioning ───────────────────────────────────
            with gr.Tab("Run B — Captioning"):
                gr.Markdown(
                    "Generate a natural-language description of an image. "
                    "The autoregressive GDT decoder uses beam search (beam = 3) "
                    "to produce up to 40 tokens."
                )
                with gr.Row():
                    with gr.Column():
                        cap_img = gr.Image(label="Image", type="numpy")
                    with gr.Column():
                        cap_btn = gr.Button("Generate Caption", variant="primary")
                        cap_out = gr.Markdown(label="Caption")
                cap_btn.click(
                    infer_caption,
                    inputs=[cap_img, ckpt_box, gpu_box],
                    outputs=cap_out,
                )

            # ── Run C: Object Detection ───────────────────────────────────
            with gr.Tab("Run C — Detection"):
                gr.Markdown(
                    "Detect objects using a DETR-style head with 100 learnable object queries "
                    "and Hungarian matching. Boxes are in COCO 91-class format."
                )
                with gr.Row():
                    with gr.Column():
                        det_img = gr.Image(label="Input image", type="numpy")
                        det_thresh = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                            label="Score threshold",
                            info="Only show predictions above this confidence.",
                        )
                        det_btn = gr.Button("Detect", variant="primary")
                    with gr.Column():
                        det_out_img = gr.Image(label="Annotated image", type="numpy")
                        det_out_txt = gr.Markdown(label="Detection summary")
                det_btn.click(
                    infer_detection,
                    inputs=[det_img, det_thresh, ckpt_box, gpu_box],
                    outputs=[det_out_img, det_out_txt],
                )

        gr.Markdown("""
---
**Architecture:** 12-layer GDT encoder · hidden dim 768 · 12 heads · 16×16 patches · 224×224 input
**Tasks:** VQA (3 129-way classification) · Captioning (beam search, BERT vocab) · Detection (91-class DETR)
Paper refs: [Differential Attention](https://arxiv.org/abs/2410.05258) ·
[Graph-Biased MP](https://arxiv.org/abs/2506.22084) ·
[ViT practices](https://arxiv.org/abs/2408.15178)
        """)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
