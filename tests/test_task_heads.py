"""
Unit tests for VQAHead, CaptioningHead, and DetectionHead.
"""

import torch
import pytest

from gdt.config import ModelConfig
from gdt.models.task_heads.vqa_head import VQAHead
from gdt.models.task_heads.captioning_head import CaptioningHead
from gdt.models.task_heads.detection_head import DetectionHead, MLP


@pytest.fixture
def small_cfg():
    return ModelConfig(
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        ffn_dim=128,
        patch_size=8,
        img_size=32,
        max_seq_len=32,
        vocab_size=256,
        dropout=0.0,
        num_object_queries=10,
    )


class TestVQAHead:
    def test_output_shape(self, small_cfg):
        head = VQAHead(small_cfg, num_answers=100)
        cls = torch.randn(4, small_cfg.hidden_dim)
        logits = head(cls)
        assert logits.shape == (4, 100)

    def test_loss_scalar(self, small_cfg):
        head = VQAHead(small_cfg, num_answers=50)
        cls = torch.randn(4, small_cfg.hidden_dim)
        logits = head(cls)
        targets = torch.rand(4, 50)
        loss = head.compute_loss(logits, targets)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_soft_label_loss_nonnegative(self, small_cfg):
        head = VQAHead(small_cfg, num_answers=20)
        cls = torch.randn(4, small_cfg.hidden_dim)
        logits = head(cls)
        targets = torch.rand(4, 20)
        loss = head.compute_loss(logits, targets)
        assert loss.item() >= 0.0

    def test_predict_shape(self, small_cfg):
        head = VQAHead(small_cfg, num_answers=30)
        cls = torch.randn(4, small_cfg.hidden_dim)
        preds = head.predict(cls)
        assert preds.shape == (4,)
        assert preds.max().item() < 30

    def test_gradient_flows(self, small_cfg):
        head = VQAHead(small_cfg, num_answers=20)
        cls = torch.randn(4, small_cfg.hidden_dim, requires_grad=True)
        logits = head(cls)
        targets = torch.rand(4, 20)
        loss = head.compute_loss(logits, targets)
        loss.backward()
        assert cls.grad is not None


class TestCaptioningHead:
    def test_forward_shape(self, small_cfg):
        head = CaptioningHead(small_cfg)
        B, Tm, Lt = 2, 20, 10
        encoder_out = torch.randn(B, Tm, small_cfg.hidden_dim)
        cap_ids = torch.randint(0, small_cfg.vocab_size, (B, Lt))
        logits = head(encoder_out, cap_ids)
        assert logits.shape == (B, Lt, small_cfg.vocab_size)

    def test_loss_scalar(self, small_cfg):
        head = CaptioningHead(small_cfg)
        B, Tm, Lt = 2, 12, 8
        encoder_out = torch.randn(B, Tm, small_cfg.hidden_dim)
        cap_ids_in = torch.randint(0, small_cfg.vocab_size, (B, Lt))
        cap_ids_out = torch.randint(0, small_cfg.vocab_size, (B, Lt))
        logits = head(encoder_out, cap_ids_in)
        loss = head.compute_loss(logits, cap_ids_out)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_generate_returns_list(self, small_cfg):
        head = CaptioningHead(small_cfg, bos_id=1, eos_id=2)
        B, Tm = 2, 12
        encoder_out = torch.randn(B, Tm, small_cfg.hidden_dim)
        tokens = head.generate(encoder_out, max_new_tokens=5, beam_size=2)
        assert isinstance(tokens, list)
        assert len(tokens) == B
        for t in tokens:
            assert isinstance(t, list)


class TestMLPLayer:
    def test_output_shape(self):
        mlp = MLP(16, 32, 4, n_layers=3)
        x = torch.randn(5, 16)
        out = mlp(x)
        assert out.shape == (5, 4)


class TestDetectionHead:
    def test_forward_output_shapes(self, small_cfg):
        head = DetectionHead(
            hidden_dim=small_cfg.hidden_dim,
            num_classes=10,
            num_queries=small_cfg.num_object_queries,
            num_heads=small_cfg.num_heads,
        )
        B, Tm = 2, 15
        encoder_out = torch.randn(B, Tm, small_cfg.hidden_dim)
        logits, boxes = head(encoder_out)
        N = small_cfg.num_object_queries
        assert logits.shape == (B, N, 11)   # 10 classes + 1 no-object
        assert boxes.shape == (B, N, 4)

    def test_boxes_in_zero_one(self, small_cfg):
        """Boxes should be in [0, 1] after sigmoid."""
        head = DetectionHead(small_cfg.hidden_dim, 10, small_cfg.num_object_queries, small_cfg.num_heads)
        encoder_out = torch.randn(2, 15, small_cfg.hidden_dim)
        _, boxes = head(encoder_out)
        assert boxes.min().item() >= 0.0
        assert boxes.max().item() <= 1.0

    def test_compute_loss_scalar(self, small_cfg):
        head = DetectionHead(small_cfg.hidden_dim, 10, small_cfg.num_object_queries, small_cfg.num_heads)
        encoder_out = torch.randn(2, 15, small_cfg.hidden_dim)
        logits, boxes = head(encoder_out)
        targets = [
            {"labels": torch.tensor([1, 3]), "boxes": torch.rand(2, 4)},
            {"labels": torch.tensor([5]), "boxes": torch.rand(1, 4)},
        ]
        loss = head.compute_loss(logits, boxes, targets)
        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert loss.item() >= 0.0
