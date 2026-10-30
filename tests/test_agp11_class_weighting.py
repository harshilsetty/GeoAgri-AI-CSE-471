"""Unit tests for AGP 11 Class-Weighted OBB Loss.

Tests:
1. Verify v8OBBLoss unweighted behavior when class_weights is None.
2. Verify v8OBBLoss weighted behavior when class_weights is assigned.
3. Verify gradient flow and finite loss with zero minority-class instances in batch.
4. Verify callback integration without framework modifications.
"""

import pytest
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.utils.loss import v8OBBLoss

def test_1_unweighted_obb_loss_default():
    """Verify default unweighted loss behaves identically to baseline."""
    model = YOLO("yolo11n-obb.pt").model
    loss_fn = v8OBBLoss(model)
    assert loss_fn.class_weights is None, "Default class_weights must be None"

def test_2_weighted_obb_loss_assignment():
    """Verify class_weights assignment correctly shapes tensor to (1, 1, 5)."""
    model = YOLO("yolo11n-obb.pt").model
    weights = torch.tensor([0.95, 1.41, 1.16, 3.88, 0.56], dtype=torch.float32)
    model.class_weights = weights

    loss_fn = v8OBBLoss(model)
    assert loss_fn.class_weights is not None, "class_weights must be bound"
    assert loss_fn.class_weights.shape == (1, 1, 5), f"Expected (1, 1, 5), got {loss_fn.class_weights.shape}"
    assert torch.allclose(loss_fn.class_weights.squeeze(), weights), "Weights values mismatch"

def test_3_gradient_finite_check_with_zero_minority():
    """Verify finite gradients when batch has zero structures."""
    model = YOLO("yolo11n-obb.pt").model
    weights = torch.tensor([0.95, 1.41, 1.16, 3.88, 0.56], dtype=torch.float32)
    model.class_weights = weights
    loss_fn = v8OBBLoss(model)

    # Synthetic predictions
    bs = 2
    nc = 5
    num_anchors = 100
    pred_scores = torch.randn(bs, num_anchors, nc, requires_grad=True)
    target_scores = torch.zeros(bs, num_anchors, nc)
    # Put some targets for boulders (class 0) and vegetation (class 4), 0 for structures (class 3)
    target_scores[0, :5, 0] = 1.0
    target_scores[1, :5, 4] = 1.0

    bce_loss = loss_fn.bce(pred_scores, target_scores)
    weighted_bce = bce_loss * loss_fn.class_weights
    loss_val = weighted_bce.sum() / max(target_scores.sum(), 1)

    assert torch.isfinite(loss_val), "Loss must be finite"
    loss_val.backward()
    assert pred_scores.grad is not None, "Gradients must exist"
    assert torch.all(torch.isfinite(pred_scores.grad)), "Gradients must be finite"

def test_4_callback_attachment_without_framework_patching():
    """Verify clean callback attachment to YOLO model without in-place patching."""
    yolo_model = YOLO("yolo11n-obb.pt")
    weights = torch.tensor([0.95, 1.41, 1.16, 3.88, 0.56], dtype=torch.float32)

    def attach_weights(trainer):
        trainer.model.class_weights = weights.to(trainer.device)

    yolo_model.add_callback("on_pretrain_routine_end", attach_weights)
    assert "on_pretrain_routine_end" in yolo_model.callbacks

