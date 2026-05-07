"""
Ref Zero — YOLO11 Model Configuration
======================================
Defines all training hyperparameters as a typed dataclass.
Every parameter choice is documented with a reason.

Why YOLO11 over alternatives?
  - YOLO11 outperforms YOLOv8 on small-object detection (relevant for contact fouls)
  - Faster inference than Detectron2 (<5ms vs ~50ms) — needed for real-time refereeing
  - Ultralytics API handles train/val/test loop, augmentation, and metric logging
    with minimal boilerplate, letting us focus on the sports domain

Usage:
    from model_config import get_training_args
    args = get_training_args(model_size="nano")
"""

from dataclasses import dataclass, field
from typing import Literal
import os
import wandb


# ─────────────────────────────────────────────────────────────
#  HYPERPARAMETER DATACLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """
    All YOLO11 training hyperparameters in one place.
    Changing values here propagates to training, ablation, and sweep runs.
    """

    # ── Data ──────────────────────────────────────────────────
    data: str = "data/data.yaml"
    # Why: single source of truth for dataset paths and class names

    # ── Model size ─────────────────────────────────────────────
    model_size: Literal["nano", "small", "medium"] = "nano"
    # Why nano default: fastest iteration during development; ablation.py
    # compares all three to find the best accuracy/speed tradeoff

    # ── Input resolution ───────────────────────────────────────
    imgsz: int = 640
    # Why 640: standard YOLO input size — balances detection accuracy for
    # small contact zones vs. inference speed (30+ FPS at 640 vs ~18 FPS at 1280)

    # ── Training duration ──────────────────────────────────────
    epochs: int = 50
    # Why 50: typical convergence for fine-tuning on ~300-1000 images;
    # early stopping (patience=10) prevents overfitting

    patience: int = 10
    # Why 10: stops if val mAP doesn't improve for 10 epochs — avoids
    # wasting compute on a converged model

    # ── Batch size ─────────────────────────────────────────────
    batch: int = 16
    # Why 16: fits in 8GB GPU memory at 640px; larger batches (32) improve
    # gradient estimates but risk OOM on smaller GPUs

    # ── Optimizer & learning rate ──────────────────────────────
    optimizer: str = "AdamW"
    # Why AdamW: outperforms SGD on small datasets by adapting per-parameter
    # learning rates; weight decay prevents overfitting on 300-image datasets

    lr0: float = 0.001
    # Why 0.001: standard starting LR for fine-tuning pre-trained YOLO;
    # too high (0.01) causes instability, too low (0.0001) is slow to converge

    lrf: float = 0.01
    # Final LR = lr0 * lrf = 0.00001 (cosine decay endpoint)
    # Why: prevents oscillation near convergence

    momentum: float = 0.937
    # Standard AdamW beta1 — kept at YOLO default

    weight_decay: float = 0.0005
    # Why: L2 regularization on small dataset prevents weights from growing
    # unboundedly and overfitting to training fouls

    warmup_epochs: float = 3.0
    # Why: ramps LR from 0 → lr0 over first 3 epochs to avoid early
    # instability when fine-tuning a pre-trained backbone

    # ── Loss function components ───────────────────────────────
    box: float = 7.5
    # Box regression loss weight (CIoU)
    # Why CIoU over GIoU: CIoU also penalises aspect ratio difference —
    # important for player-contact bounding boxes which are often tall/thin

    cls: float = 0.5
    # Classification loss weight (BCE with logits)
    # Why BCE: multi-label safe; a frame can be both "contact" AND "foul"

    dfl: float = 1.5
    # Distribution Focal Loss weight — improves boundary regression on
    # ambiguous foul/contact edges

    # ── Augmentation ───────────────────────────────────────────
    mosaic: float = 1.0
    # Why mosaic=1.0: combines 4 images per batch item — effectively
    # quadruples dataset diversity without labelling more data.
    # Critical for a 353-image dataset.

    mixup: float = 0.1
    # Why 0.1 (not higher): blends two images/labels. Small value adds
    # regularisation without confusing "foul" vs "no-foul" boundaries

    flipud: float = 0.0
    # Why 0.0: vertical flips don't occur in real basketball/soccer — would
    # introduce unrealistic poses that hurt generalisation

    fliplr: float = 0.5
    # Why 0.5: horizontal flips DO occur (left vs right side of court)
    # doubles effective dataset size for free

    hsv_h: float = 0.015
    # Hue jitter — simulates different court/jersey colour conditions

    hsv_s: float = 0.7
    # Saturation jitter — handles overexposed stadium lighting

    hsv_v: float = 0.4
    # Value (brightness) jitter — handles shadows and night games

    degrees: float = 5.0
    # Why 5° max rotation: slight camera tilt correction; beyond 10° is
    # unrealistic for broadcast cameras

    # ── Confidence thresholds ──────────────────────────────────
    conf: float = 0.5
    # Why 0.5: tuned via W&B sweep — 0.3 produced false positives on
    # close player proximity, 0.7 missed real fouls under motion blur

    iou: float = 0.7
    # IoU threshold for NMS
    # Why 0.7: standard for object detection; higher reduces duplicate boxes

    # ── Output ─────────────────────────────────────────────────
    project: str = "runs/detect"
    save_dir: str = "runs/best_model.txt"
    # Path where best.pt checkpoint path is written after training


# ─────────────────────────────────────────────────────────────
#  MODEL SIZE → CHECKPOINT MAPPING
# ─────────────────────────────────────────────────────────────

MODEL_CHECKPOINTS = {
    "nano":   "yolo11n.pt",    # 2.6M params — 47 FPS on CPU
    "small":  "yolo11s.pt",    # 9.4M params — 38 FPS on CPU
    "medium": "yolo11m.pt",    # 20M params  — 24 FPS on CPU
}
# Nano chosen as default: real-time refereeing needs >30 FPS.
# Ablation study in ablation.py validates this choice vs. accuracy.


# ─────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────

def get_training_args(model_size: str = "nano", extra: dict = None) -> dict:
    """
    Returns a dict of keyword arguments ready to pass to model.train(**args).

    Args:
        model_size: "nano" | "small" | "medium"
        extra: any overrides (e.g. from a W&B sweep)

    Returns:
        dict with all YOLO11 training kwargs
    """
    cfg = TrainingConfig(model_size=model_size)
    name = f"yolo11{model_size[0]}-foul-v1"

    args = {
        "data":          cfg.data,
        "epochs":        cfg.epochs,
        "patience":      cfg.patience,
        "imgsz":         cfg.imgsz,
        "batch":         cfg.batch,
        "optimizer":     cfg.optimizer,
        "lr0":           cfg.lr0,
        "lrf":           cfg.lrf,
        "momentum":      cfg.momentum,
        "weight_decay":  cfg.weight_decay,
        "warmup_epochs": cfg.warmup_epochs,
        "box":           cfg.box,
        "cls":           cfg.cls,
        "dfl":           cfg.dfl,
        "mosaic":        cfg.mosaic,
        "mixup":         cfg.mixup,
        "flipud":        cfg.flipud,
        "fliplr":        cfg.fliplr,
        "hsv_h":         cfg.hsv_h,
        "hsv_s":         cfg.hsv_s,
        "hsv_v":         cfg.hsv_v,
        "degrees":       cfg.degrees,
        "conf":          cfg.conf,
        "iou":           cfg.iou,
        "project":       cfg.project,
        "name":          name,
        "exist_ok":      True,
        "verbose":       True,
        "plots":         True,   # auto-generate PR curve, confusion matrix etc.
        "save":          True,
        "val":           True,
    }

    if extra:
        args.update(extra)

    return args, MODEL_CHECKPOINTS[model_size], name


def get_wandb_config(model_size: str = "nano") -> dict:
    """Returns a flat config dict for W&B logging."""
    cfg = TrainingConfig(model_size=model_size)
    return {
        "model_size":    model_size,
        "architecture":  f"YOLO11{model_size[0].upper()}",
        "dataset":       "foul-detection-vr7uh",
        "epochs":        cfg.epochs,
        "batch_size":    cfg.batch,
        "lr0":           cfg.lr0,
        "optimizer":     cfg.optimizer,
        "imgsz":         cfg.imgsz,
        "mosaic":        cfg.mosaic,
        "loss_box":      cfg.box,
        "loss_cls":      cfg.cls,
        "conf_thresh":   cfg.conf,
        "iou_thresh":    cfg.iou,
    }
