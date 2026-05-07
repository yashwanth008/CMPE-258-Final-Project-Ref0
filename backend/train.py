"""
Ref Zero — YOLO11 Training Pipeline
=====================================
Fine-tunes YOLO11 on the sports foul detection dataset.
Logs all metrics to W&B. Saves best checkpoint path to runs/best_model.txt.

Usage:
    python backend/train.py
    python backend/train.py --model-size small
    python backend/train.py --model-size medium --epochs 100

Requirements:
    pip install ultralytics wandb roboflow
    wandb login
    python data/prepare_dataset.py --api-key YOUR_ROBOFLOW_KEY
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ── Make sure we can import model_config from the same folder ──
sys.path.insert(0, str(Path(__file__).parent))
from model_config import get_training_args, get_wandb_config, MODEL_CHECKPOINTS


def parse_args():
    parser = argparse.ArgumentParser(description="Train Ref Zero YOLO11 foul detector")
    parser.add_argument(
        "--model-size",
        choices=["nano", "small", "medium"],
        default="nano",
        help="YOLO11 model size (default: nano — fastest, 47 FPS)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epoch count (default: from model_config.py)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging (useful for quick local tests)",
    )
    return parser.parse_args()


def setup_wandb(model_size: str, disabled: bool):
    """Initialise W&B run. Returns run or None."""
    if disabled:
        os.environ["WANDB_DISABLED"] = "true"
        print("⚠️  W&B logging disabled")
        return None
    try:
        import wandb
        run = wandb.init(
            project="ref-zero",
            name=f"yolo11{model_size[0]}-foul-{int(time.time())}",
            config=get_wandb_config(model_size),
            tags=["foul-detection", f"yolo11{model_size[0]}", "sports-cv"],
        )
        print(f"✅  W&B run initialised: {run.url}")
        return run
    except Exception as e:
        print(f"⚠️  W&B init failed ({e}) — continuing without logging")
        return None


def train(model_size: str, extra_args: dict = None):
    """Run YOLO11 fine-tuning. Returns path to best.pt checkpoint."""
    from ultralytics import YOLO

    checkpoint = MODEL_CHECKPOINTS[model_size]
    train_args, _, run_name = get_training_args(model_size, extra_args)

    print("\n" + "=" * 56)
    print(f"  REF ZERO — TRAINING YOLO11 ({model_size.upper()})")
    print("=" * 56)
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Dataset    : {train_args['data']}")
    print(f"  Epochs     : {train_args['epochs']}")
    print(f"  Batch size : {train_args['batch']}")
    print(f"  Input size : {train_args['imgsz']}px")
    print(f"  Run name   : {run_name}")
    print("=" * 56 + "\n")

    # Verify dataset exists
    data_yaml = Path(train_args["data"])
    if not data_yaml.exists():
        print(f"❌  Dataset not found at: {data_yaml}")
        print("   Run: python data/prepare_dataset.py --api-key YOUR_KEY")
        sys.exit(1)

    # Load pre-trained YOLO11 backbone
    print(f"📦  Loading {checkpoint}...")
    model = YOLO(checkpoint)

    # Fine-tune on foul detection dataset
    print("🚀  Starting training...\n")
    start = time.time()
    results = model.train(**train_args)
    elapsed = time.time() - start

    # ── Find best checkpoint ──────────────────────────────────
    best_pt = Path(train_args["project"]) / run_name / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"⚠️  best.pt not found at {best_pt} — checking last.pt")
        best_pt = Path(train_args["project"]) / run_name / "weights" / "last.pt"

    # Save checkpoint path so main.py can hot-swap it
    Path("runs").mkdir(exist_ok=True)
    Path("runs/best_model.txt").write_text(str(best_pt))
    print(f"\n✅  Best checkpoint saved to: {best_pt}")
    print(f"    Path written to: runs/best_model.txt")

    # ── Validation on test set ────────────────────────────────
    print("\n📊  Running final validation on test set...")
    val_results = model.val(data=train_args["data"], split="test")

    map50    = getattr(val_results.box, "map50",    "N/A")
    map5095  = getattr(val_results.box, "map",      "N/A")
    precision= getattr(val_results.box, "mp",       "N/A")
    recall   = getattr(val_results.box, "mr",       "N/A")

    # ── Log final metrics to W&B ──────────────────────────────
    try:
        import wandb
        if wandb.run:
            wandb.log({
                "test/mAP50":     map50,
                "test/mAP50-95":  map5095,
                "test/precision": precision,
                "test/recall":    recall,
                "training_time_min": elapsed / 60,
            })
            wandb.save(str(best_pt))
            print(f"✅  Final metrics logged to W&B")
    except Exception:
        pass

    # ── Print final summary ───────────────────────────────────
    print("\n" + "=" * 56)
    print("  TRAINING COMPLETE")
    print("=" * 56)
    print(f"  Model size  : YOLO11{model_size[0].upper()}")
    print(f"  mAP@0.5     : {map50:.4f}" if isinstance(map50, float) else f"  mAP@0.5     : {map50}")
    print(f"  mAP@0.5-0.95: {map5095:.4f}" if isinstance(map5095, float) else f"  mAP@0.5-0.95: {map5095}")
    print(f"  Precision   : {precision:.4f}" if isinstance(precision, float) else f"  Precision   : {precision}")
    print(f"  Recall      : {recall:.4f}" if isinstance(recall, float) else f"  Recall      : {recall}")
    print(f"  Time        : {elapsed/60:.1f} min")
    print(f"  Checkpoint  : {best_pt}")
    print("=" * 56 + "\n")

    return str(best_pt)


def copy_metrics_artifacts(run_name: str, project: str = "runs/detect"):
    """Copy auto-generated charts to assets/metrics/ for the frontend."""
    src = Path(project) / run_name
    dst = Path("assets/metrics")
    dst.mkdir(parents=True, exist_ok=True)

    charts = [
        "results.png",
        "confusion_matrix.png",
        "PR_curve.png",
        "F1_curve.png",
        "val_batch0_pred.jpg",
    ]
    copied = 0
    for chart in charts:
        src_file = src / chart
        if src_file.exists():
            import shutil
            shutil.copy(src_file, dst / chart)
            copied += 1

    print(f"📈  Copied {copied}/{len(charts)} metric charts → assets/metrics/")


def main():
    args = parse_args()

    # Override epochs if flag passed
    extra = {}
    if args.epochs:
        extra["epochs"] = args.epochs

    # Setup W&B
    run = setup_wandb(args.model_size, args.no_wandb)

    try:
        # Run training
        best_pt = train(args.model_size, extra or None)

        # Copy metric charts to frontend-accessible location
        _, _, run_name = get_training_args(args.model_size)
        copy_metrics_artifacts(run_name)

        print("🎉  Done! Next steps:")
        print("   1. Restart backend — it will auto-load the new checkpoint")
        print("   2. Run: python backend/ablation.py   (for ablation study)")
        print("   3. Check W&B dashboard for training curves")

    finally:
        try:
            import wandb
            if wandb.run:
                wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
