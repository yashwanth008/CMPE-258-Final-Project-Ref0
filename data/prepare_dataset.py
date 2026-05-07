"""
Ref Zero — Dataset preparation script
Downloads the sports foul detection dataset from Roboflow,
verifies class balance, and confirms the 70/20/10 split.

Usage:
    pip install roboflow
    python data/prepare_dataset.py --api-key YOUR_ROBOFLOW_KEY
"""

import os
import argparse
from pathlib import Path

def download_dataset(api_key: str, output_dir: str = "./data/foul-detection") -> str:
    """Download dataset from Roboflow in YOLOv11 format."""
    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError("Run: pip install roboflow")

    print("📥  Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("v-for-foul-detection-in-basketball").project("foul-detection-vr7uh")
    dataset = project.version(1).download("yolov11", location=output_dir)
    print(f"✅  Dataset downloaded to: {dataset.location}")
    return dataset.location


def verify_structure(dataset_dir: str) -> None:
    """Confirm train/valid/test folders and data.yaml exist."""
    path = Path(dataset_dir)
    required = ["train/images", "valid/images", "test/images", "data.yaml"]
    print("\n📂  Verifying folder structure...")
    all_ok = True
    for item in required:
        exists = (path / item).exists()
        status = "✅" if exists else "❌"
        print(f"  {status}  {item}")
        if not exists:
            all_ok = False
    if not all_ok:
        raise RuntimeError("Dataset structure is incomplete. Re-download.")
    print("✅  Structure OK\n")


def count_images(dataset_dir: str) -> dict:
    """Count images in each split."""
    path = Path(dataset_dir)
    splits = {"train": 0, "valid": 0, "test": 0}
    for split in splits:
        img_dir = path / split / "images"
        if img_dir.exists():
            splits[split] = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    return splits


def count_classes(dataset_dir: str) -> dict:
    """Count label occurrences per class across training set."""
    path = Path(dataset_dir)
    label_dir = path / "train" / "labels"
    class_counts: dict[int, int] = {}
    if not label_dir.exists():
        return {}
    for label_file in label_dir.glob("*.txt"):
        for line in label_file.read_text().strip().splitlines():
            if line:
                cls = int(line.split()[0])
                class_counts[cls] = class_counts.get(cls, 0) + 1
    return class_counts


def print_summary(dataset_dir: str) -> None:
    """Print a full dataset summary."""
    splits = count_images(dataset_dir)
    total = sum(splits.values())

    print("=" * 48)
    print("  REF ZERO — DATASET SUMMARY")
    print("=" * 48)

    print(f"\n{'Split':<10} {'Images':>8} {'%':>6}")
    print("-" * 28)
    for split, count in splits.items():
        pct = (count / total * 100) if total else 0
        flag = ""
        if split == "train" and pct < 60:
            flag = " ⚠️  (low)"
        print(f"{split:<10} {count:>8} {pct:>5.1f}%{flag}")
    print(f"{'TOTAL':<10} {total:>8}")

    # Target: 70/20/10
    train_pct = splits["train"] / total * 100 if total else 0
    valid_pct = splits["valid"] / total * 100 if total else 0
    test_pct  = splits["test"]  / total * 100 if total else 0
    print(f"\n  Target split: 70 / 20 / 10")
    print(f"  Actual split: {train_pct:.0f} / {valid_pct:.0f} / {test_pct:.0f}")

    # Class balance
    classes = count_classes(dataset_dir)
    if classes:
        # Try to read class names from data.yaml
        yaml_path = Path(dataset_dir) / "data.yaml"
        names = {}
        if yaml_path.exists():
            for line in yaml_path.read_text().splitlines():
                if "names:" in line:
                    import yaml
                    data = yaml.safe_load(yaml_path.read_text())
                    names = {i: n for i, n in enumerate(data.get("names", []))}
                    break

        print(f"\n{'Class':<20} {'Labels':>8}")
        print("-" * 30)
        cls_total = sum(classes.values())
        for cls_id, count in sorted(classes.items()):
            name = names.get(cls_id, f"class_{cls_id}")
            pct = count / cls_total * 100
            balance = " ⚠️  (imbalanced)" if pct < 20 or pct > 80 else ""
            print(f"  {name:<18} {count:>8}  ({pct:.1f}%){balance}")

    print("\n" + "=" * 48)
    if total < 300:
        print("⚠️  WARNING: Under 300 training images.")
        print("   Apply 3x augmentation in Roboflow UI or set augment=True in train.py")
    else:
        print("✅  Dataset ready for training.")
    print("=" * 48 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare Ref Zero training dataset")
    parser.add_argument("--api-key", required=True, help="Your Roboflow API key")
    parser.add_argument("--output-dir", default="./data/foul-detection",
                        help="Where to save the dataset (default: ./data/foul-detection)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download if dataset already exists")
    args = parser.parse_args()

    if args.skip_download and Path(args.output_dir).exists():
        print(f"⏭️   Skipping download — using existing: {args.output_dir}")
        dataset_dir = args.output_dir
    else:
        dataset_dir = download_dataset(args.api_key, args.output_dir)

    verify_structure(dataset_dir)
    print_summary(dataset_dir)


if __name__ == "__main__":
    main()
