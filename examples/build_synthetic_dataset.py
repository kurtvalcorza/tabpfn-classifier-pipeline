#!/usr/bin/env python3
"""Build a small, reproducible train/val/test classification ZIP for DIMER smoke tests."""
from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def build_dataset(
    out: Path,
    rows: int = 600,
    features: int = 12,
    classes: int = 3,
    seed: int = 42,
) -> Path:
    if rows < max(50, classes * 6):
        raise ValueError("rows is too small for a useful stratified train/val/test smoke dataset")
    if features < 2:
        raise ValueError("features must be at least 2")
    if classes < 2:
        raise ValueError("classes must be at least 2")

    informative = min(features, max(2, classes * 2))
    redundant = min(max(0, features - informative), max(0, features // 4))
    if informative + redundant > features:
        redundant = features - informative

    X, y = make_classification(
        n_samples=rows,
        n_features=features,
        n_informative=informative,
        n_redundant=redundant,
        n_repeated=0,
        n_classes=classes,
        n_clusters_per_class=1,
        class_sep=1.2,
        flip_y=0.02,
        random_state=seed,
    )
    frame = pd.DataFrame(X, columns=[f"feature_{i:02d}" for i in range(features)])
    frame["category"] = [f"segment_{int(index) % 4}" for index in range(rows)]
    frame["record_id"] = [f"row-{index:05d}" for index in range(rows)]
    frame["target"] = [f"class_{value}" for value in y]

    train_plus_val, test = train_test_split(
        frame,
        test_size=0.15,
        random_state=seed,
        stratify=frame["target"],
    )
    train, val = train_test_split(
        train_plus_val,
        test_size=0.17647058823529413,  # 15% of total after the first 15% split
        random_state=seed,
        stratify=train_plus_val["target"],
    )

    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        train.to_csv(root / "train.csv", index=False)
        val.to_csv(root / "val.csv", index=False)
        test.to_csv(root / "test.csv", index=False)
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in ("train.csv", "val.csv", "test.csv"):
                archive.write(root / name, arcname=name)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--features", type=int, default=12)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = build_dataset(args.out, args.rows, args.features, args.classes, args.seed)
    print(path)


if __name__ == "__main__":
    main()
