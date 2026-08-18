from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from examples.build_synthetic_dataset import build_dataset


def test_builder_creates_stratified_train_val_test_zip(tmp_path: Path) -> None:
    output = build_dataset(tmp_path / "smoke.zip", rows=300, features=8, classes=3, seed=9)
    assert output.exists()
    with zipfile.ZipFile(output) as archive:
        assert sorted(archive.namelist()) == ["test.csv", "train.csv", "val.csv"]
        frames = {}
        for stem in ("train", "val", "test"):
            with archive.open(f"{stem}.csv") as handle:
                frames[stem] = pd.read_csv(handle)

    schemas = [set(frame.columns) for frame in frames.values()]
    assert schemas[0] == schemas[1] == schemas[2]
    assert "target" in frames["train"].columns
    train_classes = set(frames["train"]["target"])
    assert train_classes == set(frames["val"]["target"])
    assert train_classes == set(frames["test"]["target"])
    assert sum(len(frame) for frame in frames.values()) == 300


def test_builder_is_reproducible(tmp_path: Path) -> None:
    first = build_dataset(tmp_path / "a.zip", rows=240, features=6, classes=2, seed=123)
    second = build_dataset(tmp_path / "b.zip", rows=240, features=6, classes=2, seed=123)

    def read_train(path: Path) -> pd.DataFrame:
        with zipfile.ZipFile(path) as archive:
            with archive.open("train.csv") as handle:
                return pd.read_csv(handle)

    pd.testing.assert_frame_equal(read_train(first), read_train(second))
