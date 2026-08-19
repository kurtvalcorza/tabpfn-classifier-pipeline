"""Integration tier: exercises the relocation loader with real TabPFN.

Fits a real TabPFN classifier (v2, Apache, CPU), persists a DIMER artifact,
moves it to a clean directory, and reloads it through the pipeline's own
`load_dimer_tabpfn_artifact` to predict — the production serving path. Marked
`integration`.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


def _coerce_non_json_init_params(model) -> None:
    # Mirrors the finetuner's save_artifacts workaround for TabPFN 8.1.0.
    for param, value in model.get_params(deep=False).items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            setattr(model, param, str(value))


def test_relocation_loader_roundtrips_real_artifact(tmp_path: Path) -> None:
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    from tabpfn.model_loading import save_fitted_tabpfn_model, save_tabpfn_model

    from serving.load_artifact import load_dimer_tabpfn_artifact

    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(150, 3))
    score = x_train @ np.array([2.0, -1.5, 0.5])
    y_train = np.where(score < -1.0, "a", np.where(score > 1.0, "c", "b"))
    x_test = rng.normal(size=(45, 3))
    train = pd.DataFrame(x_train, columns=["f1", "f2", "f3"])
    test = pd.DataFrame(x_test, columns=["f1", "f2", "f3"])

    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, device="cpu", random_state=0, show_progress_bar=False
    )
    model.fit(train, pd.Series(y_train))
    reference = np.asarray(model.predict_proba(test), dtype=float)

    train_dir = tmp_path / "train_out"
    train_dir.mkdir()
    _coerce_non_json_init_params(model)
    save_fitted_tabpfn_model(model, train_dir / "model.tabpfn_fit")
    save_tabpfn_model(model, train_dir / "model.ckpt")

    serve_dir = tmp_path / "clean_serving_env"
    serve_dir.mkdir()
    for name in ("model.tabpfn_fit", "model.ckpt"):
        shutil.copy(train_dir / name, serve_dir / name)

    reloaded = load_dimer_tabpfn_artifact(serve_dir, device="cpu")
    proba = np.asarray(reloaded.predict_proba(test), dtype=float)

    assert proba.shape == reference.shape
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-3)
    assert np.max(np.abs(proba - reference)) < 1e-2
