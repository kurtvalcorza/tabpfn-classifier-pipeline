"""TabPFN checkpoint loads must stay restricted to weights_only=True.

tabpfn passes weights_only=None and relies on torch's default, so both guards (the package's and the standalone serving
loader's) must refuse the two conditions that turn that default into a full unpickle. The offline tier has no torch, so the
version is passed in.
"""
from __future__ import annotations

import pytest

from serving import load_artifact
from tabpfn_classifier_pipeline import pipeline

GUARDS = [pipeline.require_weights_only_load, load_artifact.require_weights_only_load]


@pytest.mark.parametrize("guard", GUARDS)
def test_refuses_when_force_no_weights_only_is_set(monkeypatch, guard) -> None:
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    with pytest.raises(RuntimeError, match="TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"):
        guard("2.11.0")


@pytest.mark.parametrize("guard", GUARDS)
def test_passes_without_the_env_var_on_supported_torch(monkeypatch, guard) -> None:
    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)
    guard("2.11.0+cu128")


@pytest.mark.parametrize("guard", GUARDS)
def test_refuses_torch_older_than_2_6(monkeypatch, guard) -> None:
    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)
    with pytest.raises(RuntimeError, match="older than 2.6"):
        guard("2.5.1")


def test_estimator_build_is_guarded(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "true")
    pipe = pipeline.TabPFNClassifierPipeline(weights_path=tmp_path / "w.ckpt", device="cpu")
    with pytest.raises(RuntimeError, match="TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"):
        pipe._build_estimator()
