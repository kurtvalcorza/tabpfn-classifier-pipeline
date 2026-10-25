"""Bundle format v2 (model.safetensors) and the deprecated v1 layout (model.ckpt), read with weights_only=True only.

The offline tests need neither torch nor tabpfn. The real-model tests need tabpfn and the staged pinned checkpoint under
weights/ (integration tier) and skip without them.
"""
# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest
from serving import load_artifact

from tabpfn_classifier_pipeline import pipeline as api
from tabpfn_classifier_pipeline import safetensors_weights as sw

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "weights" / api.MODEL_KEY
WEIGHTS_SHA256 = "97a228454844a64364050ec34c7bf14d35610d5e9c5fc7bb0d6554b8d24bbed7"


def _fitted(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("init_params.json", json.dumps({"model_path": "/producer/model.safetensors"}))
        archive.writestr("padding.bin", b"\0" * 2048)


def _bundle(root: Path, *, legacy: bool = False) -> dict:
    root.mkdir()
    _fitted(root / api.FITTED_NAME)
    name = api.LEGACY_CHECKPOINT_NAME if legacy else api.CHECKPOINT_NAME
    (root / name).write_bytes(b"w" * 4096)
    digest = api.sha256_file(root / name)
    manifest = {"taskType": api.TASK_TYPE, "targetColumn": "target", "featureColumns": ["a"], "classes": ["x", "y"], "fittedEstimatorSha256": api.sha256_file(root / api.FITTED_NAME)}
    if legacy:
        manifest.update({"schemaVersion": 1, "fittedEstimator": api.FITTED_NAME, "foundationCheckpoint": name, "foundationCheckpointSha256": digest})
    else:
        manifest.update(sw.v2_manifest_fields(api.TASK_TYPE, digest, 4096))
    (root / api.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_print_tool_reports_the_pinned_digest() -> None:
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "print_weights_sha256.py")], capture_output=True, text=True, check=True).stdout
    assert out.split()[0] == hashlib.sha256((ROOT / "src" / "tabpfn_classifier_pipeline" / "safetensors_weights.py").read_bytes()).hexdigest() == WEIGHTS_SHA256


def test_v2_manifest_fields_name_the_safetensors_file() -> None:
    fields = sw.v2_manifest_fields(api.TASK_TYPE, "a" * 64, 123)
    assert fields["schemaVersion"] == 2 and fields["bundleFormat"] == "tabpfn-dimer-classifier-v2"
    assert fields["foundationCheckpoint"] == fields["modelPath"] == "model.safetensors" and "model.safetensors" in fields["payloadFiles"]
    assert (fields["foundationCheckpointSha256"], fields["foundationCheckpointBytes"]) == ("a" * 64, 123)
    assert sw.check_bundle_layout({"taskType": api.TASK_TYPE, **fields}) == 2


def test_both_layouts_validate_and_bind_to_their_pinned_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    v2 = _bundle(tmp_path / "v2")
    v1 = _bundle(tmp_path / "v1", legacy=True)
    assert api.validate_artifact_bundle(tmp_path / "v2")["layout"] == 2
    assert api.validate_artifact_bundle(tmp_path / "v1")["layout"] == 1
    with pytest.raises(ValueError, match="expected"):
        api.validate_artifact_bundle(tmp_path / "v2", pinned_base=True)  # the stand-in is not the pinned conversion
    monkeypatch.setattr(api, "CONVERTED_SHA256", v2["foundationCheckpointSha256"])
    monkeypatch.setattr(api, "WEIGHTS_SHA256", v1["foundationCheckpointSha256"])
    assert api.validate_artifact_bundle(tmp_path / "v2", pinned_base=True)["layout"] == 2
    assert api.validate_artifact_bundle(tmp_path / "v1", pinned_base=True)["layout"] == 1
    assert api.pinned_checkpoint_sha256(2) == api.CONVERTED_SHA256 and api.pinned_checkpoint_sha256(1) == api.WEIGHTS_SHA256


@pytest.mark.parametrize("change", ["both_files", "v1_names_safetensors", "v2_names_ckpt", "v2_without_payload", "unknown_schema", "size_mismatch"])
def test_mixed_or_unknown_layouts_are_refused_before_weights_are_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    legacy = change == "v1_names_safetensors"
    manifest = _bundle(tmp_path / "b", legacy=legacy)
    root = tmp_path / "b"
    if change == "both_files":
        (root / api.LEGACY_CHECKPOINT_NAME).write_bytes(b"pickle")
    elif change == "v1_names_safetensors":
        manifest["foundationCheckpoint"] = api.CHECKPOINT_NAME
    elif change == "v2_names_ckpt":
        manifest["foundationCheckpoint"] = manifest["modelPath"] = api.LEGACY_CHECKPOINT_NAME
    elif change == "v2_without_payload":
        del manifest["payloadFiles"]
    elif change == "unknown_schema":
        manifest["schemaVersion"] = 3
    elif change == "size_mismatch":
        manifest["foundationCheckpointBytes"] = 4097
    (root / api.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    if change != "size_mismatch":
        monkeypatch.setattr(api, "sha256_file", lambda path: pytest.fail(f"read {path} before the layout check"))
    with pytest.raises(ValueError, match="mixed|malformed|unknown bundle layout|foundationCheckpointBytes"):
        api.validate_artifact_bundle(root)


def test_serving_loader_picks_the_layout_and_refuses_mixed(tmp_path: Path) -> None:
    _bundle(tmp_path / "v2")
    _bundle(tmp_path / "v1", legacy=True)
    assert load_artifact.artifact_weights(tmp_path / "v2") == (tmp_path / "v2" / "model.safetensors", 2)
    assert load_artifact.artifact_weights(tmp_path / "v1") == (tmp_path / "v1" / "model.ckpt", 1)
    (tmp_path / "v2" / "model.ckpt").write_bytes(b"pickle")
    with pytest.raises(ValueError, match="exactly one"):
        load_artifact.artifact_weights(tmp_path / "v2")
    (tmp_path / "v1" / "model.ckpt").unlink()
    (tmp_path / "v1" / "model.safetensors").write_bytes(b"w")
    with pytest.raises(ValueError, match="disagrees"):
        load_artifact.artifact_weights(tmp_path / "v1")


def test_pickle_files_lists_weight_pickles_anywhere(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    for name in ("a/x.ckpt", "b.pth", "c.safetensors", "d.json", "e.tabpfn_fit"):
        (tmp_path / name).write_bytes(b"x")
    assert sw.pickle_files(tmp_path) == ["a/x.ckpt", "b.pth"]


@pytest.mark.integration
def test_convert_written_checkpoint_removes_the_pickle(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    source = tmp_path / "model.ckpt"
    torch.save({"state_dict": {"w": torch.ones(3)}, "config": {"a": 1}}, source)
    facts = sw.convert_written_checkpoint(source, tmp_path / "model.safetensors")
    assert not source.exists() and sw.pickle_files(tmp_path) == [] and facts["converted_sha256"] == sw.sha256_file(tmp_path / "model.safetensors")


# --------------------------------------------------------------------------- real model, both layouts


@pytest.fixture
def real_v2_bundle(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("tabpfn")
    if not (SNAPSHOT / api.WEIGHTS_FILE).is_file() and not api.converted_is_verified(SNAPSHOT):
        pytest.skip("pinned checkpoint not staged under weights/")
    weights = tmp_path / "weights"
    weights.mkdir()
    for path in SNAPSHOT.iterdir():
        if path.is_file() and path.suffix in (".json", ".md", ".ckpt", ".safetensors") or path.name == "LICENSE":
            shutil.copyfile(path, weights / path.name)
    frames = api.read_dataset_zip(api.build_synthetic_dataset(tmp_path / "sample.zip"))
    train, val = frames["train.csv"], frames["val.csv"]
    features = [c for c in train.columns if c != "target"]
    pipe = api.TabPFNClassifierPipeline.from_pretrained(device="cpu", weights_dir=weights).fit(train[features], train["target"])
    expected = pipe.predict_proba(val[features])
    manifest = pipe.save_artifact(tmp_path / "v2")
    return tmp_path, weights, manifest, val[features], expected


@pytest.mark.integration
def test_real_v2_bundle_reloads_without_torch_load(real_v2_bundle, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    from tabpfn.model_loading import _load_checkpoint_cached

    tmp_path, _weights, manifest, rows, expected = real_v2_bundle
    assert manifest["foundationCheckpointSha256"] == api.CONVERTED_SHA256 and sw.pickle_files(tmp_path / "v2") == []
    _load_checkpoint_cached.cache_clear()
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("torch.load called on the v2 path"))
    fresh = api.TabPFNClassifierPipeline.from_artifact(tmp_path / "v2", device="cpu", pinned_base=True)
    assert np.array_equal(fresh.predict_proba(rows), expected)
    served = load_artifact.load_dimer_tabpfn_artifact(tmp_path / "v2", device="cpu")
    assert np.array_equal(np.asarray(served.predict_proba(rows), dtype=float), expected)


@pytest.mark.integration
def test_real_v1_bundle_loads_with_weights_only_and_a_deprecation_warning(real_v2_bundle, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    from tabpfn.model_loading import _load_checkpoint_cached

    tmp_path, weights, manifest, rows, expected = real_v2_bundle
    if not (weights / api.WEIGHTS_FILE).is_file():
        pytest.skip("the source checkpoint is needed to build a v1 bundle")
    v1 = tmp_path / "v1"
    v1.mkdir()
    shutil.copyfile(tmp_path / "v2" / api.FITTED_NAME, v1 / api.FITTED_NAME)
    shutil.copyfile(weights / api.WEIGHTS_FILE, v1 / api.LEGACY_CHECKPOINT_NAME)
    legacy = {k: v for k, v in manifest.items() if k not in sw.V2_ONLY_KEYS}
    legacy.update({"schemaVersion": 1, "foundationCheckpoint": api.LEGACY_CHECKPOINT_NAME, "foundationCheckpointSha256": api.WEIGHTS_SHA256})
    (v1 / api.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(legacy), encoding="utf-8")
    calls: list = []
    real = torch.load

    def spy(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return real(*args, **kwargs)

    _load_checkpoint_cached.cache_clear()
    monkeypatch.setattr(torch, "load", spy)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fresh = api.TabPFNClassifierPipeline.from_artifact(v1, device="cpu", pinned_base=True)
        served = load_artifact.load_dimer_tabpfn_artifact(v1, device="cpu")
    assert calls and all(flag is True for flag in calls), calls
    assert sum(issubclass(w.category, DeprecationWarning) and "v1 TabPFN bundle" in str(w.message) for w in caught) == 2
    assert np.array_equal(fresh.predict_proba(rows), expected)
    assert np.array_equal(np.asarray(served.predict_proba(rows), dtype=float), expected)
    with pytest.raises(ValueError, match="not a safetensors file"):
        fresh.save_artifact(tmp_path / "re-export")


def test_the_default_sample_is_the_published_v2_reissue_and_v1_is_recorded_as_superseded() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("stages_for_pin", ROOT / "tools" / "tutorial_stages.py")
    stages = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stages)
    assert stages.SAMPLE_ARTIFACT_TAG == "sample-artifact-synthetic-v2" and stages.SAMPLE_ARTIFACT_V1_TAG == "sample-artifact-synthetic-v1"
    assert stages.SAMPLE_ARTIFACT_SHA256 == "8828df98a2db8047bc2d52313a5a173bfa499d8b780619899def268dee31b41c" and stages.SAMPLE_ARTIFACT_BYTES == 197034023
    assert len(stages.SAMPLE_ARTIFACT_V1_SHA256) == 64 and stages.SAMPLE_ARTIFACT_V1_SHA256 != stages.SAMPLE_ARTIFACT_SHA256
    assert stages.SAMPLE_ARTIFACT_V1_TAG not in stages.SAMPLE_ARTIFACT_URL and stages.SAMPLE_ARTIFACT_V1_BYTES > 0
