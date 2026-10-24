"""The pretrained model is served from the pinned safetensors conversion, never from the pickle ``.ckpt``.

Offline tests need neither torch nor tabpfn. The conversion tests need torch + safetensors; the equivalence test also
needs tabpfn and the staged pinned checkpoint under weights/, and skips without them. Both are in the integration tier.
"""
# ruff: noqa: E501
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tabpfn_classifier_pipeline import pipeline as api
from tabpfn_classifier_pipeline import safetensors_weights as sw

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "weights" / api.MODEL_KEY


def _raw_safetensors(header: dict, payload: bytes) -> bytes:
    text = json.dumps(header).encode("utf-8")
    return len(text).to_bytes(8, "little") + text + payload


def test_shim_module_imports_only_the_standard_library() -> None:
    tree = ast.parse((ROOT / "src" / "tabpfn_classifier_pipeline" / "safetensors_weights.py").read_text(encoding="utf-8"))
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            top.add((node.module or "").split(".")[0])
    assert top <= set(sys.stdlib_module_names) | {"__future__"}, top


def test_canonical_header_is_sorted_padded_and_idempotent() -> None:
    payload = bytes(range(16))
    header = {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}, "__metadata__": {"b": "2", "a": "1"}}
    out = sw.canonical_safetensors(_raw_safetensors(header, payload))
    size = int.from_bytes(out[:8], "little")
    assert size % 8 == 0 and out[8 + size :] == payload
    assert out[8 : 8 + size].rstrip(b" ") == json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert sw.canonical_safetensors(out) == out


def test_converted_pins_are_well_formed_and_distinct_from_the_source() -> None:
    assert api.CONVERTED_FILE.endswith(".safetensors") and "v3" in api.CONVERTED_FILE
    assert len(api.CONVERTED_SHA256) == 64 and api.CONVERTED_SHA256 != api.WEIGHTS_SHA256 and api.CONVERTED_BYTES > 0


def test_ensure_converted_reuses_a_verified_file_without_reading_the_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = b"converted" * 512
    (tmp_path / api.CONVERTED_FILE).write_bytes(data)
    monkeypatch.setattr(api, "CONVERTED_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(api, "CONVERTED_BYTES", len(data))
    monkeypatch.setattr(sw, "convert_checkpoint", lambda *a, **k: pytest.fail("must not convert again"))
    assert api.ensure_converted(tmp_path) == tmp_path / api.CONVERTED_FILE


def test_ensure_converted_replaces_a_tampered_file_through_the_pinned_conversion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / api.CONVERTED_FILE).write_bytes(b"tampered")
    calls: list[dict] = []
    monkeypatch.setattr(api, "require_weights_only_load", lambda: None)
    monkeypatch.setattr(sw, "convert_checkpoint", lambda source, target, **kwargs: calls.append({"source": source, "target": target, **kwargs}))
    api.ensure_converted(tmp_path)
    assert calls == [{"source": tmp_path / api.WEIGHTS_FILE, "target": tmp_path / api.CONVERTED_FILE, "expected_sha256": api.WEIGHTS_SHA256, "source_name": api.WEIGHTS_FILE, "expected_converted_sha256": api.CONVERTED_SHA256}]


def test_save_artifact_refuses_to_re_export_a_v1_pickle_checkpoint(tmp_path: Path) -> None:
    pipe = api.TabPFNClassifierPipeline(weights_path=tmp_path / api.LEGACY_CHECKPOINT_NAME, feature_columns=["a"], classes=["x", "y"], _model=object())
    with pytest.raises(ValueError, match="not a safetensors file"):
        pipe.save_artifact(tmp_path / "bundle", saver=lambda model, path: None)


# --------------------------------------------------------------------------- conversion (torch + safetensors)


def _tiny_checkpoint(tmp_path: Path) -> tuple[Path, str]:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    state = {"encoder.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4).t(), "encoder.bias": torch.zeros(4)}
    path = tmp_path / "tiny.ckpt"
    torch.save({"state_dict": state, "config": {"emsize": 4, "nlayers": 1}, "architecture_name": "tabpfn_v3", "inference_config": {"X": [None, "safepower"]}, "optimizer_state": None}, path)
    return path, sw.sha256_file(path)


@pytest.mark.integration
def test_conversion_is_byte_reproducible_and_decodes_like_tabpfn(tmp_path: Path) -> None:
    source, digest = _tiny_checkpoint(tmp_path)
    first = sw.convert_checkpoint(source, tmp_path / "a.safetensors", expected_sha256=digest, source_name="tiny.ckpt")
    second = sw.convert_checkpoint(source, tmp_path / "b.safetensors", expected_sha256=digest, source_name="tiny.ckpt")
    assert first == second and (tmp_path / "a.safetensors").read_bytes() == (tmp_path / "b.safetensors").read_bytes()
    meta = sw.read_converted_metadata(tmp_path / "a.safetensors")
    assert meta["config"] == {"emsize": 4, "nlayers": 1} and meta["architecture_name"] == "tabpfn_v3" and meta["optimizer_state"] is None
    assert meta[sw.PROVENANCE_KEY] == {"format": sw.CONVERSION_FORMAT, "source_file": "tiny.ckpt", "source_sha256": digest}
    import torch
    from safetensors.torch import load_file

    tensors = load_file(str(tmp_path / "a.safetensors"))
    assert torch.equal(tensors["encoder.weight"], torch.arange(12, dtype=torch.float32).reshape(3, 4).t())


@pytest.mark.integration
def test_conversion_refuses_an_unpinned_source_and_an_unexpected_result(tmp_path: Path) -> None:
    source, digest = _tiny_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="refusing to deserialise"):
        sw.convert_checkpoint(source, tmp_path / "x.safetensors", expected_sha256="0" * 64, source_name="tiny.ckpt")
    with pytest.raises(ValueError, match="not writing"):
        sw.convert_checkpoint(source, tmp_path / "x.safetensors", expected_sha256=digest, source_name="tiny.ckpt", expected_converted_sha256="0" * 64)
    assert not (tmp_path / "x.safetensors").exists()


# --------------------------------------------------------------------------- equivalence on the pinned checkpoint


@pytest.mark.integration
@pytest.mark.skipif(not (SNAPSHOT / api.WEIGHTS_FILE).is_file(), reason="pinned checkpoint not staged under weights/")
def test_served_safetensors_equals_the_pinned_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("tabpfn")
    import shutil

    from tabpfn.checkpoint import Checkpoint
    from tabpfn.model_loading import _load_checkpoint_cached

    weights = tmp_path / "weights"
    weights.mkdir()
    for entry in json.loads((SNAPSHOT / api.MANIFEST_NAME).read_text(encoding="utf-8"))["files"]:
        shutil.copyfile(SNAPSHOT / entry["path"], weights / entry["path"])
    shutil.copyfile(SNAPSHOT / api.MANIFEST_NAME, weights / api.MANIFEST_NAME)
    served = api.ensure_converted(weights)
    assert sw.sha256_file(served) == api.CONVERTED_SHA256 and served.stat().st_size == api.CONVERTED_BYTES
    source = torch.load(weights / api.WEIGHTS_FILE, map_location="cpu", weights_only=True)
    converted = Checkpoint(served).load()
    assert set(converted["state_dict"]) == set(source["state_dict"])
    assert all(torch.equal(converted["state_dict"][k], v) and converted["state_dict"][k].dtype == v.dtype for k, v in source["state_dict"].items())
    assert converted["config"] == source["config"] and converted["architecture_name"] == source["architecture_name"]

    frames = api.read_dataset_zip(api.build_synthetic_dataset(tmp_path / "sample.zip"))
    train, val = frames["train.csv"], frames["val.csv"]
    features = [c for c in train.columns if c != "target"]
    reference = api.TabPFNClassifierPipeline(weights_path=weights / api.WEIGHTS_FILE, device="cpu").fit(train[features], train["target"])
    expected = reference.predict_proba(val[features])
    _load_checkpoint_cached.cache_clear()

    def refuse(*args, **kwargs):
        raise AssertionError("torch.load called on the safetensors serving path")

    pipe = api.TabPFNClassifierPipeline.from_pretrained(device="cpu", weights_dir=weights)
    assert pipe.weights_path == served
    monkeypatch.setattr(torch, "load", refuse)
    pipe.fit(train[features], train["target"])
    proba = pipe.predict_proba(val[features])
    assert np.array_equal(proba, expected), float(np.max(np.abs(proba - expected)))
    assert list(pipe.predict(val[features])["prediction"]) == list(reference.predict(val[features])["prediction"])
