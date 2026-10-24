"""ARTIFACT_SPEC SER7-SER9 (b): ``model.safetensors`` beside the ``model.ckpt`` byte copy.

Offline tier (no torch): the deterministic writer, the torch-free header parser, and the bundle validator's refusals
(missing, tampered, wrong provenance, pre-conversion bundle). Integration tier (torch + safetensors, as in the CI
``integration`` job): conversion equality on a synthetic checkpoint (SER8), refusal of non-plain fields, refusal of a
conversion that disagrees with the checkpoint, byte-identical output across fresh interpreters, TabPFN's own loader
reading the file, and -- only when the pinned TabPFN-3 checkpoint is staged locally -- the pinned conversion digest.
"""
# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import pickletools
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tabpfn_classifier_pipeline import pipeline as api

ROOT = Path(__file__).resolve().parents[1]
PINNED_CHECKPOINT = ROOT / "weights" / api.MODEL_KEY / api.WEIGHTS_FILE


def _stand_in_bundle(root: Path) -> Path:
    """A four-member bundle written by hand: stand-in fitted archive and checkpoint, a real safetensors file."""
    root.mkdir(parents=True)
    with zipfile.ZipFile(root / api.FITTED_NAME, "w") as fitted:
        fitted.writestr("init_params.json", json.dumps({"model_path": "/producer/model.ckpt"}))
        fitted.writestr("padding.bin", b"\0" * 2048)
    blob = hashlib.sha256(b"stand-in checkpoint").digest() * 128
    (root / api.CHECKPOINT_NAME).write_bytes(blob)
    ckpt_sha = api.sha256_file(root / api.CHECKPOINT_NAME)
    converted = api.write_safetensors(root / api.CONVERTED_CHECKPOINT_NAME, {"blob": ("U8", [len(blob)], blob)}, {"config": "{}"})
    manifest = {
        "schemaVersion": api.ARTIFACT_SCHEMA_VERSION, "taskType": api.TASK_TYPE, "targetColumn": "target", "featureColumns": ["a"], "classes": ["x", "y"],
        "fittedEstimator": api.FITTED_NAME, "fittedEstimatorSha256": api.sha256_file(root / api.FITTED_NAME),
        "foundationCheckpoint": api.CHECKPOINT_NAME, "foundationCheckpointSha256": ckpt_sha,
        "convertedCheckpoint": api.CONVERTED_CHECKPOINT_NAME, "convertedCheckpointSha256": converted,
        "convertedCheckpointProvenance": {"derivedFrom": {"file": api.CHECKPOINT_NAME, "sha256": ckpt_sha}, "tool": api.SAFETENSORS_CONVERTER, "toolVersion": api.SAFETENSORS_CONVERTER_VERSION},
    }
    _write_manifest(root, manifest)
    return root


def _manifest(root: Path) -> dict:
    return json.loads((root / api.ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))


def _write_manifest(root: Path, manifest: dict) -> None:
    (root / api.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


# ---------------------------------------------------------------- offline: writer, parser, validator


def test_writer_is_deterministic_and_parses(tmp_path: Path) -> None:
    tensors = {"b": ("F32", [2], struct.pack("<2f", 1.5, -2.0)), "a": ("I64", [1, 1], struct.pack("<q", 7)), "empty": ("BF16", [0, 3], b"")}
    metadata = {"z": '{"k":[1,2]}', "config": "{}"}
    first = api.write_safetensors(tmp_path / "1.safetensors", tensors, metadata)
    second = api.write_safetensors(tmp_path / "2.safetensors", dict(reversed(list(tensors.items()))), dict(reversed(list(metadata.items()))))
    assert first == second  # insertion order of tensors or metadata never changes the bytes
    raw = (tmp_path / "1.safetensors").read_bytes()
    (length,) = struct.unpack("<Q", raw[:8])
    assert length % 8 == 0 and raw[8:9] == b"{"
    parsed = api.read_safetensors_header(tmp_path / "1.safetensors")
    assert parsed["metadata"] == metadata and sorted(parsed["tensors"]) == ["a", "b", "empty"]
    assert parsed["tensors"]["a"] == {"data_offsets": [0, 8], "dtype": "I64", "shape": [1, 1]}  # sorted names, packed in order
    assert raw[parsed["dataStart"]:] == tensors["a"][2] + tensors["b"][2]


def test_writer_bytes_identical_in_a_fresh_interpreter(tmp_path: Path) -> None:
    script = (
        "import sys, struct; sys.path.insert(0, sys.argv[1]);"
        "from tabpfn_classifier_pipeline import pipeline as api;"
        "print(api.write_safetensors(sys.argv[2], {'w': ('F32', [3], struct.pack('<3f', 1, 2, 3)), 'v': ('U8', [2], b'ab')}, {'config': '{\"a\":1}', 'name': '\"x\"'}))"
    )
    digests = {subprocess.run([sys.executable, "-c", script, str(ROOT / "src"), str(tmp_path / f"{i}.safetensors")], check=True, capture_output=True, text=True).stdout.strip() for i in range(2)}
    assert len(digests) == 1


def test_converted_file_holds_no_pickle(tmp_path: Path) -> None:
    """The file is an 8-byte length, a JSON header and raw bytes; there is no pickle stream to execute."""
    api.write_safetensors(tmp_path / "m.safetensors", {"w": ("F32", [2], struct.pack("<2f", 1, 2))}, {"config": "{}"})
    raw = (tmp_path / "m.safetensors").read_bytes()
    assert raw[8] != 0x80  # a pickle protocol 2+ stream starts with PROTO (0x80)
    with pytest.raises(Exception):  # noqa: B017 -- any failure: the bytes are not a pickle
        list(pickletools.genops(raw))
    assert not zipfile.is_zipfile(tmp_path / "m.safetensors")  # not a torch.save zip either


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda raw: struct.pack("<Q", 10**12) + raw[8:], "implausible"),
        (lambda raw: raw[:-1], "data section"),
        (lambda raw: raw + b"\0", "data section"),
        (lambda raw: raw[:8] + raw[8:].replace(b'"F32"', b'"F99"', 1), "unsupported dtype"),
        (lambda raw: raw[:8] + b"[" + raw[9:], "not JSON"),
        (lambda raw: raw[:8] + raw[8:].replace(b'"shape":[4]', b'"shape":[5]', 1), "needs 20 bytes"),
    ],
)
def test_parser_refuses_malformed_files(tmp_path: Path, mutate, message: str) -> None:
    good = tmp_path / "good.safetensors"
    api.write_safetensors(good, {"w": ("F32", [4], b"\0" * 16)}, {"config": "{}"})
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(mutate(good.read_bytes()))
    with pytest.raises(ValueError, match=message):
        api.read_safetensors_header(bad)


def test_writer_refuses_inconsistent_tensors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="do not match"):
        api.write_safetensors(tmp_path / "x.safetensors", {"w": ("F32", [3], b"\0" * 8)})
    with pytest.raises(ValueError, match="unsupported"):
        api.write_safetensors(tmp_path / "x.safetensors", {"w": ("C64", [1], b"\0" * 8)})
    with pytest.raises(ValueError, match="strings to strings"):
        api.write_safetensors(tmp_path / "x.safetensors", {"w": ("U8", [1], b"\0")}, {"config": {"a": 1}})


def test_plain_metadata_rules() -> None:
    assert api.metadata_json({"b": (None, "safepower"), "a": 1.0, "c": True}) == '{"a":1.0,"b":[null,"safepower"],"c":true}'
    for bad in ({1, 2}, Path("x"), float("nan"), {1: "non-string key"}, b"bytes"):
        with pytest.raises(ValueError):
            api.metadata_json({"field": bad})


def test_validator_accepts_the_stand_in_bundle(tmp_path: Path) -> None:
    root = _stand_in_bundle(tmp_path / "art")
    checked = api.validate_artifact_bundle(root, expected_converted_sha256=_manifest(root)["convertedCheckpointSha256"])
    assert set(checked["verifiedSha256"]) == {"fittedEstimator", "foundationCheckpoint", "convertedCheckpoint"}
    assert checked["conversionCheck"] == {"checked": "structure and digest", "tensors": 1}


def test_validator_refuses_missing_safetensors(tmp_path: Path) -> None:
    root = _stand_in_bundle(tmp_path / "art")
    (root / api.CONVERTED_CHECKPOINT_NAME).unlink()
    with pytest.raises(ValueError, match="model.safetensors is missing"):
        api.validate_artifact_bundle(root)


def test_validator_refuses_a_pre_conversion_manifest(tmp_path: Path) -> None:
    """A bundle exported before SER9 (b) (no convertedCheckpoint entry, e.g. sample-artifact-synthetic-v1) is refused."""
    root = _stand_in_bundle(tmp_path / "art")
    manifest = {k: v for k, v in _manifest(root).items() if not k.startswith("convertedCheckpoint")}
    _write_manifest(root, manifest)
    (root / api.CONVERTED_CHECKPOINT_NAME).unlink()
    with pytest.raises(ValueError, match="predates the safetensors conversion"):
        api.validate_artifact_bundle(root)


def test_validator_refuses_tampered_safetensors(tmp_path: Path) -> None:
    root = _stand_in_bundle(tmp_path / "art")
    path = root / api.CONVERTED_CHECKPOINT_NAME
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="model.safetensors SHA-256 .* != manifest"):
        api.validate_artifact_bundle(root)


def test_validator_refuses_a_rehashed_conversion_of_other_weights(tmp_path: Path) -> None:
    """Other weights behind a re-hashed manifest pass the manifest digest but not a pinned expected digest."""
    root = _stand_in_bundle(tmp_path / "art")
    pinned = _manifest(root)["convertedCheckpointSha256"]
    other = api.write_safetensors(root / api.CONVERTED_CHECKPOINT_NAME, {"blob": ("U8", [4096], b"\1" * 4096)}, {"config": "{}"})
    manifest = _manifest(root)
    manifest["convertedCheckpointSha256"] = other
    _write_manifest(root, manifest)
    api.validate_artifact_bundle(root)  # internally consistent ...
    with pytest.raises(ValueError, match="!= expected"):
        api.validate_artifact_bundle(root, expected_converted_sha256=pinned)  # ... but not the pinned conversion


@pytest.mark.parametrize(
    "key, value, message",
    [("derivedFrom", {"file": "model.ckpt", "sha256": "0" * 64}, "derivedFrom"), ("derivedFrom", {"file": "other.ckpt"}, "derivedFrom"), ("derivedFrom", None, "derivedFrom"), ("tool", "something-else", "names tool")],
)
def test_validator_refuses_provenance_that_does_not_name_the_checkpoint(tmp_path: Path, key: str, value, message: str) -> None:
    root = _stand_in_bundle(tmp_path / "art")
    manifest = _manifest(root)
    manifest["convertedCheckpointProvenance"] = {**manifest["convertedCheckpointProvenance"], key: value}
    _write_manifest(root, manifest)
    with pytest.raises(ValueError, match=message):
        api.validate_artifact_bundle(root)


def test_validator_refuses_a_safetensors_that_is_not_safetensors(tmp_path: Path) -> None:
    root = _stand_in_bundle(tmp_path / "art")
    (root / api.CONVERTED_CHECKPOINT_NAME).write_bytes(b"\x80\x02" + b"\0" * 2046)  # a pickle-looking file
    manifest = _manifest(root)
    manifest["convertedCheckpointSha256"] = api.sha256_file(root / api.CONVERTED_CHECKPOINT_NAME)
    _write_manifest(root, manifest)
    with pytest.raises(ValueError, match="header is not JSON"):
        api.validate_artifact_bundle(root)


# ---------------------------------------------------------------- integration: torch conversion (SER8)


def _synthetic_checkpoint(path: Path, *, perturb: bool = False) -> dict:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    shared = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    state = {
        "encoder.weight": shared,
        "encoder.weight_view": shared[:, 1:],  # shares storage and is non-contiguous
        "decoder.weight": torch.randn(5, 3, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16),
        "decoder.bias": torch.tensor([float("nan"), -0.0, 1e-40], dtype=torch.float32),  # NaN, signed zero, denormal
        "criterion.borders": torch.linspace(-1, 1, 7, dtype=torch.float64),
        "counts": torch.tensor([1, 2, 3], dtype=torch.int64),
        "mask": torch.tensor([True, False]),
        "scalar": torch.tensor(3.0, dtype=torch.float16),
        "padding": torch.zeros(512, dtype=torch.float32),  # keeps the file above the validator's 1 KiB floor
    }
    if perturb:
        state["counts"] = torch.tensor([1, 2, 4], dtype=torch.int64)
    obj = {"state_dict": state, "config": {"nlayers": 2, "rope_base": 100000.0, "name": "TabPFN-v3"}, "inference_config": {"TRANSFORMS": (None, "safepower"), "MAX": 160}, "architecture_name": "tabpfn_v3", "optimizer_state": None}
    torch.save(obj, path)
    return obj


@pytest.mark.integration
def test_conversion_is_faithful(tmp_path: Path) -> None:
    import torch
    from safetensors import safe_open

    obj = _synthetic_checkpoint(tmp_path / "model.ckpt")
    info = api.convert_checkpoint_to_safetensors(tmp_path / "model.ckpt", tmp_path / "model.safetensors")
    assert info["bitwiseEqual"] is True and info["tensors"] == len(obj["state_dict"]) and info["sha256"] == api.sha256_file(tmp_path / "model.safetensors")
    assert info["metadataKeys"] == sorted(k for k in obj if k != "state_dict")
    with safe_open(str(tmp_path / "model.safetensors"), framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == set(obj["state_dict"])
        for name, ref in obj["state_dict"].items():
            got = handle.get_tensor(name)
            assert got.dtype == ref.dtype and got.shape == ref.shape
            assert torch.equal(got.contiguous().view(-1).view(torch.uint8), ref.contiguous().reshape(-1).view(torch.uint8))
        metadata = handle.metadata()
    assert json.loads(metadata["inference_config"]) == {"MAX": 160, "TRANSFORMS": [None, "safepower"]}
    assert json.loads(metadata["config"])["rope_base"] == 100000.0 and json.loads(metadata["optimizer_state"]) is None


@pytest.mark.integration
def test_conversion_bytes_identical_in_fresh_interpreters(tmp_path: Path) -> None:
    _synthetic_checkpoint(tmp_path / "model.ckpt")
    script = "import sys; sys.path.insert(0, sys.argv[1]); from tabpfn_classifier_pipeline import pipeline as api; print(api.convert_checkpoint_to_safetensors(sys.argv[2], sys.argv[3])['sha256'])"
    digests = {subprocess.run([sys.executable, "-c", script, str(ROOT / "src"), str(tmp_path / "model.ckpt"), str(tmp_path / f"{i}.safetensors")], check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1] for i in range(2)}
    assert len(digests) == 1


@pytest.mark.integration
def test_conversion_disagreeing_with_the_checkpoint_is_refused(tmp_path: Path) -> None:
    _synthetic_checkpoint(tmp_path / "other.ckpt", perturb=True)
    api.convert_checkpoint_to_safetensors(tmp_path / "other.ckpt", tmp_path / "other.safetensors")
    _synthetic_checkpoint(tmp_path / "model.ckpt")
    with pytest.raises(ValueError, match="tensor 'counts' differs"):
        api.verify_safetensors_conversion(tmp_path / "model.ckpt", tmp_path / "other.safetensors")
    # In a bundle: the other weights behind a re-hashed manifest pass the digest checks, and verify_conversion refuses them.
    root = _stand_in_bundle(tmp_path / "art")
    (root / api.CHECKPOINT_NAME).write_bytes((tmp_path / "model.ckpt").read_bytes())
    (root / api.CONVERTED_CHECKPOINT_NAME).write_bytes((tmp_path / "other.safetensors").read_bytes())
    manifest = _manifest(root)
    manifest["foundationCheckpointSha256"] = api.sha256_file(root / api.CHECKPOINT_NAME)
    manifest["convertedCheckpointSha256"] = api.sha256_file(root / api.CONVERTED_CHECKPOINT_NAME)
    manifest["convertedCheckpointProvenance"]["derivedFrom"]["sha256"] = manifest["foundationCheckpointSha256"]
    _write_manifest(root, manifest)
    api.validate_artifact_bundle(root)
    with pytest.raises(ValueError, match="differs"):
        api.validate_artifact_bundle(root, verify_conversion=True)


@pytest.mark.integration
def test_non_plain_checkpoint_fields_are_refused(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    torch.save({"state_dict": {"w": torch.zeros(2)}, "config": {"dtype": torch.float32}}, tmp_path / "model.ckpt")
    with pytest.raises(ValueError, match="not plain data"):
        api.convert_checkpoint_to_safetensors(tmp_path / "model.ckpt", tmp_path / "model.safetensors")
    assert not (tmp_path / "model.safetensors").exists()


@pytest.mark.integration
def test_tabpfn_checkpoint_loader_reads_the_conversion(tmp_path: Path) -> None:
    """TabPFN's own Checkpoint loader (what model_path=*.safetensors uses) returns the same fields and tensors."""
    import torch

    pytest.importorskip("tabpfn")
    from tabpfn.checkpoint import Checkpoint

    obj = _synthetic_checkpoint(tmp_path / "model.ckpt")
    api.convert_checkpoint_to_safetensors(tmp_path / "model.ckpt", tmp_path / "model.safetensors")
    loaded = Checkpoint(tmp_path / "model.safetensors").load()
    assert set(loaded) == set(obj) and loaded["config"] == obj["config"] and loaded["architecture_name"] == "tabpfn_v3"
    assert all(torch.equal(loaded["state_dict"][k].view(-1).view(torch.uint8), v.contiguous().reshape(-1).view(torch.uint8)) for k, v in obj["state_dict"].items())


@pytest.mark.integration
@pytest.mark.skipif(not PINNED_CHECKPOINT.is_file(), reason="pinned TabPFN-3 checkpoint not staged under weights/ (non-commercial weights; not fetched in CI)")
def test_pinned_checkpoint_converts_to_the_pinned_digest(tmp_path: Path) -> None:
    assert api.sha256_file(PINNED_CHECKPOINT) == api.WEIGHTS_SHA256
    info = api.convert_checkpoint_to_safetensors(PINNED_CHECKPOINT, tmp_path / api.CONVERTED_CHECKPOINT_NAME)
    assert info["sha256"] == api.WEIGHTS_SAFETENSORS_SHA256 and (tmp_path / api.CONVERTED_CHECKPOINT_NAME).stat().st_size == api.WEIGHTS_SAFETENSORS_BYTES
    assert info["bitwiseEqual"] is True and info["tensors"] == 510


@pytest.mark.integration
def test_real_tabpfn_reloads_from_the_conversion_with_identical_predictions(tmp_path: Path) -> None:
    """Real TabPFN (v2, Apache, CPU; the checkpoint the integration job already downloads): save its checkpoint,
    convert it, and point a fitted archive at the conversion. Predictions equal the pickle-backed checkpoint's exactly."""
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("tabpfn")
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    from tabpfn.model_loading import load_fitted_tabpfn_model, save_fitted_tabpfn_model, save_tabpfn_model

    rng = np.random.default_rng(0)
    train = pd.DataFrame(rng.normal(size=(120, 3)), columns=["f1", "f2", "f3"])
    labels = pd.Series(np.where(train["f1"] - train["f2"] > 0, "a", "b"))
    test = pd.DataFrame(rng.normal(size=(30, 3)), columns=["f1", "f2", "f3"])
    model = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0, show_progress_bar=False).fit(train, labels)
    api._coerce_non_json_init_params(model)
    save_tabpfn_model(model, tmp_path / api.CHECKPOINT_NAME)
    save_fitted_tabpfn_model(model, tmp_path / api.FITTED_NAME)
    info = api.convert_checkpoint_to_safetensors(tmp_path / api.CHECKPOINT_NAME, tmp_path / api.CONVERTED_CHECKPOINT_NAME)
    assert info["bitwiseEqual"] is True
    results = {}
    for name in (api.CHECKPOINT_NAME, api.CONVERTED_CHECKPOINT_NAME):
        rewritten = api.rewrite_model_path(tmp_path / api.FITTED_NAME, tmp_path / name, tmp_path / f"via-{name}.tabpfn_fit")
        results[name] = np.asarray(load_fitted_tabpfn_model(rewritten, device="cpu").predict_proba(test), dtype=float)
    assert np.array_equal(results[api.CHECKPOINT_NAME], results[api.CONVERTED_CHECKPOINT_NAME])
