# ruff: noqa: E501  -- long docstrings and messages are kept readable
"""Safetensors weights and the DIMER TabPFN bundle formats (tabpfn==8.1.0).

CANONICAL SOURCE: kurtvalcorza/tabpfn-classifier-pipeline, src/tabpfn_classifier_pipeline/safetensors_weights.py.
Byte-identical copies, never edited there: kurtvalcorza/tabpfn-regressor-pipeline
src/tabpfn_regressor_pipeline/safetensors_weights.py, and the fine-tune workers kurtvalcorza/tabpfn-classifier-finetuner
and kurtvalcorza/tabpfn-regressor-finetuner as tabpfn_safetensors_weights.py. Every copy's tests pin this file's SHA-256.
To re-sync after a change here, run ``python tools/print_weights_sha256.py`` in this repository, copy this file over the
other three, and update each pinned digest to the printed value.

tabpfn 8.1.0 reads a ``.safetensors`` checkpoint itself: ``tabpfn.checkpoint.Checkpoint`` opens it with
``safetensors.safe_open`` and JSON-decodes every header metadata value into the checkpoint dict, and
``tabpfn.model_loading.load_model`` builds the architecture from ``config`` / ``architecture_name`` and calls
``load_state_dict`` (strict by default). No ``torch.load`` is involved, so serving needs no shim: the converted file is
passed as ``model_path``. This module only writes such files and checks the bundle layouts.

- ``convert_checkpoint`` reads a digest-checked ``.ckpt`` once with ``torch.load(map_location="cpu",
  weights_only=True)``, keeps its ``state_dict`` as contiguous CPU tensors and every other top-level field as
  ``json.dumps(value, sort_keys=True)`` header metadata (the layout of tabpfn's own ``save_as_safetensors``), adds a
  ``dimer_conversion`` provenance field, and re-emits the header with sorted keys so the bytes are reproducible.
- ``convert_written_checkpoint`` does the same for a ``.ckpt`` this process just had tabpfn write
  (``save_tabpfn_model``) and removes the pickle.
- Bundle format v2 (``tabpfn-dimer-<task>-v2``, ``schemaVersion`` 2) carries ``model.safetensors``. v1 bundles
  (``schemaVersion`` 1, ``model.ckpt``) stay readable through ``legacy_checkpoint_to_safetensors``: the pickle is read
  with ``weights_only=True`` only, converted to a temporary safetensors file, with a ``DeprecationWarning``.
  ``check_bundle_layout`` refuses mixed or unknown layouts from the manifest alone, before any file is read.

This module depends only on the standard library at import time (torch and safetensors are imported where they are
used), so a notebook can carry it and a worker can vendor it without the rest of the package.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONVERSION_FORMAT = "tabpfn-checkpoint-safetensors-v1"
PROVENANCE_KEY = "dimer_conversion"

BUNDLE_SCHEMA_VERSION = 2
LEGACY_BUNDLE_SCHEMA_VERSION = 1
BUNDLE_CHECKPOINT = "model.safetensors"
LEGACY_BUNDLE_CHECKPOINT = "model.ckpt"
FITTED_ESTIMATOR = "model.tabpfn_fit"
BUNDLE_FORMATS = {"tabular_classification": "tabpfn-dimer-classifier-v2", "tabular_regression": "tabpfn-dimer-regressor-v2"}
V2_ONLY_KEYS = ("bundleFormat", "modelPath", "payloadFiles", "foundationCheckpointBytes")
PICKLE_SUFFIXES = (".ckpt", ".pt", ".pth", ".bin", ".pkl", ".pickle")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_safetensors(data: bytes) -> bytes:
    """Re-emit a safetensors header as sorted, compact JSON padded with spaces to 8 bytes; the data is unchanged.

    ``safetensors.torch.save`` (0.8.0) writes ``__metadata__`` in hash-map order, so equal inputs can give different
    bytes. Tensor offsets are relative to the end of the header, so re-ordering the header keeps them valid."""
    size = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + size])
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 8)
    return len(text).to_bytes(8, "little") + text + data[8 + size :]


def convert_checkpoint(source: str | Path, destination: str | Path, *, expected_sha256: str, source_name: str, expected_converted_sha256: str | None = None) -> dict[str, Any]:
    """Convert a TabPFN ``.ckpt`` to a tabpfn-readable ``.safetensors`` once; return the source and converted digests.

    The source must match ``expected_sha256`` before it is read, and is read with
    ``torch.load(map_location="cpu", weights_only=True)``. With ``expected_converted_sha256`` the file is only moved
    into place when the converted bytes match it."""
    source, destination = Path(source), Path(destination)
    source_sha = sha256_file(source)
    if source_sha != expected_sha256:
        raise ValueError(f"{source.name}: sha256 {source_sha} != pinned {expected_sha256}; refusing to deserialise it")
    import torch
    from safetensors.torch import save

    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("state_dict"), Mapping) or "config" not in checkpoint:
        raise ValueError(f"{source.name}: not a TabPFN checkpoint (needs 'state_dict' and 'config')")
    if PROVENANCE_KEY in checkpoint:
        raise ValueError(f"{source.name}: already carries a {PROVENANCE_KEY!r} field")
    tensors: dict[str, Any] = {}
    for key, value in checkpoint["state_dict"].items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{source.name}: state_dict[{key!r}] is {type(value).__name__}, not a tensor")
        tensors[key] = value.detach().to("cpu").contiguous().clone()
    metadata = {key: json.dumps(value, sort_keys=True) for key, value in checkpoint.items() if key != "state_dict"}
    metadata[PROVENANCE_KEY] = json.dumps({"format": CONVERSION_FORMAT, "source_file": source_name, "source_sha256": source_sha}, sort_keys=True)
    data = canonical_safetensors(save(tensors, metadata=metadata))
    converted_sha = hashlib.sha256(data).hexdigest()
    if expected_converted_sha256 is not None and converted_sha != expected_converted_sha256:
        raise ValueError(f"{source.name}: converted sha256 {converted_sha} != pinned {expected_converted_sha256}; not writing {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, destination)
    return {"source_sha256": source_sha, "converted_sha256": converted_sha, "tensors": len(tensors), "bytes": len(data)}


def convert_written_checkpoint(source: str | Path, destination: str | Path, *, remove_source: bool = True) -> dict[str, Any]:
    """Convert a ``.ckpt`` that tabpfn just wrote in this process (``save_tabpfn_model``) and remove the pickle.

    The file is this process's own output, so its observed SHA-256 is the expected one; it is read once with
    ``torch.load(weights_only=True)``. Returns ``convert_checkpoint``'s facts plus the destination path."""
    source = Path(source)
    facts = convert_checkpoint(source, destination, expected_sha256=sha256_file(source), source_name=source.name)
    if remove_source:
        source.unlink()
    return {**facts, "path": str(destination)}


def read_converted_metadata(path: str | Path) -> dict[str, Any]:
    """The JSON-decoded header metadata of a converted file (what tabpfn's ``Checkpoint`` decodes), without tensors."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        header = handle.metadata() or {}
    metadata = {key: json.loads(value) for key, value in header.items()}
    if (metadata.get(PROVENANCE_KEY) or {}).get("format") != CONVERSION_FORMAT:
        raise ValueError(f"{path}: not a {CONVERSION_FORMAT} file")
    return metadata


# --------------------------------------------------------------------------- bundle formats


def v2_manifest_fields(task_type: str, checkpoint_sha256: str, checkpoint_bytes: int) -> dict[str, Any]:
    """The manifest fields that make a bundle v2: format, checkpoint name, model path, payload files, digest and size."""
    if task_type not in BUNDLE_FORMATS:
        raise ValueError(f"no TabPFN bundle format for taskType {task_type!r}")
    return {
        "schemaVersion": BUNDLE_SCHEMA_VERSION,
        "bundleFormat": BUNDLE_FORMATS[task_type],
        "fittedEstimator": FITTED_ESTIMATOR,
        "foundationCheckpoint": BUNDLE_CHECKPOINT,
        "modelPath": BUNDLE_CHECKPOINT,
        "payloadFiles": sorted([FITTED_ESTIMATOR, BUNDLE_CHECKPOINT]),
        "foundationCheckpointSha256": checkpoint_sha256,
        "foundationCheckpointBytes": int(checkpoint_bytes),
    }


def check_bundle_layout(manifest: Mapping[str, Any]) -> int:
    """Return the bundle layout (2 or 1) from the manifest alone; refuse anything else before a file is read.

    v2: ``schemaVersion`` 2, ``bundleFormat`` for its ``taskType``, and ``foundationCheckpoint``, ``modelPath`` and
    ``payloadFiles`` all naming ``model.safetensors``. v1 (deprecated): ``schemaVersion`` 1, ``foundationCheckpoint``
    ``model.ckpt`` and none of the v2-only fields. A v2 manifest naming a pickle, or a v1 one carrying v2 fields, is mixed."""
    version = manifest.get("schemaVersion")
    checkpoint = manifest.get("foundationCheckpoint")
    if version == BUNDLE_SCHEMA_VERSION:
        expected = BUNDLE_FORMATS.get(str(manifest.get("taskType")))
        problems = []
        if manifest.get("bundleFormat") != expected:
            problems.append(f"bundleFormat={manifest.get('bundleFormat')!r} (expected {expected!r})")
        if checkpoint != BUNDLE_CHECKPOINT or manifest.get("modelPath") != BUNDLE_CHECKPOINT:
            problems.append(f"foundationCheckpoint={checkpoint!r}, modelPath={manifest.get('modelPath')!r} (both must be {BUNDLE_CHECKPOINT!r})")
        if manifest.get("fittedEstimator") != FITTED_ESTIMATOR:
            problems.append(f"fittedEstimator={manifest.get('fittedEstimator')!r}")
        if manifest.get("payloadFiles") != sorted([FITTED_ESTIMATOR, BUNDLE_CHECKPOINT]):
            problems.append(f"payloadFiles={manifest.get('payloadFiles')!r}")
        size = manifest.get("foundationCheckpointBytes")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            problems.append(f"foundationCheckpointBytes={size!r}")
        if problems:
            raise ValueError("refusing a mixed or malformed v2 bundle manifest: " + "; ".join(problems))
        return BUNDLE_SCHEMA_VERSION
    if version == LEGACY_BUNDLE_SCHEMA_VERSION:
        extra = [key for key in V2_ONLY_KEYS if key in manifest]
        if checkpoint != LEGACY_BUNDLE_CHECKPOINT or extra:
            raise ValueError(f"refusing a mixed v1 bundle manifest: foundationCheckpoint={checkpoint!r} (v1 carries {LEGACY_BUNDLE_CHECKPOINT!r}), v2-only fields {extra}")
        return LEGACY_BUNDLE_SCHEMA_VERSION
    raise ValueError(f"unknown bundle layout: schemaVersion={version!r} (supported: {BUNDLE_SCHEMA_VERSION}, deprecated {LEGACY_BUNDLE_SCHEMA_VERSION})")


def bundle_checkpoint_name(layout: int) -> str:
    return BUNDLE_CHECKPOINT if layout == BUNDLE_SCHEMA_VERSION else LEGACY_BUNDLE_CHECKPOINT


def check_bundle_files(root: str | Path, layout: int) -> None:
    """Refuse a bundle directory holding the other layout's checkpoint too (mixed), by name only."""
    other = LEGACY_BUNDLE_CHECKPOINT if layout == BUNDLE_SCHEMA_VERSION else BUNDLE_CHECKPOINT
    if (Path(root) / other).exists():
        raise ValueError(f"refusing a mixed bundle: a v{layout} bundle must not also hold {other}")


def pickle_files(root: str | Path) -> list[str]:
    """Relative paths of pickle-format weight files (``PICKLE_SUFFIXES``) anywhere under ``root``."""
    root = Path(root)
    return sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file() and p.suffix.lower() in PICKLE_SUFFIXES)


def legacy_checkpoint_to_safetensors(source: str | Path, destination: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    """DEPRECATED: read a v1 bundle's ``model.ckpt`` (``weights_only=True`` only) into a safetensors file to load from.

    Emits a ``DeprecationWarning``; the bundle itself is not modified. Kept until the v1 artifacts are re-issued as v2."""
    warnings.warn(
        f"{Path(source).name}: loading a pickle checkpoint from a v1 TabPFN bundle (weights_only=True). This layout is "
        f"deprecated; re-issue the bundle as v2 with {BUNDLE_CHECKPOINT}.",
        DeprecationWarning,
        stacklevel=2,
    )
    return convert_checkpoint(source, destination, expected_sha256=expected_sha256, source_name=Path(source).name)
