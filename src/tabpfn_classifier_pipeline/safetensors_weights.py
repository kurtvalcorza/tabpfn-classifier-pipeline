# ruff: noqa: E501  -- long docstrings and messages are kept readable
"""Safetensors serving for the pinned TabPFN checkpoint (tabpfn==8.1.0).

tabpfn 8.1.0 reads a ``.safetensors`` checkpoint itself: ``tabpfn.checkpoint.Checkpoint`` opens it with
``safetensors.safe_open`` and JSON-decodes every header metadata value into the checkpoint dict, and
``tabpfn.model_loading.load_model`` then builds the architecture from ``config`` / ``architecture_name`` and calls
``load_state_dict`` (strict by default). No ``torch.load`` is involved. So the serving path needs no shim: it hands
the converted file to ``TabPFNClassifier(model_path=...)``. This module only writes that file.

``convert_checkpoint`` reads the digest-checked ``.ckpt`` once with ``torch.load(map_location="cpu",
weights_only=True)``, keeps its ``state_dict`` as contiguous CPU tensors and every other top-level field as
``json.dumps(value, sort_keys=True)`` header metadata (the layout of tabpfn's own ``save_as_safetensors``), adds a
``dimer_conversion`` provenance field, and re-emits the header with sorted keys so the bytes are reproducible.

This module depends only on the standard library at import time (torch and safetensors are imported where they are
used), so the notebook can carry it without importing tabpfn.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONVERSION_FORMAT = "tabpfn-checkpoint-safetensors-v1"
PROVENANCE_KEY = "dimer_conversion"


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


def read_converted_metadata(path: str | Path) -> dict[str, Any]:
    """The JSON-decoded header metadata of a converted file (what tabpfn's ``Checkpoint`` decodes), without tensors."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        header = handle.metadata() or {}
    metadata = {key: json.loads(value) for key, value in header.items()}
    if (metadata.get(PROVENANCE_KEY) or {}).get("format") != CONVERSION_FORMAT:
        raise ValueError(f"{path}: not a {CONVERSION_FORMAT} file")
    return metadata
