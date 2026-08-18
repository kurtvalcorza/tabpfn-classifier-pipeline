"""Load a DIMER TabPFN fitted artifact together with its companion checkpoint.

TabPFN's fitted-estimator archive intentionally omits foundation weights and records
an estimator ``model_path``. DIMER may relocate artifacts between training and serving,
so this helper rewrites that path in a temporary copy of the fitted archive to point at
the checkpoint beside the persisted artifact.
"""
from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def _safe_member(name: str) -> None:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe fitted-artifact member: {name!r}")


def rewrite_model_path(
    fitted_archive: Path,
    checkpoint: Path,
    destination: Path,
) -> Path:
    """Copy a ``.tabpfn_fit`` archive while replacing its init model path."""
    fitted_archive = fitted_archive.resolve()
    checkpoint = checkpoint.resolve()
    destination = destination.resolve()
    if not fitted_archive.is_file():
        raise FileNotFoundError(f"fitted estimator not found: {fitted_archive}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"companion checkpoint not found: {checkpoint}")

    saw_init = False
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(fitted_archive, "r") as source, zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED
    ) as target:
        for info in source.infolist():
            _safe_member(info.filename)
            payload = source.read(info.filename)
            if info.filename == "init_params.json":
                params = json.loads(payload.decode("utf-8"))
                params["model_path"] = str(checkpoint)
                payload = json.dumps(params, sort_keys=True).encode("utf-8")
                saw_init = True
            target.writestr(info, payload)
    if not saw_init:
        destination.unlink(missing_ok=True)
        raise ValueError("fitted artifact does not contain init_params.json")
    return destination


def load_dimer_tabpfn_artifact(
    artifact_dir: str | Path,
    *,
    device: str = "cpu",
):
    """Load the fitted DIMER estimator from an artifact directory.

    Expected files:
      - model.tabpfn_fit
      - model.ckpt

    The original archive is never modified.
    """
    artifact_dir = Path(artifact_dir).resolve()
    fitted = artifact_dir / "model.tabpfn_fit"
    checkpoint = artifact_dir / "model.ckpt"

    from tabpfn.model_loading import load_fitted_tabpfn_model

    with tempfile.TemporaryDirectory() as temp_dir:
        rewritten = Path(temp_dir) / "model.tabpfn_fit"
        rewrite_model_path(fitted, checkpoint, rewritten)
        return load_fitted_tabpfn_model(rewritten, device=device)
