"""Load a DIMER TabPFN fitted artifact together with its companion checkpoint.

TabPFN's fitted-estimator archive intentionally omits foundation weights and records
an estimator ``model_path``. DIMER may relocate artifacts between training and serving,
so this helper rewrites that path in a temporary copy of the fitted archive to point at
the checkpoint beside the persisted artifact.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


# tabpfn 8.1.0 reads checkpoints with torch.load and weights_only=None; that is restricted only on
# torch>=2.6 and only while TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD is unset. Same check as the package's
# require_weights_only_load.
def require_weights_only_load(torch_version: str | None = None) -> None:
    """Refuse to continue unless a TabPFN checkpoint would load with weights_only=True."""
    if os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "").strip().lower() in {"1", "y", "yes", "true"}:
        raise RuntimeError(
            "Refusing to load a TabPFN checkpoint: TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD is set, "
            "which would make torch.load unpickle it without weights_only; unset it."
        )
    if torch_version is None:
        import torch

        torch_version = torch.__version__
    match = re.match(r"(\d+)\.(\d+)", torch_version)
    if match is None or (int(match[1]), int(match[2])) < (2, 6):
        raise RuntimeError(
            f"Refusing to load a TabPFN checkpoint: torch {torch_version} is older than 2.6, "
            "so torch.load would not default to weights_only=True; install torch>=2.6."
        )


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

    require_weights_only_load()
    with tempfile.TemporaryDirectory() as temp_dir:
        rewritten = Path(temp_dir) / "model.tabpfn_fit"
        rewrite_model_path(fitted, checkpoint, rewritten)
        return load_fitted_tabpfn_model(rewritten, device=device)
