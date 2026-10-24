"""Load a DIMER TabPFN fitted artifact together with its companion weights.

TabPFN's fitted-estimator archive intentionally omits foundation weights and records
an estimator ``model_path``. DIMER may relocate artifacts between training and serving,
so this helper rewrites that path in a temporary copy of the fitted archive to point at
the weights beside the persisted artifact.

Bundle format v2 carries ``model.safetensors``, which tabpfn opens with ``safe_open``
(no ``torch.load``). A v1 bundle (``model.ckpt``) is still accepted but deprecated: the
pickle is read once with ``torch.load(weights_only=True)`` into a temporary safetensors
file and a ``DeprecationWarning`` is emitted. A directory holding both, or a manifest
whose layout disagrees with the files, is refused before any weights are read.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import warnings
import zipfile
from pathlib import Path, PurePosixPath

V2_WEIGHTS = "model.safetensors"
V1_WEIGHTS = "model.ckpt"


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


def artifact_weights(artifact_dir: Path) -> tuple[Path, int]:
    """The weights file and bundle layout (2 or 1) of an artifact directory; refuses mixed or unknown layouts.

    With ``artifact_manifest.json`` its ``schemaVersion`` and ``foundationCheckpoint`` decide; without one
    (a bare worker output), exactly one of ``model.safetensors`` / ``model.ckpt`` must be present."""
    present = [name for name in (V2_WEIGHTS, V1_WEIGHTS) if (artifact_dir / name).is_file()]
    if len(present) != 1:
        raise ValueError(
            f"refusing artifact {artifact_dir}: expected exactly one of "
            f"{V2_WEIGHTS} / {V1_WEIGHTS}, found {present}"
        )
    manifest_path = artifact_dir / "artifact_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {2: V2_WEIGHTS, 1: V1_WEIGHTS}.get(manifest.get("schemaVersion"))
        if expected is None or manifest.get("foundationCheckpoint") != expected or present != [expected]:
            raise ValueError(
                f"refusing artifact {artifact_dir}: manifest schemaVersion={manifest.get('schemaVersion')!r} "
                f"foundationCheckpoint={manifest.get('foundationCheckpoint')!r} "
                f"disagrees with the files {present}"
            )
    name = present[0]
    return artifact_dir / name, 2 if name == V2_WEIGHTS else 1


def load_dimer_tabpfn_artifact(
    artifact_dir: str | Path,
    *,
    device: str = "cpu",
):
    """Load the fitted DIMER estimator from an artifact directory.

    Expected files:
      - model.tabpfn_fit
      - model.safetensors (bundle format v2), or model.ckpt (v1, deprecated)

    The original archive is never modified.
    """
    artifact_dir = Path(artifact_dir).resolve()
    fitted = artifact_dir / "model.tabpfn_fit"
    weights, layout = artifact_weights(artifact_dir)

    from tabpfn.model_loading import load_fitted_tabpfn_model

    require_weights_only_load()
    with tempfile.TemporaryDirectory() as temp_dir:
        if layout == 1:
            warnings.warn(
                f"{weights.name}: loading a pickle checkpoint from a v1 TabPFN bundle (weights_only=True). "
                f"This layout is deprecated; re-issue the bundle with {V2_WEIGHTS}.",
                DeprecationWarning,
                stacklevel=2,
            )
            import torch
            from tabpfn.checkpoint import save_as_safetensors

            converted = Path(temp_dir) / "legacy-v1.safetensors"
            save_as_safetensors(torch.load(weights, map_location="cpu", weights_only=True), converted)
            weights = converted
        rewritten = Path(temp_dir) / "model.tabpfn_fit"
        rewrite_model_path(fitted, weights, rewritten)
        return load_fitted_tabpfn_model(rewritten, device=device)
