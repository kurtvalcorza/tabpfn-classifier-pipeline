from __future__ import annotations

import json
import zipfile
from pathlib import Path

from serving.load_artifact import rewrite_model_path


def test_rewrite_model_path_points_to_companion_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "source.tabpfn_fit"
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(
            "init_params.json",
            json.dumps({"__class_name__": "TabPFNClassifier", "model_path": "/old/cache/model.ckpt"}),
        )
        archive.writestr("fitted_attrs.joblib", b"state")

    destination = tmp_path / "rewritten.tabpfn_fit"
    rewrite_model_path(source, checkpoint, destination)

    with zipfile.ZipFile(destination) as archive:
        params = json.loads(archive.read("init_params.json"))
        assert params["model_path"] == str(checkpoint.resolve())
        assert archive.read("fitted_attrs.joblib") == b"state"


def test_rewrite_rejects_unsafe_members(tmp_path: Path) -> None:
    source = tmp_path / "source.tabpfn_fit"
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("init_params.json", json.dumps({"model_path": "old.ckpt"}))
        archive.writestr("../escape", b"bad")

    try:
        rewrite_model_path(source, checkpoint, tmp_path / "rewritten.tabpfn_fit")
    except ValueError as exc:
        assert "unsafe fitted-artifact member" in str(exc)
    else:
        raise AssertionError("unsafe member should have been rejected")
