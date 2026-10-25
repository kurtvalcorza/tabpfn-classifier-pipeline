"""Offline unit tests for src/tabpfn_classifier_pipeline (no tabpfn, no torch, no weights, no network)."""
# ruff: noqa: E501
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabpfn_classifier_pipeline import pipeline as api

ROOT = Path(__file__).resolve().parents[1]


class FakeClassifier:
    """Stands in for tabpfn.TabPFNClassifier: majority vote with class frequencies as probabilities."""

    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None
        self._freq: np.ndarray | None = None
        self.model_path = "fake"

    def get_params(self, deep: bool = False) -> dict:
        return {"model_path": Path(self.model_path), "n_estimators": 4}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FakeClassifier:
        counts = y.astype(str).value_counts().sort_index()
        self.classes_ = np.asarray(counts.index.tolist())
        self._freq = counts.to_numpy(dtype=float) / len(y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self._freq is not None
        return np.tile(self._freq, (len(X), 1))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.classes_ is not None
        return np.asarray([self.classes_[int(np.argmax(self._freq))]] * len(X))


def fake_saver(model: FakeClassifier, path: Path) -> None:
    api._coerce_non_json_init_params(model)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("init_params.json", json.dumps({"__class_name__": "TabPFNClassifier", "model_path": model.model_path}))
        archive.writestr("fitted_attrs.joblib", json.dumps({"classes": [str(c) for c in model.classes_], "freq": list(map(float, model._freq))}) + "x" * 2048)


def fake_converter(checkpoint: Path, destination: Path) -> dict:
    """Stands in for convert_checkpoint_to_safetensors without torch: the checkpoint's bytes as one U8 tensor."""
    blob = checkpoint.read_bytes()
    digest = api.write_safetensors(destination, {"blob": ("U8", [len(blob)], blob)}, {"config": api.metadata_json({"stand_in": True})})
    return {"sha256": digest, "tool": "safetensors", "toolVersion": "0.8.0", "torchVersion": "2.11.0", "converter": api.SAFETENSORS_CONVERTER, "converterVersion": api.SAFETENSORS_CONVERTER_VERSION, "layout": api.SAFETENSORS_LAYOUT, "tensorCount": 1, "tensorBytes": len(blob), "metadataKeys": ["config"], "bitwiseEqual": True}


def fake_loader(fitted: Path, device: str) -> FakeClassifier:
    with zipfile.ZipFile(fitted) as archive:
        params = json.loads(archive.read("init_params.json"))
        state = json.loads(archive.read("fitted_attrs.joblib").decode("utf-8").rstrip("x"))
    assert Path(params["model_path"]).name == api.SAFETENSORS_NAME  # rewritten to the companion safetensors conversion
    model = FakeClassifier()
    model.classes_ = np.asarray(state["classes"])
    model._freq = np.asarray(state["freq"], dtype=float)
    return model


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(60, 3)), columns=["f1", "f2", "f3"])
    frame["cat"] = ["a" if i % 2 else "b" for i in range(60)]
    frame["target"] = ["x" if i % 3 else "y" for i in range(60)]
    return frame.iloc[:45].reset_index(drop=True), frame.iloc[45:].reset_index(drop=True)


def _snapshot(tmp_path: Path, checkpoint_bytes: bytes = b"c" * 4096) -> Path:
    root = tmp_path / "weights"
    root.mkdir()
    (root / api.WEIGHTS_FILE).write_bytes(checkpoint_bytes)
    (root / "config.json").write_bytes(b'{"model_name": "TabPFN-v3"}')
    manifest = {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": api.MODEL_KEY,
        "modelId": api.MODEL_ID,
        "revision": api.MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 27, "sha256": api.sha256_hex(b'{"model_name": "TabPFN-v3"}')},
            {"path": api.WEIGHTS_FILE, "bytes": len(checkpoint_bytes), "sha256": api.sha256_hex(checkpoint_bytes)},
        ],
        "totalBytes": 27 + len(checkpoint_bytes),
    }
    (root / api.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return root


@pytest.fixture
def constants(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"c" * 4096
    monkeypatch.setattr(api, "WEIGHTS_SHA256", api.sha256_hex(payload))
    monkeypatch.setattr(api, "WEIGHTS_BYTES", len(payload))


def test_identity_constants_and_committed_manifest_agree() -> None:
    manifest = json.loads((ROOT / "weights" / api.MODEL_KEY / api.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert (manifest["modelId"], manifest["revision"], manifest["modelKey"]) == (api.MODEL_ID, api.MODEL_REVISION, api.MODEL_KEY)
    entry = {e["path"]: e for e in manifest["files"]}[api.WEIGHTS_FILE]
    assert (entry["sha256"], entry["bytes"]) == (api.WEIGHTS_SHA256, api.WEIGHTS_BYTES)
    assert len(api.MODEL_REVISION) == 40 and api.MODEL_VERSION == "v3"
    assert (ROOT / "weights" / api.MODEL_KEY / "config.json").read_text(encoding="utf-8").strip().endswith("}")


def test_verify_snapshot_and_rejections(tmp_path: Path, constants: None) -> None:
    root = _snapshot(tmp_path)
    verify = api.verify_snapshot(root)
    assert verify["modelId"] == api.MODEL_ID
    (root / api.WEIGHTS_FILE).write_bytes(b"d" * 4096)
    with pytest.raises(ValueError, match="sha256"):
        api.verify_snapshot(root)
    (root / api.WEIGHTS_FILE).unlink()
    with pytest.raises(FileNotFoundError):
        api.verify_snapshot(root)
    other = tmp_path / "wrong"
    other.mkdir()
    (other / api.MANIFEST_NAME).write_text(json.dumps({"modelId": "x/y", "revision": "0" * 40, "files": [{"path": "a", "bytes": 1, "sha256": "0" * 64}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected"):
        api.verify_snapshot(other)


def test_stage_missing_files_uses_injected_downloader(tmp_path: Path, constants: None) -> None:
    root = _snapshot(tmp_path)
    (root / api.WEIGHTS_FILE).unlink()
    with pytest.raises(FileNotFoundError, match="allow_download"):
        api.stage_missing_files(root)
    calls: list[dict] = []

    def downloader(**kwargs):
        calls.append(kwargs)
        (Path(kwargs["local_dir"]) / kwargs["filename"]).write_bytes(b"c" * 4096)

    assert api.stage_missing_files(root, allow_download=True, downloader=downloader) == [api.WEIGHTS_FILE]
    assert calls == [{"repo_id": api.MODEL_ID, "filename": api.WEIGHTS_FILE, "revision": api.MODEL_REVISION, "local_dir": str(root)}]
    assert api.stage_missing_files(root, allow_download=True, downloader=downloader) == []
    assert api.verify_snapshot(root)["files"]


def test_from_pretrained_binds_the_verified_checkpoint_without_tabpfn(tmp_path: Path, constants: None) -> None:
    root = _snapshot(tmp_path)
    pipe = api.TabPFNClassifierPipeline.from_pretrained(device="cpu", weights_dir=root)
    assert pipe.weights_path == root / api.WEIGHTS_FILE and pipe.source == "local-snapshot" and pipe.classes is None
    with pytest.raises(RuntimeError, match="fit"):
        pipe.predict(pd.DataFrame({"f1": [1.0]}))


def test_synthetic_dataset_reads_back_and_validates(tmp_path: Path) -> None:
    out = api.build_synthetic_dataset(tmp_path / "synthetic.zip", rows=120, features=4, classes=3, seed=7)
    frames = api.read_dataset_zip(out)
    assert set(frames) == {"train.csv", "val.csv", "test.csv"} and len(frames["train.csv"]) + len(frames["val.csv"]) + len(frames["test.csv"]) == 120
    manifest = api.validate_inputs(frames["train.csv"], "target", val=frames["val.csv"], test=frames["test.csv"], names=["synthetic"])
    assert manifest["verdict"] == "accepted" and manifest["inputs"][0]["classes"] == ["class_0", "class_1", "class_2"]
    assert manifest["inputs"][0]["feature_columns"] == [c for c in frames["train.csv"].columns if c != "target"]
    assert set(manifest["inputs"][0]["splits"]) == {"train", "val", "test"} and manifest["model_revision"] == api.MODEL_REVISION
    assert api.read_dataset_dir(tmp_path)["train.csv"].shape == frames["train.csv"].shape


def test_dataset_zip_safety_rules(tmp_path: Path) -> None:
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("train.csv", "f1,target\n1,a\n2,b\n")
        archive.writestr("../escape.csv", "x")
    with pytest.raises(ValueError, match="unsafe archive member"):
        api.read_dataset_zip(bad)
    bomb = tmp_path / "bomb.zip"
    with zipfile.ZipFile(bomb, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("train.csv", "f1,target\n" + "1,a\n" * 200000)
    with pytest.raises(ValueError, match="ratio"):
        api.read_dataset_zip(bomb, max_ratio=5)
    with pytest.raises(ValueError, match="ceiling"):
        api.read_dataset_zip(bomb, max_expanded_bytes=1000)
    with pytest.raises(ValueError, match="unsafe archive member"):
        api.safe_extract_zip(bad, tmp_path / "out")
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as archive:
        archive.writestr("a.txt", "x")
        archive.writestr("sub/b.txt", "y")
    assert api.safe_extract_zip(good, tmp_path / "out") == ["a.txt", "b.txt"] and (tmp_path / "out" / "b.txt").read_text() == "y"


def test_validate_inputs_rejections_and_warnings() -> None:
    train, val = _frames()
    with pytest.raises(api.InputRejected) as info:
        api.validate_inputs(train.rename(columns={"target": "label"}), "target")
    assert info.value.finding["code"] == "TARGET_MISSING"
    dup = train.copy()
    dup.columns = ["f1", "f1", "f3", "cat", "target"]
    with pytest.raises(api.InputRejected) as info:
        api.validate_inputs(dup, "target")
    assert info.value.finding["code"] == "DUPLICATE_COLUMNS"
    one = train.assign(target="x")
    with pytest.raises(api.InputRejected) as info:
        api.validate_inputs(one, "target")
    assert info.value.finding["code"] == "TOO_FEW_CLASSES"
    with pytest.raises(api.InputRejected) as info:
        api.validate_inputs(train, "target", val=val.drop(columns=["cat"]))
    assert info.value.finding["code"] == "SCHEMA_MISMATCH"
    with pytest.raises(api.InputRejected) as info:
        api.validate_inputs(train, "target", val=val.assign(prediction="x"))
    assert info.value.finding["code"] == "RESERVED_COLUMNS"
    manifest = api.validate_inputs(train, "target", val=val.assign(target="z"))
    assert [f["code"] for f in manifest["findings"]] == ["UNSEEN_CLASSES"]
    imbalanced = train.assign(target=["x"] * 43 + ["y"] * 2)
    assert "CLASS_IMBALANCE" in [f["code"] for f in api.validate_inputs(imbalanced, "target")["findings"]]


def test_validate_new_rows_schema() -> None:
    train, _ = _frames()
    features = ["f1", "f2", "f3", "cat"]
    assert list(api.validate_new_rows(train[features], features, target_column="target", classes=["x", "y"]).columns) == features
    with pytest.raises(api.InputRejected, match="target/prediction/proba"):
        api.validate_new_rows(train, features, target_column="target", classes=["x", "y"])
    with pytest.raises(api.InputRejected, match="schema mismatch"):
        api.validate_new_rows(train[["f1", "f2"]], features)
    with pytest.raises(api.InputRejected, match="infinite"):
        api.validate_new_rows(train[features].assign(f1=np.inf), features)


def test_fit_predict_evaluate_save_and_reload(tmp_path: Path, constants: None) -> None:
    root = _snapshot(tmp_path)
    train, val = _frames()
    features = ["f1", "f2", "f3", "cat"]
    pipe = api.TabPFNClassifierPipeline.from_pretrained(device="cpu", weights_dir=root).fit(train[features], train["target"], estimator_factory=FakeClassifier)
    assert pipe.classes == ["x", "y"] and pipe.feature_columns == features and pipe.source == "in-context-fit"
    predictions = pipe.predict(val[features])
    assert list(predictions.columns) == ["row_id", "prediction", "proba_x", "proba_y"] and predictions.attrs["decision_rule"] == "argmax"
    assert np.allclose(predictions[["proba_x", "proba_y"]].sum(axis=1), 1.0)
    metrics = pipe.evaluate(val[features], val["target"])
    assert set(metrics) >= {"accuracy", "balancedAccuracy", "logLoss"} and 0.0 <= metrics["accuracy"] <= 1.0
    with pytest.raises(api.InputRejected):
        pipe.predict(val)  # carries the target column

    artifact = tmp_path / "artifact"
    manifest = pipe.save_artifact(artifact, saver=fake_saver, converter=fake_converter)
    assert manifest["classes"] == ["x", "y"] and manifest["baseModel"]["sha256"] == api.WEIGHTS_SHA256
    assert api.sha256_file(artifact / api.CHECKPOINT_NAME) == api.WEIGHTS_SHA256
    provenance = manifest["safetensorsConversion"]
    assert manifest["safetensorsWeights"] == api.SAFETENSORS_NAME and manifest["safetensorsWeightsSha256"] == api.sha256_file(artifact / api.SAFETENSORS_NAME)
    assert provenance["derived_from"] == {"file": api.CHECKPOINT_NAME, "sha256": api.WEIGHTS_SHA256} and provenance["converter"] == api.SAFETENSORS_CONVERTER
    assert (provenance["tool"], provenance["toolVersion"], provenance["layout"], provenance["ser9Clause"]) == ("safetensors", "0.8.0", api.SAFETENSORS_LAYOUT, "b")
    assert provenance["bitwiseEqual"] is True and provenance["tensorCount"] == 1 and "sha256" not in provenance
    checked = api.validate_artifact_bundle(artifact)
    assert checked["verifiedSha256"]["fittedEstimator"] == manifest["fittedEstimatorSha256"] and checked["recordedModelPath"] == "fake"
    with pytest.raises(ValueError, match="expected"):
        api.validate_artifact_bundle(artifact, expected_checkpoint_sha256="0" * 64)

    bundle_zip = tmp_path / "bundle.zip"
    assert api.zip_artifact_bundle(artifact, bundle_zip) == api.sha256_file(bundle_zip)
    assert sorted(api.safe_extract_zip(bundle_zip, tmp_path / "unzipped")) == sorted(api.ARTIFACT_MEMBERS)
    assert api.validate_artifact_bundle(tmp_path / "unzipped")["classes"] == manifest["classes"]
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    for name in api.ARTIFACT_MEMBERS:
        (fresh / name).write_bytes((artifact / name).read_bytes())
    reloaded = api.TabPFNClassifierPipeline.from_artifact(fresh, device="cpu", loader=fake_loader)
    assert reloaded.source == "artifact" and reloaded.classes == ["x", "y"] and reloaded.feature_columns == features
    assert reloaded.evaluate(val[features], val["target"])["accuracy"] == metrics["accuracy"]
    (fresh / api.CHECKPOINT_NAME).write_bytes(b"tampered" * 512)
    with pytest.raises(ValueError, match="SHA-256"):
        api.TabPFNClassifierPipeline.from_artifact(fresh, device="cpu", loader=fake_loader)


def test_rewrite_model_path_rejects_unsafe_members(tmp_path: Path) -> None:
    source = tmp_path / "s.tabpfn_fit"
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("init_params.json", json.dumps({"model_path": "old.ckpt"}))
        archive.writestr("../escape", b"bad")
    with pytest.raises(ValueError, match="unsafe archive member"):
        api.rewrite_model_path(source, checkpoint, tmp_path / "r.tabpfn_fit")
    assert api.manifest_digest({"sha256": {"fittedEstimator": "a" * 64}}, "fittedEstimator") == "a" * 64


def test_baseline_and_evaluation_report() -> None:
    baseline = api.majority_class_baseline(["x", "x", "y"], ["x", "y", "y"], classes=["x", "y"])
    assert baseline["majorityClass"] == "x" and abs(baseline["accuracy"] - 1 / 3) < 1e-9 and baseline["logLoss"] > 0
    report = api.evaluation_report({"accuracy": 0.9, "balancedAccuracy": 0.85, "logLoss": 0.3, "rocAuc": 0.95, "extra": 1.0}, baseline=baseline, n_validation=10, class_labels=["x", "y"], reload_check={"accuracyMatches": True})
    assert report["verdict"] == "sample-sanity" and [m["id"] for m in report["metrics"]] == ["accuracy", "balancedAccuracy", "logLoss", "rocAuc"]
    assert report["baselines"]["majority_class"]["majorityClass"] == "x" and report["adaptation"].startswith("in-context")
    empty = api.evaluation_report(None, n_validation=0)
    assert empty["verdict"] == "not-measurable" and empty["metrics"] == [] and "needs" in empty


# --------------------------------------------------------------------------- re-export (ARTIFACT_SPEC 2.0 SER9 (b))


def legacy_loader(fitted: Path, device: str) -> FakeClassifier:
    """``fake_loader`` for a pre-conversion bundle, whose weights come from the digest-checked ``model.ckpt``."""
    with zipfile.ZipFile(fitted) as archive:
        params = json.loads(archive.read("init_params.json"))
        state = json.loads(archive.read("fitted_attrs.joblib").decode("utf-8").rstrip("x"))
    assert Path(params["model_path"]).name == api.CHECKPOINT_NAME
    model = FakeClassifier()
    model.classes_ = np.asarray(state["classes"])
    model._freq = np.asarray(state["freq"], dtype=float)
    return model


def pin_checkpoint(monkeypatch: pytest.MonkeyPatch, checkpoint: Path) -> None:
    """Test seam: treat ``checkpoint`` as the pinned upstream file (``save_artifact`` refuses any other)."""
    monkeypatch.setattr(api, "WEIGHTS_SHA256", api.sha256_file(checkpoint))


def _legacy(art: Path) -> Path:
    """Turn ``art`` into a pre-conversion bundle like the pinned v1 sample: no ``model.safetensors``."""
    path = art / api.ARTIFACT_MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key in ("safetensorsWeights", "safetensorsWeightsSha256", "safetensorsConversion"):
        manifest.pop(key)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (art / api.SAFETENSORS_NAME).unlink()
    return art


def _exported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpoint: Path | None = None, converter=fake_converter) -> Path:
    if checkpoint is None:
        checkpoint = tmp_path / "upstream.ckpt"
        checkpoint.write_bytes(b"c" * 4096)
    pin_checkpoint(monkeypatch, checkpoint)
    train, _ = _frames()
    features = ["f1", "f2", "f3", "cat"]
    pipe = api.TabPFNClassifierPipeline(weights_path=checkpoint).fit(train[features], train["target"], estimator_factory=FakeClassifier)
    art = tmp_path / "artifact"
    pipe.save_artifact(art, saver=fake_saver, converter=converter)
    return art


def test_a_reloaded_bundle_reexports_the_authentic_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    art = _exported(tmp_path, monkeypatch)
    reloaded = api.TabPFNClassifierPipeline.from_artifact(art, device="cpu", loader=fake_loader)
    again = tmp_path / "again"
    manifest = reloaded.save_artifact(again, saver=fake_saver, converter=fake_converter)
    assert api.sha256_file(again / api.CHECKPOINT_NAME) == api.sha256_file(art / api.CHECKPOINT_NAME) == api.WEIGHTS_SHA256
    assert (again / api.SAFETENSORS_NAME).read_bytes() == (art / api.SAFETENSORS_NAME).read_bytes()
    assert manifest["foundationCheckpointSha256"] == api.WEIGHTS_SHA256
    api.validate_artifact_bundle(again)
    assert reloaded.weights_path == art / api.CHECKPOINT_NAME


def test_a_reloaded_legacy_bundle_reexports_its_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    legacy = _legacy(_exported(tmp_path, monkeypatch))
    with pytest.raises(ValueError, match="safetensorsWeights"):
        api.TabPFNClassifierPipeline.from_artifact(legacy, device="cpu", loader=legacy_loader)
    reloaded = api.TabPFNClassifierPipeline.from_artifact(legacy, device="cpu", loader=legacy_loader, require_safetensors=False)
    again = tmp_path / "again"
    reloaded.save_artifact(again, saver=fake_saver, converter=fake_converter)
    assert api.sha256_file(again / api.CHECKPOINT_NAME) == api.sha256_file(legacy / api.CHECKPOINT_NAME) == api.WEIGHTS_SHA256
    api.validate_artifact_bundle(again)  # the re-export is a full SER9 (b) bundle
    assert reloaded.weights_path == legacy / api.CHECKPOINT_NAME


def test_export_refuses_a_checkpoint_that_is_not_the_pinned_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    art = _exported(tmp_path, monkeypatch)
    other = tmp_path / "other.ckpt"
    other.write_bytes(b"d" * 4096)
    train, _ = _frames()
    features = ["f1", "f2", "f3", "cat"]
    pipe = api.TabPFNClassifierPipeline(weights_path=other).fit(train[features], train["target"], estimator_factory=FakeClassifier)
    with pytest.raises(ValueError, match="not the pinned"):
        pipe.save_artifact(tmp_path / "fresh", saver=fake_saver, converter=fake_converter)
    assert not (tmp_path / "fresh").exists()
    reloaded = api.TabPFNClassifierPipeline.from_artifact(art, device="cpu", loader=fake_loader)
    monkeypatch.setattr(api, "WEIGHTS_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="not the pinned"):
        reloaded.save_artifact(tmp_path / "again", saver=fake_saver, converter=fake_converter)
    assert not (tmp_path / "again").exists()


@pytest.mark.integration
def test_a_reloaded_bundle_reexports_through_the_real_converter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from test_safetensors_conversion import _synthetic_checkpoint

    ckpt = tmp_path / "upstream.ckpt"
    _synthetic_checkpoint(ckpt)
    art = _exported(tmp_path, monkeypatch, checkpoint=ckpt, converter=None)
    reloaded = api.TabPFNClassifierPipeline.from_artifact(art, device="cpu", loader=fake_loader)
    again = tmp_path / "again"
    reloaded.save_artifact(again, saver=fake_saver)  # torch.load(weights_only=True) of the authentic model.ckpt
    assert api.sha256_file(again / api.CHECKPOINT_NAME) == api.sha256_file(ckpt) == api.WEIGHTS_SHA256
    assert (again / api.SAFETENSORS_NAME).read_bytes() == (art / api.SAFETENSORS_NAME).read_bytes()
    api.validate_artifact_bundle(again, verify_conversion=True)
    # legacy (pre-conversion) bundle: weights come from model.ckpt and the re-export gains the conversion
    legacy = _legacy(art)
    reloaded = api.TabPFNClassifierPipeline.from_artifact(legacy, device="cpu", loader=legacy_loader, require_safetensors=False)
    third = tmp_path / "third"
    reloaded.save_artifact(third, saver=fake_saver)
    assert api.sha256_file(third / api.CHECKPOINT_NAME) == api.WEIGHTS_SHA256
    assert (third / api.SAFETENSORS_NAME).read_bytes() == (again / api.SAFETENSORS_NAME).read_bytes()
    api.validate_artifact_bundle(third, verify_conversion=True)
