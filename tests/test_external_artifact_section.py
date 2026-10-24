"""The E2E notebook's closing external-artifact section (NOTEBOOK_SPEC 3.0 §19.1, XAS1–XAS7, SART1–SART11).

Offline and model-free (CI's light pins only): the pin, the digest refusal before extraction, the member and SART10
contract checks, that the section's cells use only names the setup cells define (XAS5), the `outputs/external_artifact/`
path (XAS6), the exact-URL ST1 rule, and a full run of the three section stages against a locally built stand-in artifact
served from a ``file://`` URL with a test-only URL/digest override (the model loader is a stand-in; no tabpfn, no torch).
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import builtins
import contextlib
import functools
import hashlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import shutil
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
E2E = ROOT / "tutorials" / "tabpfn_classifier_colab.ipynb"
PINNED_URL = "https://github.com/kurtvalcorza/tabpfn-classifier-pipeline/releases/download/sample-artifact-synthetic-v1/tabpfn_classifier_artifact.zip"
PINNED_SHA256 = "8fe248e2f9841c452316d07ee3e88e5fcde89ec458ff630954028b1c22df1054"
PINNED_FITTED_SHA256 = "1f48ecaf118c834633c8c000579e3f68227cf11c9ae0b146b68509f6f3bb9af6"
SECTION_START = "## 10. Using an artifact made elsewhere"
SECTION_END = "## 11. Optional activity"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGES = _load("xas_tutorial_stages", TOOLS / "tutorial_stages.py")
VALIDATOR = _load("xas_validate_release_assets", TOOLS / "validate_release_assets.py")


def _src(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _cells() -> list[dict]:
    return json.loads(E2E.read_text(encoding="utf-8"))["cells"]


def _section_cells() -> list[dict]:
    cells = _cells()
    start = next(i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and _src(c).startswith(SECTION_START))
    end = next(i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and _src(c).startswith(SECTION_END))
    return cells[start:end]


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    root.mkdir(parents=True)
    try:
        (root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    except OSError:  # Windows without the symlink privilege
        shutil.copytree(ROOT / "src", root / "src")
    return root


def _quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


# ---------------------------------------------------------------- the pin (SART6, SART8, XAS2)


def test_section_pins_the_release_asset_by_url_and_digests() -> None:
    assert STAGES.SAMPLE_ARTIFACT_URL == PINNED_URL
    assert STAGES.SAMPLE_ARTIFACT_SHA256 == PINNED_SHA256
    assert STAGES.SAMPLE_ARTIFACT_FITTED_SHA256 == PINNED_FITTED_SHA256
    assert STAGES.SAMPLE_ARTIFACT_PRODUCER["commit"].startswith("7fce7f9")
    assert STAGES.SAMPLE_ARTIFACT_PRODUCER["release_tag"] == STAGES.SAMPLE_ARTIFACT_TAG == "sample-artifact-synthetic-v1"


def test_default_path_fetches_the_pin_and_never_this_runs_export(tmp_path, monkeypatch) -> None:
    """XAS2: the default source downloads the pinned URL at the pinned digest; the current run's export is never read."""
    root = _root(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / f"{STAGES.STEM}_artifact.zip").write_bytes(b"this run's export: must never be opened here")
    seen = []

    def fake_fetch(cache, url, expected):
        seen.append((cache, url, expected))
        raise RuntimeError("stop after the fetch")

    monkeypatch.setattr(STAGES, "fetch_pinned_artifact", fake_fetch)
    run = STAGES.Run(root, outputs, tmp_path / "weights", {})
    with pytest.raises(RuntimeError, match="stop after the fetch"):
        _quiet(STAGES.stage_external_artifact, run)
    assert seen == [(root / "inputs" / "external_artifact.zip", PINNED_URL, PINNED_SHA256)]
    for fn in (STAGES.stage_external_artifact, STAGES.stage_external_reconstruct, STAGES.stage_external_predict, STAGES.external_serving, STAGES.external_sample_rows, STAGES.synthetic_contract):
        body = inspect.getsource(fn)
        for main_path in ("{STEM}_artifact.zip", "artifact-reload", "export_reference", "read_frame", "tables(", "data.json", "validated.json", "condition.json", "run.out / \"artifact\""):
            assert main_path not in body, (fn.__name__, main_path)


# ---------------------------------------------------------------- integrity (XAS4, VER6)


def _tiny_zip(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("artifact_manifest.json", "{}")
    return path


def test_pinned_digest_mismatch_refuses_before_extraction(tmp_path) -> None:
    archive = _tiny_zip(tmp_path / "tampered.zip")
    cache = tmp_path / "cache" / "external_artifact.zip"
    with pytest.raises(ValueError, match=r"Trusted digest mismatch .* the pinned SAMPLE_ARTIFACT_SHA256 is .*nothing was extracted"):
        STAGES.fetch_pinned_artifact(cache, archive.resolve().as_uri(), PINNED_SHA256)
    assert not cache.exists() and not cache.with_name(cache.name + ".part").exists()


def test_pinned_digest_match_is_kept_and_reused(tmp_path) -> None:
    archive = _tiny_zip(tmp_path / "standin.zip")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    cache = tmp_path / "cache" / "external_artifact.zip"
    assert STAGES.fetch_pinned_artifact(cache, archive.resolve().as_uri(), digest) == cache
    archive.unlink()  # a second call reuses the verified cache without the network
    assert STAGES.fetch_pinned_artifact(cache, "file:///nonexistent.zip", digest) == cache


def test_own_artifact_with_a_wrong_digest_is_refused_before_extraction(tmp_path) -> None:
    root = _root(tmp_path)
    archive = _tiny_zip(tmp_path / "mine.zip")
    run = STAGES.Run(root, tmp_path / "outputs", tmp_path / "weights", {"source": "own", "zip_path": str(archive), "expected_zip_sha256": "0" * 64})
    with pytest.raises(ValueError, match="EXPECTED_ZIP_SHA256 is 0{64}.*nothing was extracted"):
        _quiet(STAGES.stage_external_artifact, run)
    assert not (root / "state" / "external_bundle").exists()
    assert not (tmp_path / "outputs" / STAGES.EXTERNAL_DIR).exists()
    with pytest.raises(ValueError, match="must be 64 hexadecimal characters"):
        STAGES.check_trusted_digest(archive, "abc", "EXPECTED_ZIP_SHA256")


# ---------------------------------------------------------------- a stand-in artifact (model-free)


class _Estimator:
    classes_ = np.array(["class_0", "class_1", "class_2"])

    def predict_proba(self, X):
        return np.tile([0.6, 0.3, 0.1], (len(X), 1))


def _standin(tmp_path: Path, P, monkeypatch, *, features: list[str] | None = None, legacy: bool = False) -> tuple[Path, str]:
    """A three-member artifact with a stand-in fitted archive and weights; the package's pinned digest for the layout
    (``CONVERTED_SHA256`` for v2, ``WEIGHTS_SHA256`` for the deprecated v1) is overridden (test only) to the stand-in's,
    so ``validate_artifact_bundle`` runs unchanged. ``legacy=True`` builds a v1 ``model.ckpt`` bundle."""
    art = tmp_path / ("standin-artifact-v1" if legacy else "standin-artifact")
    art.mkdir()
    weights_name = P.LEGACY_CHECKPOINT_NAME if legacy else P.CHECKPOINT_NAME
    with zipfile.ZipFile(art / P.FITTED_NAME, "w") as fitted:
        fitted.writestr("init_params.json", json.dumps({"model_path": "/producer/model.ckpt"}))
        fitted.writestr("padding.bin", b"\0" * 2048)
    (art / weights_name).write_bytes(hashlib.sha256(b"stand-in checkpoint").digest() * 128)
    ckpt_sha = hashlib.sha256((art / weights_name).read_bytes()).hexdigest()
    pinned_name = "WEIGHTS_SHA256" if legacy else "CONVERTED_SHA256"
    monkeypatch.setattr(P, pinned_name, ckpt_sha)
    monkeypatch.setattr(P.pipeline, pinned_name, ckpt_sha)  # pinned_checkpoint_sha256 reads the module's pin
    features = features or [f"feature_{i:02d}" for i in range(12)]
    manifest = {
        "taskType": P.TASK_TYPE, "targetColumn": "target", "featureColumns": features,
        "classes": ["class_0", "class_1", "class_2"], 
        "fittedEstimatorSha256": hashlib.sha256((art / P.FITTED_NAME).read_bytes()).hexdigest(),
        "baseModel": {"modelId": P.MODEL_ID, "revision": P.MODEL_REVISION, "file": P.WEIGHTS_FILE, "sha256": P.WEIGHTS_SHA256}, "mode": "zero-shot-icl",
        "nEstimators": 4, "randomState": 42, "featureKinds": {c: "numeric" for c in features},
    }
    if legacy:
        manifest.update({"schemaVersion": P.LEGACY_ARTIFACT_SCHEMA_VERSION, "fittedEstimator": P.FITTED_NAME, "foundationCheckpoint": P.LEGACY_CHECKPOINT_NAME, "foundationCheckpointSha256": ckpt_sha})
        monkeypatch.setattr(P.safetensors_weights, "legacy_checkpoint_to_safetensors", lambda source, destination, **kw: Path(destination).write_bytes(b"converted"))
    else:
        manifest.update(P.safetensors_weights.v2_manifest_fields(P.TASK_TYPE, ckpt_sha, (art / weights_name).stat().st_size))
    (art / P.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    zip_path = tmp_path / ("tabpfn_classifier_artifact_v1.zip" if legacy else "tabpfn_classifier_artifact.zip")
    P.zip_artifact_bundle(art, zip_path)
    original = P.TabPFNClassifierPipeline.from_artifact.__func__
    monkeypatch.setattr(P.TabPFNClassifierPipeline, "from_artifact", classmethod(functools.partial(original, loader=lambda fitted, device: _Estimator())))
    return zip_path, manifest["fittedEstimatorSha256"]


def _cpu(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(__version__="2.11.0+cpu", Tensor=type("Tensor", (), {}), cuda=types.SimpleNamespace(is_available=lambda: False)))
    real = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "8.1.0" if name == "tabpfn" else real(name))


def test_section_runs_end_to_end_on_a_standin_artifact(tmp_path, monkeypatch) -> None:
    """The three stages, in order, against a stand-in served from file:// with a test-only URL/digest override:
    digest before extraction, members, bundle digests, base model, SART10 contract, reconstruction, prediction."""
    _cpu(monkeypatch)
    root = _root(tmp_path)
    P = STAGES.package(root)
    zip_path, fitted = _standin(tmp_path, P, monkeypatch)
    monkeypatch.setattr(STAGES, "SAMPLE_ARTIFACT_URL", zip_path.resolve().as_uri())
    monkeypatch.setattr(STAGES, "SAMPLE_ARTIFACT_SHA256", hashlib.sha256(zip_path.read_bytes()).hexdigest())
    monkeypatch.setattr(STAGES, "SAMPLE_ARTIFACT_FITTED_SHA256", fitted)
    outputs = tmp_path / "outputs"
    run = STAGES.Run(root, outputs, tmp_path / "weights", {"source": "pinned"})
    _quiet(STAGES.stage_external_artifact, run)
    record = json.loads((run.state / "external_artifact.json").read_text())
    assert record["trusted_digest"].startswith("verified (pinned") and record["members"] == sorted([P.ARTIFACT_MANIFEST_NAME, P.CHECKPOINT_NAME, P.FITTED_NAME])
    assert record["data_contract"].startswith("synthetic default path")
    run.options = {}
    _quiet(STAGES.stage_external_reconstruct, run)
    run.options = {"rows_path": ""}
    _quiet(STAGES.stage_external_predict, run)
    written = sorted(p.relative_to(outputs).as_posix() for p in outputs.rglob("*") if p.is_file())
    assert written == ["external_artifact/tabpfn_classifier_external_predictions.csv", "external_artifact/tabpfn_classifier_external_result.json"]
    predictions = pd.read_csv(outputs / "external_artifact" / "tabpfn_classifier_external_predictions.csv")
    assert len(predictions) == 8 and set(predictions.columns[:2]) == {"record_id", "category"} and predictions.columns[2] == "prediction"
    assert np.allclose(predictions[[f"proba_class_{i}" for i in range(3)]].sum(axis=1), 1.0)
    result = json.loads((outputs / "external_artifact" / "tabpfn_classifier_external_result.json").read_text())
    assert result["input_manifest"]["inputs"][0]["kind"] == "sample" and result["input_manifest"]["findings"][0]["verdict"] == "rejected"
    # the pinned fitted-archive digest is enforced: a different pin refuses the same ZIP
    monkeypatch.setattr(STAGES, "SAMPLE_ARTIFACT_FITTED_SHA256", "f" * 64)
    run.options = {"source": "pinned"}
    with pytest.raises(ValueError, match=r"model\.tabpfn_fit SHA-256 .* != expected f{64}"):
        _quiet(STAGES.stage_external_artifact, run)
    assert not (outputs / STAGES.EXTERNAL_DIR).exists()


def test_members_are_checked_before_extraction(tmp_path, monkeypatch) -> None:
    """A ZIP with notes.txt, with both weights files (mixed layouts), or with sub/model.safetensors that flattening would
    let overwrite model.safetensors, is refused."""
    P = STAGES.package(_root(tmp_path))
    good, _fitted = _standin(tmp_path, P, monkeypatch)
    assert STAGES.check_members(P, good) == sorted([P.ARTIFACT_MANIFEST_NAME, P.FITTED_NAME, P.CHECKPOINT_NAME])
    extra = tmp_path / "extra.zip"
    shutil.copyfile(good, extra)
    with zipfile.ZipFile(extra, "a") as archive:
        archive.writestr("notes.txt", "x")
    with pytest.raises(ValueError, match=r"unexpected=\['notes\.txt'\]"):
        STAGES.check_members(P, extra)
    nested = tmp_path / "nested.zip"
    shutil.copyfile(good, nested)
    with zipfile.ZipFile(nested, "a") as archive:
        archive.writestr("sub/model.safetensors", b"evil")
    with pytest.raises(ValueError, match=r"\['model\.safetensors'\] occur more than once"):
        STAGES.check_members(P, nested)
    mixed = tmp_path / "mixed.zip"
    shutil.copyfile(good, mixed)
    with zipfile.ZipFile(mixed, "a") as archive:
        archive.writestr("model.ckpt", b"pickle")
    with pytest.raises(ValueError, match=r"unexpected=\['model\.ckpt'\]"):
        STAGES.check_members(P, mixed)


def test_v1_artifact_is_still_accepted_bound_to_the_source_checkpoint(tmp_path, monkeypatch) -> None:
    """A deprecated v1 bundle (model.ckpt) passes the members, digest and base-model checks, bound to WEIGHTS_SHA256; a
    v1 manifest that names model.safetensors, or a v2 one that names model.ckpt, is refused before any file is read."""
    _cpu(monkeypatch)
    root = _root(tmp_path)
    P = STAGES.package(root)
    zip_path, fitted = _standin(tmp_path, P, monkeypatch, legacy=True)
    assert STAGES.check_members(P, zip_path) == sorted([P.ARTIFACT_MANIFEST_NAME, P.FITTED_NAME, P.LEGACY_CHECKPOINT_NAME])
    monkeypatch.setattr(STAGES, "SAMPLE_ARTIFACT_URL", zip_path.resolve().as_uri())
    monkeypatch.setattr(STAGES, "SAMPLE_ARTIFACT_SHA256", hashlib.sha256(zip_path.read_bytes()).hexdigest())
    monkeypatch.setattr(STAGES, "SAMPLE_ARTIFACT_FITTED_SHA256", fitted)
    run = STAGES.Run(root, tmp_path / "outputs", tmp_path / "weights", {"source": "pinned"})
    _quiet(STAGES.stage_external_artifact, run)
    record = json.loads((run.state / "external_artifact.json").read_text())
    assert "deprecated bundle format v1" in record["checkpoint_binding"] and record["manifest"]["schemaVersion"] == 1
    run.options = {}
    _quiet(STAGES.stage_external_reconstruct, run)
    base = {"taskType": P.TASK_TYPE, "fittedEstimator": P.FITTED_NAME}
    with pytest.raises(ValueError, match="mixed v1"):
        P.safetensors_weights.check_bundle_layout({**base, "schemaVersion": 1, "foundationCheckpoint": P.CHECKPOINT_NAME})
    with pytest.raises(ValueError, match="mixed or malformed v2"):
        P.safetensors_weights.check_bundle_layout({**P.safetensors_weights.v2_manifest_fields(P.TASK_TYPE, "a" * 64, 10), "taskType": P.TASK_TYPE, "foundationCheckpoint": P.LEGACY_CHECKPOINT_NAME})
    with pytest.raises(ValueError, match="unknown bundle layout"):
        P.safetensors_weights.check_bundle_layout({**base, "schemaVersion": 3})


def test_artifact_from_another_dataset_breaks_the_data_contract(tmp_path) -> None:
    """SART10: a hash-valid artifact built on another table is refused."""
    root = _root(tmp_path)
    P = STAGES.package(root)
    run = STAGES.Run(root, tmp_path / "outputs", tmp_path / "weights", {})
    contract = STAGES.synthetic_contract(run, P)
    assert contract["targetColumn"] == "target" and contract["classes"] == ["class_0", "class_1", "class_2"] and len(contract["featureColumns"]) == 12
    assert "record_id" not in contract["featureColumns"] and "category" not in contract["featureColumns"]
    STAGES.check_data_contract({k: contract[k] for k in ("featureColumns", "targetColumn", "classes", "featureKinds")}, contract)
    with pytest.raises(RuntimeError, match=r"default data contract \(SART10\)"):
        STAGES.check_data_contract({"featureColumns": ["mean radius"], "targetColumn": "target", "classes": ["benign", "malignant"], "featureKinds": {"mean radius": "numeric"}}, contract)


def test_non_numeric_value_in_a_numeric_feature_is_refused() -> None:
    rows = pd.DataFrame({"feature_00": ["1.0", "abc"], "feature_01": [1.0, 2.0]})
    with pytest.raises(ValueError, match=r"rows\.csv: numeric feature 'feature_00' has 1 non-numeric value\(s\), e\.g\. \['abc'\]"):
        STAGES.check_numeric_features("rows.csv", rows, {"feature_00": "numeric", "feature_01": "numeric"})
    kinds = STAGES.feature_kinds({"featureColumns": ["feature_00", "feature_01"]}, rows.assign(feature_00=["1.0", "2.0"]))
    assert kinds == {"feature_00": "numeric", "feature_01": "numeric"}  # without recorded kinds the majority-numeric rule applies


def test_deterministic_export_zip(tmp_path) -> None:
    """The export ZIP is deterministic: sorted members, fixed 1980-01-01 timestamps, mode 0644, same bytes → same digest."""
    sys.path.insert(0, str(ROOT / "src"))
    import tabpfn_classifier_pipeline as P

    art = tmp_path / "art"
    art.mkdir()
    manifest = json.dumps({"taskType": P.TASK_TYPE, **P.safetensors_weights.v2_manifest_fields(P.TASK_TYPE, "a" * 64, 100)}).encode()
    for name in (P.FITTED_NAME, P.CHECKPOINT_NAME):
        (art / name).write_bytes(name.encode() * 100)
    (art / P.ARTIFACT_MANIFEST_NAME).write_bytes(manifest)
    first = P.zip_artifact_bundle(art, tmp_path / "a.zip")
    for name in (P.FITTED_NAME, P.CHECKPOINT_NAME):  # new mtimes, same bytes
        (art / name).write_bytes(name.encode() * 100)
    (art / P.ARTIFACT_MANIFEST_NAME).write_bytes(manifest)
    assert P.zip_artifact_bundle(art, tmp_path / "b.zip") == first
    with zipfile.ZipFile(tmp_path / "a.zip") as archive:
        infos = archive.infolist()
    assert [i.filename for i in infos] == sorted([P.ARTIFACT_MANIFEST_NAME, P.FITTED_NAME, P.CHECKPOINT_NAME])
    assert all(i.date_time == (1980, 1, 1, 0, 0, 0) and (i.external_attr >> 16) & 0o777 == 0o644 for i in infos)
    assert not (tmp_path / "a.zip.part").exists()


# ---------------------------------------------------------------- XAS5: setup cells and its own cells only


def _defined(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.FunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def test_section_cells_reference_only_setup_cell_names() -> None:
    setup = [c for c in _cells() if c["cell_type"] == "code" and c.get("metadata", {}).get("cellView") == "form"]
    assert len(setup) == 4, "the four Infrastructure cells (Sections 1-3)"
    setup_names = set().union(*(_defined(ast.parse(_src(c))) for c in setup))
    section_code = [_src(c) for c in _section_cells() if c["cell_type"] == "code"]
    assert len(section_code) == 3
    own = set().union(*(_defined(ast.parse(s)) for s in section_code))
    used = {n.id for s in section_code for n in ast.walk(ast.parse(s)) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    foreign = used - own - setup_names - set(dir(builtins))
    assert not foreign, f"the section uses names defined outside the setup cells and itself: {sorted(foreign)}"
    assert {"run_stage", "ROOT", "Path"} <= used & setup_names


def test_section_states_its_dependency_and_its_difference_from_the_reload() -> None:
    md = "\n".join(_src(c) for c in _section_cells() if c["cell_type"] == "markdown")
    assert "**This section needs only the setup cells (Sections 1–3).**" in md
    assert "the reload shows that this run's export survives serialization, and this section shows that an artifact made elsewhere can be validated and served here" in md
    assert "**Trust boundary.**" in md and "outputs/external_artifact/" in md


def test_section_defaults_run_the_pinned_artifact_and_sample_rows_without_upload(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)
    calls: list = []
    ns = {"ROOT": tmp_path / "run", "Path": Path, "run_stage": lambda stage, **o: calls.append((stage, o))}
    for source in (_src(c) for c in _section_cells() if c["cell_type"] == "code"):
        exec(compile(source, "<section>", "exec"), ns)
    assert calls == [("external_artifact", {"source": "pinned"}), ("external_reconstruct", {}), ("external_predict", {"rows_path": ""})]


def test_section_runs_last_on_the_default_path_after_the_reload() -> None:
    code = [_src(c) for c in _cells() if c["cell_type"] == "code"]
    reload_at = next(i for i, s in enumerate(code) if "run_stage('reload')" in s)
    section_at = next(i for i, s in enumerate(code) if "run_stage('external_artifact'" in s)
    assert reload_at < section_at
    later = "\n".join(code[section_at + 3:])
    assert "run_stage(" not in later.replace("run_stage('activity'", ""), "only the optional activity may follow the section"
    meta = json.loads(E2E.read_text(encoding="utf-8"))["metadata"]["dimer"]
    assert meta["notebook_spec"] == "3.0" and meta["notebook_kind"] == "TUTORIAL"
    assert not list((ROOT / "tutorials").glob("*_artifact_inference_colab.ipynb")), "KND11: no separate artifact-inference notebook"


def test_interpretation_precedes_the_section_which_closes_the_canonical_path() -> None:
    """§7.1: export and fresh reload → interpretation → conclusion → external artifact → optional activity → troubleshooting → references."""
    heads = [line for c in _cells() if c["cell_type"] == "markdown" for line in _src(c).split("\n") if line.startswith("## ")]
    at = {h.split(" ·")[0]: i for i, h in enumerate(heads)}
    order = ["## 8. Export the artifact bundle and verify it in a fresh process", "## Interpretation and limits", "## Conclusion", SECTION_START, SECTION_END + ": how much does the ensemble size matter?", "## Troubleshooting", "## References"]
    assert [at[h] for h in order] == sorted(at[h] for h in order)


# ---------------------------------------------------------------- ST1: only the exact pinned URL

_REPO = "https://github.com/kurtvalcorza/tabpfn-classifier-pipeline"


@pytest.mark.parametrize(
    "reference",
    [
        f"{_REPO}/releases/download/../../../archive/refs/heads/main.zip",
        f"{_REPO}/releases/download/sample-artifact-synthetic-v1/../../../archive/refs/heads/main.zip",
        f"{_REPO}/releases/download/%2e%2e/%2E%2E/archive/refs/heads/main.zip",
        "https://GitHub.com/KurtValcorza/tabpfn-classifier-pipeline/releases/download/%2E%2E/archive/refs/heads/main.zip",
        f"{_REPO}/releases/download/sample-artifact-synthetic-v1/tabpfn_classifier_artifact.zip?raw=1",
        f"{_REPO}/releases/download/sample-artifact-synthetic-v1/tabpfn_classifier_artifact.zip#x",
        "https://github.com/kurtvalcorza/tabicl-classifier-pipeline/releases/download/sample-artifact-magic-v1/tabiclv2_classifier_artifact.zip",
        f"{_REPO}/releases/download/sample-bundle-v1/tabpfn_classifier_sample_bundle.zip",
        f"{_REPO}/releases/download/sample-artifact-synthetic-v2/tabpfn_classifier_artifact.zip",
        f"{_REPO}/releases/download/sample-artifact-synthetic-v1/tabpfn_classifier_artifact.zip.sig",
        f"{_REPO}.git",
    ],
    ids=["traversal", "tag-then-traversal", "pct2e", "pct2E-uppercase-host", "query", "fragment", "other-repo", "other-tag-old-bundle", "other-tag", "suffix", "clone"],
)
def test_st1_only_the_exact_pinned_release_asset_is_allowed(reference: str) -> None:
    source = f'SAMPLE_ARTIFACT_URL = "{reference}"\n'
    assert VALIDATOR.disallowed_own_references(source, (VALIDATOR.PINNED_SAMPLE_ARTIFACT_URL,)) == [reference]


def test_st1_exact_pinned_url_passes_and_matches_the_runner() -> None:
    assert VALIDATOR.PINNED_SAMPLE_ARTIFACT_URL == PINNED_URL == STAGES.SAMPLE_ARTIFACT_URL
    source = f'SAMPLE_ARTIFACT_URL = "{PINNED_URL}"  # literal\n'
    assert VALIDATOR.disallowed_own_references(source, (VALIDATOR.PINNED_SAMPLE_ARTIFACT_URL,)) == []
    assert VALIDATOR.disallowed_own_references(source) == [PINNED_URL], "kernel cells allow no repository reference at all"


def test_st1_real_notebook_passes_the_validator() -> None:
    runner = VALIDATOR.carried_runner(E2E)
    assert VALIDATOR.disallowed_own_references(runner, (VALIDATOR.PINNED_SAMPLE_ARTIFACT_URL,)) == []
    VALIDATOR.validate_notebooks()
