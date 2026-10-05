"""Regression tests for the 2026-10-05 notebook review (TPC-M1..M2, TPC-m1..m4; TPCA-M1..M3, TPCA-m1..m2).

They need only CI's dependencies (no torch, no tabpfn, no checkpoint, no Parquet engine): they exec the notebooks' own
kernel cells with stand-ins for `run_stage` and `google.colab`, run the stage runners' model-free parts (`data`,
`validate`, the bundle member and digest checks, `rows`) in-process, and check the generated notebooks statically. Each
test names its finding.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import shutil
import sys
import types
import zipfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
E2E = ROOT / "tutorials" / "tabpfn_classifier_colab.ipynb"
AI = ROOT / "tutorials" / "tabpfn_classifier_artifact_inference_colab.ipynb"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGES = _load("tpc_tutorial_stages", TOOLS / "tutorial_stages.py")
AI_STAGES = _load("tpc_tutorial_stages_ai", TOOLS / "tutorial_stages_artifact_inference.py")


def _nb(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _cell_with(path: Path, needle: str) -> str:
    found = [_src(c) for c in _nb(path)["cells"] if c["cell_type"] == "code" and needle in _src(c)]
    assert len(found) == 1, needle
    return found[0]


def _set(src: str, name: str, value) -> str:
    out = []
    for line in src.split("\n"):
        if line.startswith(f"{name} = "):
            line = f"{name} = {value!r}" + (line[line.index("  #"):] if "  #" in line else "")
        out.append(line)
    return "\n".join(out)


@contextlib.contextmanager
def _colab(queue):
    files = types.SimpleNamespace(calls=0)

    def upload():
        files.calls += 1
        return queue.pop(0)

    files.upload = upload
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
    sys.modules.update({"google": google, "google.colab": colab})
    try:
        yield files
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _no_colab(monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)


def _kernel(tmp_path: Path) -> tuple[dict, list]:
    calls: list = []
    ns = {"ROOT": tmp_path / "run", "Path": Path, "shutil": shutil, "run_stage": lambda stage, **o: calls.append((stage, o))}
    return ns, calls


def _run(root: Path, outputs: Path, options: dict | None = None, module=STAGES):
    if not (root / "src").exists():
        root.mkdir(parents=True, exist_ok=True)
        (root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    return module.Run(root, outputs, root / "weights", options or {})


def _quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def _synthetic(tmp_path: Path) -> dict[str, pd.DataFrame]:
    sys.path.insert(0, str(ROOT / "src"))
    import tabpfn_classifier_pipeline as P

    return P.read_dataset_zip(P.build_synthetic_dataset(tmp_path / "s.zip"))


def _zip(path: Path, frames: dict[str, pd.DataFrame]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, frame in frames.items():
            archive.writestr(name, frame.to_csv(index=False))
    return path


# ---------------------------------------------------------------- TPC-M1 / TPCA-M2: isolated runtime


@pytest.mark.parametrize("path", [E2E, AI], ids=["TPC-M1", "TPCA-M2"])
def test_m1_no_kernel_install_and_no_restart_instruction(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert "Restart the runtime" not in text and "restart the runtime" not in text.replace("no runtime restart", "")
    own = [_src(c) for c in _nb(path)["cells"] if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_sources")]
    for src in own:
        assert not re.search(r"['\"]-m['\"]\s*,\s*['\"]pip['\"]|^\s*[%!]\s*pip\b|['\"]pip install", src, re.M), src[:80]
    install = _cell_with(path, "# @title Infrastructure: build (or reuse) the isolated")
    assert "'--require-hashes'" in install and "--managed-python" in install
    assert "MPLBACKEND='Agg'" in install and "'PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP'" in install


def _exec_check_cell(ns: dict, path: Path, **overrides) -> None:
    src = _cell_with(path, "# @title Infrastructure: check the runtime")
    src = re.sub(r"'environment': [0-9.]+}", "'environment': 0.0}", src, count=1)
    src = re.sub(r"\{'weights': max\(0\.0, [0-9.]+", "{'weights': max(0.0, 0.0", src, count=1)
    for name, value in overrides.items():
        src = _set(src, name, value)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, "<check>", "exec"), ns)


@pytest.mark.parametrize("path", [E2E, AI], ids=["TPC-M1", "TPCA-M2"])
def test_m1_section1_is_idempotent(path: Path, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ns: dict = {}
    _exec_check_cell(ns, path)
    first = ns["ROOT"]
    (first / "tutorial_stages.py").write_text("# carried")
    _exec_check_cell(ns, path)
    assert ns["ROOT"] == first and (first / "tutorial_stages.py").is_file()
    _exec_check_cell(ns, path, NEW_RUN_DIRECTORY=True)
    assert ns["ROOT"] != first


def test_m1_second_run_all_reuses_the_matching_environment(tmp_path, monkeypatch) -> None:
    """TPC-M1: the venv is keyed on the lock digest; a second exec of the install cell builds nothing."""
    import subprocess as real_subprocess

    monkeypatch.chdir(tmp_path)
    ns: dict = {}
    _exec_check_cell(ns, E2E)
    ns["ENV_ROOT"] = tmp_path / "uvroot"
    ns["NOTEBOOK_SOURCE"] = {"revision": "test"}
    src = _cell_with(E2E, "# @title Infrastructure: build (or reuse) the isolated")
    version = re.search(r"UV = ENV_ROOT / 'uv-([0-9.]+)'", src).group(1)
    ns["ENV_ROOT"].mkdir()
    (ns["ENV_ROOT"] / f"uv-{version}").write_bytes(b"uv stand-in")
    (ns["ENV_ROOT"] / f"uv-{version}.sha256").write_text(hashlib.sha256(b"uv stand-in").hexdigest())
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append([str(c) for c in cmd])
        if cmd[1] == "venv":
            python = Path(cmd[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("")
        out = json.dumps({"python": "3.12.12", "torch": "x", "tabpfn": "x", "numpy": "x", "pandas": "x", "scikit-learn": "x", "cuda": False})
        return types.SimpleNamespace(stdout=out + "\n", returncode=0)

    fake = types.SimpleNamespace(run=fake_run, Popen=real_subprocess.Popen, PIPE=real_subprocess.PIPE, STDOUT=real_subprocess.STDOUT)
    for attempt in range(2):
        ns["subprocess"] = fake
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src.replace("import zipfile\n", "import zipfile\nsubprocess = globals()['subprocess']\n", 1), "<install>", "exec"), ns)
        ns["subprocess"] = fake
        assert ns["environment_reused"] is (attempt == 1)
    assert len([c for c in commands if c[1] in ("venv", "pip")]) == 2


# ---------------------------------------------------------------- TPC-M2 / TPCA-M3: guided layer

GUIDED = ("## How to use this notebook", "**Who this notebook is for.**", "## The task: Input → Model → Output", "## Roadmap", "<summary><strong>Glossary</strong>", "Predict before running", "**What to notice:**", "Check your reasoning", "## Troubleshooting", "## Conclusion")


@pytest.mark.parametrize("path", [E2E, AI], ids=["TPC-M2", "TPCA-M3"])
def test_m2_guided_layer_and_collapsed_infrastructure(path: Path) -> None:
    nb = _nb(path)
    md = "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")
    for heading in GUIDED:
        assert heading in md, heading
    assert "{{" not in md and "{MODEL_ID}" not in md
    infra = [c for c in nb["cells"] if c["cell_type"] == "code" and _src(c).startswith("# @title Infrastructure:")]
    assert len(infra) == 4 and all(c["metadata"].get("cellView") == "form" for c in infra)
    assert _cell_with(path, "RUN_ACTIVITY = False  # @param")


def test_M2_n_estimators_activity_and_tamper_activity() -> None:
    """TPC-M2 / TPCA-M3: the N_ESTIMATORS field becomes a guided activity; the tampered-bundle experiment is the companion's."""
    assert "ACTIVITY_N_ESTIMATORS = 1  # @param" in _cell_with(E2E, "RUN_ACTIVITY = False  # @param")
    assert "TAMPER = 'flip one byte of model.tabpfn_fit'  # @param" in _cell_with(AI, "RUN_ACTIVITY = False  # @param")
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    body = source[source.index("def stage_activity"):source.index("STAGES = {")]
    assert 'write_output(f"activity/' in body and "!= before" in body


@pytest.mark.parametrize("runner", ["tutorial_stages.py", "tutorial_stages_artifact_inference.py"])
def test_no_quality_assert_in_stage_runners(runner: str) -> None:
    assert not [n for n in ast.walk(ast.parse((TOOLS / runner).read_text(encoding="utf-8"))) if isinstance(n, ast.Assert)]


# ---------------------------------------------------------------- TPC-m1: profile and adaptation


def test_m1_profile_is_e2e_and_the_run7_deviation_is_recorded() -> None:
    """TPC-m1: the notebook fits, evaluates, exports and reloads (E2E anatomy); not fine-tuning is a recorded, proposed RUN7 deviation."""
    assert _nb(E2E)["metadata"]["dimer"]["notebook_profile"] == "E2E"
    text = E2E.read_text(encoding="utf-8")
    assert "proposed RUN7 deviation" in text and "pending the maintainer's approval" in text
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    assert 'report["adaptation_deviation"] = RUN7_DEVIATION' in source


# ---------------------------------------------------------------- TPC-m2: identifiers are not features


def test_m2_record_id_is_not_a_feature_and_travels_with_the_rows(tmp_path) -> None:
    """TPC-m2: the default manifest's features exclude record_id (and the index-derived category); the new rows keep them."""
    run = _run(tmp_path / "run", tmp_path / "outputs")
    _quiet(STAGES.stage_data, run)
    _quiet(STAGES.stage_validate, run)
    manifest = json.loads((run.out / "tabpfn_classifier_input_manifest.json").read_text())
    entry = manifest["inputs"][0]
    assert "record_id" not in entry["feature_columns"] and "category" not in entry["feature_columns"]
    assert entry["drop_columns"] == ["record_id", "category"] and len(entry["feature_columns"]) == 12
    data, _v, _train, val, test = STAGES.tables(run, "t")
    rows, _origin = STAGES.new_rows_frame(data, val, test)
    assert set(rows.columns[-2:]) == {"record_id", "category"} and "target" not in rows.columns
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    assert "passthrough = [c for c in rows.columns if c in drop]" in source


# ---------------------------------------------------------------- TPC-m3: coded findings instead of raw errors


def test_m3_renamed_target_in_a_train_only_zip_is_a_coded_finding(tmp_path) -> None:
    """TPC-m3: a train-only ZIP labelled `label` stops in Section 4 with TARGET_MISSING (not KeyError from the splitter)."""
    frames = _synthetic(tmp_path)
    path = _zip(tmp_path / "renamed.zip", {"train.csv": frames["train.csv"].rename(columns={"target": "label"})})
    run = _run(tmp_path / "run", tmp_path / "outputs", {"use_byod": True, "byod_path": str(path), "target_column": "target"})
    with pytest.raises(ValueError, match=r"\[TARGET_MISSING\] train\.csv: target column 'target' not present .*Set TARGET_COLUMN"):
        _quiet(STAGES.stage_data, run)
    run.options["target_column"] = "label"
    _quiet(STAGES.stage_data, run)
    assert json.loads((run.state / "data.json").read_text())["splits"].startswith("seeded stratified holdout")


def test_m3_unseen_validation_class_is_refused_with_its_code(tmp_path) -> None:
    """TPC-m3: a class only in val.csv stops in Section 5 as UNSEEN_CLASSES (it used to crash later in log_loss)."""
    frames = _synthetic(tmp_path)
    val = frames["val.csv"].copy()
    val.loc[val.index[:3], "target"] = "class_9"
    path = _zip(tmp_path / "unseen.zip", {"train.csv": frames["train.csv"], "val.csv": val, "test.csv": frames["test.csv"]})
    run = _run(tmp_path / "run", tmp_path / "outputs", {"use_byod": True, "byod_path": str(path)})
    _quiet(STAGES.stage_data, run)
    with pytest.raises(ValueError, match=r"\[UNSEEN_CLASSES\] classes absent from train\.csv: val\.csv: \['class_9'\]"):
        _quiet(STAGES.stage_validate, run)
    assert not (run.out / "tabpfn_classifier_input_manifest.json").exists()


def test_m3_numeric_column_with_a_stray_string_is_refused(tmp_path) -> None:
    """TPC-m3: `feature_00` with one `abc` is refused naming the file, the column and the value; TEXT_COLUMNS opts out."""
    frames = _synthetic(tmp_path)
    train = frames["train.csv"].astype({"feature_00": object})
    train.loc[train.index[5], "feature_00"] = "abc"
    path = _zip(tmp_path / "typo.zip", {"train.csv": train, "val.csv": frames["val.csv"], "test.csv": frames["test.csv"]})
    run = _run(tmp_path / "run", tmp_path / "outputs", {"use_byod": True, "byod_path": str(path)})
    with pytest.raises(ValueError, match=r"train\.csv: column 'feature_00' is numeric except for 1 value\(s\) \['abc'\]"):
        _quiet(STAGES.stage_data, run)


def test_m3_byod_path_field_and_upload_fallback(tmp_path, monkeypatch) -> None:
    """TPC-m3 / TPC-m1 BYOD: BYOD_PATH works without google.colab; an empty path outside Colab or a cancelled upload is named."""
    cell = _cell_with(E2E, "USE_BYOD = False  # @param")
    _no_colab(monkeypatch)
    ns, calls = _kernel(tmp_path)
    exec(compile(_set(_set(cell, "USE_BYOD", True), "BYOD_PATH", "/d/data.zip"), "<s4>", "exec"), ns)
    assert calls[0][1]["byod_path"] == "/d/data.zip" and calls[0][1]["drop_columns"] == ["record_id", "category"]
    ns, calls = _kernel(tmp_path)
    with pytest.raises(RuntimeError, match="BYOD_PATH is empty, and the upload dialog exists only in Google Colab"):
        exec(compile(_set(cell, "USE_BYOD", True), "<s4>", "exec"), ns)
    monkeypatch.undo()
    with _colab([{}]) as files:
        ns, calls = _kernel(tmp_path)
        with pytest.raises(RuntimeError, match="cancelled or empty"):
            exec(compile(_set(cell, "USE_BYOD", True), "<s4>", "exec"), ns)
        assert files.calls == 1 and calls == []
    ns, calls = _kernel(tmp_path)
    exec(compile(cell, "<s4>", "exec"), ns)
    assert calls == [("data", {"use_byod": False, "byod_path": "", "target_column": "target", "drop_columns": ["record_id", "category"], "text_columns": [], "validation_split": 0.2})]


# ---------------------------------------------------------------- TPC-m4: probability-level reload check


def test_m4_reload_compares_probabilities_and_clears_its_directory() -> None:
    """TPC-m4: the fresh-process reload compares probabilities on every validation row with a tolerance, records the
    largest difference, and clears the reload directory first so the cell can be re-run alone."""
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    body = source[source.index("def stage_reload"):source.index("def new_rows_frame")]
    assert "shutil.rmtree(fresh_dir, ignore_errors=True)" in body
    assert "np.testing.assert_allclose(proba, expected, rtol=RELOAD_RTOL, atol=RELOAD_ATOL" in body
    assert '"maxAbsProbabilityDifference": diff' in body and "accuracyMatches" not in body
    assert (STAGES.RELOAD_RTOL, STAGES.RELOAD_ATOL) == (1e-5, 1e-6)


def test_S2_logistic_regression_reference(tmp_path) -> None:
    """TPC-S2: the report's classical reference reproduces the review's numbers (0.911 validation / 0.944 test)."""
    run = _run(tmp_path / "run", tmp_path / "outputs")
    _quiet(STAGES.stage_data, run)
    _quiet(STAGES.stage_validate, run)
    _data, v, train, val, test = STAGES.tables(run, "t")
    ref = STAGES.classical_reference(train, val, test, "target", v["features"])
    assert (round(ref["validation"]["accuracy"], 4), ref["validation"]["errors"]) == (0.9111, 8)
    assert (round(ref["test"]["accuracy"], 4), ref["test"]["errors"]) == (0.9444, 5)


# ---------------------------------------------------------------- TPCA-M1: no sample yet — stop cleanly; builder ready


def test_M1_default_path_without_a_sample_stops_naming_the_field(tmp_path, monkeypatch) -> None:
    """TPCA-M1 (not fixed: needs a hosted run): with no carried sample the default path opens no dialog and stops naming
    ARTIFACT_ZIP_PATH, instead of blocking on an upload."""
    _no_colab(monkeypatch)
    ns, calls = _kernel(tmp_path)
    exec(compile(_cell_with(AI, "ARTIFACT_ZIP_PATH = ''  # @param"), "<s4>", "exec"), ns)
    assert calls == [("artifact", {"source": "sample", "zip_path": "", "expected_zip_sha256": "", "expected_fitted_sha256": ""})]
    root = tmp_path / "run"
    run = _run(root, tmp_path / "outputs", {"source": "sample"}, module=AI_STAGES)
    with pytest.raises(RuntimeError, match="No pinned sample bundle is carried .* Set ARTIFACT_ZIP_PATH"):
        _quiet(AI_STAGES.stage_artifact, run)


def _fake_bundle(tmp_path: Path, P) -> Path:
    art = tmp_path / "art"
    art.mkdir()
    with zipfile.ZipFile(art / P.FITTED_NAME, "w") as fitted:
        fitted.writestr("init_params.json", json.dumps({"model_path": "x"}))
        fitted.writestr("fitted_attrs.joblib", b"\0" * 2048)
    (art / P.CHECKPOINT_NAME).write_bytes(b"c" * 4096)
    manifest = {"schemaVersion": 1, "taskType": P.TASK_TYPE, "targetColumn": "target", "featureColumns": ["a"], "classes": ["x", "y"], "fittedEstimator": P.FITTED_NAME, "foundationCheckpoint": P.CHECKPOINT_NAME, "fittedEstimatorSha256": hashlib.sha256((art / P.FITTED_NAME).read_bytes()).hexdigest(), "foundationCheckpointSha256": P.WEIGHTS_SHA256}
    (art / P.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))
    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name in (P.ARTIFACT_MANIFEST_NAME, P.FITTED_NAME, P.CHECKPOINT_NAME):
            archive.write(art / name, arcname=name)
    return zip_path


def test_M1_sample_builder_writes_the_bundle_without_the_checkpoint(tmp_path) -> None:
    """TPCA-M1: tools/build_sample_bundle.py turns a recorded E2E export into examples/sample-bundle/ (no model.ckpt)."""
    builder = _load("tpc_build_sample_bundle", TOOLS / "build_sample_bundle.py")
    P = AI_STAGES.package(_run(tmp_path / "r", tmp_path / "o", module=AI_STAGES).root)
    zip_path = _fake_bundle(tmp_path, P)
    rows = tmp_path / "rows.csv"
    rows.write_text("record_id,a\nr1,1.0\n")
    out = tmp_path / "sample"
    record = builder.build(zip_path, rows, "test producer", out)
    assert sorted(p.name for p in out.iterdir()) == ["SAMPLE_BUNDLE.json", "artifact_manifest.json", "model.tabpfn_fit", "new_rows.csv"]
    assert record["checkpoint"]["sha256"] == P.WEIGHTS_SHA256 and builder.check(out) == 0
    (out / "new_rows.csv").write_text("changed\n")
    with contextlib.redirect_stderr(io.StringIO()):
        assert builder.check(out) == 1
    assert builder.check(tmp_path / "absent") == 0


# ---------------------------------------------------------------- TPCA-m2: members and digests


def test_m2_unlisted_and_nested_members_are_refused(tmp_path) -> None:
    """TPCA-m2: a bundle with notes.txt, or with sub/model.ckpt that flattening would let overwrite model.ckpt, is refused."""
    P = AI_STAGES.package(_run(tmp_path / "r", tmp_path / "o", module=AI_STAGES).root)
    good = _fake_bundle(tmp_path, P)
    assert AI_STAGES.check_members(P, good) == sorted([P.ARTIFACT_MANIFEST_NAME, P.FITTED_NAME, P.CHECKPOINT_NAME])
    extra = tmp_path / "extra.zip"
    shutil.copyfile(good, extra)
    with zipfile.ZipFile(extra, "a") as archive:
        archive.writestr("notes.txt", "x")
    with pytest.raises(ValueError, match=r"unexpected=\['notes\.txt'\]"):
        AI_STAGES.check_members(P, extra)
    nested = tmp_path / "nested.zip"
    shutil.copyfile(good, nested)
    with zipfile.ZipFile(nested, "a") as archive:
        archive.writestr("sub/model.ckpt", b"evil")
    with pytest.raises(ValueError, match=r"\['model\.ckpt'\] occur more than once"):
        AI_STAGES.check_members(P, nested)


def test_m2_trusted_digests_are_checked_and_malformed_ones_refused(tmp_path) -> None:
    """TPCA-M1/m2 (trust): a ZIP whose SHA-256 differs from EXPECTED_ZIP_SHA256 is refused before anything is loaded."""
    P = AI_STAGES.package(_run(tmp_path / "r", tmp_path / "o", module=AI_STAGES).root)
    zip_path = _fake_bundle(tmp_path, P)
    observed = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match=f"EXPECTED_ZIP_SHA256 is {'0' * 64}, the supplied file's SHA-256 is {observed}"):
        AI_STAGES.check_trusted_digest(zip_path, "0" * 64, "EXPECTED_ZIP_SHA256")
    with pytest.raises(ValueError, match="must be 64 hexadecimal characters"):
        AI_STAGES.check_digest_format("abc", "EXPECTED_FITTED_SHA256")
    assert AI_STAGES.check_trusted_digest(zip_path, observed.upper(), "EXPECTED_ZIP_SHA256") == observed


# ---------------------------------------------------------------- TPCA-m1 / TPCA-m2: rows, identifiers, numeric types


def _rows_run(tmp_path: Path, source: str, options: dict):
    run = _run(tmp_path / "run", tmp_path / "outputs", options, module=AI_STAGES)
    manifest = {"targetColumn": "target", "featureColumns": [f"feature_{i:02d}" for i in range(12)], "classes": ["class_0", "class_1", "class_2"], "featureKinds": {f"feature_{i:02d}": "numeric" for i in range(12)}}
    run.write_state("artifact.json", {"source": source, "manifest": manifest})
    return run


def _new_rows(tmp_path: Path) -> pd.DataFrame:
    frames = _synthetic(tmp_path)
    return frames["test.csv"].drop(columns=["target"]).head(8)


def test_m1_identifier_columns_are_kept_and_undeclared_ones_get_a_hint(tmp_path) -> None:
    """TPCA-m1: declared ID_COLUMNS are excluded from the features and kept; an undeclared one is refused with a hint."""
    path = tmp_path / "mine.csv"
    _new_rows(tmp_path).to_csv(path, index=False)
    run = _rows_run(tmp_path, "path", {"source": "path", "path": str(path), "id_columns": []})
    with pytest.raises(ValueError, match=r"\[SCHEMA_MISMATCH\].*If \['category', 'record_id'\] are identifiers, list them in ID_COLUMNS"):
        _quiet(AI_STAGES.stage_rows, run)
    run.options["id_columns"] = ["record_id", "category"]
    _quiet(AI_STAGES.stage_rows, run)
    state = json.loads((run.state / "rows.json").read_text())
    assert state["id_columns"] == ["record_id", "category"] and state["sample_kind"] == "BYOD"
    assert list(pd.read_csv(run.state / "rows.csv").columns[:2]) == ["record_id", "category"]
    run.options["id_columns"] = ["feature_00"]
    with pytest.raises(ValueError, match="are fitted feature columns"):
        _quiet(AI_STAGES.stage_rows, run)


def test_m2_non_numeric_value_in_a_numeric_feature_is_refused(tmp_path) -> None:
    """TPCA-m2: `abc` in feature_00 stops in the rows stage naming feature_00 and the value."""
    rows = _new_rows(tmp_path).astype({"feature_00": object})
    rows.loc[rows.index[0], "feature_00"] = "abc"
    path = tmp_path / "abc.csv"
    rows.to_csv(path, index=False)
    run = _rows_run(tmp_path, "path", {"source": "path", "path": str(path), "id_columns": ["record_id", "category"]})
    with pytest.raises(ValueError, match=r"abc\.csv: numeric feature 'feature_00' has 1 non-numeric value\(s\), e\.g\. \['abc'\]"):
        _quiet(AI_STAGES.stage_rows, run)
    del run  # without recorded feature kinds the majority-numeric rule still catches it
    run = _rows_run(tmp_path / "b", "path", {"source": "path", "path": str(path), "id_columns": ["record_id", "category"]})
    art = json.loads((run.state / "artifact.json").read_text())
    art["manifest"].pop("featureKinds")
    run.write_state("artifact.json", art)
    with pytest.raises(ValueError, match="numeric feature 'feature_00'"):
        _quiet(AI_STAGES.stage_rows, run)


def test_m1_zip_by_path_with_rows_by_upload_has_no_name_error(tmp_path, monkeypatch) -> None:
    """TPCA-m1 (review NameError class): ZIP by path, rows empty: outside Colab a message naming NEW_DATA_PATH; in Colab the dialog."""
    artifact_cell = _set(_cell_with(AI, "ARTIFACT_ZIP_PATH = ''  # @param"), "ARTIFACT_ZIP_PATH", "/d/bundle.zip")
    rows_cell = _cell_with(AI, "NEW_DATA_PATH = ''  # @param")
    _no_colab(monkeypatch)
    ns, calls = _kernel(tmp_path)
    exec(compile(artifact_cell, "<s4>", "exec"), ns)
    with pytest.raises(RuntimeError, match="set NEW_DATA_PATH"):
        exec(compile(rows_cell, "<s6>", "exec"), ns)
    monkeypatch.undo()
    with _colab([{"rows.csv": b"a\n1\n"}]) as files:
        ns, calls = _kernel(tmp_path)
        exec(compile(artifact_cell, "<s4>", "exec"), ns)
        exec(compile(rows_cell, "<s6>", "exec"), ns)
        assert files.calls == 1 and calls[1][1]["source"] == "upload" and calls[1][1]["path"].endswith("inputs/rows.csv")
