"""Regression tests for the 2026-10-05 notebook review (TPC-M1..M2, TPC-m1..m4) of the E2E notebook.

The companion's findings (TPCA-*) moved with its consumer logic into the external-artifact section and are tested in
`tests/test_external_artifact_section.py` (NOTEBOOK_SPEC 3.0 retired the companion notebook).

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


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGES = _load("tpc_tutorial_stages", TOOLS / "tutorial_stages.py")


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
    ns = {"ROOT": tmp_path / "run", "Path": Path, "shutil": shutil, "run_stage": lambda stage, **o: calls.append((stage, o)), "Refused": type("Refused", (Exception,), {})}
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


# ---------------------------------------------------------------- TPC-M1: isolated runtime


@pytest.mark.parametrize("path", [E2E], ids=["TPC-M1"])
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


@pytest.mark.parametrize("path", [E2E], ids=["TPC-M1"])
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


@pytest.mark.parametrize("path", [E2E], ids=["TPC-M2"])
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
    """TPC-M2 / TPCA-M3: the N_ESTIMATORS field becomes a guided activity; its outputs stay under outputs/activity/."""
    assert "ACTIVITY_N_ESTIMATORS = 1  # @param" in _cell_with(E2E, "RUN_ACTIVITY = False  # @param")
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    body = source[source.index("def stage_activity"):source.index("STAGES = {")]
    assert 'write_output(f"activity/' in body and "!= before" in body


@pytest.mark.parametrize("runner", ["tutorial_stages.py"])
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
    with pytest.raises(ValueError, match=r"^renamed\.zip/train\.csv: \[TARGET_MISSING\] target column 'target' not present .*Set TARGET_COLUMN"):
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
    with pytest.raises(ValueError, match=r"^unseen\.zip: \[UNSEEN_CLASSES\] classes absent from train\.csv: val\.csv: \['class_9'\]"):
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
        with pytest.raises(ns["Refused"], match=r"^\(no file\): the upload was cancelled or empty"):  # DAT20: a refusal, not a RuntimeError
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


# ---------------------------------------------------------------- carried stage processes


def test_stage_processes_import_neither_ipython_nor_google_colab() -> None:
    """Stages run in the isolated environment, which has neither IPython nor google.colab: only kernel cells may use them
    (the upload dialogs). There is no worker and no google.colab stub that would need a ModuleSpec (ENV15); a carried
    module importing either would fail on Colab (swin2sr-x4-super-resolution-pipeline 34eac6c / tirex 9ee5922 pattern)."""
    build = _load("tpc_build_notebook", TOOLS / "build_notebook.py")
    tpl = _load("tpc_notebook_template", TOOLS / "notebook_template.py").TEMPLATE
    carried = [ROOT / source for dest, source in build.carried_sources(ROOT, tpl).items() if dest.endswith(".py")]
    assert {p.name for p in carried} >= {"pipeline.py", Path(tpl["stage_runner"]).name}
    offenders = [str(p) for p in carried if re.search(r"^\s*(from|import)\s+(IPython|google)\b", p.read_text(encoding="utf-8"), re.M)]
    assert not offenders, offenders
    text = (ROOT / "tutorials" / tpl["notebook_name"]).read_text(encoding="utf-8")
    assert "sys.modules['google" not in text and 'sys.modules[\\"google' not in text and "_WORKER_SOURCE" not in text
    assert "IPython" not in text
