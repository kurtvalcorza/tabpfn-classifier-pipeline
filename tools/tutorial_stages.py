"""Stage runner for the standalone TabPFN-3 classifier E2E tutorial (NOTEBOOK_SPEC 2.2 §25.13 isolated-environment pattern).

The tutorial notebook carries this file verbatim (as ``tutorial_stages.py`` in its run directory, beside the carried
package under ``src/``) and runs every stage with the interpreter of an isolated, hash-locked environment::

    python -u tutorial_stages.py --root RUN_DIR --outputs OUTPUTS --weights WEIGHTS --stage data --options '{...}'

Nothing is installed into the notebook kernel. Each stage is a separate process and starts from files only: the
verified snapshot under ``--weights``, the partitions and records of earlier stages under ``RUN_DIR/state`` (tables as
pandas ``orient="table"`` JSON, so no Parquet engine is needed), and the learner-facing exports under ``--outputs``.
In-context conditioning (``fit``) registers the support rows and is repeated by every stage that needs the model. On
failure a stage writes ``RUN_DIR/state/<stage>.error.json``, which the notebook re-raises in the kernel. A user input the
stage refuses (a BYOD table, uploaded rows, your own artifact ZIP) is not a failure (NOTEBOOK_SPEC 3.0 DAT19/DAT20): the
stage raises ``Refused``, prints no traceback, records ``"refused": true`` with the file name and the rule, and exits with
``REFUSED_EXIT``; the notebook then shows one ``Refused: <file>: <reason>`` line and stops at that cell.

Stages: weights → data → validate → condition → report → export → reload → predict, then the closing external-artifact
section (NOTEBOOK_SPEC 3.0 §19.1) external_artifact → external_reconstruct → external_predict, plus the optional
``activity``. The external-artifact stages load the pinned sample artifact (a release asset exported by a recorded
earlier run of this notebook, pinned by URL and SHA-256), never this run's export, and read none of the main path's
state. The ``data`` and ``validate`` stages import no model library, so CI exercises them directly.
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import math
import os
import re
import shutil
import sys
import traceback
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

STEM = "tabpfn_classifier"
PACKAGE = "tabpfn_classifier_pipeline"
SEED = 42
NEW_ROWS = 8
RELOAD_RTOL, RELOAD_ATOL = 1e-5, 1e-6
DEFAULT_DROP_COLUMNS = ["record_id", "category"]  # the synthetic sample's row identifier and its row-index-derived segment
NUMERIC_LIKE_SHARE = 0.9  # an object column whose values are >= 90 % numbers is treated as a numeric column with typos
RUN7_DEVIATION = (
    "in-context conditioning is this tutorial's adaptation; the production pipeline's gradient fine-tuning "
    "(Prior Labs' FinetunedTabPFNClassifier, default on, large-GPU) is not run here — the same fallback production "
    "uses without a large GPU. Proposed RUN7 deviation, pending maintainer approval."
)

# --- External-artifact section (NOTEBOOK_SPEC 3.0 §19.1, XAS1–XAS7, SART1–SART11) ---------------------------------------
# The default artifact of the closing section is a release asset of this repository: the export
# (`outputs/tabpfn_classifier_artifact.zip`) of a recorded earlier hosted run of this notebook's default path (SART8,
# SART9). It is pinned by URL and SHA-256 and never replaced; a changed artifact gets a new tag and a new pin (SART7,
# SART11). The export of the current run is never used as the external artifact (XAS2).
SAMPLE_ARTIFACT_TAG = "sample-artifact-synthetic-v1"
SAMPLE_ARTIFACT_URL = "https://github.com/kurtvalcorza/tabpfn-classifier-pipeline/releases/download/sample-artifact-synthetic-v1/tabpfn_classifier_artifact.zip"  # literal: validators allow only this exact URL
SAMPLE_ARTIFACT_SHA256 = "8fe248e2f9841c452316d07ee3e88e5fcde89ec458ff630954028b1c22df1054"
SAMPLE_ARTIFACT_BYTES = 197068508
SAMPLE_ARTIFACT_FITTED_SHA256 = "1f48ecaf118c834633c8c000579e3f68227cf11c9ae0b146b68509f6f3bb9af6"  # the run's printed EXPECTED_FITTED_SHA256
SAMPLE_ARTIFACT_PRODUCER = {
    "notebook": "tutorials/tabpfn_classifier_colab.ipynb (default path: the deterministic synthetic sample, seed 42)",
    "commit": "7fce7f931e9f65c7d406d436b912f754d3cfb1d1",
    "run": "recorded hosted Colab T4 run; the release notes of the tag name the run record (SART8)",
    "release_tag": SAMPLE_ARTIFACT_TAG,
}
# Staged, NOT published: the v2 re-issue (bundle format v2, model.safetensors) of the v1 asset above, built
# deterministically by re-issuing v1 (model.tabpfn_fit byte-identical, weights = the pinned conversion, a reissue block in
# the manifest). The default stays on v1 until this tag is published; then the four SAMPLE_ARTIFACT_* pins above move to it.
SAMPLE_ARTIFACT_V2_TAG = "sample-artifact-synthetic-v2"
SAMPLE_ARTIFACT_V2_SHA256 = "8828df98a2db8047bc2d52313a5a173bfa499d8b780619899def268dee31b41c"
SAMPLE_ARTIFACT_V2_BYTES = 197034023
SAMPLE_ARTIFACT_MAX_BYTES = 512 * 1024 * 1024  # a download larger than this is refused before it is hashed or opened
EXTERNAL_DIR = "external_artifact"  # under outputs/: the section's predictions and provenance (XAS6)
EXTERNAL_SAMPLE_ROWS = 8
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
REFUSED_EXIT = 3  # a user input refused with a learner-facing message (DAT20); argparse already uses 2, a crash exits 1
RERUN_SECTION_4 = "Fix the file (or the Section 4 fields) and run the Section 4 cell again."


class Refused(ValueError):
    """A user-supplied input that breaks the input contract (DAT19). Not a crash: no traceback is printed for it."""

    def __init__(self, file: str, reason: str) -> None:
        self.file = str(file) or "your input"
        self.reason = str(reason).strip()
        super().__init__(f"{self.file}: {self.reason}")


def refusal(exc: Exception, files: dict[str, str], default: str, fix: str) -> Refused:
    """A rejection of the learner's input as one ``Refused`` naming the file the learner supplied, not the internal
    split name (DAT19): ``train.csv`` of a single uploaded CSV is that CSV, and a ZIP member is ``<zip>/<member>``."""
    finding = getattr(exc, "finding", None)
    message = finding["message"] if finding else str(exc)
    file = default
    for split, shown in files.items():
        if message.startswith(split + ": "):
            file, message = shown, message[len(split) + 2:]
            break
    renamed = {split: shown for split, shown in files.items() if "/" not in shown}  # a ZIP member keeps its own name
    if renamed:
        message = re.sub("|".join(re.escape(s) for s in sorted(renamed, key=len, reverse=True)), lambda m: renamed[m.group(0)], message)
    if finding:
        hint = " Set TARGET_COLUMN to the name of your label column." if finding.get("code") == "TARGET_MISSING" else ""
        message = f"[{finding['code']}] {message} (observed: {finding.get('observed')}).{hint}"
    return Refused(file, f"{message.rstrip()} {fix}")


class Run:
    """Paths of one run: carried sources and state under ``root``; learner-facing files under ``outputs``."""

    def __init__(self, root: Path, outputs: Path, weights: Path, options: dict[str, Any]) -> None:
        self.root = root
        self.out = outputs
        self.weights = weights
        self.options = options
        self.state = root / "state"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)

    def write_state(self, name: str, value: Any) -> Path:
        path = self.state / name
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from Section 4)")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_frame(self, name: str, frame) -> None:
        (self.state / name).write_text(frame.to_json(orient="table", index=False), encoding="utf-8")

    def read_frame(self, name: str, needed_by: str):
        import pandas as pd

        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from Section 4)")
        return pd.read_json(io.StringIO(path.read_text(encoding="utf-8")), orient="table")

    def write_output(self, name: str, value: Any) -> Path:
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path


def package(root: Path):
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return importlib.import_module(PACKAGE)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rounded(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [rounded(v, digits) for v in value]
    return value


def coded(P, exc: Exception, name: str) -> ValueError:
    """An InputRejected re-raised with its code, the file and the observed value, so the kernel shows the rule."""
    finding = getattr(exc, "finding", None)
    if not finding:
        return ValueError(f"{name}: {exc}")
    observed = finding.get("observed")
    hint = " Set TARGET_COLUMN to the name of your label column." if finding.get("code") == "TARGET_MISSING" else ""
    return ValueError(f"{name}: [{finding['code']}] {finding['message']} (observed: {observed}).{hint}")


# --------------------------------------------------------------------------------------------------
# data (TPC-m2, TPC-m3)
# --------------------------------------------------------------------------------------------------


def check_numeric_like(frame, name: str, skip: list[str]) -> None:
    """Refuse a column that is numeric except for a few stray strings, naming the file, the column and the values."""
    import pandas as pd

    for column in frame.columns:
        if column in skip or pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = frame[column].dropna()
        if values.empty:
            continue
        numeric = pd.to_numeric(values.astype(str).str.strip(), errors="coerce")
        if float(numeric.notna().mean()) >= NUMERIC_LIKE_SHARE:
            bad = values[numeric.isna()].astype(str).unique().tolist()[:5]
            raise ValueError(f"{name}: column {column!r} is numeric except for {int(numeric.isna().sum())} value(s) {bad}; it would be treated as a category and lose its numeric meaning. Fix those values (leave missing cells empty), or list the column in TEXT_COLUMNS if it really is categorical.")


def clear_outputs(run: Run) -> list[str]:
    removed = []
    for path in sorted(run.out.glob(f"{STEM}_*")):
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    return removed


def read_byod(P, path: Path) -> dict[str, Any]:
    """The learner's dataset; anything unreadable is refused naming the file the learner supplied (DAT19/DAT20)."""
    if str(path) in ("", ".") or not path.exists():
        raise Refused(str(path) if str(path) not in ("", ".") else "BYOD_PATH", "does not exist in this runtime: upload the file or correct BYOD_PATH, then run the Section 4 cell again.")
    if not (path.is_dir() or path.suffix.lower() in (".zip", ".csv")):
        raise Refused(path.name, "BYOD must be a dataset ZIP (train.csv, optional val.csv/test.csv), a train.csv-style CSV, or a directory holding them. Choose such a file and run the Section 4 cell again.")
    try:
        if path.is_dir():
            return P.read_dataset_dir(path)
        if path.suffix.lower() == ".zip":
            return P.read_dataset_zip(path)
        import pandas as pd

        return {"train.csv": pd.read_csv(path)}
    except (ValueError, zipfile.BadZipFile) as exc:  # pandas' parser and decoding errors are ValueErrors too
        raise Refused(path.name, f"cannot be read as a dataset: {exc}. {RERUN_SECTION_4}") from None


def shown_names(path: Path, frames: dict[str, Any]) -> dict[str, str]:
    """The name the learner knows each split by: a single uploaded CSV is ``train.csv`` here (DAT19)."""
    if path.is_file() and path.suffix.lower() == ".csv":
        return {"train.csv": path.name}
    return {name: f"{path.name}/{name}" for name in frames}


def stage_weights(run: Run) -> None:
    P = package(run.root)
    snapshot = run.weights / P.MODEL_KEY
    snapshot.mkdir(parents=True, exist_ok=True)
    carried = run.root / "weights" / P.MODEL_KEY / P.MANIFEST_NAME
    manifest = json.loads(carried.read_text(encoding="utf-8"))
    if (manifest["modelId"], manifest["revision"]) != (P.MODEL_ID, P.MODEL_REVISION):
        raise RuntimeError("the carried manifest does not name the identity carried by the package; regenerate the notebook")
    shutil.copyfile(carried, snapshot / P.MANIFEST_NAME)
    print({"model_id": P.MODEL_ID, "revision": P.MODEL_REVISION, "license": P.MODEL_LICENSE, "files": len(manifest["files"]), "total_bytes": manifest["totalBytes"]})
    fetched = P.stage_missing_files(snapshot, allow_download=True)
    print({"weights_dir": str(snapshot), "fetched": fetched})
    verified = P.verify_snapshot(snapshot)
    print({"verified_files": len(verified["files"]), "revision": verified["revision"], "checkpoint": P.WEIGHTS_FILE, "sha256": P.WEIGHTS_SHA256})
    served = P.ensure_converted(snapshot)
    print({"served_file": P.CONVERTED_FILE, "sha256": P.CONVERTED_SHA256, "bytes": P.CONVERTED_BYTES, "converted_from": P.WEIGHTS_FILE})
    run.write_state("weights.json", {"snapshot": str(snapshot), "checkpoint": str(snapshot / P.WEIGHTS_FILE), "served": str(served)})


def stage_data(run: Run) -> None:
    P = package(run.root)
    opts = run.options
    use_byod = bool(opts.get("use_byod", False))
    target = opts.get("target_column") or "target"
    drop = list(opts["drop_columns"]) if opts.get("drop_columns") is not None else list(DEFAULT_DROP_COLUMNS)
    text_columns = list(opts.get("text_columns") or [])
    split = float(opts.get("validation_split", P.DEFAULT_VALIDATION_SPLIT))
    removed = clear_outputs(run)
    for name in ("train.json", "val.json", "test.json", "data.json", "validated.json", "condition.json", "export_reference.json", "shown_names.json"):
        (run.state / name).unlink(missing_ok=True)
    if removed:
        print({"removed_previous_outputs": removed})
    if use_byod:
        path = Path(opts.get("byod_path") or "")
        frames = read_byod(P, path)
        origin = {"type": "user-supplied dataset (BYOD)", "file": path.name}
        kind, digest_source = "BYOD", path
        shown = shown_names(path, frames)
    else:
        zip_path = P.build_synthetic_dataset(run.state / "synthetic.zip", rows=600, features=12, classes=3, seed=SEED)
        frames = P.read_dataset_zip(zip_path)
        origin = {"type": "deterministic synthetic tutorial sample", "generator": "build_synthetic_dataset", "rows": 600, "features": 12, "classes": 3, "seed": SEED}
        kind, digest_source = "synthetic", zip_path
        shown = {}
    header = list(map(str, frames["train.csv"].columns))
    unknown = [c for c in drop if c not in header and c not in DEFAULT_DROP_COLUMNS]
    if unknown:
        error = ValueError(f"train.csv: DROP_COLUMNS {unknown} are not in the header {header}. Use the exact column names.")
        raise refusal(error, shown, origin.get("file", ""), RERUN_SECTION_4) if use_byod else error
    drop = [c for c in drop if c in header]  # the default identifiers are simply absent from most BYOD tables
    if target in drop:
        error = ValueError(f"train.csv: DROP_COLUMNS must not contain the target column {target!r}.")
        raise refusal(error, shown, origin.get("file", ""), RERUN_SECTION_4) if use_byod else error
    for name, frame in frames.items():
        try:
            check_numeric_like(frame, name, [target, *drop, *text_columns])
        except ValueError as exc:
            if use_byod:
                raise refusal(exc, shown, origin["file"], RERUN_SECTION_4) from None
            raise
    train, val, test = frames["train.csv"], frames.get("val.csv"), frames.get("test.csv")
    features_only = lambda f: f.drop(columns=[c for c in drop if c in f.columns]) if f is not None else None  # noqa: E731
    # Validate BEFORE any split so a renamed target or a duplicate column is a coded finding, not a KeyError (TPC-m3).
    try:
        P.validate_inputs(features_only(train), target, val=features_only(val), test=features_only(test), names=[origin["type"]])
    except P.InputRejected as exc:
        if use_byod:
            raise refusal(exc, shown, origin["file"], RERUN_SECTION_4) from None
        raise coded(P, exc, "dataset") from None
    split_origin = "explicit train/val/test from the dataset"
    if val is None:
        train, val = P.stratified_holdout(train, target, split, seed=SEED)
        split_origin = f"seeded stratified holdout of {split} drawn from train.csv (independent rows assumed)"
    for key, frame in (("train", train), ("val", val), ("test", test)):
        if frame is not None:
            run.write_frame(f"{key}.json", frame.reset_index(drop=True))
    run.write_state("shown_names.json", shown)
    record = {"sample_kind": kind, **origin, "dataset_sha256": sha256_file(digest_source) if digest_source.is_file() else None, "target": target, "drop_columns": drop, "text_columns": text_columns, "splits": split_origin, "rows": {"train": len(train), "val": len(val), "test": None if test is None else len(test)}}
    run.write_state("data.json", record)
    print(record)
    print("class counts (train):", train[target].astype(str).value_counts().sort_index().to_dict())
    if drop:
        print(f"DROP_COLUMNS {drop}: kept beside the predictions as identifiers, never given to the model.")


def stage_validate(run: Run) -> None:
    P = package(run.root)
    data = run.read_state("data.json", "validate")
    target, drop = data["target"], data["drop_columns"]
    train = run.read_frame("train.json", "validate")
    val = run.read_frame("val.json", "validate")
    test = run.read_frame("test.json", "validate") if (run.state / "test.json").is_file() else None
    strip = lambda f: f.drop(columns=drop) if f is not None else None  # noqa: E731
    print({"ceilings": {"MAX_TRAIN_ROWS": P.MAX_TRAIN_ROWS, "MAX_FEATURES": P.MAX_FEATURES, "MAX_CLASSES": P.MAX_CLASSES, "MIN_CLASSES": P.MIN_CLASSES, "MIN_ROWS_PER_CLASS": P.MIN_ROWS_PER_CLASS}, "decision_rule": P.DECISION_RULE, "model_version": P.MODEL_VERSION})
    byod = data["sample_kind"] == "BYOD"
    shown = run.read_state("shown_names.json", "validate") if byod else {}
    try:
        manifest = P.validate_inputs(strip(train), target, val=strip(val), test=strip(test), names=[data["type"]])
    except P.InputRejected as exc:
        if byod:
            raise refusal(exc, shown, data["file"], RERUN_SECTION_4) from None
        raise coded(P, exc, "dataset") from None
    unseen = [f for f in manifest["findings"] if f.get("code") == "UNSEEN_CLASSES"]
    if unseen:
        # TPC-m3: a validation/test class absent from the support rows can never be predicted and breaks log loss later.
        detail = "; ".join(f"{f['input']}: {f['observed']}" for f in unseen)
        error = ValueError(f"[UNSEEN_CLASSES] classes absent from train.csv: {detail}. Every class in val.csv/test.csv must also occur in train.csv; re-split the data or remove those rows.")
        if byod:
            raise refusal(error, shown, data["file"], RERUN_SECTION_4) from None
        raise error
    try:
        P.validate_inputs(strip(train).rename(columns={target: "label"}), target)
    except P.InputRejected as exc:
        manifest["findings"].append({"input": "renamed-target-probe", **exc.finding})
    entry = manifest["inputs"][0]
    entry["drop_columns"] = drop
    run.write_output(f"{STEM}_input_manifest.json", manifest)
    features, classes = entry["feature_columns"], entry["classes"]
    baseline = P.majority_class_baseline(train[target], val[target], classes=classes)
    run.write_state("validated.json", {"features": features, "classes": classes, "baseline": baseline})
    print(json.dumps({k: entry[k] for k in ("id", "mode", "target_column", "classes", "numeric_features", "categorical_features", "splits", "drop_columns")}, indent=2))
    print("findings:", json.dumps(manifest["findings"], indent=2, default=str))
    print("majority-class baseline on val.csv", rounded(baseline))


# --------------------------------------------------------------------------------------------------
# model stages
# --------------------------------------------------------------------------------------------------


def tables(run: Run, needed_by: str):
    data = run.read_state("data.json", needed_by)
    val_state = run.read_state("validated.json", needed_by)
    train = run.read_frame("train.json", needed_by)
    val = run.read_frame("val.json", needed_by)
    test = run.read_frame("test.json", needed_by) if (run.state / "test.json").is_file() else None
    return data, val_state, train, val, test


def fitted(run: Run, train, data, features, n_estimators: int | None = None):
    P = package(run.root)
    opts = run.read_state("condition.json", "model")["settings"] if (run.state / "condition.json").is_file() and n_estimators is None else {}
    n = int(n_estimators or opts.get("n_estimators", run.options.get("n_estimators", P.DEFAULT_N_ESTIMATORS)))
    seed = int(opts.get("seed", run.options.get("seed", SEED)))
    pipe = P.TabPFNClassifierPipeline.from_pretrained(weights_dir=run.weights / P.MODEL_KEY, n_estimators=n, random_state=seed)
    pipe.fit(train[features], train[data["target"]], target_column=data["target"])
    return pipe


def stage_condition(run: Run) -> None:
    P = package(run.root)
    data, v, train, val, test = tables(run, "condition")
    n = int(run.options.get("n_estimators", P.DEFAULT_N_ESTIMATORS))
    seed = int(run.options.get("seed", SEED))
    if not 1 <= n <= 32:
        raise ValueError(f"N_ESTIMATORS must be from 1 to 32, got {n}")
    (run.state / "condition.json").unlink(missing_ok=True)
    pipe = fitted(run, train, data, v["features"], n_estimators=n)
    if pipe.classes != v["classes"]:
        raise RuntimeError("estimator class order differs from the validated class list")
    target = data["target"]
    metrics = pipe.evaluate(val[v["features"]], val[target])
    test_metrics = None if test is None else pipe.evaluate(test[v["features"]], test[target])
    record = {"settings": {"n_estimators": n, "seed": seed}, "validation": metrics, "test": test_metrics, "device": pipe.device, "classes": pipe.classes, "adaptation": "zero-shot-icl (in-context conditioning, no gradient update)", "run7_deviation": RUN7_DEVIATION}
    run.write_state("condition.json", record)
    print(json.dumps(rounded({"mode": record["adaptation"], "device": pipe.device, "n_estimators": n, "seed": seed, "classes": pipe.classes, "validation": metrics, "test": test_metrics}), indent=2))
    print("Adaptation:", RUN7_DEVIATION)


def classical_reference(train, val, test, target: str, features: list[str]) -> dict[str, Any]:
    """Standardised logistic regression on the numeric features (TPC-S2): a non-trivial classical comparator."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    numeric = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    if not numeric:
        return {"id": "standardised_logistic_regression", "skipped": "no numeric features"}
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    model.fit(train[numeric].fillna(train[numeric].mean()), train[target].astype(str))

    def score(frame):
        pred = model.predict(frame[numeric].fillna(train[numeric].mean()))
        y = frame[target].astype(str).to_numpy()
        return {"accuracy": float(accuracy_score(y, pred)), "balancedAccuracy": float(balanced_accuracy_score(y, pred)), "errors": int((pred != y).sum())}

    out = {"id": "standardised_logistic_regression", "features": "numeric features only", "validation": score(val)}
    if test is not None:
        out["test"] = score(test)
    out["validation_rows"] = int(len(val))
    out["one_row_resolution"] = float(1.0 / len(val)) if len(val) else float(np.nan)
    return out


def stage_report(run: Run) -> None:
    P = package(run.root)
    data, v, train, val, test = tables(run, "report")
    cond = run.read_state("condition.json", "report")
    target = data["target"]
    reference = classical_reference(train, val, test, target, v["features"])
    errors = int(round((1.0 - cond["validation"]["accuracy"]) * len(val)))
    report = P.evaluation_report(cond["validation"], baseline=v["baseline"], n_validation=len(val), class_labels=v["classes"], sample_kind=data["sample_kind"])
    report["test_metrics"] = cond["test"]
    report["classical_reference"] = reference
    report["validation_errors"] = {"tabpfn": errors, "majority_class": int(round((1.0 - v["baseline"]["accuracy"]) * len(val))), **({"standardised_logistic_regression": reference["validation"]["errors"]} if "validation" in reference else {})}
    report["adaptation_deviation"] = RUN7_DEVIATION
    report["interpretation"] = f"One validation row is {1.0 / len(val):.4f} of accuracy ({len(val)} rows); differences of a row or two between models are within that resolution, and no dispersion over splits was measured."
    run.write_output(f"{STEM}_evaluation_report.json", report)
    print(json.dumps(rounded({k: report[k] for k in ("verdict", "reason", "decision_rule", "n_validation", "validation_errors", "classical_reference", "interpretation")}), indent=2))
    if cond["validation"]["accuracy"] < v["baseline"]["accuracy"]:
        print("WARNING: in-context TabPFN does not beat the majority baseline on this holdout; inspect the data before drawing any conclusion.")


def stage_export(run: Run) -> None:
    import numpy as np

    P = package(run.root)
    data, v, train, val, _test = tables(run, "export")
    pipe = fitted(run, train, data, v["features"])
    art = run.out / "artifact"
    shutil.rmtree(art, ignore_errors=True)
    manifest = pipe.save_artifact(art)
    import pandas as pd

    # Record which features the estimator saw as numeric, so whoever loads the bundle can refuse text in them (TPCA-m2).
    manifest["featureKinds"] = {c: ("numeric" if pd.api.types.is_numeric_dtype(train[c]) else "categorical") for c in v["features"]}
    (art / P.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    zip_path = run.out / f"{STEM}_artifact.zip"
    zip_sha = P.zip_artifact_bundle(art, zip_path)
    if manifest["foundationCheckpointSha256"] != P.CONVERTED_SHA256:
        raise RuntimeError("the bundled model.safetensors is not the pinned conversion of the foundation checkpoint")
    run.write_state("export_reference.json", {"proba": np.asarray(pipe.predict_proba(val[v["features"]])).tolist(), "accuracy": run.read_state("condition.json", "export")["validation"]["accuracy"], "device": pipe.device})
    record = {"artifact": sorted(p.name for p in art.iterdir()), "fittedEstimatorSha256": manifest["fittedEstimatorSha256"], "foundationCheckpointSha256": manifest["foundationCheckpointSha256"], "bundle_zip": zip_path.name, "bundle_zip_sha256": zip_sha}
    print(record)
    print(f"Trusted digests of this export: whoever loads {zip_path.name} elsewhere checks EXPECTED_ZIP_SHA256 = '{zip_sha}' and EXPECTED_FITTED_SHA256 = '{manifest['fittedEstimatorSha256']}'. Section 10 never loads this run's export; it loads the pinned sample artifact.")
    run.write_state("export.json", record)


def stage_reload(run: Run) -> None:
    """TPC-m4: a fresh process rebuilds the estimator from the bundle and its probabilities must equal the exporting
    process's on every validation row within an explicit tolerance; the fresh directory is cleared first."""
    import numpy as np

    P = package(run.root)
    data, v, _train, val, _test = tables(run, "reload")
    reference = run.read_state("export_reference.json", "reload")
    fresh_dir = run.out / "artifact-reload"
    shutil.rmtree(fresh_dir, ignore_errors=True)
    shutil.copytree(run.out / "artifact", fresh_dir)
    checked = P.validate_artifact_bundle(fresh_dir, pinned_base=True)
    fresh = P.TabPFNClassifierPipeline.from_artifact(fresh_dir, device=reference["device"], pinned_base=True)
    if fresh.classes != v["classes"] or fresh.feature_columns != v["features"]:
        raise RuntimeError("reloaded artifact disagrees with the validated schema")
    proba = np.asarray(fresh.predict_proba(val[v["features"]]))
    expected = np.asarray(reference["proba"])
    diff = float(np.max(np.abs(proba - expected)))
    labels_identical = bool((proba.argmax(axis=1) == expected.argmax(axis=1)).all())
    reload_check = {"rows": int(len(val)), "maxAbsProbabilityDifference": diff, "rtol": RELOAD_RTOL, "atol": RELOAD_ATOL, "labelsIdentical": labels_identical, "recordedAccuracy": reference["accuracy"], "recordedModelPath": checked["recordedModelPath"], "process": "fresh (separate from the exporting stage)"}
    np.testing.assert_allclose(proba, expected, rtol=RELOAD_RTOL, atol=RELOAD_ATOL, err_msg="The reloaded artifact does not reproduce the exporting process's probabilities. Do not ship this artifact.")
    report_path = run.out / f"{STEM}_evaluation_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["reload_check"] = reload_check
        run.write_output(report_path.name, report)
    run.write_state("reload.json", reload_check)
    print(json.dumps(rounded(reload_check, 8), indent=2))
    print(f"Fresh-boundary verification PASSED: probabilities equivalent on {len(val)} validation rows (rtol={RELOAD_RTOL}, atol={RELOAD_ATOL}).")


def new_rows_frame(data: dict, val, test):
    """The first eight rows of test.csv (else val.csv) without the target, identifiers kept (TPC-S1)."""
    source, label = (test, "test.csv") if test is not None else (val, "val.csv")
    rows = source.drop(columns=[data["target"]]).head(NEW_ROWS).reset_index(drop=True)
    return rows, f"first {NEW_ROWS} rows of {label} (held out from the in-context support), target removed"


def stage_predict(run: Run) -> None:
    import importlib.metadata
    import platform

    import pandas as pd

    P = package(run.root)
    data, v, train, val, test = tables(run, "predict")
    cond = run.read_state("condition.json", "predict")
    reload_check = run.read_state("reload.json", "predict")
    export = run.read_state("export.json", "predict")
    path = run.options.get("new_data_path") or ""
    drop = data["drop_columns"]
    for stale in (f"{STEM}_predictions.csv", f"{STEM}_result.json"):  # a refusal leaves no earlier result on disk (DAT20)
        (run.out / stale).unlink(missing_ok=True)
    fix = "Fix the file and run the Section 9 cell again."
    if path:
        file = Path(path)
        if not file.is_file():
            raise Refused(path, "does not exist in this runtime: upload the file or correct NEW_DATA_PATH, then run the Section 9 cell again.")
        try:
            rows = pd.read_csv(file)
        except ValueError as exc:
            raise Refused(file.name, f"cannot be read as a CSV: {exc}. {fix}") from None
        origin = f"user-supplied CSV (BYOD): {file.name}"
    else:
        rows, origin = new_rows_frame(data, val, test)
        rows.to_csv(run.out / f"{STEM}_new_rows.csv", index=False)
    passthrough = [c for c in rows.columns if c in drop]
    try:
        features = P.validate_new_rows(rows.drop(columns=passthrough), v["features"], target_column=data["target"], classes=v["classes"])
        check_numeric_like(features, Path(path).name if path else "new rows", [])
    except ValueError as exc:
        if path:
            raise refusal(exc, {"new rows": Path(path).name, Path(path).name: Path(path).name}, Path(path).name, fix) from None
        raise coded(P, exc, "new rows") from None
    # The rows are checked before the model is loaded, so a refused file costs no model load (DAT19).
    fresh = P.TabPFNClassifierPipeline.from_artifact(run.out / "artifact", device=cond["device"], pinned_base=True)
    scored = fresh.predict(features)
    out = pd.concat([rows[passthrough].reset_index(drop=True), scored.drop(columns=["row_id"]).reset_index(drop=True)], axis=1) if passthrough else scored
    out.to_csv(run.out / f"{STEM}_predictions.csv", index=False)
    report = json.loads((run.out / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((run.out / f"{STEM}_input_manifest.json").read_text(encoding="utf-8"))
    source = json.loads((run.root / "source.json").read_text(encoding="utf-8")) if (run.root / "source.json").is_file() else {}
    import torch

    payload = {
        "predictions": out.to_dict(orient="records"),
        "metrics": {"validation": cond["validation"], "test": cond["test"]},
        "majority_class_baseline": v["baseline"],
        "evaluation_report": report,
        "fresh_boundary_reload": reload_check,
        "input_manifest": manifest,
        "dataset": {k: data[k] for k in ("sample_kind", "type", "dataset_sha256", "target", "drop_columns", "splits", "rows")} | {"feature_columns": v["features"], "classes": v["classes"]},
        "artifact": export,
        "inference": {"mode": "zero-shot-icl", "adaptation": cond["run7_deviation"], **cond["settings"], "decision_rule": P.DECISION_RULE, "new_rows_origin": origin, "scored_rows": int(len(out)), "identifier_columns": passthrough},
        "notebook_source": source,
        "repository_revision": source.get("revision"),
        "model_id": P.MODEL_ID,
        "model_revision": P.MODEL_REVISION,
        "model_license": P.MODEL_LICENSE,
        "model_file": P.CONVERTED_FILE,
        "source_checkpoint": P.WEIGHTS_FILE,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "tabpfn": importlib.metadata.version("tabpfn"), "pandas": pd.__version__, "scikit_learn": importlib.metadata.version("scikit-learn"), "device": fresh.device},
    }
    run.write_output(f"{STEM}_result.json", payload)
    print("scored rows drawn from:", origin)
    if not path:
        print(f"Their unlabelled copy, with {passthrough or 'no'} identifier column(s), is outputs/{STEM}_new_rows.csv.")
    print(out.round(4).to_string(index=False))


def stage_activity(run: Run) -> None:
    """Optional activity: refit with a different ensemble size and compare validation metrics and probabilities with
    the canonical run; writes only to outputs/activity/."""
    import numpy as np

    package(run.root)
    data, v, train, val, _test = tables(run, "activity")
    n = int(run.options.get("n_estimators", 1))
    if not 1 <= n <= 32:
        raise ValueError(f"ACTIVITY_N_ESTIMATORS must be from 1 to 32, got {n}")
    cond = run.read_state("condition.json", "activity")
    before = {p.name: sha256_file(p) for p in sorted(run.out.glob(f"{STEM}_*")) if p.is_file()}
    changed = fitted(run, train, data, v["features"], n_estimators=n)
    metrics = changed.evaluate(val[v["features"]], val[data["target"]])
    reference = run.read_state("export_reference.json", "activity") if (run.state / "export_reference.json").is_file() else None
    record = {"changed": {"n_estimators": [cond["settings"]["n_estimators"], n]}, "validation": {"canonical": cond["validation"], "changed": metrics}, "validation_rows": len(val), "one_row_resolution": 1.0 / len(val)}
    if reference is not None:
        diff = np.abs(np.asarray(changed.predict_proba(val[v["features"]])) - np.asarray(reference["proba"]))
        record["max_abs_probability_change"] = float(diff.max())
        record["labels_changed"] = int((np.asarray(changed.predict_proba(val[v["features"]])).argmax(axis=1) != np.asarray(reference["proba"]).argmax(axis=1)).sum())
    run.write_output(f"activity/{STEM}_activity_n_estimators_{n}.json", record)
    if {p.name: sha256_file(p) for p in sorted(run.out.glob(f"{STEM}_*")) if p.is_file()} != before:
        raise RuntimeError("The activity changed a canonical output; it must write only to outputs/activity/.")
    print(json.dumps(rounded(record), indent=2))
    print({"canonical_outputs_unchanged": True})


# --- External-artifact section (§19.1): stages external_artifact → external_reconstruct → external_predict ---------------
# These stages read only the carried files, the run directory's own `inputs/` cache and their own state; they never read
# a file that the data, validate, condition, report, export, reload or predict stages wrote (XAS5).


def check_trusted_digest(path: Path, expected: str, field: str) -> str:
    """Refuse a file whose SHA-256 is not ``expected``; nothing is extracted or deserialised before this passes."""
    expected = str(expected).strip().lower()
    if not SHA256_HEX.match(expected):
        raise ValueError(f"{field} must be 64 hexadecimal characters (the digest the producer printed when it exported), got {expected!r}.")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"Trusted digest mismatch for {path.name}: {field} is {expected}, the file's SHA-256 is {observed}. This is not the artifact the digest was issued for; nothing was extracted. Obtain the artifact and its digest from the producer again.")
    return observed


def fetch_pinned_artifact(cache: Path, url: str, expected: str) -> Path:
    """The pinned sample artifact: reused from ``cache`` when its digest matches, else downloaded (no authentication),
    size-capped, and accepted only at the pinned SHA-256. The digest is checked before anything opens the file."""
    if cache.is_file() and sha256_file(cache) == expected:
        return cache
    import urllib.request

    cache.parent.mkdir(parents=True, exist_ok=True)
    part = cache.with_name(cache.name + ".part")
    size = 0
    with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            if size > SAMPLE_ARTIFACT_MAX_BYTES:
                out.close()
                part.unlink(missing_ok=True)
                raise ValueError(f"The pinned sample artifact at {url} exceeds {SAMPLE_ARTIFACT_MAX_BYTES} bytes: refused before hashing or extraction.")
            out.write(chunk)
    try:
        check_trusted_digest(part, expected, "the pinned SAMPLE_ARTIFACT_SHA256")
    except ValueError:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, cache)
    return cache


def check_members(P, zip_path: Path) -> list[str]:
    """The ZIP must hold exactly the three bundle members of one layout (v2 ``model.safetensors`` or the deprecated v1
    ``model.ckpt``), each once, at the top level; anything else (both checkpoints, an extra file, a sub-directory copy
    that would overwrite a member when flattened) is refused before extraction."""
    with zipfile.ZipFile(zip_path) as archive:
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
    bare = [PurePosixPath(n.replace("\\", "/")).name for n in names]
    weights = P.LEGACY_CHECKPOINT_NAME if P.LEGACY_CHECKPOINT_NAME in bare and P.CHECKPOINT_NAME not in bare else P.CHECKPOINT_NAME
    expected = {P.ARTIFACT_MANIFEST_NAME, P.FITTED_NAME, weights}
    duplicates = sorted({b for b in bare if bare.count(b) > 1})
    if duplicates:
        raise ValueError(f"{zip_path.name}: members {duplicates} occur more than once (e.g. in a sub-directory); extraction would overwrite one with the other. Refusing the artifact.")
    nested = [n for n in names if "/" in n.replace("\\", "/")]
    unexpected = sorted(set(bare) - expected)
    missing = sorted(expected - set(bare))
    if nested or unexpected or missing:
        raise ValueError(f"{zip_path.name}: an artifact holds exactly {sorted(expected)} at the top level; unexpected={unexpected}, nested={nested}, missing={missing}. Refusing the artifact.")
    return sorted(bare)


def check_base_model(P, manifest: dict[str, Any]) -> dict[str, Any]:
    """The manifest must name the pinned base model carried by this notebook, and its weights must BE that model: a v2
    ``model.safetensors`` the pinned conversion (``CONVERTED_SHA256``), a deprecated v1 ``model.ckpt`` the pinned
    checkpoint (``WEIGHTS_SHA256``); ``validate_artifact_bundle(pinned_base=True)`` checks the bytes. A swapped
    checkpoint behind a re-hashed manifest is refused."""
    base = manifest.get("baseModel") or {}
    declared = (base.get("modelId"), base.get("revision"), base.get("sha256"))
    if declared != (P.MODEL_ID, P.MODEL_REVISION, P.WEIGHTS_SHA256):
        raise RuntimeError(f"Artifact was not produced on the pinned base model carried by this notebook: declares {declared}, expected {(P.MODEL_ID, P.MODEL_REVISION, P.WEIGHTS_SHA256)}.")
    layout = manifest.get("schemaVersion")
    pinned = P.pinned_checkpoint_sha256(layout)
    if manifest.get("foundationCheckpointSha256") != pinned:
        raise RuntimeError(f"The artifact's {manifest.get('foundationCheckpoint')} must be the pinned TabPFN-3 weights for bundle layout v{layout} (SHA-256 {pinned}); its manifest records {manifest.get('foundationCheckpointSha256')}. Refusing to load a different checkpoint.")
    binding = "identical to the pinned conversion of the base checkpoint (SHA-256 checked; bundle format v2)" if layout == P.ARTIFACT_SCHEMA_VERSION else "identical to the pinned base checkpoint (SHA-256 checked; deprecated bundle format v1, read with weights_only=True)"
    return {"mode": manifest.get("mode"), "checkpoint_binding": binding}


def check_numeric_features(name: str, rows, kinds: dict[str, str]):
    """Columns the fitted estimator saw as numeric must hold numbers; refuse naming the column and the first bad values,
    instead of handing strings to the model."""
    import pandas as pd

    out = rows.copy()
    for column, kind in kinds.items():
        if kind != "numeric" or column not in out.columns:
            continue
        original = out[column]
        converted = pd.to_numeric(original, errors="coerce")
        bad = original.notna() & converted.isna()
        if bad.any():
            examples = original[bad].astype(str).unique().tolist()[:5]
            raise ValueError(f"{name}: numeric feature {column!r} has {int(bad.sum())} non-numeric value(s), e.g. {examples}. Fix them (leave missing cells empty).")
        out[column] = converted
    return out


def feature_kinds(manifest: dict[str, Any], rows) -> dict[str, str]:
    """The kinds the export recorded (``featureKinds``); for an artifact without them, a column whose values are mostly
    numbers is treated as numeric, so a stray string in it is still caught."""
    import pandas as pd

    recorded = manifest.get("featureKinds")
    if recorded:
        return dict(recorded)
    kinds = {}
    for c in manifest["featureColumns"]:
        if c not in rows.columns:
            continue
        values = rows[c].dropna()
        share = float(pd.to_numeric(values.astype(str).str.strip(), errors="coerce").notna().mean()) if len(values) else 1.0
        kinds[c] = "numeric" if pd.api.types.is_numeric_dtype(rows[c]) or share > 0.5 else "categorical"
    return kinds


def synthetic_frames(run: Run, P) -> dict[str, Any]:
    """The default path's deterministic synthetic table, drawn again in code under ``inputs/`` (no download and no
    main-path state): the same generator, size and seed as the ``data`` stage."""
    zip_path = P.build_synthetic_dataset(run.root / "inputs" / "external_synthetic.zip", rows=600, features=12, classes=3, seed=SEED)
    return P.read_dataset_zip(zip_path)


def synthetic_contract(run: Run, P) -> dict[str, Any]:
    """The data contract of the current default path: features (identifiers dropped), target, class labels and kinds."""
    train = synthetic_frames(run, P)["train.csv"]
    features = [c for c in train.columns if c != "target" and c not in DEFAULT_DROP_COLUMNS]
    return {"task": P.TASK_TYPE, "featureColumns": features, "targetColumn": "target", "classes": sorted(str(c) for c in train["target"].unique()), "featureKinds": {c: "numeric" for c in features}}


def check_data_contract(manifest: dict[str, Any], contract: dict[str, Any]) -> None:
    """SART10: the artifact must follow the current default path's features, target, labels and feature kinds."""
    for key in ("featureColumns", "targetColumn", "classes", "featureKinds"):
        if manifest.get(key) != contract[key]:
            raise RuntimeError(f"The artifact does not follow this notebook's default data contract (SART10): its {key} is {manifest.get(key)!r}, the synthetic default path's is {contract[key]!r}. A sample artifact built from another dataset or an earlier default is refused even when its digest verifies.")


def external_out(run: Run) -> Path:
    return run.out / EXTERNAL_DIR


def stage_external_artifact(run: Run) -> None:
    P = package(run.root)
    opts = run.options
    source = opts.get("source", "pinned")
    shutil.rmtree(external_out(run), ignore_errors=True)  # a failed check leaves no earlier success record behind (VER6)
    (run.state / "external_artifact.json").unlink(missing_ok=True)
    bundle = run.state / "external_bundle"
    shutil.rmtree(bundle, ignore_errors=True)
    expected_fitted = ""
    if source == "pinned":
        # SART3: provenance, format and integrity mechanism, stated before anything is downloaded or deserialised.
        print({"artifact": "pinned sample artifact (created outside this run)", "url": SAMPLE_ARTIFACT_URL, "pinned_sha256": SAMPLE_ARTIFACT_SHA256, "bytes": SAMPLE_ARTIFACT_BYTES, "pinned_fitted_sha256": SAMPLE_ARTIFACT_FITTED_SHA256, "producer": SAMPLE_ARTIFACT_PRODUCER, "format": f"artifact_manifest.json schemaVersion {P.ARTIFACT_SCHEMA_VERSION} + {P.FITTED_NAME} + {P.CHECKPOINT_NAME}", "integrity": "SHA-256 of the whole ZIP, checked before extraction; then the member list, the fitted archive against its pinned digest, and the checkpoint against the pinned base model"})
        zip_path = fetch_pinned_artifact(run.root / "inputs" / "external_artifact.zip", SAMPLE_ARTIFACT_URL, SAMPLE_ARTIFACT_SHA256)
        expected_fitted = SAMPLE_ARTIFACT_FITTED_SHA256
        trust = {"trusted_digest": "verified (pinned SAMPLE_ARTIFACT_SHA256 and SAMPLE_ARTIFACT_FITTED_SHA256)", "zip_sha256": SAMPLE_ARTIFACT_SHA256, "url": SAMPLE_ARTIFACT_URL, "producer": SAMPLE_ARTIFACT_PRODUCER}
    elif source == "own":
        given = str(opts.get("zip_path") or "")
        zip_path = Path(given)
        if not given or not zip_path.is_file():
            raise Refused(given or "ARTIFACT_ZIP_PATH", "is not a file in this runtime: point ARTIFACT_ZIP_PATH at the artifact ZIP you were given, or upload it, then run this cell again.")
        expected = str(opts.get("expected_zip_sha256") or "").strip()
        if expected:
            try:
                digest = check_trusted_digest(zip_path, expected, "EXPECTED_ZIP_SHA256")
            except ValueError as exc:
                raise Refused(zip_path.name, f"{exc} Then run this cell again.") from None
            trust = {"trusted_digest": "verified (EXPECTED_ZIP_SHA256)", "zip_sha256": digest}
        else:
            trust = {"trusted_digest": "not supplied", "zip_sha256": sha256_file(zip_path)}
            print("No EXPECTED_ZIP_SHA256 was supplied: the checks below establish internal consistency only, not that this is the artifact you were sent. Ask the producer for the digest it printed when it exported.")
    else:
        raise ValueError(f"ARTIFACT_SOURCE must be the pinned sample or your own ZIP, got {source!r}")
    try:
        members = check_members(P, zip_path)
        P.safe_extract_zip(zip_path, bundle)
        artifact = P.validate_artifact_bundle(bundle, expected_fitted_sha256=expected_fitted, pinned_base=True)
        manifest = {k: v for k, v in artifact.items() if k not in ("verifiedSha256", "recordedModelPath", "layout")}
        binding = check_base_model(P, manifest)
    except (ValueError, RuntimeError, FileNotFoundError, zipfile.BadZipFile) as exc:
        if source != "own":
            raise  # the pinned sample failing these checks is an integrity error, not a refusal: keep the traceback
        shutil.rmtree(bundle, ignore_errors=True)
        raise Refused(zip_path.name, f"is not a usable artifact for this notebook: {exc} Use an artifact ZIP that this notebook exported in Section 8, and run this cell again.") from None
    contract = None
    if source == "pinned":
        contract = synthetic_contract(run, P)
        check_data_contract(manifest, contract)
    record = {"source": source, "zip": zip_path.name, **trust, **binding, "members": members, "bundle": str(bundle), "manifest": manifest, "verifiedSha256": artifact["verifiedSha256"], "data_contract": "synthetic default path (features, target, classes and feature kinds match)" if contract else "not checked (your own artifact; your rows must follow its manifest)"}
    run.write_state("external_artifact.json", record)
    print({k: record[k] for k in ("source", "zip", "zip_sha256", "trusted_digest", "mode", "checkpoint_binding", "data_contract")})
    print({"schemaVersion": manifest.get("schemaVersion"), "bundleFormat": manifest.get("bundleFormat", "v1 (deprecated: model.ckpt)"), "taskType": manifest.get("taskType"), "baseModel": manifest.get("baseModel"), "nEstimators": manifest.get("nEstimators"), "randomState": manifest.get("randomState"), "fittedEstimatorSha256": artifact["verifiedSha256"]["fittedEstimator"], "foundationCheckpointSha256": artifact["verifiedSha256"]["foundationCheckpoint"]})
    print({"featureColumns": len(manifest["featureColumns"]), "targetColumn": manifest["targetColumn"], "classes": manifest["classes"], "featureKinds": sorted(set((manifest.get("featureKinds") or {}).values()))})


def external_serving(run: Run):
    """Serving state rebuilt from the validated artifact's files alone: the fitted estimator (which holds the support
    rows and the fitted preprocessing) and the pinned checkpoint. Nothing is refitted and nothing is downloaded."""
    import torch

    P = package(run.root)
    art = run.read_state("external_artifact.json", "external_reconstruct")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    serving = P.TabPFNClassifierPipeline.from_artifact(Path(art["bundle"]), device=device, expected_fitted_sha256=art["verifiedSha256"]["fittedEstimator"], pinned_base=True)
    manifest = art["manifest"]
    if serving.classes != [str(c) for c in manifest["classes"]] or serving.feature_columns != list(manifest["featureColumns"]):
        raise RuntimeError("Reconstructed estimator disagrees with the validated manifest.")
    return P, art, manifest, serving


def stage_external_reconstruct(run: Run) -> None:
    _P, art, _manifest, serving = external_serving(run)
    print({"source": serving.source, "device": serving.device, "classes": serving.classes, "features": len(serving.feature_columns or []), "n_estimators": serving.n_estimators, "random_state": serving.random_state, "refit": False, "network_fallback_for_weights": False, "checkpoint_binding": art["checkpoint_binding"]})


def external_sample_rows(run: Run, P):
    """The default input: the first eight rows of the synthetic sample's ``test.csv`` (never among the artifact's support
    rows), label removed, identifiers kept. They follow the default data contract (SART10)."""
    test = synthetic_frames(run, P)["test.csv"]
    return test.drop(columns=["target"]).head(EXTERNAL_SAMPLE_ROWS).reset_index(drop=True)


def stage_external_predict(run: Run) -> None:
    import importlib.metadata
    import platform

    import pandas as pd

    P = package(run.root)
    art = run.read_state("external_artifact.json", "external_predict")
    manifest = art["manifest"]
    features, classes, target = list(manifest["featureColumns"]), [str(c) for c in manifest["classes"]], manifest["targetColumn"]
    path = str(run.options.get("rows_path") or "")
    for stale in (f"{STEM}_external_predictions.csv", f"{STEM}_external_result.json"):  # no earlier result survives a refusal (DAT20)
        (external_out(run) / stale).unlink(missing_ok=True)
    fix = "Fix the file and run this cell again."
    if path:
        file = Path(path)
        if not file.is_file():
            raise Refused(path, "does not exist in this runtime: upload the file or correct EXTERNAL_ROWS_PATH, then run this cell again.")
        try:
            rows, name, kind = pd.read_csv(file), file.name, "BYOD"
        except ValueError as exc:
            raise Refused(file.name, f"cannot be read as a CSV: {exc}. {fix}") from None
    else:
        if art["source"] != "pinned":
            raise Refused(art["zip"], "is your own artifact, and the synthetic sample rows follow the pinned sample artifact's contract only. Tick USE_OWN_ROWS (or set EXTERNAL_ROWS_PATH) with rows that follow its manifest, and run this cell again.")
        rows, name, kind = external_sample_rows(run, P), f"synthetic-test-first-{EXTERNAL_SAMPLE_ROWS}-rows.csv", "sample"
    passthrough = [c for c in rows.columns if c in DEFAULT_DROP_COLUMNS and c not in features]
    try:
        frame = P.validate_new_rows(rows.drop(columns=passthrough), features, target_column=target, classes=classes)
        frame = check_numeric_features(name, frame, feature_kinds(manifest, frame))
    except ValueError as exc:
        if kind == "BYOD":
            raise refusal(exc, {"new rows": name, name: name}, name, fix) from None
        raise coded(P, exc, name) from None
    findings = []
    try:  # a probe that breaks the artifact's schema must be refused; the finding is recorded, not swallowed
        P.validate_new_rows(frame.drop(columns=[features[0]]), features, target_column=target, classes=classes)
    except P.InputRejected as exc:
        findings.append({"input": "missing-column-probe", "verdict": "rejected", **exc.finding})
    else:
        raise RuntimeError("The missing-column probe was accepted: input validation is not working, so no prediction is written.")
    _P, art, manifest, serving = external_serving(run)  # loaded only after the rows passed their checks (DAT19)
    scored = serving.predict(frame)
    out = pd.concat([rows[passthrough].reset_index(drop=True), scored.drop(columns=["row_id"]).reset_index(drop=True)], axis=1) if passthrough else scored
    target_dir = external_out(run)
    target_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(target_dir / f"{STEM}_external_predictions.csv", index=False)
    import torch

    source = json.loads((run.root / "source.json").read_text(encoding="utf-8")) if (run.root / "source.json").is_file() else {}
    payload = {
        "section": "external artifact (NOTEBOOK_SPEC 3.0 §19.1): an artifact made outside this run, validated and served here",
        "predictions": out.to_dict(orient="records"),
        "input_manifest": {"inputs": [{"id": name, "kind": kind, "rows": int(len(frame)), "feature_columns": features, "identifier_columns": passthrough, "missing_values": {k: int(v) for k, v in frame.isna().sum().items() if v}}], "verdict": "accepted", "findings": findings},
        "artifact": {k: art.get(k) for k in ("source", "zip", "zip_sha256", "url", "producer", "trusted_digest", "mode", "checkpoint_binding", "data_contract", "members", "verifiedSha256")} | {"manifest": manifest},
        "reconstruction": {"loader": "tabpfn_classifier_pipeline.TabPFNClassifierPipeline.from_artifact", "device": serving.device, "refit": False, "network_fallback_for_weights": False},
        "inference": {"nEstimators": serving.n_estimators, "randomState": serving.random_state, "decisionRule": P.DECISION_RULE + " over uncalibrated class probabilities", "threshold": None, "classes": classes},
        "notebook_source": source,
        "model_id": P.MODEL_ID,
        "model_revision": P.MODEL_REVISION,
        "model_license": P.MODEL_LICENSE,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "tabpfn": importlib.metadata.version("tabpfn"), "pandas": pd.__version__, "device": serving.device},
    }
    run.write_output(f"{EXTERNAL_DIR}/{STEM}_external_result.json", payload)
    print("scored rows drawn from:", name, f"({kind})")
    print(out.round(4).to_string(index=False))
    print({"outputs": [f"outputs/{EXTERNAL_DIR}/{STEM}_external_predictions.csv", f"outputs/{EXTERNAL_DIR}/{STEM}_external_result.json"], "input_kind": kind, "artifact_source": art["source"], "zip_sha256": art["zip_sha256"]})


STAGES = {
    "weights": stage_weights,
    "data": stage_data,
    "validate": stage_validate,
    "condition": stage_condition,
    "report": stage_report,
    "export": stage_export,
    "reload": stage_reload,
    "predict": stage_predict,
    "activity": stage_activity,
    "external_artifact": stage_external_artifact,
    "external_reconstruct": stage_external_reconstruct,
    "external_predict": stage_external_predict,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--options", default="{}")
    args = parser.parse_args(argv)
    run = Run(args.root.resolve(), args.outputs.resolve(), args.weights.resolve(), json.loads(args.options))
    error_file = run.state / f"{args.stage}.error.json"
    error_file.unlink(missing_ok=True)
    try:
        STAGES[args.stage](run)
    except Refused as exc:  # the learner's input was refused, not a crash: no traceback, a dedicated exit code (DAT20)
        error_file.write_text(json.dumps({"type": "Refused", "refused": True, "file": exc.file, "message": str(exc)}), encoding="utf-8")
        return REFUSED_EXIT
    except BaseException as exc:  # noqa: BLE001 -- every failure is reported to the kernel with its own message
        traceback.print_exc()
        error_file.write_text(json.dumps({"type": type(exc).__name__, "message": str(exc)}), encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
