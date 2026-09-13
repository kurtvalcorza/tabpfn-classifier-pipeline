"""TabPFN-3 tabular classification — public tutorial API (DIMER pipeline, standalone-notebook carrier).

Inference-only: the pinned TabPFN-3 classifier checkpoint conditions on the labelled training rows **in context**
(``fit`` registers the support set; no gradient update) and scores query rows in a forward pass. The DIMER
fine-tuning path of this pipeline lives in the private ``tabpfn-classifier-finetuner`` worker and is NOT carried
here (private code; TabPFN-3 weights are non-commercial). This module owns the pinned snapshot scheme, the input
contract (DAT24), the ICL fit / predict / evaluate calls, the ``model.tabpfn_fit`` + ``model.ckpt`` +
``artifact_manifest.json`` bundle the serving path consumes, its pre-load validation, and the evaluation report
(EVAL21). ``tabpfn`` / ``torch`` are imported lazily inside the functions that need them.
"""
# ruff: noqa: E501  -- contract dictionaries and messages are kept on single lines
from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

MODEL_ID = "Prior-Labs/tabpfn_3"
MODEL_REVISION = "24a16a89d245878b846555110985634aa2e656d7"
# Non-commercial licence: testing, evaluation and internal benchmarking only (no production deployment).
MODEL_LICENSE = "tabpfn-3-license-v1.0"
MODEL_KEY = "tabpfn-3-classifier"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
WEIGHTS_FILE = "tabpfn-v3-classifier-v3_default.ckpt"
WEIGHTS_SHA256 = "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988"
WEIGHTS_BYTES = 212804803
MODEL_VERSION = "v3"
TABPFN_VERSION = "8.1.0"

TASK_TYPE = "tabular_classification"
# Ceilings of the selected generation (contract/model-versions.json, `v3`): rows, features, classes.
MAX_TRAIN_ROWS = 1_000_000
MAX_FEATURES = 2000
MAX_CLASSES = 160
MIN_CLASSES = 2
MIN_ROWS_PER_CLASS = 2
DEFAULT_N_ESTIMATORS = 4
DEFAULT_SEED = 42
DEFAULT_VALIDATION_SPLIT = 0.2
MAX_ZIP_EXPANDED_BYTES = 512 * 1024 * 1024  # BYOD ZIP expansion ceiling (512 MiB)
MAX_ZIP_RATIO = 200  # compression-bomb guard: expanded / compressed
DECISION_RULE = "argmax"  # predict() is argmax over predict_proba(); the probabilities are raw ensemble outputs, uncalibrated
METRIC_IDS = ("accuracy", "balancedAccuracy", "logLoss", "rocAuc")
ARTIFACT_SCHEMA_VERSION = 1
FITTED_NAME = "model.tabpfn_fit"
CHECKPOINT_NAME = "model.ckpt"
ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"


# --------------------------------------------------------------------------- snapshot scheme


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("modelId"), manifest.get("revision")) != (MODEL_ID, MODEL_REVISION):
        raise ValueError(f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, expected {MODEL_ID}@{MODEL_REVISION}")
    if not isinstance(manifest.get("files"), list) or not manifest["files"]:
        raise ValueError("manifest has no file entries")
    return manifest


def verify_snapshot(weights_dir: str | Path | None = None) -> dict[str, Any]:
    """Re-hash every manifest entry under ``weights_dir``; raise on any missing file, size or digest mismatch."""
    root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
    manifest = _read_manifest(root)
    checkpoint_ok = False
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {path}")
        size = path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
        if entry["path"] == WEIGHTS_FILE:
            checkpoint_ok = digest == WEIGHTS_SHA256 and size == WEIGHTS_BYTES
    if not checkpoint_ok:
        raise ValueError(f"manifest does not pin {WEIGHTS_FILE} at sha256 {WEIGHTS_SHA256} / {WEIGHTS_BYTES} bytes")
    return manifest


def stage_missing_files(weights_dir: str | Path | None = None, *, allow_download: bool = False, downloader: Callable[..., Any] | None = None) -> list[str]:
    """Fetch the manifest entries absent from ``weights_dir`` (revision-pinned, never ``main``); refuse without ``allow_download``."""
    root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
    manifest = _read_manifest(root)
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(f"snapshot at {root} is missing {missing}; pass allow_download=True to stage them from {MODEL_ID}@{MODEL_REVISION[:12]}")
    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download
    for rel in missing:
        downloader(repo_id=MODEL_ID, filename=rel, revision=MODEL_REVISION, local_dir=str(root))
        if not (root / rel).is_file():
            raise FileNotFoundError(f"download did not produce {root / rel}")
    return missing


# --------------------------------------------------------------------------- datasets


def build_synthetic_dataset(out: str | Path, rows: int = 600, features: int = 12, classes: int = 3, seed: int = DEFAULT_SEED) -> Path:
    """Deterministic train/val/test classification ZIP (mirrors examples/build_synthetic_dataset.py); tutorial data only."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    if rows < max(50, classes * 6):
        raise ValueError("rows is too small for a useful stratified train/val/test dataset")
    if features < 2 or classes < 2:
        raise ValueError("features must be at least 2 and classes at least 2")
    informative = min(features, max(2, classes * 2))
    redundant = min(max(0, features - informative), max(0, features // 4))
    X, y = make_classification(n_samples=rows, n_features=features, n_informative=informative, n_redundant=redundant, n_repeated=0, n_classes=classes, n_clusters_per_class=1, class_sep=1.2, flip_y=0.02, random_state=seed)
    frame = pd.DataFrame(X, columns=[f"feature_{i:02d}" for i in range(features)])
    frame["category"] = [f"segment_{int(index) % 4}" for index in range(rows)]
    frame["record_id"] = [f"row-{index:05d}" for index in range(rows)]
    frame["target"] = [f"class_{value}" for value in y]
    train_plus_val, test = train_test_split(frame, test_size=0.15, random_state=seed, stratify=frame["target"])
    train, val = train_test_split(train_plus_val, test_size=0.17647058823529413, random_state=seed, stratify=train_plus_val["target"])
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, part in (("train.csv", train), ("val.csv", val), ("test.csv", test)):
            archive.writestr(name, part.to_csv(index=False))
    return out


def _safe_member(name: str) -> str:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise ValueError(f"unsafe archive member: {name!r}")
    return path.name


def safe_extract_zip(zip_path: str | Path, destination: str | Path, max_expanded_bytes: int = MAX_ZIP_EXPANDED_BYTES, max_ratio: int = MAX_ZIP_RATIO) -> list[str]:
    """Extract a ZIP member by member (bare file names only, no directories/absolute/traversing paths); never ``extractall``."""
    zip_path, destination = Path(zip_path), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        expanded = sum(info.file_size for info in infos)
        compressed = max(1, sum(info.compress_size for info in infos))
        if expanded > max_expanded_bytes:
            raise ValueError(f"archive expands to {expanded} bytes > ceiling {max_expanded_bytes}")
        if expanded / compressed > max_ratio:
            raise ValueError(f"archive compression ratio {expanded / compressed:.0f} > ceiling {max_ratio}")
        for info in infos:
            name = _safe_member(info.filename)
            with archive.open(info) as source, open(destination / name, "wb") as target:
                shutil.copyfileobj(source, target)
            written.append(name)
    return written


def read_dataset_zip(zip_path: str | Path, *, max_expanded_bytes: int = MAX_ZIP_EXPANDED_BYTES, max_ratio: int = MAX_ZIP_RATIO) -> dict[str, pd.DataFrame]:
    """Read ``train.csv`` (+ optional ``val.csv`` / ``test.csv``) from a ZIP with archive-safety checks; never ``extractall``."""
    zip_path = Path(zip_path)
    frames: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(zip_path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        expanded = sum(info.file_size for info in infos)
        compressed = max(1, sum(info.compress_size for info in infos))
        if expanded > max_expanded_bytes:
            raise ValueError(f"archive expands to {expanded} bytes > ceiling {max_expanded_bytes}")
        if expanded / compressed > max_ratio:
            raise ValueError(f"archive compression ratio {expanded / compressed:.0f} > ceiling {max_ratio}")
        for info in infos:
            name = _safe_member(info.filename)
            if name in ("train.csv", "val.csv", "test.csv"):
                if name in frames:
                    raise ValueError(f"archive carries {name} more than once")
                frames[name] = pd.read_csv(io.BytesIO(archive.read(info.filename)))
    if "train.csv" not in frames:
        raise ValueError("archive has no train.csv")
    return frames


def read_dataset_dir(dataset_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read a ZIP (``*.zip``) or loose ``train.csv`` / ``val.csv`` / ``test.csv`` from a directory."""
    root = Path(dataset_dir)
    zips = sorted(root.glob("*.zip"))
    if zips:
        if len(zips) > 1:
            raise ValueError(f"expected one ZIP in {root}, found {len(zips)}")
        return read_dataset_zip(zips[0])
    frames = {name: pd.read_csv(root / name) for name in ("train.csv", "val.csv", "test.csv") if (root / name).is_file()}
    if "train.csv" not in frames:
        raise ValueError(f"{root} has neither a ZIP nor train.csv")
    return frames


def stratified_holdout(train: pd.DataFrame, target_column: str, validation_split: float = DEFAULT_VALIDATION_SPLIT, seed: int = DEFAULT_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded stratified split for a single ``train.csv``; wrong for temporal/grouped data (supply explicit splits instead)."""
    from sklearn.model_selection import train_test_split

    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be in (0, 1)")
    fit_part, val_part = train_test_split(train, test_size=validation_split, random_state=seed, stratify=train[target_column].astype(str))
    return fit_part.reset_index(drop=True), val_part.reset_index(drop=True)


# --------------------------------------------------------------------------- input contract (DAT24)

INPUT_SCHEMA = {
    "format": "CSV table(s): train.csv with an optional val.csv / test.csv (ZIP or loose files); one row per observation",
    "target": "one categorical target column (declared name), no missing values, >= MIN_CLASSES classes with >= MIN_ROWS_PER_CLASS rows each",
    "features": "every other column; numeric or categorical (strings); unique column names; identical schema across splits",
    "ceilings": {"MAX_TRAIN_ROWS": MAX_TRAIN_ROWS, "MAX_FEATURES": MAX_FEATURES, "MAX_CLASSES": MAX_CLASSES},
    "splits": "explicit val.csv/test.csv are preserved; without val.csv a seeded stratified holdout is drawn (independent rows assumed)",
    "reserved_columns": "no `prediction` or `proba_<class>` columns in inputs",
}


class InputRejected(ValueError):
    """Raised by ``validate_inputs`` with a structured finding."""

    def __init__(self, finding: dict[str, Any]) -> None:
        super().__init__(finding["message"])
        self.finding = finding


def _reject(code: str, message: str, observed: Any = None) -> InputRejected:
    return InputRejected({"code": code, "verdict": "rejected", "message": message, "observed": observed})


def _check_frame(name: str, frame: pd.DataFrame, target_column: str, feature_columns: list[str] | None) -> list[str]:
    columns = list(frame.columns)
    duplicates = sorted({c for c in columns if columns.count(c) > 1})
    if duplicates:
        raise _reject("DUPLICATE_COLUMNS", f"{name}: duplicate column names", duplicates)
    if target_column not in columns:
        raise _reject("TARGET_MISSING", f"{name}: target column {target_column!r} not present", columns)
    features = [c for c in columns if c != target_column]
    reserved = [c for c in features if c == "prediction" or c.startswith("proba_")]
    if reserved:
        raise _reject("RESERVED_COLUMNS", f"{name}: reserved output columns present", reserved)
    if feature_columns is not None and features != feature_columns:
        raise _reject("SCHEMA_MISMATCH", f"{name}: feature columns differ from train.csv", {"expected": feature_columns, "observed": features})
    if frame[target_column].isna().any():
        raise _reject("TARGET_MISSING_VALUES", f"{name}: target has missing values", int(frame[target_column].isna().sum()))
    if len(frame) == 0:
        raise _reject("EMPTY_SPLIT", f"{name}: no rows", 0)
    return features


def validate_inputs(train: pd.DataFrame, target_column: str, *, val: pd.DataFrame | None = None, test: pd.DataFrame | None = None, names: Sequence[str] | None = None) -> dict[str, Any]:
    """Apply the input contract to the supplied splits and return the DAT24 input manifest (raises ``InputRejected``)."""
    features = _check_frame("train.csv", train, target_column, None)
    if not features:
        raise _reject("NO_FEATURES", "train.csv has no feature columns", list(train.columns))
    if len(features) > MAX_FEATURES:
        raise _reject("TOO_MANY_FEATURES", f"train.csv has {len(features)} features > MAX_FEATURES {MAX_FEATURES}", len(features))
    if len(train) > MAX_TRAIN_ROWS:
        raise _reject("TOO_MANY_ROWS", f"train.csv has {len(train)} rows > MAX_TRAIN_ROWS {MAX_TRAIN_ROWS}", len(train))
    counts = train[target_column].astype(str).value_counts().sort_index()
    classes = counts.index.tolist()
    if len(classes) < MIN_CLASSES:
        raise _reject("TOO_FEW_CLASSES", f"train.csv target has {len(classes)} class(es) < MIN_CLASSES {MIN_CLASSES}", classes)
    if len(classes) > MAX_CLASSES:
        raise _reject("TOO_MANY_CLASSES", f"train.csv target has {len(classes)} classes > MAX_CLASSES {MAX_CLASSES}", len(classes))
    rare = {c: int(n) for c, n in counts.items() if n < MIN_ROWS_PER_CLASS}
    if rare:
        raise _reject("RARE_CLASSES", f"classes with fewer than {MIN_ROWS_PER_CLASS} training rows", rare)
    findings: list[dict[str, Any]] = []
    splits: dict[str, Any] = {"train": {"rows": int(len(train)), "class_counts": {c: int(n) for c, n in counts.items()}}}
    for name, frame in (("val.csv", val), ("test.csv", test)):
        if frame is None:
            continue
        _check_frame(name, frame, target_column, features)
        unseen = sorted(set(frame[target_column].astype(str)) - set(classes))
        if unseen:
            findings.append({"input": name, "verdict": "warning", "code": "UNSEEN_CLASSES", "message": f"{name} carries classes absent from train.csv; they can never be predicted", "observed": unseen})
        splits[name.split(".")[0]] = {"rows": int(len(frame)), "class_counts": {str(c): int(n) for c, n in frame[target_column].astype(str).value_counts().sort_index().items()}}
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    for c in numeric:
        values = train[c].dropna().to_numpy(dtype=float)
        if values.size and not np.isfinite(values).all():
            raise _reject("NON_FINITE_FEATURE", f"train.csv numeric feature {c!r} contains infinite values", c)
    imbalance = float(counts.max() / counts.min())
    if imbalance >= 10:
        findings.append({"input": "train.csv", "verdict": "warning", "code": "CLASS_IMBALANCE", "message": "majority/minority class ratio >= 10; read balancedAccuracy, not accuracy", "observed": imbalance})
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [{"id": names[0] if names else "dataset-0", "mode": "in-context-fit", "target_column": target_column, "feature_columns": features, "numeric_features": len(numeric), "categorical_features": len(features) - len(numeric), "classes": classes, "splits": splits, "train_sha256": sha256_hex(train.to_csv(index=False).encode("utf-8"))}],
        "verdict": "accepted",
        "findings": findings,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_version": MODEL_VERSION,
    }


def validate_new_rows(frame: pd.DataFrame, feature_columns: Sequence[str], *, target_column: str | None = None, classes: Sequence[str] | None = None) -> pd.DataFrame:
    """Rows for inference must carry exactly the artifact's feature columns and no target/output columns."""
    columns = list(frame.columns)
    duplicates = sorted({c for c in columns if columns.count(c) > 1})
    if duplicates:
        raise _reject("DUPLICATE_COLUMNS", "new rows: duplicate column names", duplicates)
    reserved = [c for c in [target_column, "prediction", *[f"proba_{c}" for c in (classes or [])]] if c and c in columns]
    if reserved:
        raise _reject("RESERVED_COLUMNS", "new rows: remove target/prediction/proba columns before inference", reserved)
    missing = [c for c in feature_columns if c not in columns]
    extra = [c for c in columns if c not in feature_columns]
    if missing or extra:
        raise _reject("SCHEMA_MISMATCH", "new rows: feature schema mismatch", {"missing": missing, "extra": extra})
    if len(frame) == 0:
        raise _reject("EMPTY_INPUT", "new rows: no rows", 0)
    frame = frame.loc[:, list(feature_columns)].copy()
    for c in frame.select_dtypes(include=np.number).columns:
        values = frame[c].dropna().to_numpy(dtype=float)
        if values.size and not np.isfinite(values).all():
            raise _reject("NON_FINITE_FEATURE", f"new rows: numeric feature {c!r} contains infinite values", c)
    return frame


# --------------------------------------------------------------------------- metrics / baseline


def classification_metrics(y_true: Sequence[Any], y_pred: Sequence[Any], proba: np.ndarray, classes: Sequence[str]) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score

    y_true = np.asarray([str(v) for v in y_true])
    y_pred = np.asarray([str(v) for v in y_pred])
    classes = [str(c) for c in classes]
    proba = np.asarray(proba, dtype=float)
    out = {"accuracy": float(accuracy_score(y_true, y_pred)), "balancedAccuracy": float(balanced_accuracy_score(y_true, y_pred)), "logLoss": float(log_loss(y_true, proba, labels=classes))}
    present = sorted(set(y_true))
    if len(present) == len(classes) and len(classes) >= 2:
        if len(classes) == 2:
            out["rocAuc"] = float(roc_auc_score(y_true, proba[:, 1], labels=classes))
        else:
            out["rocAuc"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes))
    return out


def majority_class_baseline(train_labels: Sequence[Any], eval_labels: Sequence[Any], classes: Sequence[str] | None = None) -> dict[str, Any]:
    """Always predict the most frequent training class, scored on the evaluation rows (log loss uses the training frequencies)."""
    train_labels = [str(v) for v in train_labels]
    eval_labels = [str(v) for v in eval_labels]
    if not train_labels or not eval_labels:
        raise ValueError("baseline needs training and evaluation labels")
    counts = pd.Series(train_labels).value_counts().sort_index()
    names = [str(c) for c in classes] if classes is not None else counts.index.tolist()
    majority = str(counts.idxmax())
    freq = {c: float(counts.get(c, 0) / len(train_labels)) for c in names}
    proba = np.array([[freq[c] for c in names]] * len(eval_labels), dtype=float)
    metrics = classification_metrics(eval_labels, [majority] * len(eval_labels), proba, names)
    return {"majorityClass": majority, "trainClassCounts": {c: int(n) for c, n in counts.items()}, **{k: metrics[k] for k in ("accuracy", "balancedAccuracy", "logLoss")}}


# --------------------------------------------------------------------------- artifact bundle


def _safe_fitted_archive(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for name in names:
            _safe_member(name)
        if "init_params.json" not in names:
            raise ValueError("fitted archive lacks init_params.json")
        return json.loads(archive.read("init_params.json"))


def manifest_digest(manifest: dict[str, Any], key: str) -> str:
    """Both manifest shapes: flat ``<key>Sha256`` (this module, the worker) or nested ``sha256[<key>]``."""
    value = manifest.get(key + "Sha256") or (manifest.get("sha256") or {}).get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"manifest has no SHA-256 for {key}")
    return value


def validate_artifact_bundle(artifact_dir: str | Path, *, expected_fitted_sha256: str = "", expected_checkpoint_sha256: str = "") -> dict[str, Any]:
    """Check manifest schema, member names, sizes, digests and archive safety BEFORE any model state is deserialised."""
    root = Path(artifact_dir)
    manifest_path = root / ARTIFACT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{ARTIFACT_MANIFEST_NAME} missing in {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schemaVersion") != ARTIFACT_SCHEMA_VERSION or manifest.get("taskType") != TASK_TYPE:
        raise ValueError(f"unsupported manifest: schemaVersion={manifest.get('schemaVersion')} taskType={manifest.get('taskType')}")
    for key in ("targetColumn", "featureColumns", "classes"):
        if key not in manifest:
            raise ValueError(f"manifest lacks {key}")
    digests: dict[str, str] = {}
    for key, override in (("fittedEstimator", expected_fitted_sha256), ("foundationCheckpoint", expected_checkpoint_sha256)):
        name = manifest.get(key)
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"manifest {key} must be a bare file name, got {name!r}")
        path = root / name
        if not path.is_file() or path.stat().st_size < 1024:
            raise ValueError(f"{name} is missing or implausibly small")
        digest = sha256_file(path)
        if digest != manifest_digest(manifest, key):
            raise ValueError(f"{name} SHA-256 {digest} != manifest {manifest_digest(manifest, key)}")
        if override and digest != override.strip().lower():
            raise ValueError(f"{name} SHA-256 {digest} != expected {override.strip().lower()}")
        digests[key] = digest
    init_params = _safe_fitted_archive(root / manifest["fittedEstimator"])
    return {**manifest, "verifiedSha256": digests, "recordedModelPath": init_params.get("model_path")}


def rewrite_model_path(fitted_archive: Path, checkpoint: Path, destination: Path) -> Path:
    """Copy a ``.tabpfn_fit`` archive while pointing its ``init_params.json`` ``model_path`` at ``checkpoint`` (original untouched)."""
    fitted_archive, checkpoint, destination = fitted_archive.resolve(), checkpoint.resolve(), destination.resolve()
    if not fitted_archive.is_file():
        raise FileNotFoundError(f"fitted estimator not found: {fitted_archive}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"companion checkpoint not found: {checkpoint}")
    saw_init = False
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(fitted_archive, "r") as source, zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as target:
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


def zip_artifact_bundle(artifact_dir: str | Path, zip_path: str | Path) -> str:
    """Zip the three bundle members flat (bare names) for transport to the artifact-inference notebook; returns the ZIP SHA-256."""
    root, zip_path = Path(artifact_dir), Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in (ARTIFACT_MANIFEST_NAME, FITTED_NAME, CHECKPOINT_NAME):
            archive.write(root / name, arcname=name)
    return sha256_file(zip_path)


def _coerce_non_json_init_params(model: Any) -> None:
    """tabpfn 8.1.0 serialises ``get_params()`` as JSON; stringify the values that are not JSON-encodable (e.g. a Path)."""
    for param, value in model.get_params(deep=False).items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            setattr(model, param, str(value))


# --------------------------------------------------------------------------- pipeline


@dataclass
class TabPFNClassifierPipeline:
    """The verified TabPFN-3 checkpoint plus, after ``fit`` or ``from_artifact``, an in-context-fitted estimator."""

    weights_path: Path
    device: str = "cpu"
    source: str = "local-snapshot"
    n_estimators: int = DEFAULT_N_ESTIMATORS
    random_state: int = DEFAULT_SEED
    target_column: str | None = None
    feature_columns: list[str] | None = None
    classes: list[str] | None = None
    _model: Any = None

    @classmethod
    def from_pretrained(cls, device: str | None = None, weights_dir: str | Path | None = None, allow_download: bool = False, *, n_estimators: int = DEFAULT_N_ESTIMATORS, random_state: int = DEFAULT_SEED) -> TabPFNClassifierPipeline:
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(weights_path=root / WEIGHTS_FILE, device=device, n_estimators=n_estimators, random_state=random_state)

    def _build_estimator(self) -> Any:
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(model_path=str(self.weights_path), device=self.device, n_estimators=self.n_estimators, random_state=self.random_state, show_progress_bar=False)

    def fit(self, X: pd.DataFrame, y: Sequence[Any], *, target_column: str = "target", estimator_factory: Callable[[], Any] | None = None) -> TabPFNClassifierPipeline:
        """In-context fit: register the training rows as the support set. No gradient update is performed."""
        model = (estimator_factory or self._build_estimator)()
        model.fit(X, pd.Series([str(v) for v in y], name=target_column))
        self._model = model
        self.target_column = target_column
        self.feature_columns = list(X.columns)
        self.classes = [str(c) for c in model.classes_]
        self.source = "in-context-fit"
        return self

    def _require(self) -> Any:
        if self._model is None or self.feature_columns is None or self.classes is None:
            raise RuntimeError("no fitted estimator: call fit() or from_artifact() first")
        return self._model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        model = self._require()
        X = validate_new_rows(X, self.feature_columns or [], target_column=self.target_column, classes=self.classes)
        return np.asarray(model.predict_proba(X), dtype=float)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """``prediction`` = argmax class label; ``proba_<class>`` columns follow the stored class order (uncalibrated)."""
        proba = self.predict_proba(X)
        classes = self.classes or []
        out = pd.DataFrame({"row_id": np.asarray(X.index), "prediction": [classes[i] for i in proba.argmax(axis=1)]})
        for i, label in enumerate(classes):
            out[f"proba_{label}"] = proba[:, i]
        out.attrs["decision_rule"] = DECISION_RULE
        return out

    def evaluate(self, X: pd.DataFrame, y: Sequence[Any]) -> dict[str, float]:
        proba = self.predict_proba(X)
        classes = self.classes or []
        predicted = [classes[i] for i in proba.argmax(axis=1)]
        return classification_metrics(list(y), predicted, proba, classes)

    def save_artifact(self, artifact_dir: str | Path, *, saver: Callable[[Any, Path], None] | None = None) -> dict[str, Any]:
        """Write ``model.tabpfn_fit`` + ``model.ckpt`` (byte copy of the verified checkpoint) + ``artifact_manifest.json``."""
        model = self._require()
        root = Path(artifact_dir)
        root.mkdir(parents=True, exist_ok=True)
        if saver is None:
            from tabpfn.model_loading import save_fitted_tabpfn_model

            def saver(estimator: Any, path: Path) -> None:
                _coerce_non_json_init_params(estimator)
                save_fitted_tabpfn_model(estimator, path)

        saver(model, root / FITTED_NAME)
        shutil.copyfile(self.weights_path, root / CHECKPOINT_NAME)
        manifest = {
            "schemaVersion": ARTIFACT_SCHEMA_VERSION,
            "taskType": TASK_TYPE,
            "targetColumn": self.target_column,
            "featureColumns": list(self.feature_columns or []),
            "classes": list(self.classes or []),
            "fittedEstimator": FITTED_NAME,
            "foundationCheckpoint": CHECKPOINT_NAME,
            "fittedEstimatorSha256": sha256_file(root / FITTED_NAME),
            "foundationCheckpointSha256": sha256_file(root / CHECKPOINT_NAME),
            "portableLoader": "tabpfn_classifier_pipeline.TabPFNClassifierPipeline.from_artifact",
            "baseModel": {"modelId": MODEL_ID, "revision": MODEL_REVISION, "file": WEIGHTS_FILE, "sha256": WEIGHTS_SHA256, "modelVersion": MODEL_VERSION, "license": MODEL_LICENSE},
            "mode": "zero-shot-icl",
            "nEstimators": self.n_estimators,
            "randomState": self.random_state,
        }
        (root / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest

    @classmethod
    def from_artifact(cls, artifact_dir: str | Path, device: str | None = None, *, expected_fitted_sha256: str = "", expected_checkpoint_sha256: str = "", loader: Callable[[Path, str], Any] | None = None) -> TabPFNClassifierPipeline:
        """Validate the bundle, then reconstruct the estimator from the fitted archive + companion checkpoint (no refit, no download)."""
        root = Path(artifact_dir)
        manifest = validate_artifact_bundle(root, expected_fitted_sha256=expected_fitted_sha256, expected_checkpoint_sha256=expected_checkpoint_sha256)
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        if loader is None:
            from tabpfn.model_loading import load_fitted_tabpfn_model

            def loader(fitted: Path, dev: str) -> Any:
                return load_fitted_tabpfn_model(fitted, device=dev)

        checkpoint = root / manifest["foundationCheckpoint"]
        with tempfile.TemporaryDirectory() as temp_dir:
            rewritten = rewrite_model_path(root / manifest["fittedEstimator"], checkpoint, Path(temp_dir) / FITTED_NAME)
            model = loader(rewritten, device)
        classes = [str(c) for c in getattr(model, "classes_", manifest["classes"])]
        if classes != [str(c) for c in manifest["classes"]]:
            raise RuntimeError("reconstructed classes disagree with the manifest")
        return cls(weights_path=checkpoint, device=device, source="artifact", n_estimators=int(manifest.get("nEstimators", DEFAULT_N_ESTIMATORS)), random_state=int(manifest.get("randomState", DEFAULT_SEED)), target_column=manifest["targetColumn"], feature_columns=list(manifest["featureColumns"]), classes=classes, _model=model)


# --------------------------------------------------------------------------- evaluation report (EVAL21)


def evaluation_report(metrics: dict[str, float] | None, *, baseline: dict[str, Any] | None = None, n_validation: int = 0, class_labels: Sequence[str] | None = None, sample_kind: str = "synthetic", reload_check: dict[str, Any] | None = None, split_name: str = "val.csv") -> dict[str, Any]:
    """Evaluation stage: the metrics are honest about what they are (single holdout, tutorial sample) or ``not-measurable``."""
    base = {
        "task": TASK_TYPE,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_version": MODEL_VERSION,
        "decision_rule": DECISION_RULE,
        "adaptation": "in-context conditioning only (no gradient update); the private-worker fine-tune path is not carried",
        "sample_kind": sample_kind,
        "n_validation": int(n_validation),
        "split": split_name,
        "class_labels": None if class_labels is None else [str(c) for c in class_labels],
        "reload_check": reload_check,
        "caveats": ["single holdout, no dispersion estimate", "probabilities are raw ensemble outputs, not calibrated", "synthetic sample metrics are plumbing evidence only" if sample_kind == "synthetic" else "BYOD metrics are one holdout of one table"],
    }
    if not metrics or n_validation == 0:
        return {**base, "metrics": [], "verdict": "not-measurable", "reason": "no labelled validation split was scored", "needs": "a labelled, leakage-safe holdout from the deployment domain scored with accuracy / balancedAccuracy / logLoss against majority_class_baseline; repeated splits for any dispersion estimate"}
    entries = [{"id": k, "value": float(v)} for k, v in metrics.items() if k in METRIC_IDS]
    return {**base, "metrics": entries, "baselines": {"majority_class": baseline} if baseline else {}, "verdict": "sample-sanity", "reason": f"{n_validation} validation row(s) from one holdout; tutorial evidence, not a benchmark", "needs": "a domain-representative labelled test set, subgroup breakdowns, calibration on held-out data and repeated splits for any generalisable claim"}
