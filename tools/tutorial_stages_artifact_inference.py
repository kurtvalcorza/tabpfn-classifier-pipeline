"""Stage runner for the standalone TabPFN-3 classifier ARTIFACT-INFERENCE tutorial (NOTEBOOK_SPEC 2.2 §25.13 pattern).

The companion notebook carries this file verbatim as ``tutorial_stages.py`` beside the carried package under ``src/``,
and runs every stage with the interpreter of an isolated, hash-locked environment. Nothing is installed into the
notebook kernel and no bundle is created here.

A bundle is ``artifact_manifest.json`` + ``model.tabpfn_fit`` (the fitted estimator state, written by tabpfn's
``save_fitted_tabpfn_model``) + ``model.ckpt`` (a byte copy of the pinned foundation checkpoint). Because ``model.ckpt``
must equal the pinned checkpoint, which Section 3 stages and digest-verifies, a pinned sample bundle needs only the
manifest and the fitted archive: when ``examples/sample-bundle/`` is carried, the default path assembles it with the
verified checkpoint. The sample can only be produced by a real TabPFN fit, so it is added from a recorded hosted E2E run
with ``tools/build_sample_bundle.py``; until then the default path stops with a message naming ``ARTIFACT_ZIP_PATH``.

Stages: weights → artifact → reconstruct → rows → predict, plus the optional ``activity`` (a tampered-bundle check).
``check_trusted_digest``, ``check_members``, ``check_numeric_features`` and the ``rows`` stage import no model library,
so CI exercises them directly.
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
import shutil
import sys
import traceback
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

STEM = "tabpfn_classifier_artifact_inference"
PACKAGE = "tabpfn_classifier_pipeline"
SAMPLE_DIR = "sample-bundle"
SAMPLE_RECORD = "sample-bundle/SAMPLE_BUNDLE.json"
SAMPLE_ROWS = "sample-bundle/new_rows.csv"
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
NO_SAMPLE = (
    "No pinned sample bundle is carried by this notebook revision: a TabPFN bundle's fitted archive can only be produced "
    "by a real fit, and none has been recorded yet. Set ARTIFACT_ZIP_PATH to the bundle the E2E notebook exported "
    "(outputs/tabpfn_classifier_artifact.zip) and paste the digests it printed, or tick UPLOAD_ARTIFACT in Colab."
)


class Run:
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


def check_digest_format(value: str, field: str) -> str:
    value = (value or "").strip().lower()
    if value and not SHA256_HEX.match(value):
        raise ValueError(f"{field} must be 64 hexadecimal characters (the digest the E2E notebook printed), got {value!r}.")
    return value


def check_trusted_digest(path: Path, expected: str, field: str) -> str:
    expected = check_digest_format(expected, field)
    observed = sha256_file(path)
    if expected and observed != expected:
        raise ValueError(f"Trusted digest mismatch for {path.name}: {field} is {expected}, the supplied file's SHA-256 is {observed}. This is not the file the digest was issued for; nothing was loaded. Obtain the bundle and its digests from the producer again.")
    return observed


def check_members(P, zip_path: Path) -> list[str]:
    """TPCA-m2: the ZIP must hold exactly the three bundle members, each once, at the top level; anything else (an
    extra file, a sub-directory copy that would overwrite a member when flattened) is refused before extraction."""
    expected = {P.ARTIFACT_MANIFEST_NAME, P.FITTED_NAME, P.CHECKPOINT_NAME}
    with zipfile.ZipFile(zip_path) as archive:
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
    bare = [PurePosixPath(n.replace("\\", "/")).name for n in names]
    duplicates = sorted({b for b in bare if bare.count(b) > 1})
    if duplicates:
        raise ValueError(f"{zip_path.name}: members {duplicates} occur more than once (e.g. in a sub-directory); extraction would overwrite one with the other. Refusing the bundle.")
    nested = [n for n in names if "/" in n.replace("\\", "/")]
    unexpected = sorted(set(bare) - expected)
    missing = sorted(expected - set(bare))
    if nested or unexpected or missing:
        raise ValueError(f"{zip_path.name}: a bundle holds exactly {sorted(expected)} at the top level; unexpected={unexpected}, nested={nested}, missing={missing}. Refusing the bundle.")
    return sorted(bare)


def check_numeric_features(name: str, rows, train_dtypes: dict[str, str]):
    """TPCA-m2: columns the fitted estimator saw as numeric must hold numbers; refuse naming the column and values."""
    import pandas as pd

    out = rows.copy()
    for column, kind in train_dtypes.items():
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
    run.write_state("weights.json", {"snapshot": str(snapshot), "checkpoint": str(snapshot / P.WEIGHTS_FILE)})


def clear_outputs(run: Run) -> list[str]:
    removed = []
    for path in sorted(run.out.glob(f"{STEM}_*")):
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    return removed


def assemble_sample(run: Run, P, bundle: Path) -> dict[str, Any]:
    record_path = run.root / SAMPLE_RECORD
    if not record_path.is_file():
        raise RuntimeError(NO_SAMPLE)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    bundle.mkdir(parents=True)
    for name in (P.ARTIFACT_MANIFEST_NAME, P.FITTED_NAME):
        source = run.root / SAMPLE_DIR / name
        if sha256_file(source) != record["files"][name]["sha256"]:
            raise ValueError(f"Trusted digest mismatch for the carried sample {name}: it does not match SAMPLE_BUNDLE.json.")
        shutil.copyfile(source, bundle / name)
    base = Path(run.read_state("weights.json", "artifact")["checkpoint"])
    target = bundle / P.CHECKPOINT_NAME
    try:
        target.hardlink_to(base)
    except OSError:
        shutil.copyfile(base, target)
    return {"trusted_digest": "verified (sample files against SAMPLE_BUNDLE.json; model.ckpt is the checkpoint verified in Section 3)", "expected_fitted_sha256": record["files"][P.FITTED_NAME]["sha256"], "producer": record.get("producer")}


def stage_artifact(run: Run) -> None:
    P = package(run.root)
    opts = run.options
    source = opts.get("source", "sample")
    removed = clear_outputs(run)
    if removed:
        print({"removed_previous_outputs": removed})
    for name in ("artifact.json", "rows.json", "rows.csv"):
        (run.state / name).unlink(missing_ok=True)
    bundle = run.state / "bundle"
    shutil.rmtree(bundle, ignore_errors=True)
    expected_fitted = check_digest_format(opts.get("expected_fitted_sha256", ""), "EXPECTED_FITTED_SHA256")
    if source == "sample":
        trust = assemble_sample(run, P, bundle)
        expected_fitted = expected_fitted or trust["expected_fitted_sha256"]
        zip_name = zip_sha = None
    elif source in ("path", "upload"):
        zip_path = Path(opts.get("zip_path") or "")
        if not str(zip_path) or not zip_path.is_file():
            raise FileNotFoundError(f"ARTIFACT_ZIP_PATH {str(zip_path)!r} is not a file in this runtime: point it at the bundle ZIP the E2E notebook exported.")
        zip_sha = check_trusted_digest(zip_path, opts.get("expected_zip_sha256", ""), "EXPECTED_ZIP_SHA256")
        members = check_members(P, zip_path)
        P.safe_extract_zip(zip_path, bundle)
        zip_name = zip_path.name
        pinned = {"EXPECTED_ZIP_SHA256": bool(opts.get("expected_zip_sha256")), "EXPECTED_FITTED_SHA256": bool(expected_fitted)}
        trust = {"trusted_digest": "verified (" + ", ".join(k for k, v in pinned.items() if v) + ")" if any(pinned.values()) else "not supplied: trusting the manifest only", "members": members, "pinned": pinned}
        if not any(pinned.values()):
            print("No trusted digest was supplied: the checks below establish internal consistency only, not that this is the bundle you were sent. Ask the producer for the digests the E2E notebook printed.")
    else:
        raise ValueError(f"unknown artifact source {source!r}")
    artifact = P.validate_artifact_bundle(bundle, expected_fitted_sha256=expected_fitted, expected_checkpoint_sha256=P.WEIGHTS_SHA256)
    if artifact["verifiedSha256"]["foundationCheckpoint"] != P.WEIGHTS_SHA256:
        raise RuntimeError("the bundled model.ckpt is not the pinned TabPFN-3 checkpoint carried by this notebook")
    record = {"source": source, "zip": zip_name, "zip_sha256": zip_sha, **{k: v for k, v in trust.items() if k != "expected_fitted_sha256"}, "bundle": str(bundle), "manifest": {k: v for k, v in artifact.items() if k not in ("verifiedSha256",)}, "verifiedSha256": artifact["verifiedSha256"]}
    run.write_state("artifact.json", record)
    print({k: record.get(k) for k in ("source", "zip", "trusted_digest")})
    print({"pinned_digests": {"zip": (zip_sha or "")[:16] or None, "fittedEstimator": artifact["verifiedSha256"]["fittedEstimator"][:16], "foundationCheckpoint (= pinned base)": artifact["verifiedSha256"]["foundationCheckpoint"][:16]}})
    print({"featureColumns": len(artifact["featureColumns"]), "targetColumn": artifact["targetColumn"], "classes": artifact["classes"], "mode": artifact.get("mode"), "nEstimators": artifact.get("nEstimators"), "baseModel": artifact.get("baseModel", {}).get("modelId")})


def reconstructed(run: Run):
    P = package(run.root)
    art = run.read_state("artifact.json", "reconstruct")
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fresh = P.TabPFNClassifierPipeline.from_artifact(Path(art["bundle"]), device=device, expected_checkpoint_sha256=P.WEIGHTS_SHA256)
    if fresh.classes != [str(c) for c in art["manifest"]["classes"]] or fresh.feature_columns != list(art["manifest"]["featureColumns"]):
        raise RuntimeError("reconstructed estimator disagrees with the manifest")
    return P, art, fresh


def stage_reconstruct(run: Run) -> None:
    _P, _art, fresh = reconstructed(run)
    print({"source": fresh.source, "device": fresh.device, "classes": fresh.classes, "n_estimators": fresh.n_estimators, "random_state": fresh.random_state, "refit": False, "network_fallback_for_weights": False})


def feature_kinds(art: dict[str, Any], rows) -> dict[str, str]:
    """The kinds the E2E export recorded in the manifest (``featureKinds``); for a bundle without them, a column whose
    values are mostly numbers is treated as numeric, so a stray string in it is still caught."""
    import pandas as pd

    recorded = art["manifest"].get("featureKinds")
    if recorded:
        return recorded
    kinds = {}
    for c in art["manifest"]["featureColumns"]:
        if c not in rows.columns:
            continue
        values = rows[c].dropna()
        share = float(pd.to_numeric(values.astype(str).str.strip(), errors="coerce").notna().mean()) if len(values) else 1.0
        kinds[c] = "numeric" if pd.api.types.is_numeric_dtype(rows[c]) or share > 0.5 else "categorical"
    return kinds


def stage_rows(run: Run) -> None:
    import pandas as pd

    P = package(run.root)
    art = run.read_state("artifact.json", "rows")
    manifest = art["manifest"]
    features, classes, target = list(manifest["featureColumns"]), [str(c) for c in manifest["classes"]], manifest["targetColumn"]
    source = run.options.get("source", "sample")
    id_columns = list(run.options.get("id_columns") or [])
    if source == "sample":
        if art["source"] != "sample":
            raise ValueError("The pinned sample rows match only the pinned sample bundle. With your own bundle, set NEW_DATA_PATH to your rows (or, in Colab, tick UPLOAD_NEW_DATA).")
        path = run.root / SAMPLE_ROWS
        id_columns = id_columns or [c for c in pd.read_csv(path, nrows=0).columns if c not in features]
    else:
        path = Path(run.options.get("path") or "")
        if not str(path) or not path.is_file():
            raise FileNotFoundError(f"NEW_DATA_PATH {str(path)!r} does not exist in this runtime: upload the file or correct the path.")
    raw = pd.read_csv(path)
    clash = [c for c in id_columns if c in features]
    if clash:
        raise ValueError(f"ID_COLUMNS {clash} are fitted feature columns; an identifier must not be a model input.")
    absent = [c for c in id_columns if c not in raw.columns]
    if absent:
        raise ValueError(f"{path.name}: ID_COLUMNS {absent} are not in the header {list(map(str, raw.columns))}.")
    try:
        frame = P.validate_new_rows(raw.drop(columns=id_columns), features, target_column=target, classes=classes)
    except P.InputRejected as exc:
        extra = (exc.finding.get("observed") or {}).get("extra") if isinstance(exc.finding.get("observed"), dict) else None
        hint = f" If {extra} are identifiers, list them in ID_COLUMNS." if extra else ""
        raise ValueError(f"{path.name}: [{exc.finding['code']}] {exc.finding['message']} (observed: {exc.finding.get('observed')}).{hint}") from None
    frame = check_numeric_features(path.name, frame, feature_kinds(art, frame))
    sample_kind = "sample" if source == "sample" and art["source"] == "sample" else "BYOD"
    input_manifest = {
        "schema": {"format": "CSV of unlabelled rows", "columns": "the artifact feature columns (any order) plus declared ID_COLUMNS", "reserved": [target, "prediction", *[f"proba_{c}" for c in classes]], "numeric": "finite numbers in the columns the estimator saw as numeric"},
        "inputs": [{"id": path.name, "mode": "artifact-inference", "rows": int(len(frame)), "feature_columns": features, "id_columns": id_columns, "missing_values": {k: int(v) for k, v in frame.isna().sum().items() if v}, "sha256": sha256_file(path)}],
        "verdict": "accepted",
        "findings": [],
        "model_id": P.MODEL_ID,
        "model_revision": P.MODEL_REVISION,
    }
    try:
        P.validate_new_rows(frame.assign(unexpected_column=0), features, target_column=target, classes=classes)
    except P.InputRejected as exc:
        input_manifest["findings"].append({"input": "extra-column-probe", **exc.finding})
    run.write_output(f"{STEM}_input_manifest.json", input_manifest)
    pd.concat([raw[id_columns].reset_index(drop=True), frame.reset_index(drop=True)], axis=1).to_csv(run.state / "rows.csv", index=False)
    run.write_state("rows.json", {"source": source, "name": path.name, "rows": len(frame), "id_columns": id_columns, "sample_kind": sample_kind})
    print(json.dumps(input_manifest["inputs"][0], indent=2, default=str))
    print("findings:", json.dumps(input_manifest["findings"], indent=2, default=str))


def stage_predict(run: Run) -> None:
    import importlib.metadata
    import platform

    import pandas as pd
    import torch

    P, art, fresh = reconstructed(run)
    rows_state = run.read_state("rows.json", "predict")
    rows = pd.read_csv(run.state / "rows.csv")
    ids = rows_state["id_columns"]
    scored = fresh.predict(rows.drop(columns=ids))
    out = pd.concat([rows[ids].reset_index(drop=True), scored.drop(columns=["row_id"]).reset_index(drop=True)], axis=1) if ids else scored
    out.to_csv(run.out / f"{STEM}_predictions.csv", index=False)
    report = P.evaluation_report(None, n_validation=0, class_labels=fresh.classes, sample_kind=rows_state["sample_kind"])
    run.write_output(f"{STEM}_evaluation_report.json", report)
    source = json.loads((run.root / "source.json").read_text(encoding="utf-8")) if (run.root / "source.json").is_file() else {}
    payload = {
        "predictions": out.to_dict(orient="records"),
        "artifact": art["manifest"],
        "artifact_verified_sha256": art["verifiedSha256"],
        "artifact_source": {k: art.get(k) for k in ("source", "zip", "zip_sha256", "trusted_digest", "pinned")},
        "reconstruction": {"loader": "tabpfn_classifier_pipeline.TabPFNClassifierPipeline.from_artifact", "device": fresh.device, "network_fallback_for_weights": False, "refit": False, "decision_rule": P.DECISION_RULE},
        "input_manifest": json.loads((run.out / f"{STEM}_input_manifest.json").read_text(encoding="utf-8")),
        "evaluation_report": report,
        "input": rows_state,
        "notebook_source": source,
        "repository_revision": source.get("revision"),
        "model_id": P.MODEL_ID,
        "model_revision": P.MODEL_REVISION,
        "model_license": P.MODEL_LICENSE,
        "model_file": P.WEIGHTS_FILE,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "tabpfn": importlib.metadata.version("tabpfn"), "pandas": pd.__version__, "device": fresh.device},
    }
    run.write_output(f"{STEM}_result.json", payload)
    print(out.round(4).to_string(index=False))
    print(json.dumps({k: report[k] for k in ("verdict", "sample_kind", "reason")}, indent=2))


def stage_activity(run: Run) -> None:
    """Optional activity (TPCA-M3): tamper with a copy of the verified bundle and watch which check refuses it. Writes
    only to outputs/activity/; the verified bundle is never modified."""
    P = package(run.root)
    art = run.read_state("artifact.json", "activity")
    kind = run.options.get("tamper", "flip one byte of model.tabpfn_fit")
    work = run.state / "tampered"
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(Path(art["bundle"]), work)
    if kind == "flip one byte of model.tabpfn_fit":
        data = bytearray((work / P.FITTED_NAME).read_bytes())
        data[len(data) // 2] ^= 0xFF
        (work / P.FITTED_NAME).write_bytes(bytes(data))
    elif kind == "rewrite the manifest digest too":
        data = bytearray((work / P.FITTED_NAME).read_bytes())
        data[len(data) // 2] ^= 0xFF
        (work / P.FITTED_NAME).write_bytes(bytes(data))
        manifest = json.loads((work / P.ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        manifest["fittedEstimatorSha256"] = sha256_file(work / P.FITTED_NAME)
        (work / P.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    elif kind == "add an unlisted member to the ZIP":
        zip_path = run.state / "tampered.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            for name in (P.ARTIFACT_MANIFEST_NAME, P.FITTED_NAME):
                archive.write(work / name, arcname=name)
            archive.writestr(P.CHECKPOINT_NAME, b"stand-in")
            archive.writestr("notes.txt", "an extra member")
    else:
        raise ValueError(f"unknown tamper option {kind!r}")
    expected = art["verifiedSha256"]["fittedEstimator"]
    try:
        if kind == "add an unlisted member to the ZIP":
            check_members(P, run.state / "tampered.zip")
        else:
            P.validate_artifact_bundle(work, expected_fitted_sha256=expected, expected_checkpoint_sha256=P.WEIGHTS_SHA256)
        outcome = {"refused": False, "message": "accepted — no check caught this change"}
    except (ValueError, RuntimeError) as exc:
        outcome = {"refused": True, "message": str(exc)}
    record = {"tamper": kind, "trusted_fitted_digest_used": expected[:16], **outcome}
    run.write_output(f"activity/{STEM}_activity_tamper.json", record)
    print(json.dumps(record, indent=2))


STAGES = {
    "weights": stage_weights,
    "artifact": stage_artifact,
    "reconstruct": stage_reconstruct,
    "rows": stage_rows,
    "predict": stage_predict,
    "activity": stage_activity,
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
    except BaseException as exc:  # noqa: BLE001 -- every failure is reported to the kernel with its own message
        traceback.print_exc()
        error_file.write_text(json.dumps({"type": type(exc).__name__, "message": str(exc)}), encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
