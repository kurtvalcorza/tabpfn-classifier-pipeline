#!/usr/bin/env python3
# ruff: noqa: E501  -- messages and the record text are kept on one line
"""Add (or ``--check``) the pinned sample bundle for the ARTIFACT-INFERENCE notebook's default path (TPCA-M1).

A TabPFN bundle's fitted archive (``model.tabpfn_fit``) is written by tabpfn's ``save_fitted_tabpfn_model`` after a real
fit with the pinned checkpoint, so it cannot be produced offline or in CI. This tool takes the exports of a recorded,
hosted run of the E2E notebook and writes ``examples/sample-bundle/``:

- ``artifact_manifest.json`` and ``model.tabpfn_fit`` from the bundle ZIP (``model.ckpt`` is not stored: the manifest
  must record it as the pinned checkpoint, which the companion stages and digest-verifies in Section 3);
- ``new_rows.csv``: the E2E notebook's unlabelled rows (``outputs/tabpfn_classifier_new_rows.csv``);
- ``SAMPLE_BUNDLE.json``: sizes and SHA-256 digests of those files, the bundle ZIP digest and the producing record.

Once the directory exists, regenerating the companion notebook carries it and its default path needs no upload.

    python tools/build_sample_bundle.py --zip outputs/tabpfn_classifier_artifact.zip \\
        --rows outputs/tabpfn_classifier_new_rows.csv --producer "Colab T4 run of <commit>, <date>"
    python tools/build_sample_bundle.py --check   # exit 1 if a carried sample's files differ from its record
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "sample-bundle"
FILES = ("artifact_manifest.json", "model.tabpfn_fit", "new_rows.csv")


def _package():
    sys.path.insert(0, str(ROOT / "src"))
    import tabpfn_classifier_pipeline as P

    return P


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(zip_path: Path, rows: Path, producer: str, out: Path) -> dict:
    P = _package()
    with tempfile.TemporaryDirectory() as tmp:
        members = P.safe_extract_zip(zip_path, Path(tmp))
        expected = {P.ARTIFACT_MANIFEST_NAME, P.FITTED_NAME, P.CHECKPOINT_NAME}
        if sorted(members) != sorted(expected):
            raise SystemExit(f"{zip_path.name}: expected exactly {sorted(expected)}, got {sorted(members)}")
        manifest = json.loads((Path(tmp) / P.ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("foundationCheckpointSha256") != P.WEIGHTS_SHA256:
            raise SystemExit("the bundle's checkpoint is not the pinned checkpoint; only a bundle built on it can be a sample")
        if sha256(Path(tmp) / P.FITTED_NAME) != manifest.get("fittedEstimatorSha256"):
            raise SystemExit("model.tabpfn_fit does not match the manifest digest")
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(tmp) / P.ARTIFACT_MANIFEST_NAME, out / "artifact_manifest.json")
        shutil.copyfile(Path(tmp) / P.FITTED_NAME, out / "model.tabpfn_fit")
    shutil.copyfile(rows, out / "new_rows.csv")
    record = {
        "description": "Pinned sample bundle for the ARTIFACT-INFERENCE notebook's default path: the bundle a recorded E2E run exported, without model.ckpt (the pinned checkpoint, staged and verified by the companion).",
        "producer": producer,
        "bundle_zip_sha256": sha256(zip_path),
        "checkpoint": {"file": P.WEIGHTS_FILE, "sha256": P.WEIGHTS_SHA256},
        "files": {name: {"bytes": (out / name).stat().st_size, "sha256": sha256(out / name)} for name in FILES},
    }
    (out / "SAMPLE_BUNDLE.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def check(out: Path = OUT) -> int:
    record_path = out / "SAMPLE_BUNDLE.json"
    if not record_path.is_file():
        print("No sample bundle is carried yet (examples/sample-bundle/SAMPLE_BUNDLE.json absent); nothing to check.")
        return 0
    record = json.loads(record_path.read_text(encoding="utf-8"))
    stale = [name for name, facts in record["files"].items() if not (out / name).is_file() or sha256(out / name) != facts["sha256"]]
    if stale:
        print(f"STALE: {stale} differ from SAMPLE_BUNDLE.json", file=sys.stderr)
        return 1
    print("OK: examples/sample-bundle/ matches SAMPLE_BUNDLE.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--producer", default="")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        return check()
    if not (args.zip and args.rows and args.producer):
        parser.error("--zip, --rows and --producer are required (or use --check)")
    record = build(args.zip, args.rows, args.producer, OUT)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
