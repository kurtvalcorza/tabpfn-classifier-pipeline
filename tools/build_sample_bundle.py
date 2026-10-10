#!/usr/bin/env python3
# ruff: noqa: E501  -- messages and the record text are kept on one line
"""Build (or ``--verify``) the trusted sample bundle for the ARTIFACT-INFERENCE notebook's default path (TPCA-M1).

NOTEBOOK_SPEC SART6-SART8: the default sample is a release asset of this repository, pinned in the notebook by URL, size
and SHA-256, never overwritten (a changed sample gets a new tag), and produced by a recorded run whose notebook, commit
and record the release notes name.

A TabPFN bundle's fitted archive (``model.tabpfn_fit``) is written by tabpfn's ``save_fitted_tabpfn_model`` after a real
fit with the pinned checkpoint, so it cannot be produced offline or in CI. This tool takes the exports of a recorded run
of the E2E notebook and writes:

- the release asset ``tabpfn_classifier_sample_bundle.zip`` (fixed member order and timestamps), holding
  ``artifact_manifest.json`` and ``model.tabpfn_fit`` from the bundle ZIP (``model.ckpt`` is not redistributed: the
  manifest must record it as the pinned checkpoint, which the companion stages and digest-verifies in Section 3),
  ``new_rows.csv`` (the E2E notebook's unlabelled rows, ``outputs/tabpfn_classifier_new_rows.csv``) and
  ``SAMPLE_BUNDLE.json`` (sizes and SHA-256 digests of those three files, the source bundle ZIP digest, the producer);
- ``examples/sample_bundle_pin.json``: the pin the companion template renders into ``SAMPLE_ARTIFACT``.

    python tools/build_sample_bundle.py --zip outputs/tabpfn_classifier_artifact.zip \\
        --rows outputs/tabpfn_classifier_new_rows.csv --producer-json producer.json --out dist/
    python tools/build_sample_bundle.py --verify dist/tabpfn_classifier_sample_bundle.zip   # exit 1 unless it equals the pin

Regenerate the companion notebook after writing a new pin. Publishing the asset is a maintainer action.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / "examples" / "sample_bundle_pin.json"
REPOSITORY = "kurtvalcorza/tabpfn-classifier-pipeline"
TAG = "sample-bundle-v1"
ASSET = "tabpfn_classifier_sample_bundle.zip"
FILES = ("artifact_manifest.json", "model.tabpfn_fit", "new_rows.csv")
MEMBERS = ("SAMPLE_BUNDLE.json", *FILES)
_EPOCH = (2026, 1, 1, 0, 0, 0)


def _package():
    sys.path.insert(0, str(ROOT / "src"))
    import tabpfn_classifier_pipeline as P

    return P


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    """A reproducible ZIP: fixed member order, timestamps and permissions, deflate level 9."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in MEMBERS:
            info = zipfile.ZipInfo(name, date_time=_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, members[name], compresslevel=9)
    return buffer.getvalue()


def build(zip_path: Path, rows: Path, producer: dict, out_dir: Path, *, tag: str = TAG) -> dict:
    """Write the release asset into ``out_dir`` and return its pin (url, bytes, sha256, tag, producer)."""
    P = _package()
    with tempfile.TemporaryDirectory() as tmp:
        members = P.safe_extract_zip(zip_path, Path(tmp))
        expected = {P.ARTIFACT_MANIFEST_NAME, P.FITTED_NAME, P.CHECKPOINT_NAME}
        if sorted(members) != sorted(expected):
            raise SystemExit(f"{zip_path.name}: expected exactly {sorted(expected)}, got {sorted(members)}")
        manifest_bytes = (Path(tmp) / P.ARTIFACT_MANIFEST_NAME).read_bytes()
        fitted_bytes = (Path(tmp) / P.FITTED_NAME).read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("foundationCheckpointSha256") != P.WEIGHTS_SHA256:
        raise SystemExit("the bundle's checkpoint is not the pinned checkpoint; only a bundle built on it can be a sample")
    if sha256_bytes(fitted_bytes) != manifest.get("fittedEstimatorSha256"):
        raise SystemExit("model.tabpfn_fit does not match the manifest digest")
    payload = {"artifact_manifest.json": manifest_bytes, "model.tabpfn_fit": fitted_bytes, "new_rows.csv": rows.read_bytes()}
    record = {
        "description": "Trusted sample bundle for the ARTIFACT-INFERENCE notebook's default path: the bundle a recorded E2E run exported, without model.ckpt (the pinned checkpoint, staged and verified by the companion), plus that run's eight unlabelled rows.",
        "producer": producer,
        "source_bundle_zip_sha256": sha256_bytes(zip_path.read_bytes()),
        "checkpoint": {"file": P.WEIGHTS_FILE, "sha256": P.WEIGHTS_SHA256},
        "files": {name: {"bytes": len(data), "sha256": sha256_bytes(data)} for name, data in payload.items()},
    }
    payload["SAMPLE_BUNDLE.json"] = (json.dumps(record, indent=2) + "\n").encode("utf-8")
    data = _zip_bytes(payload)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ASSET).write_bytes(data)
    return {
        "url": f"https://github.com/{REPOSITORY}/releases/download/{tag}/{ASSET}",
        "tag": tag,
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "producer": producer,
    }


def verify(asset: Path, pin_path: Path = PIN) -> int:
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    data = asset.read_bytes()
    if len(data) != pin["bytes"] or sha256_bytes(data) != pin["sha256"]:
        print(f"MISMATCH: {asset.name} is {len(data)} bytes, SHA-256 {sha256_bytes(data)}; the pin is {pin['bytes']} bytes, {pin['sha256']}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(asset) as archive:
        names = sorted(archive.namelist())
        record = json.loads(archive.read("SAMPLE_BUNDLE.json")) if "SAMPLE_BUNDLE.json" in names else {}
        stale = [n for n, facts in record.get("files", {}).items() if sha256_bytes(archive.read(n)) != facts["sha256"]]
    if names != sorted(MEMBERS) or stale:
        print(f"MISMATCH: members {names}, stale {stale}", file=sys.stderr)
        return 1
    print(f"OK: {asset.name} equals the pin in {pin_path.name} ({pin['url']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--producer-json", type=Path, help="JSON object naming the producing notebook, blob, commit, runtime and run record")
    parser.add_argument("--out", type=Path, default=ROOT / "dist")
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--write-pin", action="store_true", help=f"also write {PIN.relative_to(ROOT)}")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if args.verify:
        return verify(args.verify)
    if not (args.zip and args.rows and args.producer_json):
        parser.error("--zip, --rows and --producer-json are required (or use --verify)")
    pin = build(args.zip, args.rows, json.loads(args.producer_json.read_text(encoding="utf-8")), args.out, tag=args.tag)
    if args.write_pin:
        PIN.write_text(json.dumps(pin, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(pin, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
