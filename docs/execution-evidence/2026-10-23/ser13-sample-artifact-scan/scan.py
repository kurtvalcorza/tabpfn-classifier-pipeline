"""Static pickle scan of one release asset (ARTIFACT_SPEC 2.0 SER13), failing closed.

Run (pinned):
  uv run --no-project --python 3.13 --with picklescan==1.0.5 --with joblib==1.6.0 \\
      --with numpy==2.5.4 python -I scan.py ASSET.zip --report scan-report.txt

Every file in the asset, and in every archive nested inside it, gets its own verdict:
  clean      every global is innocuous or suspicious, and the whole file was read;
  dangerous  any global picklescan classifies as dangerous;
  incomplete the file could not be read to the end by any recorded method.
Content decides, not the file name: picklescan reads every file's bytes, and a file is pickle
content if its extension says so or if parsing it yields globals that are well-formed Python
identifiers (this finds protocol 0/1 pickles under any name). Every zip and tar, PyTorch zips
included, is extracted with member-path, link and duplicate-name checks and its members are queued,
so archives inside archives are reached; a pickle stream in front of an archive is judged too.
gzip, bzip2, xz, zstd, lz4, zlib, 7z and rar content is not decompressed and is incomplete. A file
that cannot be read as a pickle to the end, or that picklescan itself fails on, is incomplete,
never clean. Pickles that interleave raw numpy bytes with instructions (joblib) get a restricted
load when picklescan stops early: joblib's own reader runs with find_class replaced so that only
numpy's array and scalar rebuilders, numpy.dtype and joblib's NumpyArrayWrapper are real; every
other global is an inert stub, recorded, and classified with picklescan's own tables. Payloads of
torch.storage._load_from_bytes are captured as files and scanned like any other file, never loaded.
Not covered: data a loader decompresses or decodes with no format marker (for example raw deflate).

Exit status: 0 all clean; 1 any dangerous; 2 any incomplete (and none dangerous); 3 the asset could
not be scanned (bad archive, unsafe or duplicate member, unexpected error). The report holds only
paths relative to the asset.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import pathlib
import pickle
import pickletools
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile
from importlib.metadata import version

import joblib.numpy_pickle as jnp
from picklescan.scanner import (
    SafetyLevel,
    _build_scan_result_from_raw_globals,
    scan_pickle_bytes,
)

PYTORCH_EXT = {".pt", ".ckpt", ".pth", ".bin"}
PICKLE_EXT = {".pkl", ".pickle", ".dat", ".data", ".joblib"}
ALLOW = {
    ("joblib.numpy_pickle", "NumpyArrayWrapper"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
}


class ScanAbort(Exception):
    """The asset cannot be scanned at all (exit 3)."""


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_member(name: str, root: pathlib.Path, seen: set[str]) -> pathlib.Path:
    target = (root / name).resolve()
    if name.startswith(("/", "\\")) or ".." in pathlib.PurePosixPath(name.replace("\\", "/")).parts \
            or not target.is_relative_to(root.resolve()):
        raise ScanAbort(f"unsafe member path {name!r}")
    key = pathlib.PurePosixPath(name.replace("\\", "/")).as_posix().rstrip("/").lower()
    if key in seen:
        raise ScanAbort(f"duplicate member name {name!r}")
    seen.add(key)
    return target


def safe_extract(archive: pathlib.Path, dest: pathlib.Path) -> list[pathlib.Path]:
    """Extract a zip or tar completely, or abort: bad CRC, absolute or escaping paths, links,
    special files, duplicate member names (a loader and this scan could read different copies)."""
    out, seen = [], set()
    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                bad = zf.testzip()
                if bad is not None:
                    raise ScanAbort(f"CRC error in {bad}")
                for info in zf.infolist():
                    target = _check_member(info.filename, dest, seen)
                    if (info.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ScanAbort(f"symlink member {info.filename!r}")
                    if info.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    out.append(target)
        else:
            with tarfile.open(archive) as tf:
                for info in tf.getmembers():
                    target = _check_member(info.name, dest, seen)
                    if info.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    if not info.isfile():
                        raise ScanAbort(f"non-regular tar member {info.name!r}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(info) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    out.append(target)
        return out
    except ScanAbort:
        raise
    except (zipfile.BadZipFile, tarfile.TarError, OSError, RuntimeError, NotImplementedError,
            EOFError) as exc:
        raise ScanAbort(f"cannot extract {archive.name}: {type(exc).__name__}: {exc}") from exc


def reads_to_end(data: bytes, max_pickles: int | None = None) -> bool:
    """True if data is one or more complete pickle streams back to back, ending at EOF.
    With max_pickles, only that many leading streams must parse (legacy torch layout)."""
    bio = io.BytesIO(data)
    count = 0
    try:
        while bio.tell() < len(data):
            for _ in pickletools.genops(bio):
                pass
            count += 1
            if max_pickles is not None and count >= max_pickles:
                return True
    except Exception:
        return False
    return count > 0 and max_pickles is None


def picklescan_globals(data: bytes, file_id: str):
    """picklescan on the bytes themselves, whatever the file is called (no extension dispatch)."""
    try:
        res = scan_pickle_bytes(io.BytesIO(data), file_id, multiple_pickles=True)
    except Exception:  # picklescan itself can raise on malformed streams; never treat that as clean
        return False, []
    return True, [(g.module, g.name, g.safety.value) for g in res.globals]


IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
# Formats a loader could decompress before unpickling; their contents are not scanned here.
OPAQUE = {b"\x1f\x8b": "gzip", b"BZh": "bzip2", b"\xfd7zXZ\x00": "xz", b"\x28\xb5\x2f\xfd": "zstd",
          b"\x04\x22\x4d\x18": "lz4", b"7z\xbc\xaf\x27\x1c": "7z", b"Rar!\x1a\x07": "rar",
          b"\x78\x01": "zlib", b"\x78\x5e": "zlib", b"\x78\x9c": "zlib", b"\x78\xda": "zlib"}


def classify(raw: set[tuple[str, str]]):
    res = _build_scan_result_from_raw_globals(raw, "inventory")
    return [(g.module, g.name, g.safety.value) for g in res.globals]


def restricted_inventory(path: pathlib.Path, nested_dir: pathlib.Path):
    """joblib's reader with every global outside ALLOW stubbed. Returns (parsed, raw globals, payloads)."""
    seen: set[tuple[str, str]] = set()
    payloads: list[pathlib.Path] = []

    class Stub:
        def __new__(cls, *a, **k):
            return object.__new__(cls)

        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            pass

        def __call__(self, *a, **k):
            return Stub()

        # container protocol for SETITEM(S), APPEND(S) and ADDITEMS on stubbed dict/list/set types
        def __setitem__(self, key, value):
            pass

        def append(self, value):
            pass

        def extend(self, values):
            pass

        def add(self, value):
            pass

    def capture(data=None, *a, **k):
        if isinstance(data, bytes | bytearray):
            digest = hashlib.sha256(data).hexdigest()
            out = nested_dir / f"nested-{digest[:16]}.pt"
            out.write_bytes(bytes(data))
            payloads.append(out)
        return Stub()

    probe = pickle.Unpickler(io.BytesIO(b"."))

    class Restricted(jnp.NumpyUnpickler):
        def find_class(self, module, name):
            seen.add((module, name))
            if (module, name) in ALLOW:
                return pickle.Unpickler.find_class(probe, module, name)
            if (module, name) == ("torch.storage", "_load_from_bytes"):
                return capture
            return type(f"Stub[{module}.{name}]", (Stub,), {})

    with open(path, "rb") as fh:
        try:
            Restricted(str(path), fh, ensure_native_byte_order=True, mmap_mode=None).load()
            parsed = fh.read(1) == b""
        except Exception:
            parsed = False
    return parsed, seen, payloads


def scan_asset(asset: pathlib.Path, work: pathlib.Path):
    root = work / "x"
    nested_dir = work / "nested"
    root.mkdir(parents=True)
    nested_dir.mkdir()
    queue = collections.deque(safe_extract(asset, root))
    results = []

    def rel(p: pathlib.Path) -> str:
        p = p.resolve()
        for base, label in ((root.resolve(), ""), (nested_dir.resolve(), "<nested>/")):
            if p.is_relative_to(base):
                return label + p.relative_to(base).as_posix()
        raise AssertionError("path outside the scan root")

    while queue:
        f = queue.popleft()
        data = f.read_bytes()
        ext = f.suffix.lower()
        is_zip = zipfile.is_zipfile(f)
        is_tar = not is_zip and tarfile.is_tarfile(f)
        ps_ok, globs = picklescan_globals(data, rel(f))
        hinted = ext in PICKLE_EXT or ext in PYTORCH_EXT or f.parent == nested_dir
        # Content decides, not the name: a file is pickle content if its name says so, or if parsing
        # it yields globals that are well-formed Python identifiers. Running code from a pickle needs
        # at least one global, so this catches dangerous pickles under any name and any protocol;
        # tensor bytes (which may start with 0x80 by chance) do not parse into identifiers.
        is_pickle = (hinted and not (is_zip or is_tar)) or (
            bool(globs) and all(IDENT.match(m) and IDENT.match(n) for m, n, _ in globs))
        if is_zip or is_tar:
            # Every archive is unpacked, PyTorch zips included, so each member gets its own verdict.
            # A pickle stream in front of an archive (a polyglot) is judged as well.
            members = safe_extract(f, f.with_name(f.name + ".x"))
            queue.extend(members)
            results.append({"file": rel(f), "kind": "pytorch-zip" if ext in PYTORCH_EXT else "archive",
                            "verdict": "extracted", "detail": f"{len(members)} members queued"})
            if not is_pickle:
                continue
        if not is_pickle:
            for magic, fmt in OPAQUE.items():
                if data.startswith(magic):
                    results.append({"file": rel(f), "kind": f"compressed ({fmt})", "verdict": "incomplete",
                                    "methods": ["not decompressed: contents unscanned"], "globals": []})
                    break
            continue
        methods = ["picklescan 1.0.5 (bytes)" if ps_ok else "picklescan 1.0.5 (bytes): parser error"]
        if f.parent == nested_dir or (ext in PYTORCH_EXT and not is_zip):
            kind = "legacy-torch"
            # magic, protocol, sys_info, object, storage keys; raw storage bytes follow
            complete = ps_ok and reads_to_end(data, max_pickles=5)
        else:
            kind = "pickle+archive" if (is_zip or is_tar) else ("joblib" if ext == ".joblib" else "pickle")
            complete = ps_ok and not (is_zip or is_tar) and reads_to_end(data)
        if not complete and kind in ("pickle", "joblib"):
            # joblib (under any name) interleaves raw numpy bytes with pickle instructions
            parsed, raw, payloads = restricted_inventory(f, nested_dir)
            methods.append("restricted load (joblib 1.6.0, numpy 2.5.4)")
            # A complete restricted load saw every global the reader resolved; picklescan's partial
            # list (it loses its place after the first raw buffer) is kept only if the load also stopped.
            if not parsed:
                raw |= {(m, n) for m, n, _ in globs}
            globs = classify(raw)
            complete = parsed
            queue.extend(payloads)
        dangerous = [g for g in globs if g[2] == SafetyLevel.Dangerous.value]
        verdict = "dangerous" if dangerous else ("clean" if complete else "incomplete")
        results.append({"file": rel(f), "kind": kind, "verdict": verdict, "methods": methods,
                        "globals": sorted(set(globs))})
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("asset", type=pathlib.Path)
    ap.add_argument("--report", type=pathlib.Path, required=True)
    args = ap.parse_args(argv)
    lines = [f"asset={args.asset.name}", f"sha256={sha256(args.asset)}",
             f"tools: picklescan {version('picklescan')}, joblib {version('joblib')}, "
             f"numpy {version('numpy')}, python {sys.version.split()[0]}"]
    work = pathlib.Path(tempfile.mkdtemp(prefix="ser13-"))
    try:
        try:
            results = scan_asset(args.asset, work)
            aborted = None
        except ScanAbort as exc:
            results, aborted = [], str(exc)
        except Exception as exc:  # anything unexpected means the asset was not scanned
            results, aborted = [], f"unexpected {type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
    verdicts = collections.Counter(r["verdict"] for r in results)
    for r in results:
        lines.append(f"-- {r['file']} [{r['kind']}] {r['verdict'].upper()}")
        if "methods" in r:
            lines.append("   method: " + "; ".join(r["methods"]))
        if "detail" in r:
            lines.append("   " + r["detail"])
        for m, n, s in r.get("globals", []):
            lines.append(f"   {s:10} {m}.{n}")
    if aborted:
        status = 3
        lines.append(f"ABORTED: {aborted}")
    elif verdicts["dangerous"]:
        status = 1
    elif verdicts["incomplete"]:
        status = 2
    else:
        status = 0
    scanned = sum(1 for r in results if r["verdict"] in ("clean", "dangerous", "incomplete"))
    lines.append(f"summary: {scanned} pickle files; clean={verdicts['clean']} "
                 f"dangerous={verdicts['dangerous']} incomplete={verdicts['incomplete']}")
    lines.append(f"scan_status={status}")
    text = "\n".join(lines) + "\n"
    assert not any(tok in text for tok in (str(work), ":\\", ":/", "/home/", "/Users/", "AppData")), \
        "absolute path in report"
    args.report.write_bytes(text.encode("utf-8"))
    print(text.splitlines()[-2], text.splitlines()[-1])
    return status


if __name__ == "__main__":
    sys.exit(main())
