"""Static pickle scan of one release asset (ARTIFACT_SPEC 2.0 SER13), failing closed.

Run (pinned):
  uv run --no-project --python 3.13 --with picklescan==1.0.5 --with joblib==1.6.0 \\
      --with numpy==2.5.4 python -I scan.py ASSET.zip --report scan-report.txt

Every file in the asset, and in every archive nested inside it, gets its own verdict:
  clean      every global is innocuous or suspicious, and the whole file was read;
  dangerous  any global picklescan classifies as dangerous;
  incomplete the file could not be read to the end by any recorded method.
A PyTorch zip (.pt/.ckpt/.pth/.bin) is scanned in place; any other zip is extracted (path-checked)
and its members are queued, so archives inside archives are reached. A file that cannot be read as
a pickle to the end is incomplete, never clean. joblib files, which interleave raw numpy bytes with
pickle instructions, get a restricted load when picklescan stops early: joblib's own reader runs
with find_class replaced so that only numpy's array and scalar rebuilders, numpy.dtype and joblib's
NumpyArrayWrapper are real; every other global is an inert stub, recorded, and classified with
picklescan's own tables. Payloads of torch.storage._load_from_bytes are captured as files and
scanned like any other file, never loaded.

Exit status: 0 all clean; 1 any dangerous; 2 any incomplete (and none dangerous); 3 the asset could
not be scanned (bad archive, unsafe member path). The report holds only paths relative to the asset.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import pathlib
import pickle
import pickletools
import shutil
import sys
import tempfile
import zipfile
from importlib.metadata import version

import joblib.numpy_pickle as jnp
from picklescan.scanner import (
    SafetyLevel,
    _build_scan_result_from_raw_globals,
    scan_file_path,
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


def safe_extract(archive: pathlib.Path, dest: pathlib.Path) -> list[pathlib.Path]:
    """Extract every member, or abort: bad CRC, absolute or escaping paths, links."""
    try:
        with zipfile.ZipFile(archive) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise ScanAbort(f"CRC error in {bad}")
            out = []
            root = dest.resolve()
            for info in zf.infolist():
                name = info.filename
                target = (dest / name).resolve()
                if name.startswith(("/", "\\")) or ".." in pathlib.PurePosixPath(name).parts \
                        or not target.is_relative_to(root):
                    raise ScanAbort(f"unsafe member path {name!r}")
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ScanAbort(f"symlink member {name!r}")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                out.append(target)
            return out
    except (zipfile.BadZipFile, OSError, RuntimeError, NotImplementedError) as exc:
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


def picklescan_globals(path: pathlib.Path):
    res = scan_file_path(str(path))
    return res, [(g.module, g.name, g.safety.value) for g in res.globals]


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
        ext = f.suffix.lower()
        head = f.read_bytes()[:2]
        if zipfile.is_zipfile(f) and ext not in PYTORCH_EXT:
            members = safe_extract(f, f.with_name(f.name + ".x"))
            queue.extend(members)
            results.append({"file": rel(f), "kind": "archive", "verdict": "extracted",
                            "detail": f"{len(members)} members queued"})
            continue
        if f.read_bytes()[:3] in (b"\x1f\x8b\x08",) or head == b"BZ" or f.read_bytes()[:6] == b"\xfd7zXZ\x00":
            results.append({"file": rel(f), "kind": "compressed", "verdict": "incomplete",
                            "methods": ["not decompressed: contents unscanned"], "globals": []})
            continue
        is_pickle = ext in PICKLE_EXT or ext in PYTORCH_EXT or head[:1] == b"\x80" \
            or f.parent == nested_dir
        if not is_pickle:
            continue
        res, globs = picklescan_globals(f)
        methods = ["picklescan 1.0.5"]
        if zipfile.is_zipfile(f):
            kind = "pytorch-zip"
            with zipfile.ZipFile(f) as zf:
                pkls = [n for n in zf.namelist() if n.endswith(".pkl")]
                complete = bool(pkls) and all(reads_to_end(zf.read(n)) for n in pkls)
        elif f.parent == nested_dir or (ext in PYTORCH_EXT and not zipfile.is_zipfile(f)):
            kind = "legacy-torch"
            # magic, protocol, sys_info, object, storage keys; raw storage bytes follow
            complete = not res.scan_err and reads_to_end(f.read_bytes(), max_pickles=5)
        else:
            kind = "joblib" if ext == ".joblib" else "pickle"
            complete = not res.scan_err and reads_to_end(f.read_bytes())
        if kind == "joblib" and not complete:
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
    scanned = sum(1 for r in results if r["kind"] != "archive")
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
