"""Static pickle scan of one release asset (ARTIFACT_SPEC 2.0 SER13), failing closed.

Run (pinned):
  uv run --no-project --python 3.13 --with picklescan==1.0.5 --with joblib==1.6.0 \\
      --with numpy==2.5.4 python -I scan.py ASSET.zip --report scan-report.txt

Every file in the asset, and in every archive nested inside it, gets its own verdict:
  clean      every global is innocuous or suspicious, and the whole file was read;
  dangerous  any global picklescan classifies as dangerous;
  incomplete the file could not be read to the end by any recorded method.
Content decides, not the file name. Two readers see every file's bytes: picklescan, and a
restricted load (joblib's own reader with find_class replaced, so only numpy's array and scalar
rebuilders, numpy.dtype, joblib's NumpyArrayWrapper, _codecs.encode and bytearray are real and
run; every other global is an inert stub). find_class sees the true name of every global a stream
resolves, memo aliases included, and the stubs' names are classified with picklescan's own tables.
A file is pickle content if its extension says so, it is an object-dtype .npy, or either reader
finds a global that is dangerous or a well-formed Python identifier (protocol 0/1 pickles under any
name); tensor bytes do not parse into identifiers. The asset itself and every zip and tar inside it,
PyTorch zips included, are extracted with member-path, link and duplicate-name checks (a file that
is both a zip and a tar is extracted both ways), and members are queued, so archives inside
archives are reached; each nested archive unpacks into a fresh directory, so nothing can write over
a member queued for scanning, and members are reported as `outer!/member`. A pickle stream in front
of an archive, including the asset itself, is judged too. Every record of a .npy file is walked, as
successive numpy.load calls would read them: each header is parsed and its dtype judged by NumPy
(dtype.hasobject), as numpy.load does; any object record makes the file pickle content, and a record
whose header or extent cannot be parsed makes it incomplete. Payloads of
torch.storage._load_from_bytes are captured as files and scanned like any other file, never loaded;
a payload that cannot be rebuilt as bytes makes its file incomplete.
gzip, bzip2, xz, zstd, lz4, zlib (any window size), 7z and rar content is not decompressed and is
incomplete. A file that cannot be read to the end by either reader is incomplete, never clean.
Threat model and scope (frozen 2026-10-24; review findings inside it are fixed, findings outside it
are recorded as out of scope):
  Goal: before an asset is published, find every byte sequence in it that a standard loader could
  unpickle, and report whether it references a callable picklescan classifies as dangerous.
  In scope: zip and tar archives at any depth; pickle streams of any protocol under any file name,
  and in front of an archive; PyTorch zip and legacy checkpoints; joblib files; object-dtype .npy
  and .npz; torch payloads embedded through torch.storage._load_from_bytes.
  Out of scope, reported incomplete where recognised, otherwise not detected:
  - data a loader decodes before unpickling with no format marker (raw deflate, base64, custom or
    encrypted encodings);
  - code that is not pickle (Python source, notebooks, TorchScript, ONNX custom operators, shared
    libraries) and loaders that read a pickle from a non-standard offset inside another format;
  - harm done through callables picklescan classifies as safe or suspicious (for example a library
    constructor with side effects): picklescan's tables are the policy, not a judgement made here.
  Assumes: the scanned bytes are the bytes that get published (the asset digest is pinned), and the
  environment running the scan is trusted.

Exit status: 0 all clean; 1 any dangerous; 2 any incomplete (and none dangerous); 3 the asset could
not be scanned (bad archive, unsafe or duplicate member, unexpected error). The report holds only
paths relative to the asset.
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import io
import math
import pathlib
import pickle
import pickletools
import re
import shutil
import sys
import tarfile
import tempfile
import zlib
import zipfile
from importlib.metadata import version

import joblib.numpy_pickle as jnp
import numpy as np
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
    ("_codecs", "encode"),
    ("builtins", "bytearray"),
    ("__builtin__", "bytearray"),
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


def safe_extract(archive: pathlib.Path, dest: pathlib.Path, interpretation: str = "zip") -> list[pathlib.Path]:
    """Extract a zip or tar completely, or abort: bad CRC, absolute or escaping paths, links,
    special files, duplicate member names (a loader and this scan could read different copies)."""
    out, seen = [], set()
    try:
        if interpretation == "zip" and zipfile.is_zipfile(archive):
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
          }


def opaque_format(data: bytes) -> str | None:
    """Name the compression or container format a loader could decode before unpickling, if any."""
    for magic, fmt in OPAQUE.items():
        if data.startswith(magic):
            return fmt
    # zlib (RFC 1950), any window size: CM=8, CINFO<=7, header checksum, and a trial decompress, so
    # tensor bytes that happen to pass the 2-byte checksum are not mistaken for zlib.
    if len(data) >= 2 and data[0] & 0x0F == 8 and data[0] >> 4 <= 7 and (data[0] << 8 | data[1]) % 31 == 0:
        try:
            if zlib.decompressobj().decompress(data[:65536], 64):
                return "zlib"
        except zlib.error:
            pass
    return None


NPY_MAGIC = b"\x93NUMPY"


class NpyUnreadable(Exception):
    """A file with the NumPy magic whose header cannot be parsed (it is then incomplete)."""


def npy_object_payload(data: bytes) -> bytes | None:
    """Walk every record of a .npy file, as successive numpy.load calls would read them, and return the
    pickle streams behind its object-dtype records, joined; None if the file is not a .npy or holds
    only plain records. Each header is parsed (ast.literal_eval) and its dtype resolved by NumPy
    itself, as numpy.load does, so every spelling of an object dtype ('O', '|O8', 'object',
    structured) counts. A record whose header or extent cannot be parsed raises NpyUnreadable; an
    object record whose pickle cannot be delimited carries the rest of the file, so it is never clean."""
    if not data.startswith(NPY_MAGIC):
        return None
    payloads: list[bytes] = []
    off = 0
    while off < len(data):
        try:
            if not data.startswith(NPY_MAGIC, off):
                raise ValueError("no NumPy magic")
            major = data[off + 6]
            if major == 1:
                hlen, start = int.from_bytes(data[off + 8:off + 10], "little"), off + 10
            else:
                hlen, start = int.from_bytes(data[off + 8:off + 12], "little"), off + 12
            if start + hlen > len(data):
                raise ValueError("header runs past the end of the file")
            header = ast.literal_eval(data[start:start + hlen].decode("latin-1"))
            dtype = np.dtype(header["descr"])
            body = start + hlen
            if not dtype.hasobject:
                shape = tuple(header["shape"])
                if not all(isinstance(n, int) and n >= 0 for n in shape):
                    raise ValueError(f"bad shape {shape!r}")
                end = body + dtype.itemsize * math.prod(shape)
                if end > len(data):
                    raise ValueError("array data runs past the end of the file")
        except Exception as exc:
            raise NpyUnreadable(f"record at offset {off}: {exc}") from exc
        if not dtype.hasobject:
            off = end
            continue
        # numpy.load unpickles one stream behind an object header; the next record starts after it
        bio = io.BytesIO(data)
        bio.seek(body)
        try:
            for _ in pickletools.genops(bio):
                pass
        except Exception:
            payloads.append(data[body:])
            break
        payloads.append(data[body:bio.tell()])
        off = bio.tell()
    return b"".join(payloads) if payloads else None


def classify(raw: set[tuple[str, str]]):
    res = _build_scan_result_from_raw_globals(raw, "inventory")
    return [(g.module, g.name, g.safety.value) for g in res.globals]


def restricted_inventory(data: bytes, nested_dir: pathlib.Path, max_pickles: int | None = None):
    """joblib's reader over the bytes, with every global outside ALLOW stubbed, across consecutive
    pickle streams (up to max_pickles). find_class sees the true module and name of every global the
    stream resolves, memo aliases included. Returns (parsed_to_end, raw globals, payloads, unresolved)."""
    seen: set[tuple[str, str]] = set()
    payloads: list[pathlib.Path] = []
    unresolved: list[str] = []

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

    def capture(arg=None, *a, **k):
        if isinstance(arg, bytes | bytearray):
            digest = hashlib.sha256(arg).hexdigest()
            out = nested_dir / f"nested-{digest[:16]}.pt"
            out.write_bytes(bytes(arg))
            payloads.append(out)
        else:  # a payload this reader could not rebuild as bytes was never scanned
            unresolved.append(type(arg).__name__)
        return Stub()

    probe = pickle.Unpickler(io.BytesIO(b"."))

    class Restricted(jnp.NumpyUnpickler):
        def persistent_load(self, pid):
            # torch checkpoints reference their storages by persistent id; nothing is resolved or loaded
            return Stub()

        def find_class(self, module, name):
            seen.add((module, name))
            if (module, name) in ALLOW:
                return pickle.Unpickler.find_class(probe, module, name)
            # pickle resolves module + name as one dotted path (protocol 4 allows a dotted name), so every
            # split of torch.storage._load_from_bytes is the loader; an attribute of it cannot be captured
            qualified = f"{module}.{name}"
            if qualified == "torch.storage._load_from_bytes":
                return capture
            if qualified.startswith("torch.storage._load_from_bytes."):
                unresolved.append(qualified)
            return type(f"Stub[{module}.{name}]", (Stub,), {})

    fh = io.BytesIO(data)
    count, parsed = 0, False
    try:
        while fh.tell() < len(data) and (max_pickles is None or count < max_pickles):
            Restricted("<bytes>", fh, ensure_native_byte_order=True, mmap_mode=None).load()
            count += 1
        parsed = count > 0 and (fh.tell() == len(data) or max_pickles is not None)
    except Exception:
        parsed = False
    return parsed, seen, payloads, unresolved


def scan_asset(asset: pathlib.Path, work: pathlib.Path):
    root = work / "x"
    nested_dir = work / "nested"
    root.mkdir(parents=True)
    nested_dir.mkdir()
    outer = work / "in" / asset.name
    outer.parent.mkdir()
    shutil.copyfile(asset, outer)
    queue = collections.deque([outer])
    results = []
    # every extraction root, with the label its members are reported under; nested archives get a
    # fresh directory of their own, so no archive can write over a member queued for scanning
    bases = [(outer.parent.resolve(), ""), (root.resolve(), ""), (nested_dir.resolve(), "<nested>/")]

    def rel(p: pathlib.Path) -> str:
        p = p.resolve()
        for base, label in sorted(bases, key=lambda b: len(str(b[0])), reverse=True):
            if p.is_relative_to(base):
                return label + p.relative_to(base).as_posix()
        raise AssertionError("path outside the scan root")

    while queue:
        f = queue.popleft()
        data = f.read_bytes()
        ext = f.suffix.lower()
        is_zip = zipfile.is_zipfile(f)
        is_tar = tarfile.is_tarfile(f)
        hinted = (ext in PICKLE_EXT or ext in PYTORCH_EXT or f.parent == nested_dir) and not (is_zip or is_tar)
        if f == outer and not (is_zip or is_tar):
            raise ScanAbort(f"asset {asset.name} is not a zip or tar archive")
        if is_zip or is_tar:
            # Every archive is unpacked, PyTorch zips included, so each member gets its own verdict; a
            # file that is both a ZIP and a tar is unpacked both ways. The outer asset's own members
            # are reported under their plain names.
            for interpretation, extract in (("zip", is_zip), ("tar", is_tar)):
                if not extract:
                    continue
                if f == outer and interpretation == "zip":
                    dest = root
                else:
                    dest = work / "a" / str(len(bases))
                    dest.mkdir(parents=True)
                    suffix = "!/" if not (is_zip and is_tar) else f"!{interpretation}/"
                    bases.append((dest.resolve(), rel(f) + suffix))
                members = safe_extract(f, dest, interpretation)
                queue.extend(members)
                results.append({"file": rel(f), "kind": ("pytorch-zip" if ext in PYTORCH_EXT else "archive")
                                + ("" if not (is_zip and is_tar) else f" ({interpretation})"),
                                "verdict": "extracted", "detail": f"{len(members)} members queued"})
        try:
            npy_payload = npy_object_payload(data)
        except NpyUnreadable as exc:
            results.append({"file": rel(f), "kind": "npy", "verdict": "incomplete",
                            "methods": [f"NumPy header unreadable: {exc}"], "globals": []})
            continue
        if npy_payload is not None:
            # an object-dtype .npy (also inside an .npz) is a pickle stream behind the NumPy header
            data = npy_payload
        legacy = f.parent == nested_dir or (ext in PYTORCH_EXT and not is_zip)
        # Content decides, not the name. picklescan reads the bytes, and the restricted load reads
        # them too: its find_class sees the true name of every global the stream resolves (picklescan
        # 1.0.5 can misread memo aliases), and it captures torch payloads. A file is pickle content if
        # its name says so, it is an object .npy, or either reader finds a global that is dangerous
        # or a well-formed Python identifier; tensor bytes do not parse into identifiers.
        ps_ok, ps_globs = picklescan_globals(data, rel(f))
        parsed, raw, payloads, unresolved = restricted_inventory(data, nested_dir, 5 if legacy else None)
        ps_raw = {(m, n) for m, n, _ in ps_globs}
        ps_dangerous = {(m, n) for m, n, s in ps_globs if s == SafetyLevel.Dangerous.value}
        # A restricted load that read the whole file saw every resolved global; picklescan's list is then
        # kept only for its dangerous findings (it loses its place after raw buffers and misreads memo
        # aliases). Otherwise both lists are kept.
        globs = classify(raw | (ps_dangerous if parsed else ps_raw))
        is_pickle = hinted or npy_payload is not None or any(
            s == SafetyLevel.Dangerous.value or (IDENT.match(m) and IDENT.match(n)) for m, n, s in globs)
        if not is_pickle:
            if not (is_zip or is_tar):
                fmt = opaque_format(data)
                if fmt is not None:
                    results.append({"file": rel(f), "kind": f"compressed ({fmt})", "verdict": "incomplete",
                                    "methods": ["not decompressed: contents unscanned"], "globals": []})
            continue
        methods = ["picklescan 1.0.5 (bytes)" if ps_ok else "picklescan 1.0.5 (bytes): parser error",
                   "restricted load (joblib 1.6.0, numpy 2.5.4)"]
        if legacy:
            kind = "legacy-torch"
            # magic, protocol, sys_info, object, storage keys; raw storage bytes follow
            complete = ps_ok and parsed and reads_to_end(data, max_pickles=5)
        else:
            kind = ("pickle+archive" if (is_zip or is_tar) else "npy-object" if npy_payload is not None
                    else "joblib" if ext == ".joblib" else "pickle")
            complete = not (is_zip or is_tar) and parsed and (reads_to_end(data) or kind in ("joblib", "pickle"))
        if unresolved:
            methods.append(f"torch payload not rebuilt as bytes ({', '.join(unresolved)}): unscanned")
            complete = False
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
