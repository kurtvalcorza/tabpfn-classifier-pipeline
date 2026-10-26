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
resolves, memo aliases included, and the stubs' names are classified with picklescan's own tables;
for protocols below 3 the name pickle's compatibility mapping resolves it to (urllib.urlopen to
urllib.request.urlopen, for example) is classified too, and the more severe rating wins. It reads
consecutive streams twice, with a fresh memo per stream (separate pickle.load calls) and with one
memo shared across them (one Unpickler). After a STOP inside a frame it follows both continuations,
right after the STOP (a loader that reads ahead) and at the frame's end (one that reads the whole
frame); paths are told apart by position, stream count and memo contents, the globals of every path
are judged together, and every path must read to the end. In a legacy torch checkpoint the bytes
after its five pickles are read as further streams while they parse, and the rest must be exactly
the raw storages its keys describe; anything else is incomplete.
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
whose header or extent cannot be parsed makes it incomplete (object records before it are still
judged, so a dangerous one wins). When an object record's pickle stops inside a FRAME, the next record
is looked for both right after its STOP and at the frame's end, as loaders with and without read-ahead
would resume; every record either path reaches is judged, each object record's pickle is read on its
own, and a path that reaches no valid record makes the file incomplete. Bytes without NumPy magic
after a record are a pickle, as numpy.load(allow_pickle=True) falls back to pickle.load for them.
Extension-registry opcodes (EXT1/2/4) make a file pickle content and incomplete, also in a stream
that fails to load if it began with a PROTO opcode: the global they name is resolved only by the
loading process's copyreg registry; joblib's nested reader for object arrays records them the same
way. Payloads of torch.storage._load_from_bytes are captured as files
and scanned like any other file, never loaded; a payload that cannot be rebuilt as bytes makes its
file incomplete.
gzip, bzip2, xz, LZMA-alone, zstd, lz4, zlib (any window size, recognised by its header and a bounded
trial decompress), 7z and rar content is not decompressed and is incomplete, also when it sits in
front of an archive. A file that cannot be read to the end by either reader is incomplete, never
clean.
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
    constructor with side effects): picklescan's tables are the policy, not a judgement made here;
  - an extension-registry opcode in bytes that have no PROTO opcode and do not load to a STOP (a known
    gap, ruled 2026-10-26): flagging those would turn 4 of the 6 published sample assets incomplete
    through 15 raw tensor members whose bytes happen to begin with those opcodes;
  - a callable already registered in the scanning process's copyreg extension registry: a fresh
    scanner's registry is empty, an unregistered extension code raises, the forwarded find_class stubs
    any name a registered code resolves to, and nothing in a scanned file can register one (the
    environment running the scan is trusted).
  Assumes: the scanned bytes are the bytes that get published (the asset digest is pinned), and the
  environment running the scan is trusted.

Exit status: 0 all clean; 1 any dangerous; 2 any incomplete (and none dangerous); 3 the asset could
not be scanned (unreadable asset, bad archive, unsafe or duplicate member, unexpected error, a
report that would name a host path, or a report that cannot be written). The report holds only paths
relative to the asset; text that cannot be encoded as UTF-8 (names with surrogate escapes) is written
backslash-escaped.
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import io
import lzma
import math
import _compat_pickle
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
    # LZMA-alone (joblib's .lzma, header 5d 00), with a trial decompress so tensor bytes are not mistaken
    if data.startswith(b"\x5d\x00"):
        try:
            lzma.LZMADecompressor(lzma.FORMAT_ALONE).decompress(data[:65536], 64)
            return "lzma"
        except lzma.LZMAError:
            pass
    # zlib (RFC 1950), any window size: CM=8, CINFO<=7, header checksum, and a bounded trial decompress
    # that must raise no error, so tensor bytes that happen to pass the 2-byte checksum are not mistaken
    # for zlib. Output is not required: empty stored blocks can delay it past any bound.
    if len(data) >= 2 and data[0] & 0x0F == 8 and data[0] >> 4 <= 7 and (data[0] << 8 | data[1]) % 31 == 0:
        try:
            zlib.decompressobj().decompress(data[:65536], 65536)
            return "zlib"
        except zlib.error:
            pass
    return None


NPY_MAGIC = b"\x93NUMPY"


class NpyUnreadable(Exception):
    """A .npy record whose header or extent cannot be parsed (the file is then incomplete). payload holds
    the pickle streams of the object records reached, which are still judged."""

    def __init__(self, message: str, payload: list[bytes] | None = None):
        super().__init__(message)
        self.payload = payload


def npy_object_payloads(data: bytes) -> list[bytes] | None:
    """Walk every record of a .npy file, as successive numpy.load calls would read them, and return the
    pickle stream behind each object-dtype record, one per record; None if the file is not a .npy or
    holds only plain records. Each header is parsed (ast.literal_eval) and its dtype resolved by NumPy
    itself, as numpy.load does, so every spelling of an object dtype ('O', '|O8', 'object',
    structured) counts. After an object record whose pickle stops inside a FRAME, the next record is
    looked for both right after the STOP and at the frame's end, and every path is followed. A record
    whose header or extent cannot be parsed raises NpyUnreadable, carrying the object payloads read on
    every path; an object record whose pickle cannot be delimited carries the rest of the file, so it
    is never clean."""
    if not data.startswith(NPY_MAGIC):
        return None
    payloads: list[bytes] = []
    errors: list[str] = []
    todo, done = [0], set()
    while todo:
        off = todo.pop()
        if off in done or off == len(data):
            continue
        done.add(off)
        if len(done) > 10000:
            errors.append("too many record paths")
            break
        if off and not data.startswith(NPY_MAGIC, off):
            # numpy.load(allow_pickle=True) falls back to pickle.load for bytes without NumPy magic
            bio = io.BytesIO(data)
            bio.seek(off)
            frame_end = None
            try:
                for op, arg, pos in pickletools.genops(bio):
                    if op.name == "FRAME":
                        frame_end = pos + 9 + arg
            except Exception:
                payloads.append(data[off:])
                errors.append(f"record at offset {off}: no NumPy magic, and not a complete pickle")
                continue
            stop = bio.tell()
            payloads.append(data[off:stop])
            todo.append(stop)
            if frame_end is not None and stop < frame_end <= len(data):
                todo.append(frame_end)
            continue
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
            errors.append(f"record at offset {off}: {exc}")
            continue
        if not dtype.hasobject:
            todo.append(end)
            continue
        # numpy.load unpickles one stream behind an object header; the next record starts after it
        bio = io.BytesIO(data)
        bio.seek(body)
        frame_end = None
        try:
            for op, arg, pos in pickletools.genops(bio):
                if op.name == "FRAME":
                    frame_end = pos + 9 + arg
        except Exception:
            payloads.append(data[body:])
            continue
        stop = bio.tell()
        payloads.append(data[body:stop])
        todo.append(stop)
        # a STOP inside a FRAME: a loader with read-ahead resumes after the STOP, one without at the
        # frame's end; both are followed
        if frame_end is not None and stop < frame_end <= len(data):
            todo.append(frame_end)
    if errors:
        raise NpyUnreadable("; ".join(errors), payloads or None)
    return payloads or None


def with_compat_names(pairs: set[tuple[str, str]]) -> set[tuple[str, str]]:
    """The names plus what pickle's protocol-<3 compatibility mapping resolves each to (picklescan does
    not report the protocol, so the mapping is applied to every name it finds)."""
    out = set(pairs)
    for module, name in pairs:
        mapped = _compat_pickle.NAME_MAPPING.get((module, name))
        if mapped is None and module in _compat_pickle.IMPORT_MAPPING:
            mapped = (_compat_pickle.IMPORT_MAPPING[module], name)
        if mapped is not None:
            out.add(mapped)
    return out


def memo_key(memo: dict) -> tuple:
    """A fingerprint of a memo's contents, so reader paths that reach one position with different memos
    are both followed: plain values by hash, other objects by identity."""
    return tuple(sorted((k, type(v).__name__, hash(v) if isinstance(v, str | bytes | int | float | bool | None)
                         else id(v)) for k, v in memo.items()))


def classify(raw: set[tuple[str, str]]):
    res = _build_scan_result_from_raw_globals(raw, "inventory")
    return [(g.module, g.name, g.safety.value) for g in res.globals]


# bytes per element of each torch storage type, for the raw storages of a legacy checkpoint
ELEMENT_SIZE = {"Double": 8, "Float": 4, "Half": 2, "BFloat16": 2, "Long": 8, "Int": 4, "Short": 2,
                "Char": 1, "Byte": 1, "Bool": 1, "ComplexDouble": 16, "ComplexFloat": 8, "Untyped": 1,
                "QUInt8": 1, "QInt8": 1, "QInt32": 4, "QUInt4x2": 1, "QUInt2x4": 1}


def legacy_tail_is_storages(tail: bytes, keys, storages: dict) -> bool:
    """True if the bytes after a legacy checkpoint's pickles are exactly its raw storages, in key order,
    each an int64 element count and then its bytes, as torch's legacy save writes them."""
    if not isinstance(keys, list):
        return False
    off = 0
    for key in keys:
        qualname, _ = storages.get(str(key), ("", None))
        m = re.fullmatch(r"torch(?:\.cuda)?\.(\w+)Storage", qualname)
        if m is None or m.group(1) not in ELEMENT_SIZE or off + 8 > len(tail):
            return False
        count = int.from_bytes(tail[off:off + 8], "little", signed=True)
        off += 8 + count * ELEMENT_SIZE[m.group(1)]
        if count < 0 or off > len(tail):
            return False
    return off == len(tail)


def restricted_inventory(data: bytes, nested_dir: pathlib.Path, legacy: bool = False):
    """joblib's reader over the bytes, with every global outside ALLOW stubbed, across consecutive
    pickle streams, once with a fresh memo per stream and once with one shared memo. find_class sees the true module and name of
    every global the stream resolves, memo aliases included. For a legacy torch checkpoint the bytes
    after its five pickles are read as streams while they parse, and the rest must be its raw storages.
    Returns (parsed_to_end, raw globals, payloads, unresolved, extension codes)."""
    seen: set[tuple[str, str]] = set()
    storages: dict[str, tuple[str, object]] = {}
    extensions: set[int] = set()   # EXT1/2/4 codes (of loaded streams, or failed ones that began with PROTO)
    pending: list[int] = []
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
            if out not in payloads:  # the same payload is met once per memo mode
                payloads.append(out)
        else:  # a payload this reader could not rebuild as bytes was never scanned
            unresolved.append(type(arg).__name__)
        return Stub()

    probe = pickle.Unpickler(io.BytesIO(b"."))

    class Restricted(jnp.NumpyUnpickler):
        def persistent_load(self, pid):
            # torch checkpoints reference their storages by persistent id; nothing is resolved or loaded,
            # but a legacy checkpoint's storage types are kept to account for its raw tail
            if isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage":
                storages[str(pid[2])] = (getattr(pid[1], "qualname", ""), pid[4])
            return Stub()

        def get_extension(self, code):
            # the global behind an extension code lives in the loading process's copyreg registry
            pending.append(code)
            self.append(type(f"Stub[extension {code}]", (Stub,), {}))

        def find_class(self, module, name):
            seen.add((module, name))
            if self.proto < 3 and self.fix_imports:  # what pickle's own find_class resolves the name to
                mapped = _compat_pickle.NAME_MAPPING.get((module, name))
                if mapped is None and module in _compat_pickle.IMPORT_MAPPING:
                    mapped = (_compat_pickle.IMPORT_MAPPING[module], name)
                if mapped is not None and mapped != (module, name):
                    seen.add(mapped)
            if (module, name) in ALLOW:
                return pickle.Unpickler.find_class(probe, module, name)
            # pickle resolves module + name as one dotted path (protocol 4 allows a dotted name), so every
            # split of torch.storage._load_from_bytes is the loader; an attribute of it cannot be captured
            qualified = f"{module}.{name}"
            if qualified == "torch.storage._load_from_bytes":
                return capture
            if qualified.startswith("torch.storage._load_from_bytes."):
                unresolved.append(qualified)
            return type(f"Stub[{module}.{name}]", (Stub,), {"qualname": f"{module}.{name}"})

    def read(shared_memo: bool) -> bool:
        """One pass over the streams: shared_memo as one Unpickler calling load() repeatedly, otherwise a
        fresh memo per stream, as separate pickle.load calls. After a STOP inside a frame both
        continuations are followed: right after the STOP and at the frame's end. True only if every
        path reads to the end (for a legacy checkpoint: its five pickles, further streams, then exactly
        its raw storages)."""
        ok, todo, done = True, [(0, {}, 0, None)], set()
        while todo:
            pos, memo, count, keys = todo.pop()
            key = (pos, min(count, 5), memo_key(memo) if shared_memo else ())
            if key in done:
                continue
            done.add(key)
            if len(done) > 10000:
                return False
            if pos == len(data):
                ok = ok and count > 0 and (not legacy or (count >= 5 and legacy_tail_is_storages(b"", keys, storages)))
                continue
            fh = io.BytesIO(data)
            fh.seek(pos)
            unpickler = Restricted("<bytes>", fh, ensure_native_byte_order=True, mmap_mode=None)
            if shared_memo:
                unpickler.memo = dict(memo)
            pending.clear()
            before = set(seen)
            try:
                obj = unpickler.load()
            except Exception:
                if pending and data[pos:pos + 1] == b"\x80":  # a failed stream that announced a protocol
                    extensions.update(pending)
                if legacy and count >= 5:
                    # not a stream: the raw storages start here; globals their bytes happen to spell are
                    # dropped unless dangerous
                    junk = seen - before
                    seen.clear()
                    seen.update(before)
                    if junk:
                        seen.update((m, n) for m, n, s in classify(junk) if s == SafetyLevel.Dangerous.value)
                    ok = ok and legacy_tail_is_storages(data[pos:], keys, storages)
                else:
                    ok = False
                continue
            extensions.update(pending)
            if count == 4:  # the legacy storage keys
                keys = obj
            following = [fh.tell()]
            frame = unpickler._unframer.current_frame
            rest = 0 if frame is None else len(frame.getvalue()) - frame.tell()
            if rest:  # a STOP inside a frame: also resume right after the STOP
                following.append(fh.tell() - rest)
            for nxt in following:
                todo.append((nxt, unpickler.memo if shared_memo else {}, count + 1, keys))
        return ok

    class NestedRestricted(pickle._Unpickler):
        # joblib's NumpyArrayWrapper.read_array unpickles an object array with a fresh Unpickler and
        # forwards only find_class; this class takes its place, so extension codes are recorded too
        def get_extension(self, code):
            pending.append(code)
            self.append(type(f"Stub[extension {code}]", (Stub,), {}))

    # both memo behaviours a standard loader can have; the file is read to the end only if both read it
    nested_unpickler, jnp.Unpickler = jnp.Unpickler, NestedRestricted
    try:
        parsed = read(shared_memo=False) & read(shared_memo=True)
    finally:
        jnp.Unpickler = nested_unpickler
    return parsed, seen, payloads, unresolved, sorted(extensions)


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
        npy_error = None
        try:
            npy_payloads = npy_object_payloads(data)
        except NpyUnreadable as exc:
            if exc.payload is None:
                results.append({"file": rel(f), "kind": "npy", "verdict": "incomplete",
                                "methods": [f"NumPy header unreadable: {exc}"], "globals": []})
                continue
            npy_payloads, npy_error = exc.payload, str(exc)
        # an object-dtype .npy (also inside an .npz) holds a pickle stream behind each object header; each
        # is read on its own, as each numpy.load call unpickles one
        streams = npy_payloads if npy_payloads is not None else [data]
        legacy = f.parent == nested_dir or (ext in PYTORCH_EXT and not is_zip)
        # Content decides, not the name. picklescan reads the bytes, and the restricted load reads
        # them too: its find_class sees the true name of every global the stream resolves (picklescan
        # 1.0.5 can misread memo aliases), and it captures torch payloads. A file is pickle content if
        # its name says so, it is an object .npy, or either reader finds a global that is dangerous
        # or a well-formed Python identifier; tensor bytes do not parse into identifiers.
        ps_ok, ps_globs, parsed, raw, payloads, unresolved, extensions = True, [], True, set(), [], [], set()
        for stream in streams:
            ok, found = picklescan_globals(stream, rel(f))
            p_ok, p_raw, p_payloads, p_unresolved, p_ext = restricted_inventory(stream, nested_dir, legacy)
            ps_ok, parsed = ps_ok and ok, parsed and p_ok
            ps_globs += found
            raw |= p_raw
            payloads += [x for x in p_payloads if x not in payloads]
            unresolved += p_unresolved
            extensions.update(p_ext)
        extensions = sorted(extensions)
        ps_raw = with_compat_names({(m, n) for m, n, _ in ps_globs})
        ps_dangerous = {(m, n) for m, n, s in ps_globs if s == SafetyLevel.Dangerous.value}
        # A restricted load that read the whole file saw every resolved global; picklescan's list is then
        # kept only for its dangerous findings (it loses its place after raw buffers and misreads memo
        # aliases). Otherwise both lists are kept.
        globs = classify(raw | (ps_dangerous if parsed else ps_raw))
        is_pickle = hinted or npy_payloads is not None or bool(extensions) or any(
            s == SafetyLevel.Dangerous.value or (IDENT.match(m) and IDENT.match(n)) for m, n, s in globs)
        if not is_pickle:
            # also in front of an archive; a tar that tarfile itself decompressed was unpacked and scanned
            fmt = opaque_format(data)
            if fmt is not None and not (is_tar and fmt in ("gzip", "bzip2", "xz")):
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
            kind = ("pickle+archive" if (is_zip or is_tar) else "npy-object" if npy_payloads is not None
                    else "joblib" if ext == ".joblib" else "pickle")
            complete = not (is_zip or is_tar) and parsed and (all(reads_to_end(s) for s in streams)
                                                              or kind in ("joblib", "pickle"))
        if npy_error:
            methods.append(f"NumPy header unreadable: {npy_error}")
            complete = False
        if extensions:
            methods.append(f"extension-registry opcode (code {', '.join(map(str, extensions))}): global "
                           "resolved only by the loader's copyreg registry, unscanned")
            complete = False
        if unresolved:
            methods.append(f"torch payload not rebuilt as bytes ({', '.join(unresolved)}): unscanned")
            complete = False
        queue.extend(payloads)
        dangerous = [g for g in globs if g[2] == SafetyLevel.Dangerous.value]
        verdict = "dangerous" if dangerous else ("clean" if complete else "incomplete")
        results.append({"file": rel(f), "kind": kind, "verdict": verdict, "methods": methods,
                        "globals": sorted(set(globs))})
    return results


def safe_text(text: str) -> str:
    """Report text that always encodes as UTF-8: lone surrogates (from names decoded with surrogate
    escapes) become backslash escapes."""
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def host_path(text: str, work: pathlib.Path, asset: pathlib.Path) -> str | None:
    """Name the host-specific location a report would leak, if any: the scan's work or temp directory,
    the home or current directory, the asset's directory, or any absolute drive path. Relative member
    paths such as models/home/... or AppData/... belong to the asset and are allowed."""
    places = {"work directory": work, "temp directory": pathlib.Path(tempfile.gettempdir()),
              "home directory": pathlib.Path.home(), "current directory": pathlib.Path.cwd(),
              "asset directory": asset.absolute().parent}
    for label, place in places.items():
        for form in {str(place), place.as_posix()}:
            if len(form) > 1 and re.search(r"(?<![\w.\-/\\])" + re.escape(form), text):
                return label
    if re.search(r"(?<![\w.\-/\\])[A-Za-z]:[\\/]", text):
        return "absolute drive path"
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("asset", type=pathlib.Path)
    ap.add_argument("--report", type=pathlib.Path, required=True)
    args = ap.parse_args(argv)
    work = pathlib.Path(tempfile.mkdtemp(prefix="ser13-"))
    digest = "unreadable"
    try:
        try:
            digest = sha256(args.asset)  # inside the protected path: an unreadable asset is status 3
            results = scan_asset(args.asset, work)
            aborted = None
        except ScanAbort as exc:
            results, aborted = [], str(exc)
        except Exception as exc:  # anything unexpected means the asset was not scanned
            results, aborted = [], f"unexpected {type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
    lines = [f"asset={args.asset.name}", f"sha256={digest}",
             f"tools: picklescan {version('picklescan')}, joblib {version('joblib')}, "
             f"numpy {version('numpy')}, python {sys.version.split()[0]}"]
    head = len(lines)
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
    summary = (f"summary: {scanned} pickle files; clean={verdicts['clean']} "
               f"dangerous={verdicts['dangerous']} incomplete={verdicts['incomplete']}")
    text = safe_text("\n".join(lines + [summary, f"scan_status={status}"]) + "\n")
    leak = host_path(text, work, args.asset)
    if leak:  # refused through the status-3 path; the offending lines are not written
        status = 3
        text = safe_text("\n".join(lines[:head] + [f"ABORTED: report would name a host path ({leak})", summary,
                                                   f"scan_status={status}"]) + "\n")
    try:
        args.report.write_bytes(text.encode("utf-8"))
    except Exception as exc:  # the report could not be written: status 3, never a crash
        status = 3
        text = safe_text("\n".join(lines[:head] + [f"ABORTED: report could not be written ({type(exc).__name__})",
                                                   summary, f"scan_status={status}"]) + "\n")
        try:
            args.report.write_bytes(text.encode("utf-8"))
        except Exception:
            pass
    print(text.splitlines()[-2], text.splitlines()[-1])
    return status


if __name__ == "__main__":
    sys.exit(main())
