"""Regression tests for scan.py. Fixtures pickle references to dangerous callables (os.system);
pickling a reference never calls it, and nothing here unpickles a fixture outside scan.py's
restricted reader. Run with the same pinned environment as scan.py, plus pytest."""
import collections
import gzip
import io
import os
import pickle
import zipfile

import joblib
import numpy as np
import pytest

import scan

EVIL = pickle.dumps(os.system, protocol=4)          # GLOBAL posix/nt system: dangerous
CLEAN = pickle.dumps(collections.OrderedDict(a=1), protocol=4)


def make_zip(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def joblib_bytes(obj, tmp_path, name):
    p = tmp_path / name
    joblib.dump(obj, p)
    return p.read_bytes()


def run(tmp_path, asset):
    report = tmp_path / "report.txt"
    status = scan.main([str(asset), "--report", str(report)])
    return status, report.read_text(encoding="utf-8")


def test_clean_asset_is_status_0(tmp_path):
    jb = joblib_bytes({"a": np.arange(100.0)}, tmp_path, "c.joblib")
    asset = make_zip(tmp_path / "a.zip", {"c.joblib": jb, "m.pkl": CLEAN, "info.json": b"{}"})
    status, text = run(tmp_path, asset)
    assert status == 0, text
    assert "scan_status=0" in text


def test_dangerous_plain_pickle_beside_joblib_is_not_downgraded(tmp_path):
    # review N1: expected joblib parse errors must not hide a dangerous global elsewhere
    jb = joblib_bytes({"a": np.arange(100.0)}, tmp_path, "c.joblib")
    asset = make_zip(tmp_path / "a.zip", {"c.joblib": jb, "evil.pkl": EVIL})
    status, text = run(tmp_path, asset)
    assert status == 1, text
    assert "evil.pkl [pickle] DANGEROUS" in text


def test_dangerous_global_after_joblib_buffer(tmp_path):
    # review N2 / Codex P1: a dangerous global behind raw array bytes is found by the restricted load
    jb = joblib_bytes({"a": np.arange(1000.0), "b": os.system}, tmp_path, "e.joblib")
    asset = make_zip(tmp_path / "a.zip", {"e.joblib": jb})
    status, text = run(tmp_path, asset)
    assert status == 1, text
    assert "restricted load" in text and "DANGEROUS" in text


def test_joblib_inventory_lists_only_real_globals(tmp_path):
    jb = joblib_bytes({"a": np.arange(1000.0), "od": collections.OrderedDict(x=1)}, tmp_path, "c.joblib")
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"c.joblib": jb}))
    assert status == 0, text
    names = [line.split()[-1] for line in text.splitlines()
             if line.split()[:1] in (["innocuous"], ["suspicious"], ["dangerous"])]
    assert all(n.split(".")[0] in {"numpy", "joblib", "collections", "builtins", "copyreg"} for n in names), names


def test_dangerous_object_array_in_joblib(tmp_path):
    arr = np.empty(2, dtype=object)
    arr[0], arr[1] = os.system, "x"
    jb = joblib_bytes({"pad": np.arange(100.0), "obj": arr}, tmp_path, "o.joblib")
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"o.joblib": jb}))
    assert status == 1, text


def test_gzipped_pickle_is_incomplete_not_clean(tmp_path):
    # review N3: picklescan warns and exits 0 on a file it cannot parse
    for name in ("g.pkl", "g.bin.gz"):
        asset = make_zip(tmp_path / f"{name}.zip", {name: gzip.compress(EVIL)})
        status, text = run(tmp_path, asset)
        assert status == 2, text


def test_nested_zip_inside_zip_is_reached(tmp_path):
    # Codex P1: archives created during extraction must be scanned too
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("deep/evil.pkl", EVIL)
    middle = io.BytesIO()
    with zipfile.ZipFile(middle, "w") as zf:
        zf.writestr("inner.zip", inner.getvalue())
    asset = make_zip(tmp_path / "a.zip", {"model.fit": middle.getvalue()})
    status, text = run(tmp_path, asset)
    assert status == 1, text
    assert "deep/evil.pkl" in text


def test_dangerous_legacy_torch_payload_is_not_masked(tmp_path):
    # Codex P1: a dangerous nested torch payload must decide the status even beside joblib files
    legacy = b"".join(pickle.dumps(x, protocol=2) for x in (0x1950A86A20F9469CFC6C, 1001, {"x": 1})) + EVIL
    jb = joblib_bytes({"a": np.arange(100.0)}, tmp_path, "c.joblib")
    asset = make_zip(tmp_path / "a.zip", {"c.joblib": jb, "payload.pt": legacy})
    status, text = run(tmp_path, asset)
    assert status == 1, text


def test_pytorch_zip_members_judged_once(tmp_path):
    pt = io.BytesIO()
    with zipfile.ZipFile(pt, "w") as zf:
        zf.writestr("archive/data.pkl", CLEAN)
        zf.writestr("archive/version", b"3\n")
    asset = make_zip(tmp_path / "a.zip", {"model.ckpt": pt.getvalue()})
    status, text = run(tmp_path, asset)
    assert status == 0, text
    assert text.count("[pytorch-zip]") == 1 and text.count("data.pkl [") == 1


@pytest.mark.parametrize("data", [b"not a zip", b"PK\x03\x04truncated"])
def test_bad_asset_fails_closed(tmp_path, data):
    asset = tmp_path / "bad.zip"
    asset.write_bytes(data)
    status, text = run(tmp_path, asset)
    assert status == 3, text


def test_member_path_escape_fails_closed(tmp_path):
    asset = make_zip(tmp_path / "a.zip", {"../escape.pkl": CLEAN})
    status, text = run(tmp_path, asset)
    assert status == 3, text


def test_report_has_no_absolute_paths(tmp_path):
    asset = make_zip(tmp_path / "a.zip", {"m.pkl": CLEAN})
    _, text = run(tmp_path, asset)
    assert str(tmp_path) not in text and str(tmp_path.as_posix()) not in text


PROTO0_EVIL = b"cos\nsystem\n(t."      # protocol 0: starts with an opcode letter, not 0x80


def test_protocol0_pickle_under_a_text_name_is_found(tmp_path):
    # Claude review SCAN-1: detection by extension or 0x80 missed protocol 0/1 pickles
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"notes.txt": PROTO0_EVIL}))
    assert status == 1, text
    assert "notes.txt [pickle] DANGEROUS" in text


def test_pickle_zip_polyglot_is_scanned_as_both(tmp_path):
    # Claude review SCAN-2: a pickle in front of a zip was only extracted as an archive
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("x.txt", b"x")
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"model.pkl": PROTO0_EVIL + inner.getvalue()}))
    assert status == 1, text
    assert "[pickle+archive] DANGEROUS" in text


def test_every_pytorch_zip_member_is_judged(tmp_path):
    # Claude review SCAN-3: only *.pkl members of a PyTorch zip were checked
    pt = io.BytesIO()
    with zipfile.ZipFile(pt, "w") as zf:
        zf.writestr("archive/data.pkl", CLEAN)
        zf.writestr("archive/payload.xyz", PROTO0_EVIL)
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"model.pt": pt.getvalue()}))
    assert status == 1, text
    assert "payload.xyz [pickle] DANGEROUS" in text


def test_truncated_pickle_member_of_pytorch_zip_is_incomplete(tmp_path):
    pt = io.BytesIO()
    with zipfile.ZipFile(pt, "w") as zf:
        zf.writestr("archive/data.pkl", CLEAN)
        zf.writestr("archive/extra.data", CLEAN[:-3])
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"model.pt": pt.getvalue()}))
    assert status == 2, text


@pytest.mark.parametrize("fmt,data", [
    ("zlib", __import__("zlib").compress(EVIL)),
    ("bzip2", __import__("bz2").compress(EVIL)),
    ("xz", __import__("lzma").compress(EVIL)),
])
def test_opaque_compression_is_incomplete(tmp_path, fmt, data):
    # Claude review SCAN-4: only gzip/bz2/xz were recognised, by name
    status, text = run(tmp_path, make_zip(tmp_path / f"{fmt}.zip", {"model.z": data}))
    assert status == 2, text
    assert f"compressed ({fmt})" in text


def test_duplicate_member_names_fail_closed(tmp_path):
    # Claude review SCAN-5: a loader and the scan could read different copies
    asset = tmp_path / "a.zip"
    with zipfile.ZipFile(asset, "w") as zf:
        zf.writestr("model.pkl", EVIL)
        with pytest.warns(UserWarning):
            zf.writestr("model.pkl", CLEAN)
    status, text = run(tmp_path, asset)
    assert status == 3, text


def test_tar_members_are_scanned(tmp_path):
    import tarfile
    tar = tmp_path / "inner.tar"
    with tarfile.open(tar, "w") as tf:
        info = tarfile.TarInfo("deep/evil.pkl")
        info.size = len(EVIL)
        tf.addfile(info, io.BytesIO(EVIL))
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"bundle.tar": tar.read_bytes()}))
    assert status == 1, text


def test_random_tensor_bytes_are_not_pickle_content(tmp_path):
    rng = np.random.default_rng(0)
    pt = io.BytesIO()
    with zipfile.ZipFile(pt, "w") as zf:
        zf.writestr("archive/data.pkl", CLEAN)
        for i in range(64):
            zf.writestr(f"archive/data/{i}", rng.bytes(4096))
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"model.pt": pt.getvalue()}))
    assert status == 0, text


def test_nested_archive_cannot_overwrite_a_queued_member(tmp_path):
    # Codex (v2): inner.zip used to unpack into the predictable sibling inner.zip.x/
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("evil.pkl", CLEAN)
    asset = make_zip(tmp_path / "a.zip", {"inner.zip": inner.getvalue(), "inner.zip.x/evil.pkl": EVIL})
    status, text = run(tmp_path, asset)
    assert status == 1, text
    assert "inner.zip.x/evil.pkl [pickle] DANGEROUS" in text
    assert "inner.zip!/evil.pkl [pickle] CLEAN" in text


def test_one_invalid_global_does_not_hide_a_dangerous_one(tmp_path):
    # Codex (v2): all(IDENT) let a later non-identifier global suppress os.system
    stream = b"cos\nsystem\n(S'true'\ntRcbad-module\nunused\n."
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"notes.txt": stream}))
    assert status == 1, text


def test_object_npy_inside_npz_is_scanned(tmp_path):
    # Codex (v2): the pickle behind an object-dtype .npy header was skipped
    arr = np.empty(1, dtype=object)
    arr[0] = os.system
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=True)
    npz = io.BytesIO()
    with zipfile.ZipFile(npz, "w") as zf:
        zf.writestr("arr_0.npy", buf.getvalue())
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"data.npz": npz.getvalue()}))
    assert status == 1, text
    assert "[npy-object] DANGEROUS" in text


def test_plain_npy_is_not_pickle(tmp_path):
    buf = io.BytesIO()
    np.save(buf, np.arange(100.0))
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"x.npy": buf.getvalue()}))
    assert status == 0, text


def test_embedded_torch_payload_in_a_complete_pickle_is_scanned(tmp_path):
    # Codex (v2): payload capture only ran when the outer pickle did not read to the end.
    # Raw protocol-3 opcodes: GLOBAL torch.storage._load_from_bytes, BINBYTES payload, TUPLE1, REDUCE, STOP.
    legacy = b"".join(pickle.dumps(x, protocol=2) for x in (0x1950A86A20F9469CFC6C, 1001, {"x": 1})) + EVIL
    outer = (b"\x80\x03ctorch.storage\n_load_from_bytes\n" + b"B" + len(legacy).to_bytes(4, "little")
             + legacy + b"\x85R.")
    assert scan.reads_to_end(outer)
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"state.pkl": outer}))
    assert status == 1, text
    assert "[legacy-torch] DANGEROUS" in text


@pytest.mark.parametrize("wbits", [9, 12, 15])
def test_zlib_any_window_size_is_incomplete(tmp_path, wbits):
    # Codex (v2): only the 32K-window zlib headers were recognised
    co = __import__("zlib").compressobj(9, 8, wbits)
    data = co.compress(EVIL) + co.flush()
    status, text = run(tmp_path, make_zip(tmp_path / f"z{wbits}.zip", {"model.bin.z": data}))
    assert status == 2, text
    assert "compressed (zlib)" in text


# --- Codex round on the frozen-scope heads (scan.py v4) ---------------------------------------------

def _zip_bytes(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _npy(descr_literal: str, payload: bytes) -> bytes:
    header = "{'descr': " + descr_literal + ", 'fortran_order': False, 'shape': (1,), }"
    header = header.ljust(117 - 10) + "\n"
    return b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header.encode("latin-1") + payload


def _legacy_torch(final: bytes) -> bytes:
    return b"".join(pickle.dumps(x, protocol=2) for x in (0x1950A86A20F9469CFC6C, 1001, {"x": 1})) + final


def test_pickle_in_front_of_the_outer_asset_is_judged(tmp_path):
    asset = tmp_path / "a.zip"
    asset.write_bytes(PROTO0_EVIL + _zip_bytes({"info.json": b"{}"}))
    status, text = run(tmp_path, asset)
    assert status == 1, text
    assert "a.zip [pickle+archive] DANGEROUS" in text


def test_torch_payload_inside_an_object_npy_is_captured(tmp_path):
    payload = _legacy_torch(EVIL)
    outer = (b"\x80\x03ctorch.storage\n_load_from_bytes\n" + b"B" + len(payload).to_bytes(4, "little")
             + payload + b"\x85R.")
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"x.npy": _npy("'|O'", outer)}))
    assert status == 1, text
    assert "[legacy-torch] DANGEROUS" in text


@pytest.mark.parametrize("descr", ['"O"', "'|O8'", "'object'", "[('a', 'O')]"])
def test_every_object_dtype_spelling_is_scanned(tmp_path, descr):
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"x.npy": _npy(descr, EVIL)}))
    assert status == 1, text


def test_unreadable_npy_header_is_incomplete(tmp_path):
    data = b"\x93NUMPY\x01\x00" + (20).to_bytes(2, "little") + b"{'descr': <<broken>>"
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"x.npy": data}))
    assert status == 2, text


def test_memo_aliased_global_is_resolved(tmp_path):
    # protocol 4: memoize 'os' and 'system' at 0/1, copy them to 2/3, resolve from the copies
    stream = (b"\x80\x04\x8c\x02os\x94\x8c\x06system\x94"
              b"h\x00q\x02h\x01q\x03" b"0000" b"h\x02h\x03\x93.")
    assert scan.reads_to_end(stream)
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"state.pkl": stream}))
    assert status == 1, text
    assert "os.system" in text


def test_zip_tar_polyglot_is_unpacked_both_ways(tmp_path):
    import tarfile
    tar = tmp_path / "t.tar"
    with tarfile.open(tar, "w") as tf:
        info = tarfile.TarInfo("evil.pkl")
        info.size = len(EVIL)
        tf.addfile(info, io.BytesIO(EVIL))
    dual = tar.read_bytes() + _zip_bytes({"info.json": b"{}"})
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"dual.bin2": dual}))
    assert status == 1, text
    assert "dual.bin2!tar/evil.pkl [pickle] DANGEROUS" in text


def test_protocol2_bytes_payload_is_rebuilt_and_scanned(tmp_path):
    payload = _legacy_torch(EVIL)
    text_u = payload.decode("latin-1").encode("utf-8")
    outer = (b"\x80\x02ctorch.storage\n_load_from_bytes\n"
             b"c_codecs\nencode\n" + b"X" + len(text_u).to_bytes(4, "little") + text_u
             + b"X\x06\x00\x00\x00latin1\x86R" + b"\x85R.")
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"state.pkl": outer}))
    assert status == 1, text
    assert "[legacy-torch] DANGEROUS" in text


def test_picklescan_error_does_not_discard_a_dangerous_global(tmp_path):
    # a standalone STACK_GLOBAL in a second stream makes picklescan raise; the first stream's global stays
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"blob.xyz": EVIL + b"\x93."}))
    assert status == 1, text


def test_persistent_ids_in_torch_pickles_parse_to_the_end(tmp_path):
    # torch's data.pkl references storages with BINPERSID; the restricted load must read past them
    class P(pickle.Pickler):
        def persistent_id(self, obj):
            return ("storage", "FloatStorage", "0", "cpu", 4) if obj == "STORAGE" else None
    buf = io.BytesIO()
    P(buf, protocol=2).dump(collections.OrderedDict(w="STORAGE"))
    pt = io.BytesIO()
    with zipfile.ZipFile(pt, "w") as zf:
        zf.writestr("archive/data.pkl", buf.getvalue())
        zf.writestr("archive/data/0", b"\x00" * 16)
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"model.pt": pt.getvalue()}))
    assert status == 0, text
    assert "data.pkl [pickle] CLEAN" in text


# --- Codex round on scan.py v7: concatenated and framed streams, memo aliases, multi-record .npy ---------
# Raw opcodes, so the fixtures are the same bytes on every platform (os.system pickles as nt.system on
# Windows and posix.system elsewhere); nt.system is dangerous in picklescan's tables on every platform.

P4 = b"\x80\x04"
NT_SYSTEM = b"\x80\x02cnt\nsystem\nX\x0e\x00\x00\x00echo never-run\x85R."   # GLOBAL, arg, TUPLE1, REDUCE
ALIASED = (b"\x8c\x02nt\x94\x8c\x06system\x94"   # memoize 'nt' at 0 and 'system' at 1
           b"N0"                                  # NONE, POP: the strings are not adjacent to STACK_GLOBAL
           b"h\x00h\x01\x93"                      # BINGET 0, BINGET 1, STACK_GLOBAL
           b"\x8c\x0eecho never-run\x85R.")


def _frame(body: bytes) -> bytes:
    return P4 + b"\x95" + len(body).to_bytes(8, "little") + body


@pytest.mark.parametrize("stream", [
    pytest.param(b"\x80\x02N." + NT_SYSTEM, id="clean-then-dangerous-stream"),
    pytest.param(_frame(b"N." + NT_SYSTEM), id="both-streams-in-one-frame"),
    pytest.param(P4 + ALIASED, id="memo-aliased-alone"),
    pytest.param(P4 + b"N." + P4 + ALIASED, id="memo-aliased-after-clean-stream"),
    pytest.param(_frame(b"N." + P4 + ALIASED), id="memo-aliased-in-one-frame"),
])
def test_later_or_aliased_dangerous_stream_is_found(tmp_path, stream):
    # Codex P1s at scan.py:163 and :300 (not reproduced): a dangerous global in a second stream, inside
    # one protocol-4 frame, or resolved through memo aliases
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"model.pkl": stream}))
    assert status == 1, text
    assert "model.pkl [pickle] DANGEROUS" in text and "nt.system" in text


def _npy_records(*arrays) -> bytes:
    buf = io.BytesIO()
    for a in arrays:
        np.save(buf, a, allow_pickle=False)
    return buf.getvalue()


def test_object_record_after_a_numeric_npy_record_is_scanned(tmp_path):
    # Codex P1 at scan.py:227: only the first record's dtype was judged
    data = _npy_records(np.arange(3)) + _npy("'|O'", NT_SYSTEM)
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"arrays.npy": data}))
    assert status == 1, text
    assert "arrays.npy [npy-object] DANGEROUS" in text


def test_object_record_after_a_numeric_record_inside_npz_is_scanned(tmp_path):
    npz = _zip_bytes({"arr_0.npy": _npy_records(np.arange(3.0)) + _npy("'|O'", NT_SYSTEM)})
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"data.npz": npz}))
    assert status == 1, text
    assert "data.npz!/arr_0.npy [npy-object] DANGEROUS" in text


def test_record_after_an_object_record_is_walked(tmp_path):
    clean_obj = b"\x80\x02N."
    data = _npy_records(np.arange(3)) + _npy("'|O'", clean_obj) + _npy("'|O'", NT_SYSTEM)
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"arrays.npy": data}))
    assert status == 1, text


def test_numeric_multi_record_npy_is_not_pickle(tmp_path):
    data = _npy_records(np.arange(3), np.ones((2, 2)), np.zeros(0, dtype=np.int8))
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"arrays.npy": data}))
    assert status == 0, text
    assert "summary: 0 pickle files" in text


@pytest.mark.parametrize("tail", [
    pytest.param(lambda: _npy_records(np.arange(100))[:-8], id="truncated-data"),
    pytest.param(lambda: b"\x93NUMPY\x01\x00\xff\x00{'descr'", id="truncated-header"),
    pytest.param(lambda: b"trailing", id="no-magic"),
])
def test_unparseable_later_npy_record_is_incomplete(tmp_path, tail):
    data = _npy_records(np.arange(3)) + tail()
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"arrays.npy": data}))
    assert status == 2, text
    assert "arrays.npy [npy] INCOMPLETE" in text


# --- Codex round 2 on scan.py v8: qualified loader names, memos shared across consecutive streams -------

def _su(s: str) -> bytes:
    b = s.encode()
    return b"\x8c" + bytes([len(b)]) + b                       # SHORT_BINUNICODE


def _loader_call(module: str, name: str, payload: bytes) -> bytes:
    # protocol 4: STACK_GLOBAL module/name, BINBYTES payload, TUPLE1, REDUCE, STOP
    return P4 + _su(module) + _su(name) + b"\x93" + b"B" + len(payload).to_bytes(4, "little") + payload + b"\x85R."


NT_SYSTEM_P4 = P4 + _su("nt") + _su("system") + b"\x93" + _su("echo never-run") + b"\x85R."


@pytest.mark.parametrize("module,name", [
    pytest.param("torch.storage", "_load_from_bytes", id="exact-spelling"),
    pytest.param("torch", "storage._load_from_bytes", id="dotted-name"),
])
def test_torch_loader_under_any_qualified_split_is_captured(tmp_path, module, name):
    # Codex P1 (round 2): ('torch', 'storage._load_from_bytes') resolves to the same loader
    stream = _loader_call(module, name, NT_SYSTEM_P4)
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"model.pkl": stream}))
    assert status == 1, text
    assert "[legacy-torch] DANGEROUS" in text


def test_attribute_of_the_torch_loader_is_incomplete(tmp_path):
    stream = _loader_call("torch.storage", "_load_from_bytes.__call__", NT_SYSTEM_P4)
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"model.pkl": stream}))
    assert status == 2, text
    assert "unscanned" in text


@pytest.mark.parametrize("member", ["blob.xyz", "model.pkl"])
def test_memo_shared_across_consecutive_streams_is_found(tmp_path, member):
    # Codex P1 (round 2, not reproduced): stream 1 memoizes 'nt'/'system', stream 2 resolves them
    s1 = P4 + _su("nt") + b"\x94" + _su("system") + b"\x94" + b"00" + b"N."
    s2 = P4 + b"h\x00h\x01\x93" + _su("echo never-run") + b"\x85R."
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {member: s1 + s2}))
    assert status == 1, text
    assert "nt.system" in text


# --- Codex round 3 on scan.py v9: shared memos, frames, legacy tails, LZMA-alone, delayed zlib output,
# partial .npy payloads, report paths, unreadable assets. Fixtures are raw opcodes, never unpickled.

OS_SYSTEM = P4 + _su("os") + _su("system") + b"\x93" + _su("echo never-run") + b"\x85R."
COPIED_OS_SYSTEM = (P4 + _su("os") + b"\x94" + _su("system") + b"\x94" + b"h\x00q\x02h\x01q\x03" + b"0000"
                    + b"h\x02h\x03\x93" + _su("echo never-run") + b"\x85R.")   # resolved from memo copies 2/3


def test_memo_copies_shared_across_consecutive_streams_are_resolved(tmp_path):
    # Codex 4369187019: stream 1 copies os/system from memo 0/1 to 2/3; stream 2 resolves 2/3
    s1 = P4 + _su("os") + b"q\x00" + _su("system") + b"q\x01" + b"00" + b"h\x00q\x02h\x01q\x03" + b"00" + b"N."
    s2 = P4 + b"h\x02h\x03\x93" + _su("echo never-run") + b"\x85R."
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"blob.xyz": s1 + s2}))
    assert status == 1, text
    assert "os.system" in text


@pytest.mark.parametrize("data", [
    pytest.param(_frame(b"N." + COPIED_OS_SYSTEM), id="second-pickle-in-frame"),
    pytest.param(_frame(_su("os") + b"\x94" + _su("system") + b"\x94" + b"h\x00q\x02h\x01q\x03" + b"0000" + b"N."
                        + P4 + b"h\x02h\x03\x93" + _su("echo never-run") + b"\x85R."), id="memo-shared-in-frame"),
    pytest.param(P4 + b"N." + COPIED_OS_SYSTEM, id="control-no-frame"),
])
def test_stream_after_a_stop_inside_a_frame_is_read(tmp_path, data):
    # Codex 4369190584: EOF was trusted after the first load consumed the whole frame
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"model.pkl": data}))
    assert status == 1, text
    assert "os.system" in text


def _legacy(obj: bytes, keys: bytes, tail: bytes = b"") -> bytes:
    magic = b"\x80\x02\x8a\x0a" + (0x1950A86A20F9469CFC6C).to_bytes(10, "little") + b"."
    sys_info = b"\x80\x02}X\x0d\x00\x00\x00little_endian\x88s."
    return magic + b"\x80\x02M\xe9\x03." + sys_info + obj + keys + tail


EMPTY_OBJ, EMPTY_KEYS = b"\x80\x02}.", b"\x80\x02]."
# {'w': persistent id ('storage', torch.FloatStorage, '0', 'cpu', 4)} and storage keys ['0']
FLOAT_OBJ = (b"\x80\x02}X\x01\x00\x00\x00w(X\x07\x00\x00\x00storagectorch\nFloatStorage\n"
             b"X\x01\x00\x00\x000X\x03\x00\x00\x00cpuK\x04tQs.")
FLOAT_KEYS = b"\x80\x02]X\x01\x00\x00\x000a."
FLOAT_TAIL = (4).to_bytes(8, "little") + bytes(16)


@pytest.mark.parametrize("data", [
    pytest.param(_legacy(EMPTY_OBJ, EMPTY_KEYS, COPIED_OS_SYSTEM), id="sixth-stream"),
    pytest.param(_legacy(EMPTY_OBJ, b"\x80\x02" + _su("os") + b"q\x00" + _su("system") + b"q\x01"
                         + b"h\x00q\x02h\x01q\x03" + b"0000" + b"].",
                         P4 + b"h\x02h\x03\x93" + _su("echo never-run") + b"\x85R."), id="memo-from-fifth"),
    pytest.param(_legacy(FLOAT_OBJ, FLOAT_KEYS, OS_SYSTEM + FLOAT_TAIL), id="stream-before-storages"),
    pytest.param(_legacy(EMPTY_OBJ, b"\x80\x02]q\x00.", COPIED_OS_SYSTEM), id="earlier-memo-entries"),
])
def test_stream_after_the_five_legacy_pickles_is_read(tmp_path, data):
    # Codex 4369190587: the legacy branch stopped after five streams
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"model.bin": data}))
    assert status == 1, text
    assert "model.bin [legacy-torch] DANGEROUS" in text


def test_legacy_raw_storage_tail_is_clean(tmp_path):
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"model.bin": _legacy(FLOAT_OBJ, FLOAT_KEYS, FLOAT_TAIL)}))
    assert status == 0, text
    assert "model.bin [legacy-torch] CLEAN" in text


@pytest.mark.parametrize("tail", [FLOAT_TAIL[:-1], FLOAT_TAIL + b"x", b"\x10junk"])
def test_legacy_tail_that_is_not_its_storages_is_incomplete(tmp_path, tail):
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"model.bin": _legacy(FLOAT_OBJ, FLOAT_KEYS, tail)}))
    assert status == 2, text


@pytest.mark.parametrize("member", ["state.lzma", "model.lzma"])
def test_lzma_alone_is_incomplete(tmp_path, member):
    # Codex 4369190590 / 4369189319: joblib reads .lzma (FORMAT_ALONE, header 5d 00) transparently
    import lzma
    buf = io.BytesIO()
    with lzma.LZMAFile(buf, "wb", format=lzma.FORMAT_ALONE) as f:
        f.write(OS_SYSTEM)
    assert buf.getvalue()[:2] == b"\x5d\x00"
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {member: buf.getvalue()}))
    assert status == 2, text
    assert "compressed (lzma)" in text


def test_bytes_starting_5d00_that_are_not_lzma_are_not_opaque(tmp_path):
    data = b"\x5d\x00\x00\x80\x00" + b"\xff" * 8 + b"\x07" * 256
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"data/0": data}))
    assert status == 0, text


def test_zlib_with_delayed_output_is_incomplete(tmp_path):
    # Codex 4365649255: 14,000 empty stored blocks push the first output past the trial window
    import zlib
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    body = b"\x00\x00\x00\xff\xff" * 14000 + co.compress(OS_SYSTEM) + co.flush()
    data = b"\x78\x01" + body + zlib.adler32(OS_SYSTEM).to_bytes(4, "big")
    assert zlib.decompress(data) == OS_SYSTEM
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"model.z": data}))
    assert status == 2, text
    assert "compressed (zlib)" in text


@pytest.mark.parametrize("tail", [
    pytest.param(lambda: _npy_records(np.arange(100))[:-8], id="truncated-record"),
    pytest.param(lambda: b"trailing junk", id="trailing-junk"),
])
def test_dangerous_object_record_before_a_malformed_one_wins(tmp_path, tail):
    # Codex 4369144484: the collected payload was dropped when a later record was malformed
    data = _npy("'|O'", NT_SYSTEM) + tail()
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"x.npy": data}))
    assert status == 1, text
    assert "x.npy [npy-object] DANGEROUS" in text and "NumPy header unreadable" in text


def test_clean_object_record_before_a_malformed_one_is_incomplete(tmp_path):
    data = _npy("'|O'", b"\x80\x02N.") + b"trailing junk"
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"x.npy": data}))
    assert status == 2, text
    assert "x.npy [npy-object] INCOMPLETE" in text


@pytest.mark.parametrize("member", ["models/home/model.pkl", "AppData/model.pkl", "Users/x/model.pkl"])
def test_relative_member_paths_like_home_are_reported(tmp_path, member):
    # Codex 4369144486: the report check rejected relative components and crashed without a report
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {member: CLEAN}))
    assert status == 0, text
    assert f"{member} [pickle] CLEAN" in text


def test_a_report_that_would_name_a_host_path_is_refused_with_status_3(tmp_path, monkeypatch):
    import tempfile

    def boom(asset, work):
        raise RuntimeError(f"failed under {work} and {tempfile.gettempdir()}")
    monkeypatch.setattr(scan, "scan_asset", boom)
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"m.pkl": CLEAN}))
    assert status == 3, text
    assert "report would name a host path" in text and tempfile.gettempdir() not in text


def test_host_path_finds_drive_paths_but_not_relative_components(tmp_path):
    assert scan.host_path("x C:\\data\\m.pkl", tmp_path, tmp_path / "a.zip") == "absolute drive path"
    assert scan.host_path("-- models/home/AppData/model.pkl [pickle] CLEAN", tmp_path, tmp_path / "a.zip") is None


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_unreadable_asset_is_status_3_with_a_report(tmp_path, kind):
    # Codex 4369186436: the asset was read outside the protected path
    asset = tmp_path / "asset.zip"
    if kind == "directory":
        asset.mkdir()
    status, text = run(tmp_path, asset)
    assert status == 3, text
    assert "sha256=unreadable" in text and "scan_status=3" in text and str(tmp_path) not in text


def test_dangerous_pickle_followed_by_a_bare_stack_global_is_found(tmp_path):
    # Codex 4365660852 (not reproduced): picklescan raises on the trailing STACK_GLOBAL
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"blob.xyz": OS_SYSTEM + b"\x93."}))
    assert status == 1, text
    assert "os.system" in text


# --- Codex round 4 on scan.py v10: STOP inside a FRAME in a .npy, EXT opcodes, compressed prefix + archive ---

def _framed_object_record(body: bytes) -> bytes:
    return _npy("'|O'", _frame(body))


@pytest.mark.parametrize("second,expected", [
    pytest.param(lambda: _npy_records(np.arange(3)), 0, id="benign-second-record"),
    pytest.param(lambda: _npy("'|O'", NT_SYSTEM), 1, id="dangerous-second-record"),
])
def test_record_after_a_stop_inside_a_frame_is_reached(tmp_path, second, expected):
    # Codex 4369383122: the bytes left in the frame after STOP were read as the next header
    data = _framed_object_record(b"N." + b"PADDING") + second()
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"arrays.npy": data}))
    assert status == expected, text


def test_record_after_a_full_frame_is_reached(tmp_path):
    data = _framed_object_record(b"N.") + _npy_records(np.arange(3))
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"arrays.npy": data}))
    assert status == 0, text


def test_no_record_at_either_resume_point_is_incomplete(tmp_path):
    data = _framed_object_record(b"N." + b"PADDING") + b"trailing junk"
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"arrays.npy": data}))
    assert status == 2, text


@pytest.mark.parametrize("stream", [
    pytest.param(b"\x80\x02\x82*.", id="EXT1"),
    pytest.param(b"\x80\x02\x83" + (42).to_bytes(2, "little") + b".", id="EXT2"),
    pytest.param(b"\x80\x02\x84" + (42).to_bytes(4, "little") + b".", id="EXT4"),
])
def test_extension_registry_opcode_is_incomplete(tmp_path, stream):
    # Codex 4370794576: an unregistered extension code made the restricted load fail and the file vanish
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"blob.xyz": stream}))
    assert status == 2, text
    assert "blob.xyz [pickle] INCOMPLETE" in text and "extension-registry opcode (code 42)" in text


def test_extension_opcode_beside_a_dangerous_global_is_dangerous(tmp_path):
    stream = b"\x80\x02\x82*0" + NT_SYSTEM[2:]
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"blob.xyz": stream}))
    assert status == 1, text


def test_stream_without_extension_opcodes_is_unchanged(tmp_path):
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"blob.xyz": b"\x80\x02N."}))
    assert status == 0, text
    assert "extension-registry" not in text


@pytest.mark.parametrize("fmt,compress", [
    ("zlib", lambda b: __import__("zlib").compress(b)),
    ("gzip", lambda b: gzip.compress(b)),
])
def test_compressed_prefix_in_front_of_an_archive_is_incomplete(tmp_path, fmt, compress):
    # Codex 4370794577: the archive guard skipped compression detection
    data = compress(OS_SYSTEM) + _zip_bytes({"info.json": b"{}"})
    assert zipfile.is_zipfile(io.BytesIO(data))
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"model.dat2": data}))
    assert status == 2, text
    assert f"model.dat2 [compressed ({fmt})] INCOMPLETE" in text and "model.dat2 [archive] EXTRACTED" in text


def test_plain_zip_member_is_not_compressed(tmp_path):
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"inner.zip": _zip_bytes({"m.pkl": CLEAN})}))
    assert status == 0, text
    assert "compressed" not in text


def test_gzipped_tar_is_unpacked_and_scanned_not_opaque(tmp_path):
    import tarfile
    tgz = tmp_path / "t.tar.gz"
    with tarfile.open(tgz, "w:gz") as tf:
        info = tarfile.TarInfo("m.pkl")
        info.size = len(CLEAN)
        tf.addfile(info, io.BytesIO(CLEAN))
    status, text = run(tmp_path, make_zip(tmp_path / "a.zip", {"bundle.tgz": tgz.read_bytes()}))
    assert status == 0, text
    assert "bundle.tgz!/m.pkl [pickle] CLEAN" in text
