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
    names = [line.split()[-1] for line in text.splitlines() if line.startswith("   ") and "method" not in line]
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
