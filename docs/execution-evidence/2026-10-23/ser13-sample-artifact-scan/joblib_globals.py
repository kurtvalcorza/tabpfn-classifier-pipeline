"""List every global a joblib file references, by a restricted load (not a static parse).

joblib interleaves raw numpy buffers with pickle opcodes, so pure static parsers
(picklescan, pickletools) stop at the first buffer. This reads the file with joblib's
own unpickler, but find_class returns an inert stub for every global outside ALLOW.
Only the ALLOW entries run: numpy's array and scalar rebuilders, numpy.dtype, and
joblib's NumpyArrayWrapper, which reads the raw array bytes. torch.storage._load_from_bytes
payloads are saved to --dump-nested for a separate static scan, not loaded. Object-dtype
arrays are read with the same restricted unpickler. Pinned: joblib 1.6.0, numpy 2.5.4
(it relies on joblib.numpy_pickle internals). Run with python -I.
"""
import collections
import pickle
import sys

import joblib.numpy_pickle as jnp

ALLOW = {
    ("joblib.numpy_pickle", "NumpyArrayWrapper"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
}
seen = collections.Counter()


class Stub:
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def __call__(self, *args, **kwargs):
        return Stub()


NESTED_DIR = None  # set by --dump-nested DIR: payloads of torch.storage._load_from_bytes are saved, not loaded
nested = []


def _capture_bytes(data, *args, **kwargs):
    if NESTED_DIR is not None and isinstance(data, bytes | bytearray):
        import hashlib
        import pathlib

        digest = hashlib.sha256(data).hexdigest()
        out = pathlib.Path(NESTED_DIR) / f"nested-{digest[:16]}.pt"
        out.write_bytes(bytes(data))
        nested.append((out.name, len(data), digest))
    return Stub()


def make_stub(module, name):
    if (module, name) == ("torch.storage", "_load_from_bytes"):
        return _capture_bytes
    return type(f"Stub[{module}.{name}]", (Stub,), {})


def find_class(module, name):
    seen[(module, name)] += 1
    if (module, name) in ALLOW:
        return pickle.Unpickler.find_class(_probe, module, name)
    return make_stub(module, name)


class Restricted(jnp.NumpyUnpickler):
    def find_class(self, module, name):
        return find_class(module, name)


class RestrictedPlain(pickle.Unpickler):
    def find_class(self, module, name):
        return find_class(module, name)


class _PickleShim:
    """Stands in for the pickle module inside joblib.numpy_pickle (object arrays)."""

    def __getattr__(self, attr):
        return getattr(pickle, attr)

    @staticmethod
    def load(fh, *args, **kwargs):
        return RestrictedPlain(fh).load()


args = sys.argv[1:]
if args[:1] == ["--dump-nested"]:
    NESTED_DIR, args = args[1], args[2:]
_probe = pickle.Unpickler(open(args[0], "rb"))
jnp.pickle = _PickleShim()

incomplete = False
for path in args:
    seen.clear()
    with open(path, "rb") as fh:
        try:
            Restricted(path, fh, ensure_native_byte_order=True, mmap_mode=None).load()
            status = "parsed"
        except Exception as exc:  # report, never raise: this is an audit
            status = f"stopped: {type(exc).__name__}: {exc}"
            incomplete = True
    print(f"== {path}\n   {status}")
    for (module, name), count in sorted(seen.items()):
        tag = "allow" if (module, name) in ALLOW else "stub"
        print(f"   {tag:5} {module}.{name} x{count}")
for name, size, digest in nested:
    print(f"== nested payload {name}: {size} bytes, sha256={digest} (saved, not loaded)")
sys.exit(2 if incomplete else 0)
