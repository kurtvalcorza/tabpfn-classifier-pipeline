"""List every global a joblib file references without executing any of them.

joblib interleaves raw numpy buffers with pickle opcodes, so pure static
parsers (picklescan, pickletools) stop at the first buffer. This walks the
file with joblib's own unpickler, but find_class returns an inert stub for
every global outside a small numpy/joblib allowlist, and object-dtype arrays
are read with the same restricted unpickler. Run with python -I.
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


def make_stub(module, name):
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


_probe = pickle.Unpickler(open(sys.argv[1], "rb"))
jnp.pickle = _PickleShim()

for path in sys.argv[1:]:
    seen.clear()
    with open(path, "rb") as fh:
        try:
            Restricted(path, fh, ensure_native_byte_order=True, mmap_mode=None).load()
            status = "parsed"
        except Exception as exc:  # report, never raise: this is an audit
            status = f"stopped: {type(exc).__name__}: {exc}"
    print(f"== {path}\n   {status}")
    for (module, name), count in sorted(seen.items()):
        tag = "allow" if (module, name) in ALLOW else "stub"
        print(f"   {tag:5} {module}.{name} x{count}")
