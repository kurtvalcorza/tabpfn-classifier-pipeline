"""ARTIFACT-INFERENCE companion: shared package keys, the optional pinned sample bundle, and no self-production.

The PAR1–PAR3 and ST1 checks for the companion run in `tests/test_notebook_parity.py`, which is parametrised over
both templates.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEMPLATE_MODULE = _load("notebook_template_artifact_inference")
TEMPLATE = TEMPLATE_MODULE.TEMPLATE
PRIMARY = _load("notebook_template").TEMPLATE


def test_companion_shares_the_primary_package_keys() -> None:
    assert TEMPLATE["profile"] == "ARTIFACT-INFERENCE"
    assert TEMPLATE["notebook_name"] != PRIMARY["notebook_name"]
    for key in ("package", "repo_name", "weights_key", "modules", "entry_module", "lock", "managed_python", "uv", "install_flags", "license_file"):
        assert TEMPLATE.get(key) == PRIMARY.get(key), key


def test_sample_bundle_is_carried_exactly_when_it_exists() -> None:
    """TPCA-M1: the sample cannot be produced offline; when tools/build_sample_bundle.py has added it, it is carried."""
    record_path = ROOT / "examples/sample-bundle/SAMPLE_BUNDLE.json"
    carried = set(TEMPLATE["carried_extra"]) | set(TEMPLATE["carried_binary"])
    if not record_path.is_file():
        assert carried == set() and TEMPLATE_MODULE.SAMPLE is None
        assert "carries no pinned sample bundle yet" in TEMPLATE["run_all"]
        return
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert carried == {"sample-bundle/artifact_manifest.json", "sample-bundle/model.tabpfn_fit", "sample-bundle/new_rows.csv", "sample-bundle/SAMPLE_BUNDLE.json"}
    for name, facts in record["files"].items():
        data = (ROOT / "examples/sample-bundle" / name).read_bytes()
        assert len(data) == facts["bytes"] and hashlib.sha256(data).hexdigest() == facts["sha256"], name


def test_companion_never_self_produces() -> None:
    runner = (TOOLS / "tutorial_stages_artifact_inference.py").read_text(encoding="utf-8")
    nb = json.loads((ROOT / "tutorials" / TEMPLATE["notebook_name"]).read_text(encoding="utf-8"))
    own = "\n".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"] if c["cell_type"] == "code" and not c.get("metadata", {}).get("dimer", {}).get("embedded_sources"))
    for marker in ("build_synthetic_dataset(", "save_artifact(", "zip_artifact_bundle(", ".fit("):
        assert marker not in runner and marker not in own, marker
    calls = {n.func.attr for n in ast.walk(ast.parse(runner)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert {"safe_extract_zip", "validate_artifact_bundle", "from_artifact", "validate_new_rows", "evaluation_report"} <= calls
