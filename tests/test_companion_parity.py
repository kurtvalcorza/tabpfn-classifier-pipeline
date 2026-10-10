"""ARTIFACT-INFERENCE companion: shared package keys, the optional pinned sample bundle, and no self-production.

The PAR1–PAR3 and ST1 checks for the companion run in `tests/test_notebook_parity.py`, which is parametrised over
both templates.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
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


def test_sample_bundle_is_a_pinned_release_asset_not_carried() -> None:
    """TPCA-M1 / SART6: the default sample is the release asset pinned in examples/sample_bundle_pin.json (URL, size,
    SHA-256, producer); nothing of it is carried in the notebook, and the pin is a real one, not a placeholder."""
    pin = json.loads((ROOT / "examples/sample_bundle_pin.json").read_text(encoding="utf-8"))
    assert pin == TEMPLATE_MODULE.SAMPLE_ARTIFACT
    assert not any("sample-bundle" in k for k in (*TEMPLATE["carried_extra"], *TEMPLATE["carried_binary"]))
    assert not (ROOT / "examples/sample-bundle").exists()
    assert pin["url"] == f"https://github.com/kurtvalcorza/tabpfn-classifier-pipeline/releases/download/{pin['tag']}/tabpfn_classifier_sample_bundle.zip"
    assert len(pin["sha256"]) == 64 and set(pin["sha256"]) <= set("0123456789abcdef") and pin["sha256"] != "0" * 64
    assert pin["bytes"] > 1000
    assert {"notebook", "notebook_blob", "commit", "runtime", "run_record"} <= set(pin["producer"])
    nb = (ROOT / "tutorials" / TEMPLATE["notebook_name"]).read_text(encoding="utf-8")
    assert pin["sha256"] in nb and str(pin["bytes"]) in nb


def test_companion_never_self_produces() -> None:
    runner = (TOOLS / "tutorial_stages_artifact_inference.py").read_text(encoding="utf-8")
    nb = json.loads((ROOT / "tutorials" / TEMPLATE["notebook_name"]).read_text(encoding="utf-8"))
    own = "\n".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"] if c["cell_type"] == "code" and not c.get("metadata", {}).get("dimer", {}).get("embedded_sources"))
    for marker in ("build_synthetic_dataset(", "save_artifact(", "zip_artifact_bundle(", ".fit("):
        assert marker not in runner and marker not in own, marker
    calls = {n.func.attr for n in ast.walk(ast.parse(runner)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert {"safe_extract_zip", "validate_artifact_bundle", "from_artifact", "validate_new_rows", "evaluation_report"} <= calls
