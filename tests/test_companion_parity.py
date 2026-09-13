"""NOTEBOOK_SPEC 1.1 parity tests (PAR1–PAR3, ST1) for the ARTIFACT-INFERENCE companion notebook.

`tests/test_notebook_parity.py` is the fleet's copy and covers the primary template only; this file runs the
same checks against `tools/notebook_template_artifact_inference.py` and the notebook it generates.
"""

# ruff: noqa: E501  -- long docstrings, messages and single-line test fixtures are kept readable
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook")
TEMPLATE = _load("notebook_template_artifact_inference").TEMPLATE
PRIMARY = _load("notebook_template").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]
MANIFEST = ROOT / "weights" / TEMPLATE["weights_key"] / "dimer-base-manifest.json"


@pytest.fixture(scope="module")
def notebook() -> dict:
    if not NOTEBOOK.exists():
        pytest.skip(f"{NOTEBOOK.name} not generated yet")
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cells(notebook: dict, cell_type: str) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == cell_type]


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def test_companion_shares_the_primary_package_keys() -> None:
    assert TEMPLATE["profile"] == "ARTIFACT-INFERENCE"
    assert TEMPLATE["notebook_name"] != PRIMARY["notebook_name"]
    for key in ("package", "repo_name", "pipeline_class", "weights_key", "modules", "entry_module"):
        assert TEMPLATE.get(key) == PRIMARY.get(key), key


def test_par1_embedded_modules_equal_repository_modules(notebook: dict) -> None:
    tagged = [
        c for c in _cells(notebook, "code") if c.get("metadata", {}).get("dimer", {}).get("embedded_module")
    ]
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    ctx = build.load_context(ROOT, TEMPLATE, recorded)
    assert [c["metadata"]["dimer"]["embedded_module"] for c in tagged] == ctx["module_rels"]
    for cell, module in zip(tagged, ctx["modules"], strict=True):
        rel = f"{ctx['pkg_rel']}/{module}"
        assert cell["metadata"]["dimer"]["module_sha256"] == ctx["per_module_sha256"][rel]
        assert _source(cell).rstrip("\n") + "\n" == ctx["embedded"][module], rel


def test_par2_inline_manifest_and_pins_match_repository(notebook: dict) -> None:
    code = "\n".join(_source(c) for c in _cells(notebook, "code"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    inline = re.search(r"^MANIFEST = (\{.*?^\})$", code, re.MULTILINE | re.DOTALL)
    assert inline and json.loads(inline.group(1)) == manifest
    pins_block = re.search(r"^PINS = \[(.*?)^\]", code, re.MULTILINE | re.DOTALL)
    assert pins_block and re.findall(r"'([^']+)'", pins_block.group(1)) == build._pins(ROOT, TEMPLATE)
    meta = notebook["metadata"]["dimer"]
    assert meta["standalone"] is True
    assert meta["notebook_profile"] == "ARTIFACT-INFERENCE"
    assert meta["generated_from"]["module_sha256"] == build.load_context(ROOT, TEMPLATE)["module_sha256"]


def test_par3_generator_check_is_clean(notebook: dict) -> None:
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    rendered = build.to_bytes(build.render(ROOT, TEMPLATE, recorded))
    current = NOTEBOOK.read_bytes().replace(b"\r\n", b"\n")
    assert current == rendered, "companion notebook is stale; regenerate it with --template"


def test_st1_primary_path_has_no_repository_dependency(notebook: dict) -> None:
    code = "\n".join(_source(c) for c in _cells(notebook, "code"))
    assert "git" not in re.findall(r"subprocess\.run\(\[([^\]]*)\]", code).__str__()
    assert f"import {TEMPLATE['package']}" not in code
    assert f"from {TEMPLATE['package']}" not in code
    assert "github.com/kurtvalcorza" not in code
    assert "worker.run(" not in code and "worker_cli(" not in code
