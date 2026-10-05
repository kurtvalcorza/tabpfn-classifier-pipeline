"""NOTEBOOK_SPEC 2.2 parity tests (PAR1–PAR4, ST1) for both standalone tutorial notebooks (generator /3).

Each notebook carries the repository's package, its stage runner, the hash lock and the snapshot manifest as text in
one carrier cell (``metadata.dimer.embedded_sources``); these tests fail whenever a carried file, the lock or the
notebook bytes diverge from the repository at HEAD.
"""
# ruff: noqa: E501  -- assertion messages and paths are kept on one line

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
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
TEMPLATES = {name: _load(name).TEMPLATE for name in ("notebook_template", "notebook_template_artifact_inference")}


def _notebook(template: dict) -> dict:
    path = ROOT / "tutorials" / template["notebook_name"]
    if not path.exists():
        pytest.skip(f"{path.name} not generated yet")
    return json.loads(path.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _carrier(notebook: dict) -> dict[str, dict]:
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code" and c.get("metadata", {}).get("dimer", {}).get("embedded_sources")]
    assert len(cells) == 1, "exactly one carrier cell"
    values = {}
    for node in ast.parse(_source(cells[0])).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in ("CARRIED_FILES", "CARRIED_BINARY", "CARRIED_HASHES"):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    return values


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_par1_carried_files_equal_repository_files(name: str) -> None:
    template = TEMPLATES[name]
    notebook = _notebook(template)
    carried = _carrier(notebook)
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    ctx = build.load_context(ROOT, template, recorded)
    assert carried["CARRIED_FILES"] == ctx["files"], "a carried file drifted from the repository; regenerate the notebook"
    assert carried["CARRIED_BINARY"] == ctx["binary"]
    for dest, source in build.carried_sources(ROOT, template).items():
        assert carried["CARRIED_FILES"][dest] == (ROOT / source).read_text(encoding="utf-8").replace("\r\n", "\n"), dest
    for dest, text in carried["CARRIED_FILES"].items():
        assert carried["CARRIED_HASHES"][dest] == hashlib.sha256(text.encode("utf-8")).hexdigest(), dest
    for dest, data in carried["CARRIED_BINARY"].items():
        assert base64.b64decode(data) == (ROOT / template["carried_binary"][dest]).read_bytes(), dest
        assert carried["CARRIED_HASHES"][dest] == hashlib.sha256(base64.b64decode(data)).hexdigest(), dest


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_par2_lock_pins_every_runtime_pin_with_hashes(name: str) -> None:
    template = TEMPLATES[name]
    lock = (ROOT / template["lock"]).read_text(encoding="utf-8")
    build.check_lock(build._pins(ROOT), lock)  # raises SystemExit on any drift or unhashed entry
    meta = _notebook(template)["metadata"]["dimer"]
    assert meta["standalone"] is True
    assert meta["notebook_spec"] == build.NOTEBOOK_SPEC
    assert meta["generated_from"]["module"] == f"src/{template['package']}/{template.get('entry_module', 'pipeline.py')}"
    assert meta["generated_from"]["module_sha256"] == build.load_context(ROOT, template)["module_sha256"]


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_par3_generator_check_is_clean(name: str) -> None:
    template = TEMPLATES[name]
    notebook = _notebook(template)
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    rendered = build.to_bytes(build.render(ROOT, template, recorded))
    current = (ROOT / "tutorials" / template["notebook_name"]).read_bytes().replace(b"\r\n", b"\n")
    assert current == rendered, "notebook is stale; run python tools/build_notebook.py (with --template for the companion)"


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_st1_primary_path_has_no_repository_dependency(name: str) -> None:
    notebook = _notebook(TEMPLATES[name])
    own = "\n".join(_source(c) for c in notebook["cells"] if c["cell_type"] == "code" and not c.get("metadata", {}).get("dimer", {}).get("embedded_sources"))
    assert f"import {TEMPLATES[name]['package']}" not in own
    assert f"from {TEMPLATES[name]['package']}" not in own
    assert "github.com/kurtvalcorza" not in own
    assert "git clone" not in own
    assert "worker.run(" not in own and "worker_cli(" not in own
