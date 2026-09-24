"""The standalone tutorial pair (NOTEBOOK_SPEC 2.0 §4): validator passes, profiles, pins, no worker path."""
# ruff: noqa: E501
from __future__ import annotations

import importlib.util
import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"
TASK = "tabpfn_classifier_colab.ipynb"
ARTIFACT_INFERENCE = "tabpfn_classifier_artifact_inference_colab.ipynb"


def _load(name: str) -> dict:
    return json.loads((TUTORIALS / name).read_text(encoding="utf-8"))


def _code(nb: dict) -> str:
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def _validator():
    spec = importlib.util.spec_from_file_location("validate_release_assets", ROOT / "tools" / "validate_release_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_validator_passes() -> None:
    assert _validator().validate_all() == ["model-card", "identity-consistency", "weight-facts", "release-status", "notebooks+parity"]


def test_profiles_and_standalone_metadata() -> None:
    for name, profile in ((TASK, "TASK-INFERENCE"), (ARTIFACT_INFERENCE, "ARTIFACT-INFERENCE")):
        dimer = _load(name)["metadata"]["dimer"]
        assert dimer["notebook_profile"] == profile and dimer["notebook_spec"] == "2.0" and dimer["standalone"] is True
        assert dimer["generated_from"]["repository"] == "tabpfn-classifier-pipeline" and dimer["generated_from"]["module"] == "src/tabpfn_classifier_pipeline/pipeline.py"


def test_worker_and_token_paths_are_gone() -> None:
    for name in (TASK, ARTIFACT_INFERENCE):
        code = _code(_load(name))
        for marker in ("worker.run(", "GITHUB_TOKEN", "git clone", "colab-bootstrap", "COMPONENTS.json", "TUTORIAL_REF", "import train as worker", "FINE_TUNE"):
            assert marker not in code, (name, marker)
        assert re.search(r"github\.com/kurtvalcorza", code) is None, name


def test_inline_pins_equal_pyproject_runtime_dependencies() -> None:
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    assert all("==" in dep for dep in deps) and "tabpfn==8.1.0" in deps and "torch==2.11.0" in deps
    for name in (TASK, ARTIFACT_INFERENCE):
        block = re.search(r"^PINS = \[(.*?)^\]", _code(_load(name)), re.M | re.S)
        assert block is not None and re.findall(r"'([^']+)'", block.group(1)) == deps, name


def test_release_notebooks_have_cleared_execution_state() -> None:
    for name in (TASK, ARTIFACT_INFERENCE):
        for index, cell in enumerate(_load(name)["cells"]):
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None and not cell.get("outputs"), (name, index)
