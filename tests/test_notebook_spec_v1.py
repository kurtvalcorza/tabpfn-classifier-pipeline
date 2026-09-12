import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"
E2E = "tabpfn_classifier_colab.ipynb"
ARTIFACT_INFERENCE = "tabpfn_classifier_artifact_inference_colab.ipynb"

sys.path.insert(0, str(ROOT / "scripts"))
from validate_colab_tutorial import strip_bootstrap  # noqa: E402


def _load(name: str):
    return json.loads((TUTORIALS / name).read_text(encoding="utf-8"))


def _source(nb: dict) -> str:
    return "\n".join("".join(c.get("source", "")) if isinstance(c.get("source"), list) else str(c.get("source", "")) for c in nb["cells"])


def test_static_validator_passes():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_colab_tutorial.py")], check=True)


def test_e2e_notebook_declares_and_exercises_release_profile():
    nb = _load("tabpfn_classifier_colab.ipynb")
    source = _source(nb)
    assert nb["metadata"]["dimer"] == {"notebook_profile": "E2E", "notebook_spec": "1.0"}
    assert "worker.run(" in source
    assert "COMPONENTS.json" in source and "!= pinned" in source
    assert "load_dimer_tabpfn_artifact(" in source
    assert "reproduces the recorded validation metric" in source
    assert "fineTuneSkippedReason" in source
    assert "does **not** establish" in source


def test_e2e_private_source_bootstrap_is_ephemeral_and_fail_closed():
    source = _source(_load("tabpfn_classifier_colab.ipynb"))
    assert "GITHUB_TOKEN" in source
    assert "GIT_CONFIG_KEY_0" in source
    assert "http.https://github.com/.extraheader" in source
    assert "x-access-token:{token}@github.com" not in source
    assert "git config --global" not in source
    assert "git config --local" not in source
    assert "Private finetuner access requires" in source


def test_artifact_inference_is_external_and_never_self_produces():
    nb = _load("tabpfn_classifier_artifact_inference_colab.ipynb")
    source = _source(nb)
    assert nb["metadata"]["dimer"] == {"notebook_profile": "ARTIFACT-INFERENCE", "notebook_spec": "1.0"}
    assert "produced **outside this execution**" in source
    assert "files.upload(" in source
    assert "load_dimer_tabpfn_artifact(" in source
    assert "worker.run(" not in source
    assert "build_synthetic_dataset" not in source
    assert ".fit(" not in source


def test_skip_bootstrap_drops_every_clone_and_install():
    """The harness runs against a pre-staged checkout; nothing may re-clone or re-install over it."""
    for name in (E2E, ARTIFACT_INFERENCE):
        cell = next(c for c in _load(name)["cells"] if c["cell_type"] == "code")
        stripped, regions = strip_bootstrap("".join(cell["source"]))
        assert regions >= 1, f"{name}: no colab-bootstrap region"
        for token in ("git clone", "pip install", "shutil.rmtree", "GITHUB_TOKEN"):
            assert token not in stripped, f"{name}: {token!r} survives --skip-bootstrap"
        compile(stripped, name, "exec")


def test_skip_bootstrap_keeps_the_pinned_worker_identity():
    cell = next(c for c in _load(E2E)["cells"] if c["cell_type"] == "code")
    stripped, _ = strip_bootstrap("".join(cell["source"]))
    assert "FT_COMMIT = COMPONENTS" in stripped
    assert "PIPE_DIR = Path(" in stripped and "FT_DIR = Path(" in stripped


def test_e2e_scores_rows_even_without_a_supplied_holdout():
    """BYOD may supply only train.csv, which leaves val_df and test_df None."""
    source = _source(_load(E2E))
    assert "(test_df if test_df is not None else val_df)[FEATURES]" not in source
    assert "source_df, source_label" in source and "source_df[FEATURES]" in source


def test_requirements_colab_pins_everything():
    for line in (TUTORIALS / "requirements-colab.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            assert "==" in line, line


def test_release_notebooks_have_cleared_execution_state():
    for name in ("tabpfn_classifier_colab.ipynb", "tabpfn_classifier_artifact_inference_colab.ipynb"):
        for cell in _load(name)["cells"]:
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []
