import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"
E2E = "tabpfn_classifier_colab.ipynb"
ARTIFACT_INFERENCE = "tabpfn_classifier_artifact_inference_colab.ipynb"

sys.path.insert(0, str(ROOT / "scripts"))
from validate_colab_tutorial import strip_bootstrap  # noqa: E402


def _harness():
    spec = importlib.util.spec_from_file_location("harness", ROOT / "scripts" / "execute_notebook_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest_digest():
    """Lift manifest_digest() out of the E2E notebook and make it callable."""
    cell = next(c for c in _load(E2E)["cells"]
                if c["cell_type"] == "code" and "def manifest_digest" in "".join(c["source"]))
    src = "".join(cell["source"])
    body = src[src.index("def manifest_digest"):src.index("RELOAD = Path")]
    namespace: dict = {}
    exec(compile(body, "manifest_digest", "exec"), namespace)
    return namespace["manifest_digest"]


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


@pytest.mark.parametrize("manifest", [
    {"fittedEstimator": "model.tabpfn_fit"},                      # no digest block at all
    {"fittedEstimator": "model.tabpfn_fit", "sha256": None},      # block present but null
    {"fittedEstimator": "model.tabpfn_fit", "sha256": {}},        # block present but empty
    {"fittedEstimatorSha256": "short"},                           # flat shape, wrong length
])
def test_manifest_digest_refuses_a_missing_digest_by_value_error(manifest):
    """A manifest without a usable digest must hit the guard, not a stray AttributeError."""
    with pytest.raises(ValueError):
        _manifest_digest()(manifest, "fittedEstimator")


def test_manifest_digest_reads_both_worker_shapes():
    digest = "a" * 64
    assert _manifest_digest()({"fittedEstimatorSha256": digest}, "fittedEstimator") == digest
    assert _manifest_digest()({"sha256": {"fittedEstimator": digest}}, "fittedEstimator") == digest


def test_set_override_values_reach_the_kernel_as_literals():
    """A bare word would otherwise be spliced in as a name and raise NameError mid-run."""
    parse = _harness()._parse_overrides
    assert parse(["FINE_TUNE=True"]) == {"FINE_TUNE": "True"}
    assert parse(["EPOCHS=5"]) == {"EPOCHS": "5"}
    assert parse(["MODEL_VERSION=v3"]) == {"MODEL_VERSION": "'v3'"}
    for name, value in parse(["MODEL_VERSION=v3", "FINE_TUNE=True"]).items():
        exec(f"{name} = {value}", {})  # must not raise NameError
    with pytest.raises(SystemExit):
        parse(["NOT_AN_ASSIGNMENT"])


def test_set_override_that_matches_nothing_is_refused(tmp_path):
    """A silently ignored override would record evidence for the value that actually ran."""
    harness = _harness()
    with pytest.raises(RuntimeError, match="matched no"):
        harness._prepare(TUTORIALS / E2E, tmp_path, True, {"NO_SUCH_PARAM": "True"})


def test_skip_bootstrap_applies_a_real_override(tmp_path):
    nb = _harness()._prepare(TUTORIALS / E2E, tmp_path, True, {"FINE_TUNE": "True"})
    source = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "FINE_TUNE = True  # @param" in source
    assert "git clone" not in source and "pip install" not in source


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
