from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]


def load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_contract_identity_and_model_limits():
    contract = load("contract/contract-v1.json")
    assert contract["contractVersion"] == 1
    assert contract["taskType"] == "tabular_classification"
    assert contract["datasetFingerprintAlgorithm"] == "sha256-path-content-v1"
    assert contract["modelVersions"]["v2"]["max_features"] == 500
    assert contract["modelVersions"]["v3"]["max_classes"] == 160


def test_all_json_schemas_are_valid_draft_2020_12():
    for path in (ROOT / "contract/schemas").glob("*.schema.json"):
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))


def test_validation_success_and_failure_envelopes():
    schema = load("contract/schemas/validation-result.schema.json")
    validator = Draft202012Validator(schema)
    meta = {"contractVersion": 1, "taskType": "tabular_classification", "datasetFingerprint": "a" * 64, "datasetFingerprintAlgorithm": "sha256-path-content-v1", "resolvedModel": {"version": "v3"}}
    validator.validate({"successful": True, "contractVersion": 1, "checks": [], "metadata": meta})
    validator.validate({"successful": False, "contractVersion": 1, "code": "INVALID_PARAMETER", "metadata": {**meta, "datasetFingerprint": None}})


def test_training_success_requires_portable_artifact_and_reload_check():
    schema = load("contract/schemas/training-result.schema.json")
    validator = Draft202012Validator(schema)
    meta = {"contractVersion": 1, "taskType": "tabular_classification", "datasetFingerprint": "b" * 64, "datasetFingerprintAlgorithm": "sha256-path-content-v1", "resolvedModel": {"version": "v2"}}
    good = {"successful": True, "contractVersion": 1, "metadata": meta, "metrics": {}, "provenance": {}, "artifacts": {"modelArtifact": {"path": "fine-tuning/r/artifacts/model.tabpfn_fit"}, "foundationCheckpoint": {"path": "fine-tuning/r/artifacts/model.ckpt"}, "manifest": {"path": "fine-tuning/r/artifacts/artifact_manifest.json"}, "reloadCheck": {"passed": True}}}
    validator.validate(good)
    bad = json.loads(json.dumps(good))
    del bad["artifacts"]["reloadCheck"]
    assert list(validator.iter_errors(bad))


def test_worker_parameter_schema_rejects_out_of_range_and_unknown():
    schema = load("contract/schemas/worker-parameters.schema.json")
    validator = Draft202012Validator(schema)
    assert not list(validator.iter_errors({"preprocessing": {"validation_split": 0.2}, "hyperparameters": {"epochs": 30}}))
    assert list(validator.iter_errors({"preprocessing": {"validation_split": 0.9}, "hyperparameters": {}}))
    assert list(validator.iter_errors({"preprocessing": {"surprise": 1}, "hyperparameters": {}}))


def test_component_candidate_snapshots_are_canonical():
    subprocess.run([sys.executable, "scripts/check_component_contracts.py"], cwd=ROOT, check=True)
