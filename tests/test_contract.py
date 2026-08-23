from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contract"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _contract_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted(CONTRACT.rglob("*.json")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\n")
    digest.update(b"dimer-pipeline.json\0")
    digest.update((ROOT / "dimer-pipeline.json").read_bytes())
    return digest.hexdigest()


def test_contract_and_manifest_have_one_authority():
    contract = _load(CONTRACT / "contract.json")
    manifest = _load(ROOT / "dimer-pipeline.json")
    assert contract["contractVersion"] == 1
    assert contract["taskType"] == manifest["taskType"] == "tabular_classification"
    assert contract["manifestAuthority"] == "dimer-pipeline.json"
    assert len(_contract_digest()) == 64


def test_all_json_schemas_are_draft_2020_12_valid():
    for path in sorted((CONTRACT / "schemas").glob("*.schema.json")):
        schema = _load(path)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        Draft202012Validator.check_schema(schema)


def test_parameter_schema_matches_public_manifest():
    manifest = _load(ROOT / "dimer-pipeline.json")
    schema = _load(CONTRACT / "schemas" / "worker-parameters.schema.json")
    assert set(schema["properties"]["preprocessing"]["properties"]) == set(manifest["datasetPreprocessing"])
    assert set(schema["properties"]["hyperparameters"]["properties"]) == set(manifest["modelFinetuning"])
    for section, manifest_key in (("preprocessing", "datasetPreprocessing"), ("hyperparameters", "modelFinetuning")):
        for name, declared in manifest[manifest_key].items():
            actual = schema["properties"][section]["properties"][name]
            for key in ("type", "default", "minimum", "maximum", "minLength", "maxLength", "enum"):
                if key in declared:
                    assert actual[key] == declared[key], (section, name, key)


def test_model_capability_table_matches_v1_contract():
    models = _load(CONTRACT / "model-versions.json")
    assert models["taskType"] == "tabular_classification"
    assert models["versions"]["v2"] == {"maxSamples": 10000, "maxFeatures": 500, "maxClasses": 10}
    assert models["versions"]["v3"] == {"maxSamples": 1000000, "maxFeatures": 2000, "maxClasses": 160}


def test_result_schemas_accept_canonical_success_and_failure():
    validation = Draft202012Validator(_load(CONTRACT / "schemas" / "validation-result.schema.json"))
    training = Draft202012Validator(_load(CONTRACT / "schemas" / "training-result.schema.json"))
    validation.validate({
        "contractVersion": 1,
        "successful": True,
        "message": "ok",
        "metadata": {"taskType": "tabular_classification"},
        "datasetFingerprint": "0" * 64,
    })
    validation.validate({
        "contractVersion": 1,
        "successful": False,
        "message": "bad dataset",
        "metadata": {},
        "error": {"code": "DATASET_INVALID", "phase": "validation", "message": "bad dataset"},
    })
    training.validate({
        "contractVersion": 1,
        "successful": True,
        "message": "ok",
        "metadata": {"taskType": "tabular_classification"},
        "provenance": {},
        "artifacts": {},
    })
    training.validate({
        "contractVersion": 1,
        "successful": False,
        "message": "failed",
        "metadata": {},
        "provenance": {},
        "error": {"code": "TRAINING_FAILED", "phase": "training", "message": "failed"},
    })


def test_error_taxonomy_contains_required_operational_classes():
    codes = set(_load(CONTRACT / "error-codes.json")["codes"])
    assert {
        "CONTRACT_INVALID", "DATASET_INVALID", "MODEL_IDENTITY_MISMATCH",
        "MODEL_INTEGRITY_FAILED", "RESOURCE_UNAVAILABLE", "VALIDATION_FAILED",
        "TRAINING_FAILED", "ARTIFACT_FAILED", "RELOAD_FAILED", "INTERRUPTED",
        "INTERNAL_ERROR",
    } <= codes
