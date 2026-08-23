# TabPFN classifier contract v1

This directory is the machine-readable authority for the DIMER-facing classifier worker contract introduced by `NAIRA-SEU/dimer-backend#4`.

## Authority and migration

`contract-v1.json` owns task identity, parameter constraints, TabPFN generation limits, dataset normalization/fingerprinting, and result/artifact requirements. Worker images vendor the same semantic contract and enforce it before invoking their existing task implementation. Missing `DIMER_CONTRACT_VERSION` is treated as v1 during the current DIMER migration; an explicit unsupported version fails closed.

`DIMER_MODEL_CONFIG_JSON` is authoritative when it identifies a concrete TabPFN generation. Legacy `model_version` remains a compatibility input; a concrete disagreement is a contract error.

## Dataset identity

`sha256-path-content-v1` hashes the logical dataset after path normalization and accepted single-nested-ZIP unwrapping. Files are ordered by normalized path and each entry contributes its normalized path and SHA-256 content digest. Equivalent directory, ZIP, and accepted nested-ZIP representations therefore have the same fingerprint.

A validator pass is intended to bind to the same fingerprint at training time. DIMER must persist/compare that fingerprint; backend implementation is outside this repository-side change.

## Component verification

`component-candidates.json` records immutable worker commits under review without changing release `COMPONENTS.json`. Vendored snapshots under `components/` are checked semantically against the canonical contract. When `DIMER_COMPONENT_TOKEN` is available, `scripts/check_component_contracts.py` also fetches `contract-v1.json` from each exact private commit and verifies both its recorded Git blob SHA and semantic equality. Release pins advance only after component CI/review and conformance pass.

## DIMER requirements (documentation only)

DIMER must eventually pass validator and finetuner the same preprocessing/hyperparameter/model context, support native `tabular_classification`, resolve model identity once, bind validation to the dataset fingerprint used for training, and enforce registration/activation compatibility. No DIMER backend implementation is part of this repo-side change.
