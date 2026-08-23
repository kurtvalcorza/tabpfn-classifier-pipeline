# DIMER Contract v1 authority

This directory is the canonical TabPFN classifier worker-contract authority. The root `dimer-pipeline.json` is the canonical DIMER parameter manifest; validator and finetuner copies are deployable vendored snapshots and must remain conformant.

`component-candidates.json` is a **review/conformance manifest**, not a release manifest. It pins immutable candidate commits and expected Git blobs for the validator and finetuner currently under review. `scripts/check_component_contracts.py` verifies their expected manifest/contract semantics locally and, when `DIMER_COMPONENT_TOKEN` is available, verifies every recorded file against the exact private GitHub commit. Mutable branch heads are never used as contract authority.

`COMPONENTS.json` remains the released component-pin manifest and must not be advanced merely to test candidates. Advance it only after the candidate component PRs pass their required review and CI gates.

## DIMER integration requirements

These are platform requirements only; they are **not implemented in this repository or the worker repositories**.

**DIMER integration requirement (not implemented in this repository):** DIMER should pass the same resolved preprocessing, hyperparameter, model, pipeline, contract-version, and immutable run context to validation and finetuning.

**DIMER integration requirement (not implemented in this repository):** DIMER should natively preserve `tabular_classification` task identity without falling through object-detection/YOLO defaults.

**DIMER integration requirement (not implemented in this repository):** DIMER should persist the validator's logical dataset fingerprint and bind finetuning to that same fingerprint, invalidating validation when the logical dataset changes.

**DIMER integration requirement (not implemented in this repository):** DIMER should provide one immutable resolved model specification, including expected checkpoint/model digest when known, to both validator and finetuner.

**DIMER integration requirement (not implemented in this repository):** DIMER should treat `result.json` as terminal authority, callbacks as completion notifications, and user/system cancellation separately from model/training failure where orchestration state permits.

**DIMER integration requirement (not implemented in this repository):** DIMER export/promotion/serving should preserve and validate the complete TabPFN portable model bundle rather than treating the fitted estimator alone as self-contained.

**DIMER integration requirement (not implemented in this repository):** Workbench activation should eventually validate registered pipeline parameters/result protocol against the versioned worker contract before activation.
