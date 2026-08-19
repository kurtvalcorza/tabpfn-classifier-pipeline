# DIMER TabPFN tabular-classification pipeline contract

Authoritative contract for the TabPFN classification trio (validator + finetuner +
this umbrella). Complements the existing docs: dataset shape lives in
[`TABULAR_CLASSIFICATION_DATASET_SPEC.md`](TABULAR_CLASSIFICATION_DATASET_SPEC.md),
the release procedure in [`DEPLOYMENT.md`](DEPLOYMENT.md), and the last hardware
run in [`PHASE2_ACCEPTANCE_REPORT.md`](PHASE2_ACCEPTANCE_REPORT.md).

## 1. Effective task type and the portal fallback

The effective task type is **`tabular_classification`**. DIMER does not yet expose
a native tabular task, so the portal persists the pipeline as `Custom / Other`.
The runtime does **not** rely on that portal value:

- Both container images set `DIMER_TASK_TYPE=tabular_classification`.
- Result payloads report `metadata.taskType`, defaulting to
  `tabular_classification` when `DIMER_PIPELINE_METADATA_JSON.taskType` is absent.

When DIMER adds a native tabular task type, switch the portal type and drop the
fallback; no runtime code change is required.

## 2. `dimer-pipeline.json` is the authoritative parameter API

Every key declared in the finetuner's `dimer-pipeline.json` maps to concrete
runtime behavior in `train.py`. No silently-ignored controls, and **`model_id` is
not a hyperparameter** — the base model is selected by DIMER's Base Model, surfaced
as `model_version` plus an optional pinned checkpoint (§3).

| Manifest key | Runtime effect |
|---|---|
| `target_column`, `drop_columns` | column selection |
| `validation_split` | stratified holdout fraction when `val.csv` is absent |
| `max_train_rows` | deterministic stratified row cap (further capped to the version limit, §4) |
| `fine_tune` | `FinetunedTabPFNClassifier` (CUDA) vs zero-shot ICL |
| `model_version` | TabPFN generation → per-version limits + checkpoint selection |
| `epochs`, `learning_rate`, `weight_decay`, `time_limit_seconds`, `early_stopping_patience` | AdamW fine-tuning schedule |
| `eval_metric` | early-stopping / validation objective |
| `n_finetune_ctx_plus_query_samples`, `n_estimators_finetune/validation/final_inference` | ensembling + context sizing |
| `prediction_batch_rows` | eval chunk size (bounds VRAM/RAM) |
| `seed` | sampling / holdout / fine-tuning RNG |

## 3. Base-model handoff and provenance

DIMER's selected Base Model reaches the finetuner via **generation**
(`model_version`, `v3` default) and an optional **pinned checkpoint**
(`DIMER_TABPFN_MODEL_PATH`, required for gated `v3`/`v2.6` weights;
`v2`/`v2.5` auto-download under Apache). Provenance is verifiable, not inferred
from cache ordering: `provenance.model` records `tabpfnVersion`, `torchVersion`,
`modelVersionRequested`, `modelPathRequested`, `baseModelSha256` (SHA-256 of the
pinned checkpoint when set) + `baseModelPathHashed`.

## 4. TabPFN-specific limits (not copied from Mitra/TabICL)

Enforced by both validator and finetuner from an identical table, verified against
tabpfn 8.1.0 (v2/v2.5 source `InferenceConfig`, v3 checkpoint `inference_config`;
v2.6 from the model card):

| version | max_samples | max_features | max_classes |
|---|---|---|---|
| v2 | 10,000 | 500 | 10 |
| v2.5 | 50,000 | 2,000 | 10 |
| v2.6 | 100,000 | 2,000 | 10 |
| v3 | 1,000,000 | 2,000 | 160 |

Features and **class count** are hard rejects; rows are deterministically
(stratified) subsampled to the version cap. The validator reads the selected
version via `DIMER_HYPERPARAMETERS_JSON` passthrough; unknown/`default` → strictest
(v2). VRAM is a hard CUDA requirement for `fine_tune=true` — see `DEPLOYMENT.md`
for the production profile.

## 5. Classification split semantics

- **At least two classes**; the smallest class must support a safe stratified
  holdout.
- When `val.csv` is absent, the finetuner uses a **stratified** auto-validation
  split and fails clearly when class counts cannot support the requested split.
- An explicit `val.csv` must contain **every training class** — the finetuner
  computes ROC AUC / log loss on that exact set, and a missing class makes those
  metrics NaN and early stopping unreliable (validator check
  `val_covers_train_classes`).
- `test.csv`, if present, is scored only after training and reported separately.

## 6. Common result / provenance schema

```jsonc
{
  "successful": true,
  "metrics": { "mode": "fine-tune|zero-shot-icl", "deviceName": "...",
               "validation": { "accuracy": .., "balancedAccuracy": .., "logLoss": .., "rocAuc*": .. },
               "test": { .. | null } },
  "artifacts": { "fittedEstimator": "model.tabpfn_fit", "foundationCheckpoint": "model.ckpt",
                 "manifest": "artifact_manifest.json", "fittedEstimatorSha256": "..",
                 "foundationCheckpointSha256": "..",
                 "reloadCheck": { "passed": true, "device": "cpu", "rows": N } },
  "provenance": { "model": {..}, "dataset": {"sha256": ".."}, "seed": N },
  "metadata": { "template": "..", "taskType": "tabular_classification" }
}
```

`artifacts.reloadCheck` is written only after the finetuner reloads the saved
artifact in-process and runs `predict_proba` — success is never reported for an
artifact that cannot be reloaded.

## 7. Classification smoke matrix

Cover: binary and multiclass; class imbalance; nullable targets; mixed
numeric/categorical features; explicit `val.csv`; stratified auto-validation;
optional held-out `test.csv`; duplicate split-name rejection; and at least one
dataset near the selected version's limits. The integration CI job exercises the
real-TabPFN fit → save → relocate → load → `predict_proba` path on every PR.

## 8. Release gate

Per `DEPLOYMENT.md`: production enablement requires a successful DIMER smoke
validation + training run, exact model-generation/checkpoint provenance, a verified
saved-artifact reload/inference (enforced in-runtime via `reloadCheck` and in CI
via the integration tier), and proof that manifest controls affect runtime
behavior. Gated weights additionally require a commercial license for any
non-evaluation use.

## 9. Cross-repo synchronization

`COMPONENTS.json` pins immutable 40-char validator/finetuner commit SHAs; the
pipeline CI manifest check enforces the format and `kurtvalcorza/` ownership. The
`MODEL_VERSION_LIMITS` table and the artifact/result schema are duplicated verbatim
across validator and finetuner (DIMER isolation forbids importing shared source);
any change must update both copies in the same change set and re-pin here.
