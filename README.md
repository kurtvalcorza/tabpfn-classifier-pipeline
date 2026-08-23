# TabPFN Classifier — DIMER Pipeline

A DIMER pipeline that adapts [TabPFN](https://github.com/PriorLabs/TabPFN), Prior Labs'
tabular foundation model, to your own tabular-classification dataset. You supply a table of
rows with one categorical target column. The pipeline validates the table, fine-tunes TabPFN
on a GPU (or runs it zero-shot as an in-context classifier when no GPU is present), and
produces a saved model artifact plus a holdout score.

Unlike some tabular foundation models, TabPFN's newer generations do **not** ship under an
unrestricted open-weights licence: the code and the older **v2** weights carry Prior Labs'
Apache-derived licence (commercial use permitted), while the **v2.5 / v2.6 / v3** weights are
**non-commercial** and **v3 is Hugging Face-gated**. Because this pipeline defaults to v3, the
model weights themselves, not only the training data, gate what a served pipeline may do — see
[Data licence governs the served model](#data-licence-governs-the-served-model).

For platform-administrator setup and operations — resource profiles, weights delivery, network
egress, and the release gate — see [DEPLOYMENT.md](DEPLOYMENT.md). Model provenance, adaptation
modes, and licensing detail are in [MODEL_CARD.md](MODEL_CARD.md); the authoritative
result/artifact envelope is in [CONTRACT.md](CONTRACT.md).

---

## The model: TabPFN

TabPFN (Tabular Prior-data Fitted Network) is a tabular foundation model maintained by
[Prior Labs](https://priorlabs.ai). Like Mitra and TabICL, it is an **in-context learner**: it
reads a table of labelled examples as context and predicts on new rows, with no gradient update
required for a base prediction. This pipeline drives TabPFN through `tabpfn==8.1.0` and defaults
to the **v3 (TabPFN-3)** generation. The classifier uses `TabPFNClassifier` for zero-shot
in-context inference and `FinetunedTabPFNClassifier` for task-specific fine-tuning. See
[MODEL_CARD.md](MODEL_CARD.md) for provenance, checksums, licence, and how to supply the weights
to DIMER.

The generation is chosen explicitly in DIMER configuration (`model_version`) rather than inferred
from whichever checkpoint happens to be cached — the runtime records the requested generation and,
where a concrete checkpoint file is mounted, its SHA-256.

### Generations and capacity limits

TabPFN's generations differ in the table sizes they were trained for. This pipeline exposes four
via the `model_version` field, each with hard capacity caps enforced by both the validator and the
fine-tuner from an identical table (verified against `tabpfn==8.1.0`: v2 and v2.5 from the source
`InferenceConfig`, v3 from the checkpoint-stored `inference_config`, v2.6 from the published model
card):

| `model_version` | max samples | max features | max classes | Weight licence |
|---|---:|---:|---:|---|
| `v2` | 10,000 | 500 | 10 | Apache-derived (commercial OK) |
| `v2.5` | 50,000 | 2,000 | 10 | non-commercial |
| `v2.6` | 100,000 | 2,000 | 10 | non-commercial |
| `v3` (default) | 1,000,000 | 2,000 | 160 | non-commercial, HF-gated |

`default` follows the installed package's own default generation and applies no version-specific
tightening beyond the generic platform limits. Feature count and **class count** are hard rejects;
rows are deterministically, stratified-sampled down to the version cap by the fine-tuner. The
validator resolves the selected version from `DIMER_HYPERPARAMETERS_JSON` (falling back to
`DIMER_PIPELINE_METADATA_JSON`); an unknown or `default` value is treated conservatively as the
strictest (`v2`) tier for validation. `max_classes` applies to classification only.

### Applicability

TabPFN is an in-context learner, so its accuracy depends on whether the features carry signal
about the class. Treat published benchmark strength as evidence of strong performance where signal
exists, not as a guarantee on any table. The dataset validator does **not** impose an arbitrary
class ceiling of its own; the selected TabPFN generation remains the final authority on class and
feature support (see the table above). High-cardinality categoricals, unusual missingness,
distribution shift, leakage, and non-independent rows can all invalidate an apparently strong
holdout score.

### Fine-tuning and zero-shot

TabPFN runs in two modes, both exposed through fine-tuning fields:

- **Fine-tune** (`fine_tune=true`, the default) adapts the pretrained weights to the uploaded
  table through `FinetunedTabPFNClassifier`, performing gradient updates under an AdamW schedule.
  It **requires a CUDA GPU**.
- **Zero-shot / ICL** (`fine_tune=false`) runs `TabPFNClassifier` with no weight update. `.fit(X, y)`
  establishes the labelled in-context task but performs no task-specific gradient step; results are
  labelled `zero-shot-icl` in `result.json`.

**CPU is the default deployment; GPU is opt-in and off by default.** This matters for how a
fine-tune request behaves without a GPU: rather than failing, the run **auto-falls back to
zero-shot ICL**, records `metrics.fineTuneSkippedReason` with `metrics.fineTuneEffective=false`,
and completes normally. `DIMER_TRAIN_DEVICE` is honoured, and a bare device index such as `"0"` is
normalized to `cuda:0`. The effective device and mode are recorded in `result.json`
(`metrics.mode`, `metrics.selectedDevice`, `metadata.device`).

Because TabPFN remains an in-context model after fine-tuning, the served artifact carries both the
fitted estimator and its foundation checkpoint — see [Outputs](#outputs).

### Binary and multiclass

The fine-tuner infers the problem type from the target's distinct-value count: two classes give a
binary problem, more give multiclass, up to the selected generation's class cap. Both use the same
classifier. Binary problems report `rocAuc`; multiclass problems report one-vs-rest macro ROC AUC
(`rocAucOvrMacro`) where computable. The class count is recorded in
`result.json` (`metrics.numClasses`).

---

## When to use this pipeline

Use this pipeline for tabular classification: predicting a categorical label from a row of
features. Risk tiers, demand bands, churn/no-churn, quality grades, and any row-per-record
categorical prediction fit here. Do not use it for images — for vision tasks use the Image
Classification, Object Detection, or Segmentation pipelines. For a numeric target, use the
[TabPFN regressor pipeline](https://github.com/kurtvalcorza/tabpfn-regressor-pipeline).

---

## Repositories

The pipeline is two deployable containers, one repository each, plus this **umbrella** repository
for the contract, specs, serving loader, and dataset helpers. This umbrella deliberately does
**not** copy `validate.py` or `train.py`; `COMPONENTS.json` pins the reviewed component commits so
the deployable source has a single home and cannot drift between an umbrella copy and its container
repository.

| Component | Repository | Runs on |
|---|---|---|
| Validator | [`tabpfn-classifier-dataset-validator`](https://github.com/kurtvalcorza/tabpfn-classifier-dataset-validator) | CPU |
| Fine-tuner | [`tabpfn-classifier-finetuner`](https://github.com/kurtvalcorza/tabpfn-classifier-finetuner) | CUDA GPU for fine-tuning; CPU (zero-shot) otherwise |
| Umbrella (this repo) | `tabpfn-classifier-pipeline` | Docs, contracts, serving loader, examples |

```text
tabpfn-classifier-pipeline/            (this umbrella)
├── README.md
├── CONTRACT.md          authoritative result/artifact envelope
├── MODEL_CARD.md
├── TABULAR_CLASSIFICATION_DATASET_SPEC.md
├── DEPLOYMENT.md
├── PHASE2_ACCEPTANCE_REPORT.md
├── COMPONENTS.json      pinned validator + finetuner commits
├── serving/load_artifact.py   directory-parameterized fitted+checkpoint loader
└── examples/build_synthetic_dataset.py

tabpfn-classifier-dataset-validator/   (CPU, own repo)
├── Dockerfile
├── validate.py          DIMER-facing entrypoint + validation logic
├── requirements.txt
└── tests/

tabpfn-classifier-finetuner/           (GPU/CPU, own repo)
├── Dockerfile           tabpfn==8.1.0 on a CUDA PyTorch runtime
├── train.py             DIMER-facing entrypoint
└── dimer-pipeline.json  preprocessing + fine-tuning fields
```

DIMER builds each container repository from its root and launches the container by the portal
naming convention: `validate.py` for the validator and `train.py` for the fine-tuner. The
validator's logic lives directly in `validate.py` (there is no separate implementation module).

Keep `dimer-pipeline.json` at the fine-tuner repository root. It defines the preprocessing and
fine-tuning fields end users see. Without it, the workbench preprocessing step renders empty and
the fine-tuning step stays locked.

The authoritative dataset contract
([`TABULAR_CLASSIFICATION_DATASET_SPEC.md`](TABULAR_CLASSIFICATION_DATASET_SPEC.md)), the
result/artifact contract ([`CONTRACT.md`](CONTRACT.md)), deployment notes
([`DEPLOYMENT.md`](DEPLOYMENT.md)), the portable serving loader, and the dataset-building helpers
live in this umbrella repository, not in the container repositories.

---

## Creating the pipeline

Prerequisites: portal access as AI Engineer, and both repositories reachable by the portal's
GitHub App.

1. Open **AI Engineer → New Pipeline** and set these fields:

   | Field | Value |
   |---|---|
   | Pipeline Name | `TabPFN Tabular Classification` |
   | Description | `Fine-tune the TabPFN tabular foundation model for classification using your own tabular dataset. Supports binary and multiclass classification, dataset validation, configurable preprocessing, evaluation, and export of the trained model.` |
   | Task Type | `Custom / Other` |
   | Base Model | TabPFN / an approved checkpoint |
   | Validator repository | `https://github.com/kurtvalcorza/tabpfn-classifier-dataset-validator` |
   | Fine-tuner repository | `https://github.com/kurtvalcorza/tabpfn-classifier-finetuner` |

2. Build the validator image, then the CUDA fine-tuner image.
3. Mount or allow download of the approved TabPFN checkpoint (required for the gated `v2.5` /
   `v2.6` / `v3` generations — see [DEPLOYMENT.md → Model-weight delivery](DEPLOYMENT.md)).
4. Run the smoke test with a small dataset.
5. Enable the pipeline **only after** the on-platform serving check in the release gate passes, and
   only once the model-weight licence has been cleared for the intended use.

### Portal implementation notes

- **`Custom / Other` is the correct portal card** for tabular pipelines; DIMER has no native
  tabular task type. The pipeline declares its own task identity: both container images set
  `DIMER_TASK_TYPE=tabular_classification` and rely on that baked fallback, treating any value the
  platform sends as an override. When DIMER adds a native tabular task type, switch the portal type
  and drop the fallback; no runtime code change is required.
- **`dimer-pipeline.json` stays at the fine-tuner repository root.** The portal reads it there to
  render the preprocessing and fine-tuning fields.
- **Field-to-runtime mapping.** `datasetPreprocessing` keys reach the fine-tuner as
  `DIMER_PREPROCESSING_ARGS_JSON`; `modelFinetuning` keys as `DIMER_HYPERPARAMETERS_JSON`. Every
  declared manifest key maps one-to-one to runtime behaviour in `train.py`.
- **`model_id` is not a hyperparameter.** The base model is selected by DIMER's **Base Model**
  field, surfaced to the fine-tuner as `model_version` (the generation) plus an optional pinned
  checkpoint. For the non-commercial generations (`v2.5` / `v2.6` / `v3`), mount the approved
  checkpoint and point the run at it with **`DIMER_TABPFN_MODEL_PATH`**; only `v2` auto-downloads
  freely under the Apache-derived licence.

---

## The dataset

### Format

A zip of CSV files (a directory of the same files is also accepted). The full contract is in
[`TABULAR_CLASSIFICATION_DATASET_SPEC.md`](TABULAR_CLASSIFICATION_DATASET_SPEC.md).

```text
dataset.zip
├── train.csv          (required)   one row per example; one categorical target column
├── val.csv            (optional)   same columns as train; a stratified holdout is split if absent
└── test.csv           (optional)   scored after fitting, never used for selection
```

The target column is named `target` by default; change it with the `target_column` preprocessing
field. Its distinct values are the class labels (at least two). Every other column, except those
listed in `drop_columns`, is a feature; features may be numeric or categorical. Rows with a missing
target are dropped before the row-count and class checks. Column names must match across the CSVs,
though physical column order may differ — the fine-tuner reorders features to the training order
before inference. Duplicate `train.csv` / `val.csv` / `test.csv` candidates in the archive are
rejected so the validator and fine-tuner cannot resolve different files.

### How to build a dataset

TabPFN consumes a feature table, not raw records. Convert a time series or transaction log
(`entity, date, value`) into a training table by engineering one row per `(entity, date)`:

- **features** — history and context at that point: lags, rolling means and standard deviations,
  calendar fields, and any known covariates (promotions, holidays, weather, stock status);
- **target** — the categorical label to predict, e.g. a demand band a chosen number of days ahead,
  or a binary event flag.

[`examples/build_synthetic_dataset.py`](examples/build_synthetic_dataset.py) is a runnable helper
that emits a small, reproducible train/val/test classification ZIP for smoke tests — a synthetic
`scikit-learn` classification problem written as `train.csv` / `val.csv` / `test.csv` inside a
single archive:

```bash
python examples/build_synthetic_dataset.py --out ./tmp/tabpfn-smoke.zip
```

It defaults to 600 rows, 12 numeric features (plus a categorical and an id column to exercise
`drop_columns`), and 3 classes; `--rows`, `--features`, `--classes`, and `--seed` override those.
This is a smoke fixture for exercising the validator and fine-tuner end to end, not a benchmark
dataset.

### Row, feature, and class ceilings

Two kinds of limit apply. The **model** limits are the per-generation caps in
[Generations and capacity limits](#generations-and-capacity-limits): feature and class counts are
hard rejects, and rows above the cap are stratified-sampled down. The **operating** limits are
platform controls: `max_train_rows` (default 100,000; the fine-tuner samples larger tables down to
it, deterministically and stratified), the validator's 2,000-feature ceiling
(`DIMER_TABPFN_MAX_FEATURES`), and a minimum of 50 usable training rows, 2 classes, and 2 usable
rows per class so a stratified holdout can preserve every class. Archive guards: 1 GiB top-level
ZIP, 2 GiB total uncompressed, 512 MiB per member, 200:1 per-member compression ratio, 200 files;
path-traversal members are rejected, and a single inner ZIP is transparently unwrapped and
validated (only multiple top-level ZIPs are rejected). All of these are overridable by platform
environment variables for an intentionally larger profile.

### Data licence governs the served model

Two licences bind a served pipeline, and both must be cleared before enablement:

- **The model weights.** Unlike a fully Apache-licensed tabular model, TabPFN's default `v3`
  weights (and `v2.5` / `v2.6`) are **non-commercial**; only `v2` carries the Apache-derived
  licence with commercial use permitted. Production or externally offered DIMER service use of a
  non-commercial generation requires a commercial licence from Prior Labs, regardless of any local
  evaluation.
- **The training data.** A model fine-tuned on non-commercial data — for example CC BY-NC — may not
  be appropriate to expose as a hosted service, and the fine-tuned weights may encode information
  derived from the uploaded rows. Confirm the licence of any corpus, and its privacy and retention
  constraints, before you enable a pipeline built from it.

Review the exact checkpoint licence, not only the Python package source licence. See
[MODEL_CARD.md → Licensing](MODEL_CARD.md) and the upstream
[Prior Labs licensing FAQ](https://docs.priorlabs.ai/faq#licensing).

---

## Configurable fields

These tables are the field set declared in the fine-tuner's `dimer-pipeline.json`.

Preprocessing (`datasetPreprocessing`):

| Field | Default | Purpose |
|---|---|---|
| `target_column` | `target` | Name of the categorical column to predict; every other non-dropped column is a feature |
| `drop_columns` | — | Comma-separated columns to exclude from features (ids, raw timestamps) |
| `max_train_rows` | `100000` | Deterministic stratified cap on training rows (range 50–100,000); larger tables are sampled to it, then further capped to the version limit |
| `validation_split` | `0.2` | Stratified holdout fraction used only when `val.csv` is absent (range 0.05–0.4) |

Fine-tuning (`modelFinetuning`):

| Field | Default | Purpose |
|---|---|---|
| `fine_tune` | `true` | Fine-tune TabPFN weights (requires CUDA) or run zero-shot ICL. On a no-GPU deployment a `true` request falls back to zero-shot |
| `model_version` | `v3` | TabPFN generation → per-version limits + checkpoint selection (`v3`, `default`, `v2.6`, `v2.5`, `v2`) |
| `epochs` | `30` | Maximum fine-tuning epochs (range 1–200) |
| `learning_rate` | `1e-5` | AdamW learning rate (range 1e-7–1e-3) |
| `weight_decay` | `0.01` | AdamW weight decay (range 0–0.2) |
| `time_limit_seconds` | `3600` | Maximum fine-tuning wall-clock budget (range 60–21,600) |
| `early_stopping_patience` | `8` | Validation epochs without improvement before early stopping (range 1–50) |
| `eval_metric` | `roc_auc` | Primary validation metric used during fine-tuning (`roc_auc` or `log_loss`) |
| `n_finetune_ctx_plus_query_samples` | `10000` | Maximum context-plus-query samples per fine-tuning batch (range 1,000–200,000) |
| `n_estimators_finetune` | `2` | Ensemble estimators inside the fine-tuning loop (range 1–16) |
| `n_estimators_validation` | `2` | Estimators used for validation during fine-tuning (range 1–16) |
| `n_estimators_final_inference` | `8` | Estimators retained for the final fitted inference model (range 1–32) |
| `prediction_batch_rows` | `4096` | Validation/test prediction chunk size, to bound peak memory (range 128–32,768) |
| `seed` | `0` | RNG seed for stratified sampling, the holdout split, and TabPFN fine-tuning (range 0–2,147,483,647) |

---

## Outputs

A successful run writes the DIMER artifact layout under the run's output directory
(`DIMER_OUTPUT_DIR` = `/data/fine-tuning/<run_id>/`) alongside a `result.json` describing the run:

```text
/data/fine-tuning/<run_id>/
├── artifacts/
│   ├── model.tabpfn_fit         fitted estimator (foundation weights omitted from this archive)
│   ├── model.ckpt               foundation / fine-tuned checkpoint
│   └── artifact_manifest.json   target, ordered features, classes, artifact names + SHA-256s
├── evaluation/report.json       validation / optional test metrics + provenance
├── logs/run-summary.json        run parameters, device, provenance summary
├── progress/epoch_*.json        best-effort per-epoch telemetry (never fails a run)
├── checkpoints/                 fine-tuning checkpoints when fine-tuning
└── result.json
```

`result.json` follows the common schema in [`CONTRACT.md`](CONTRACT.md). Its `metrics` block records
`mode` (`fine-tune` or `zero-shot-icl`), `fineTuneEffective` / `fineTuneSkippedReason`, the resolved
device, row and class counts, and a `validation` block (accuracy, balanced accuracy, log loss, and
binary or one-vs-rest macro ROC AUC where computable); a `test` block appears when the zip includes
a `test.csv`:

```jsonc
{
  "successful": true,
  "message": "TabPFN fine-tuning succeeded.",
  "metrics": {
    "mode": "fine-tune", "fineTuneEffective": true, "fineTuneSkippedReason": null,
    "selectedDevice": "cuda:0", "deviceName": "...", "numClasses": 3,
    "validation": { "accuracy": .., "balancedAccuracy": .., "logLoss": .., "rocAucOvrMacro": .. },
    "test": { "..": ".." }
  },
  "artifacts": {
    "modelArtifact":      { "path": "fine-tuning/<run_id>/artifacts/model.tabpfn_fit", "name": "model.tabpfn_fit", "contentType": "application/octet-stream", "sizeBytes": N },
    "evaluationReport":   { "path": "fine-tuning/<run_id>/evaluation/report.json", "..": ".." },
    "logArtifact":        { "path": "fine-tuning/<run_id>/logs/run-summary.json", "..": ".." },
    "foundationCheckpoint": { "path": "fine-tuning/<run_id>/artifacts/model.ckpt", "..": ".." },
    "manifest": { "..": ".." },
    "fittedSha256": "..", "foundationCheckpointSha256": "..",
    "reloadCheck": { "passed": true, "device": "cpu", "rows": N }
  },
  "provenance": { "model": { "..": ".." }, "dataset": { "sha256": ".." }, "seed": 0 },
  "metadata": { "taskType": "tabular_classification", "device": { "..": ".." }, "..": ".." }
}
```

DIMER's `export-to-repository` resolves the exportable model by **`artifacts.modelArtifact`** —
`model.tabpfn_fit`, keyed by a `/data`-relative `path` (in GPU-burst mode it instead carries the
uploaded S3 model **`key`**). There is **no literal `best.pt` requirement** on the default path.
`foundationCheckpoint` (`model.ckpt`) is the second file the in-context learner needs at serve time;
`manifest`, `evaluationReport`, and `logArtifact` accompany them.

`artifacts.reloadCheck` is written only after the fine-tuner reloads the just-saved artifact
in-process and runs `predict_proba` — success is never reported for an artifact that cannot be
reloaded. On failure the same envelope is written with `successful: false` and an `error` object,
with `metadata.baseModel` / `selectedModelId` still populated.

### Serving

`model.tabpfn_fit` intentionally omits the foundation weights and records an estimator
`model_path`. Because DIMER may relocate artifacts between training and serving,
[`serving/load_artifact.py`](serving/load_artifact.py) is directory-parameterized: given the
artifact directory, it rewrites the fitted archive's `model_path` (in a temporary copy — the
original is never modified) to point at the colocated `model.ckpt`, then loads the estimator. Both
files sit under `artifacts/`. This is the same rewrite the fine-tuner's in-runtime reload check
performs.

---

## Reproducibility

TabPFN fine-tuning is stochastic: two runs on identical data can differ unless the seed is fixed.
`seed` is a first-class field and seeds the stratified sampling, the holdout split, and TabPFN
itself (Python `random`, NumPy, and torch, including CUDA where present). GPU kernel autotuning can
still leave small residual variation, so runs are reproducible in ranking but not guaranteed
byte-identical. Starting from identical weights is guaranteed only when a specific checkpoint is
mounted via `DIMER_TABPFN_MODEL_PATH`; without it, an auto-downloaded generation can change if the
upstream weights are updated.

---

## Resource profile

Each fine-tuning run executes as a Kubernetes job. The default DIMER deployment provisions **no GPU
node pool** — GPU is opt-in and off by default — so the paved path runs the validator on CPU and,
without a GPU, runs the fine-tuner as zero-shot ICL on CPU. Fine-tuning requires a CUDA GPU.

| Component | Initial profile |
|---|---|
| Validator | CPU, 2–4 vCPU, 4–8 GB RAM |
| Fine-tuner (GPU) | CUDA GPU, 8+ vCPU, 32+ GB host RAM |

[DEPLOYMENT.md §3](DEPLOYMENT.md) starts the GPU profile at ~80 GB VRAM, following Prior Labs'
official fine-tuning example. That is a conservative starting point, not a measured requirement:
the [Phase 2 acceptance run](PHASE2_ACCEPTANCE_REPORT.md) measured a peak host-total GPU memory of
**5,403 MiB for `v2`** and **7,316 MiB for `v3`** on a 600-row / 14-feature / 3-class smoke dataset
(wall-clock 8 s and 18 s respectively, on an RTX 5070 Ti Laptop, 12 GB). Keep the 80 GB starting
point until you have measured peak VRAM/RAM on representative data, then reduce the profile only
with evidence. `prediction_batch_rows` and the estimator/context-sampling budgets bound
evaluation-time memory but do not guarantee any dataset fits a smaller GPU.

**GPU burst (S3) mode.** When `GPU_BURST_MODE` is set, `/data` is not mounted: the fine-tuner reads
the dataset from and writes `result.json` and the model back to S3
(`GPU_BURST_S3_BUCKET` / `GPU_BURST_DATASET_PREFIX` / `GPU_BURST_RESULT_KEY` / `GPU_BURST_MODEL_KEY`,
via `boto3`), and `modelArtifact` points at `GPU_BURST_MODEL_KEY`. The path is inactive unless the
platform sets those variables.

---

## Provenance and traceability

### How this pipeline was authored

The validator, fine-tuner, configuration, and documentation in this repository were drafted with AI
assistance (Anthropic Claude Opus 4.8, via Claude Code) and are pending human review before
production deployment. The following were verified by execution on local GPU hardware in
[Phase 2](PHASE2_ACCEPTANCE_REPORT.md) (2026-08-19, `tabpfn==8.1.0`, `torch 2.11.0+cu128`, RTX 5070
Ti / Blackwell sm_120), not only generated:

- the validator passes its full check set (18/18) on the synthetic sample, and fails closed on the
  negative cases (duplicate `train.csv`, unseen validation class, too-few usable rows,
  path-traversal member, zip-bomb expansion);
- the fine-tuner CUDA image builds and fine-tunes TabPFN for **both `v2` and `v3`** end to end,
  writing valid `model.tabpfn_fit` + `model.ckpt` artifacts;
- the saved artifact reloads in a fresh container through `serving/load_artifact.py` and reproduces
  the training metrics — exactly on CUDA, within float tolerance on CPU;
- `test.csv` is scored when present, and the uploaded dataset's SHA-256 is recorded; a mounted
  checkpoint's SHA-256 is recorded when `DIMER_TABPFN_MODEL_PATH` is set.

Two real defects surfaced and were fixed during that run: a PEP 668 build failure in the CUDA base
image, and a TabPFN 8.1.0 artifact-save `TypeError` on non-JSON init params. Both are merged and
re-verified from the merged build.

Not yet verified, and requiring human sign-off: **DIMER serving end-to-end inference** (gate 8 —
wiring the artifact into the platform's serving layer and issuing a real inference request), the
on-platform portal image build and smoke test, and the production resource-profile request. Treat
the generated code as a reviewed draft, not audited production code, and do not production-enable
until the end-to-end serving check in [DEPLOYMENT.md](DEPLOYMENT.md) passes on-platform. The `v3`
weights additionally remain non-commercial — production enablement requires a commercial licence
from Prior Labs.

### Model lineage

| Field | Value |
|---|---|
| Base model | [TabPFN](https://github.com/PriorLabs/TabPFN), Prior Labs |
| Default generation | `v3` (TabPFN-3) via `tabpfn==8.1.0` |
| Estimators | `TabPFNClassifier` (zero-shot ICL), `FinetunedTabPFNClassifier` (fine-tune) |
| Gated v3 weights | [`Prior-Labs/tabpfn_3`](https://huggingface.co/Prior-Labs/tabpfn_3) (Hugging Face-gated, non-commercial) |
| Licence | `v2` Apache-derived (commercial OK); `v2.5` / `v2.6` / `v3` non-commercial |
| Component pins | `COMPONENTS.json` (immutable validator + finetuner commit SHAs) |

Strongest provenance comes from mounting a specific approved checkpoint via
`DIMER_TABPFN_MODEL_PATH`: the run hashes that exact file into `provenance.model.baseModelSha256`
rather than inferring the weights from a package/model cache. `provenance.model` also records
`tabpfnVersion`, `torchVersion`, `modelVersionRequested`, and `modelPathRequested`.

### Data lineage

A trained model inherits the provenance and licence of the table it was fine-tuned on. Each dataset
should carry its source, its licence, and — for a derived table — the deterministic, seeded
transformation that produced it. The uploaded dataset's SHA-256 is recorded in every run's
`provenance.dataset`. Because fine-tuned weights may encode information derived from the uploaded
rows, treat the served artifact with the same data-governance controls as the source training
dataset.

### Per-run record

Every run writes a `result.json` that serves as the run's provenance record: the requested
generation and any resolved checkpoint path/SHA-256, the target and dropped columns, the seed, time
budget, and eval metric, the resolved training device, row and class counts, the mode and whether
fine-tuning was effective, the resulting scores, the fitted-artifact and foundation-checkpoint
SHA-256s, and the input dataset SHA-256. Paired with the container image digest and the
`COMPONENTS.json` commit pins, this record forms a chain from data to served model.

---

## References

- [TabPFN source code](https://github.com/PriorLabs/TabPFN), Prior Labs.
- [Prior Labs documentation](https://docs.priorlabs.ai/) and
  [licensing FAQ](https://docs.priorlabs.ai/faq#licensing).
- [`Prior-Labs/tabpfn_3`](https://huggingface.co/Prior-Labs/tabpfn_3) gated model repository (v3
  weights, non-commercial), Hugging Face.
- Pipeline docs: [`CONTRACT.md`](CONTRACT.md) (result/artifact envelope),
  [`MODEL_CARD.md`](MODEL_CARD.md), [`TABULAR_CLASSIFICATION_DATASET_SPEC.md`](TABULAR_CLASSIFICATION_DATASET_SPEC.md),
  [`DEPLOYMENT.md`](DEPLOYMENT.md), [`PHASE2_ACCEPTANCE_REPORT.md`](PHASE2_ACCEPTANCE_REPORT.md).
