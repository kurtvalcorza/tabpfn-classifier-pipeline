# TabPFN Classifier — DIMER Pipeline

A DIMER pipeline for tabular classification with Prior Labs' TabPFN foundation model. The default configuration targets **TabPFN-3** through `tabpfn==8.1.0` and supports both task-specific fine-tuning and zero-shot/in-context classification.

This repository is the **integration and specification repository**. Deployable source code lives in the two component repositories and is intentionally not copied here:

| Component | Repository | Runtime |
|---|---|---|
| Dataset validator | `kurtvalcorza/tabpfn-classifier-dataset-validator` | CPU |
| Fine-tuner | `kurtvalcorza/tabpfn-classifier-finetuner` | CUDA GPU for fine-tuning |

`COMPONENTS.json` pins reviewed component commits. This avoids the source duplication/drift problem that occurs when deployable files are maintained both in an umbrella repository and in their own container repositories.

## Flow

```text
CSV dataset / ZIP
      |
      v
TabPFN dataset validator
      |
      | validated canonical train/val/test contract
      v
TabPFN classifier fine-tuner
      |
      +--> model.tabpfn_fit
      +--> model.ckpt
      +--> artifact_manifest.json
      +--> result.json
```

## Dataset contract

```text
dataset.zip
├── train.csv          required
├── val.csv            optional
└── test.csv           optional
```

- One categorical target column, `target` by default.
- Other non-dropped columns are tabular features.
- `val.csv` and `test.csv`, when supplied, must use the same schema as `train.csv` and may not introduce unseen classes.
- If `val.csv` is absent, the fine-tuner creates a deterministic **stratified** validation holdout.
- `test.csv` is evaluated only after fitting; it is never used for training or early stopping.
- Duplicate/ambiguous `train.csv`, `val.csv`, or `test.csv` candidates are rejected rather than resolved differently by separate containers.
- The validator enforces usable-target rows, class/split viability, feature limits, and resource-bounded ZIP handling.

See [`TABULAR_CLASSIFICATION_DATASET_SPEC.md`](TABULAR_CLASSIFICATION_DATASET_SPEC.md).

## Training modes

### Fine-tuning

`fine_tune=true` uses `FinetunedTabPFNClassifier` to update the pretrained TabPFN weights on the uploaded task. CUDA is mandatory. A missing GPU is a failed fine-tuning run, not an implicit mode change.

The default DIMER fields expose:

- model generation (`v3` by default)
- epochs
- learning rate and weight decay
- wall-clock time limit
- early-stopping patience
- validation metric (`roc_auc` or `log_loss`)
- fine-tuning and validation ensemble sizes
- final inference ensemble size
- context-plus-query sampling budget
- deterministic seed
- evaluation batch size

The authoritative DIMER form definition is `dimer-pipeline.json` in the **fine-tuner repository root**.

### Zero-shot / ICL

`fine_tune=false` uses `TabPFNClassifier` without gradient updates. This remains a valid comparison/baseline mode but is reported explicitly as `zero-shot-icl` in `result.json`.

## Evaluation

Every successful run reports validation metrics:

- accuracy
- balanced accuracy
- log loss, when computable
- ROC AUC for binary classification
- one-vs-rest macro ROC AUC for multiclass classification, when computable

If `test.csv` is present, the same metrics are reported independently under `metrics.test` after fitting.

## Artifacts

A successful run writes:

```text
/data/output/
├── model.tabpfn_fit
├── model.ckpt
├── artifact_manifest.json
└── checkpoints/             fine-tuning checkpoints when applicable
```

`model.tabpfn_fit` is the fitted estimator state for round-trip inference. `model.ckpt` is the underlying foundation/fine-tuned model checkpoint. The manifest records target, feature order, classes, and artifact filenames.

## Provenance

`result.json` records:

- TabPFN and PyTorch package versions
- requested TabPFN generation
- requested/resolved local model path where available
- SHA-256 of an explicitly resolved checkpoint file when available
- SHA-256 of the uploaded dataset
- RNG seed

For production reproducibility, mount a specific approved checkpoint and set `DIMER_TABPFN_MODEL_PATH`; do not rely on an opaque model cache as evidence of the weights actually used.

## Model licensing

**Do not treat the Python package license and the current model-weight license as identical.** Current Prior Labs documentation distinguishes the code/older TabPFN-2 weights from newer TabPFN-2.5/2.6/3 weights, which have non-commercial terms. Review the exact checkpoint license and DIMER's intended service use before production enablement. See [`MODEL_CARD.md`](MODEL_CARD.md).

## DIMER setup

Create a Custom / Other pipeline with:

| Field | Value |
|---|---|
| Task identity | `tabular_classification` |
| Base model | TabPFN / approved checkpoint |
| Validator repository | `tabpfn-classifier-dataset-validator` |
| Fine-tuner repository | `tabpfn-classifier-finetuner` |

Then:

1. Build the validator image.
2. Build the CUDA fine-tuner image.
3. Mount or allow download of the approved TabPFN checkpoint.
4. Run the included synthetic smoke dataset through validation.
5. Execute one GPU fine-tuning smoke test.
6. Reload `model.tabpfn_fit` and verify predictions against the trained process.
7. Only after the serving layer can load the artifact, consider pipeline enablement.

Detailed operational steps are in [`DEPLOYMENT.md`](DEPLOYMENT.md).

## Repository layout

```text
.
├── README.md
├── MODEL_CARD.md
├── TABULAR_CLASSIFICATION_DATASET_SPEC.md
├── DEPLOYMENT.md
├── COMPONENTS.json
├── examples/
│   └── build_synthetic_dataset.py
├── tests/
│   └── test_example_builder.py
└── .github/workflows/ci.yml
```

The pipeline repository deliberately does **not** contain copies of `validate.py` or `train.py`. Review and deploy the pinned component commits listed in `COMPONENTS.json`.
