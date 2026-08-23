# Deployment Runbook — TabPFN Classifier

This runbook covers DIMER deployment of the TabPFN classifier validator and fine-tuner.

## 1. Components

Deploy the exact reviewed component commits pinned in `COMPONENTS.json`:

- `kurtvalcorza/tabpfn-classifier-dataset-validator`
- `kurtvalcorza/tabpfn-classifier-finetuner`

The umbrella repository intentionally contains no duplicate container source.

## 2. Pre-deployment gates

Do not enable the pipeline until all of the following are true:

1. Validator CI passes.
2. Fine-tuner unit/config CI passes.
3. Fine-tuner CUDA image builds successfully.
4. The intended TabPFN checkpoint/model generation has been reviewed for licensing.
5. An approved model-distribution path is available: mounted checkpoint, controlled cache, or explicitly approved network retrieval.
6. A real GPU fine-tuning smoke test succeeds.
7. `model.tabpfn_fit` can be reloaded in a clean runtime and reproduces predictions.
8. DIMER's serving layer can load the fitted TabPFN artifact and execute an inference request end-to-end.

A successful training container alone is not sufficient evidence that the hosted inference path works.

## 3. Initial resource profile

Start conservatively:

| Component | Initial profile |
|---|---|
| Validator | CPU, 2–4 vCPU, 4–8 GB RAM |
| Fine-tuner | CUDA GPU with ~80 GB VRAM, 8+ vCPU, 32+ GB host RAM |

The 80 GB GPU starting point follows Prior Labs' official fine-tuning example. After representative DIMER runs, record actual peak VRAM/RAM and reduce the profile only with evidence.

## 4. Container builds

### Validator

Build from the validator repository root:

```bash
docker build -t tabpfn-classifier-dataset-validator .
```

### Fine-tuner

Build from the fine-tuner repository root:

```bash
docker build -t tabpfn-classifier-finetuner .
```

The fine-tuner image pins `tabpfn==8.1.0` and uses a CUDA-enabled PyTorch runtime.

Record the final immutable image digest in the DIMER deployment record. Tags alone are not sufficient provenance.

## 5. Model-weight delivery

### Preferred production path: explicit mounted checkpoint

Mount an approved checkpoint read-only and set:

```text
DIMER_TABPFN_MODEL_PATH=/models/tabpfn/model.ckpt
```

Advantages:

- no runtime model download;
- deterministic weight selection;
- SHA-256 can be recorded against the exact file;
- easier license/security review;
- avoids ambiguity from multiple cached checkpoints.

### Download/cache path

If runtime download is permitted, make the network requirement explicit and preserve the resulting checkpoint in controlled storage. Do not claim exact checkpoint provenance merely because a model name/version string was configured.

## 6. DIMER pipeline creation

Use the platform's Custom / Other pipeline path and connect:

| DIMER field | Component |
|---|---|
| Validator repo | `tabpfn-classifier-dataset-validator` |
| Fine-tuner repo | `tabpfn-classifier-finetuner` |
| Task identity | `tabular_classification` |

The fine-tuner repository root contains `dimer-pipeline.json`, which defines the user-facing preprocessing and fine-tuning fields.

## 7. Runtime paths

Validator:

```text
DIMER_DATASET_DIR=/data/dataset
DIMER_RESULT_PATH=/data/dataset-validations/result.json
```

Fine-tuner:

```text
DIMER_DATASET_DIR=/data/dataset
DIMER_OUTPUT_DIR=/data/fine-tuning/<run_id>   # DIMER sets this per run
DIMER_RESULT_PATH=/data/fine-tuning/<run_id>/result.json
DIMER_TRAIN_DEVICE=cpu                          # default deployment; "0"/cuda when GPU is enabled
```

Both support `DIMER_DONE_CALLBACK`, `DIMER_PREPROCESSING_ARGS_JSON`, and `DIMER_PIPELINE_METADATA_JSON`; the fine-tuner also consumes `DIMER_HYPERPARAMETERS_JSON`.

## 8. Validator acceptance test

Use the synthetic example builder:

```bash
python examples/build_synthetic_dataset.py --out ./tmp/tabpfn-smoke.zip
```

Expected validator properties:

- succeeds on the generated train/val/test package;
- reports usable training rows and class count;
- confirms matching schemas;
- confirms validation/test classes exist in training;
- returns exit code 0.

Then deliberately test invalid packages:

- duplicate `train.csv` paths;
- unseen validation class;
- too few usable target rows;
- path-traversal ZIP member;
- excessive archive expansion.

They must fail closed.

## 9. Fine-tuning smoke test

Run on a CUDA host with the synthetic dataset and conservative settings:

```json
{
  "fine_tune": true,
  "model_version": "v3",
  "epochs": 2,
  "time_limit_seconds": 600,
  "early_stopping_patience": 2,
  "n_estimators_finetune": 1,
  "n_estimators_validation": 1,
  "n_estimators_final_inference": 2,
  "seed": 0
}
```

Acceptance criteria:

- mode is `fine-tune` on a GPU host (on a CPU host it is `zero-shot-icl` with `metrics.fineTuneSkippedReason` set);
- the selected device is recorded in `metadata.device`;
- validation metrics are present;
- test metrics are present when `test.csv` is supplied;
- `artifacts/model.tabpfn_fit`, `artifacts/model.ckpt`, and `artifacts/artifact_manifest.json` exist (plus `evaluation/report.json`, `logs/run-summary.json`, `progress/epoch_*.json`);
- dataset hash is recorded;
- explicitly mounted model checkpoint is hashed when configured;
- callback behavior is correct.

## 10. Artifact round-trip test

In a clean environment with the same supported TabPFN package version:

1. Load `artifacts/model.tabpfn_fit` with TabPFN's fitted-model loading utility (its companion `artifacts/model.ckpt` sits beside it).
2. Load a fixed inference fixture.
3. Run predictions and probabilities.
4. Compare them with the predictions captured immediately after training.
5. Establish an acceptable numeric tolerance for probabilities.

Do this before wiring the artifact into DIMER serving.

## 11. End-to-end serving test

The final production gate is:

```text
upload dataset
  -> validator
  -> fine-tuner
  -> persisted artifact
  -> DIMER deployment/serving layer
  -> API inference request
  -> expected classification response
```

Verify:

- feature order and names come from `artifact_manifest.json`;
- missing/unexpected feature behavior is defined;
- probability/class serialization is stable;
- model initialization fits within serving memory;
- concurrency limits do not exhaust GPU/host memory;
- logs never expose uploaded dataset rows or secrets.

## 12. Observability

Capture at minimum:

- image digest;
- component Git commit;
- TabPFN and PyTorch versions;
- model checkpoint SHA-256 where explicitly mounted;
- dataset SHA-256;
- seed and fine-tuning configuration;
- train/validation/test row counts;
- wall-clock duration;
- peak host RAM and GPU VRAM;
- validation/test metrics;
- artifact size;
- serving startup latency and inference latency.

## 13. Rollback

Rollback should pin all of the following together:

- validator component commit/image digest;
- fine-tuner component commit/image digest;
- approved TabPFN checkpoint hash;
- DIMER configuration revision;
- serving adapter/runtime version.

Do not roll back only the model while leaving an incompatible serving runtime in place.

## 14. Known deployment blocker

**Model-weight licensing must be cleared for the intended use before enabling the default TabPFN-3 configuration as a production/external service.** Current upstream terms for newer TabPFN generations are not equivalent to an unrestricted Apache-2.0 model-weight grant.
