---
license: other
license_name: tabpfn-3-license-v1.0
model_card_spec: "1.0"
pipeline_tag: tabular-classification
tags:
  - tabular-classification
  - tabular-foundation-model
  - in-context-learning
  - tabpfn
base_model: Prior-Labs/tabpfn_3
---

# TabPFN-3 Classifier (tabpfn 8.1.0)


###### Description

TabPFN Classifier packages Prior Labs' TabPFN (Tabular Prior-data Fitted Network) through the `tabpfn==8.1.0` package, with the TabPFN-3 generation selected by default (`model_version: v3`) and v2, v2.5, and v2.6 selectable through DIMER configuration. TabPFN is a Transformer trained on a prior over synthetic tabular tasks so that it performs supervised classification in a single forward pass: the labelled training rows are the in-context support, the query rows attend to them, and the head emits class probabilities. The v3 checkpoint's stored inference config admits up to 1,000,000 samples, 2,000 features, and 160 classes; the earlier generations are narrower.

At inference the model conditions on the operator's training table; adaptation happens through in-context conditioning by default and, when `fine_tune=true` and a CUDA GPU is present, through gradient fine-tuning with Prior Labs' `FinetunedTabPFNClassifier` on explicit train and validation sets. What this repository adds is the DIMER composition: the pipeline contract (`dimer-pipeline.json`, `CONTRACT.md`), the component manifest pinning the validator and fine-tuner commits (`COMPONENTS.json`), an artifact loader for serving (`serving/load_artifact.py`), tests, and a synthetic example builder; the validator and fine-tuner workers live in the sibling `tabpfn-classifier-dataset-validator` and `tabpfn-classifier-finetuner` repositories. The upstream weights are not modified by this repository.

#### Intended Use and Limitations

The use cases below are the ones envisioned during development; the limits are the ones the workers enforce.

###### Primary Intended Uses

Supervised classification of tabular data where each observation is one row of mixed numeric and categorical predictor columns and one categorical target. The pipeline takes a `train.csv` (optionally `val.csv`/`test.csv`) with a declared target column and produces a fitted TabPFN artifact pair (`model.tabpfn_fit` + `model.ckpt`), per-row class labels and class probabilities, and holdout metrics.

Concrete application domains envisioned during development: binary and multiclass classification for risk categorisation, quality grading, churn and event prediction, and scientific classification on feature tables of small to medium size, where the operator wants strong performance without hyperparameter search. The pipeline is meant to play the role of a strong zero-shot baseline, or a fine-tuned model when a GPU is available, inside DIMER. Enforced ceilings follow the selected generation (v3: 1,000,000 rows, 2,000 features, 160 classes; v2: 10,000 / 500 / 10), further capped by the DIMER `max_train_rows` preprocessing argument. A provided `test.csv` is used only for post-fit evaluation, never for early stopping or checkpoint selection.

###### Primary Intended Users

Machine-learning engineers, data scientists, and researchers building predictive systems from structured datasets, and DIMER integrators provisioning the workers. The envisioned deployment setting is internal enterprise or research use through the DIMER platform — CPU by default, GPU opt-in — and, because `tabpfn-3-license-v1.0` limits the TabPFN-3 weights to a Non-Commercial Purpose that its own definition says excludes production deployment and revenue generation, testing and evaluation only until a production enablement has been cleared against those terms.

The pipeline assumes its users understand dataset provenance, holdout evaluation, leakage, class imbalance, and distribution shift; know that `predict()` is an argmax over class probabilities not calibrated for their domain; can read `metrics.fineTuneEffective` to tell a fine-tuned run from a zero-shot fallback; and understand that random stratified splitting is wrong for temporal or grouped data, where they must supply explicit splits. A user who would ship a zero-shot fallback believing it was fine-tuned is outside the assumed competency.

###### Out-of-scope use cases

- **Capability boundaries:** numeric-target regression (the sibling `tabpfn-regressor-pipeline` does that); image, text, object-detection, or segmentation tasks; unsupervised clustering; causal-effect estimation; time-series forecasting without tabular feature construction.
- **Input boundaries:** tables exceeding the selected generation's caps (v3: 1,000,000 rows, 2,000 features, 160 classes — the validator refuses, the fine-tuner re-checks); a target with a single class; datasets whose rows are not exchangeable (temporal or grouped) unless explicit `val.csv`/`test.csv` splits are provided, because the default stratified random split leaks across time and groups.
- **Decision boundaries:** autonomous high-impact decisions — health, safety, criminal justice, credit, employment, housing — without application-specific validation and a human decision-maker; treating TabPFN benchmark strength as a guarantee of task-level superiority; production deployment or commercial hosted use of the TabPFN-3 weights without clearing the model-weight terms, which define Non-Commercial Purpose to exclude both.

#### Factors

TabPFN's behaviour varies with the structure of the table it is given, not with a physical capture condition; the three subsections below say what that means for groups, instruments, and environment.

###### Groups

This pipeline is not human-centric by construction: TabPFN's pretraining prior is synthetic, so no demographic group is an intrinsic development group of the foundation model. Whether any real-world tables entered the TabPFN-3 pretraining mix is not enumerated in this repository and is treated as unknown; in either case the pretraining corpus is not group-audited, demographic fairness or subgroup parity has **not** been established for the checkpoint, and the pipeline measures no subgroup metric.

Where the operator's downstream table describes people, the obligation transfers to the operator: identify the relevant groups in their own data, compute per-group accuracy, balanced accuracy, log loss, and ROC-AUC on the holdout split, and check for disparate error rates before deployment. The validator result records `classNames` and row counts, not any demographic structure.

###### Instrumentation

TabPFN consumes an abstract tabular representation rather than a raw sensor stream. The instrument does not disappear because a table sits between it and the model: the operator's rows are produced by whatever systems fed the CSV — transactional databases, ETL pipelines, sensors, survey instruments — and their sampling rate, resolution, calibration, and encoding of missing values determine feature quality.

Instrument error reaches the model as feature error. Drift, miscalibration, or a changed collection procedure between training and inference is not detectable by this pipeline; the validator checks schema, row and class counts, and generation caps, not whether a column's meaning has changed. Operators should document the instrumentation of downstream datasets separately.

###### Environment

**Operating environment.** CPU is the default deployment. Fine-tuning requires a CUDA GPU — Prior Labs' official example recommends 80 GB — and when `fine_tune=true` is requested without a usable CUDA device the run does **not** fail: it falls back to zero-shot in-context learning and records `metrics.fineTuneSkippedReason` with `metrics.fineTuneEffective=false`. Prediction is chunked at `prediction_batch_rows` to bound memory. The fine-tuner records `cudaAvailable` and the device name in provenance.

**Data environment.** The reported behaviour assumes the inference rows are exchangeable with the training rows: same feature semantics, same encoding, same class prevalence. Performance degrades, without warning from the pipeline, under distribution shift, high-cardinality categoricals, unusual missingness, and non-independent rows; an apparently strong holdout score on temporal or grouped data split at random is not evidence of anything. Robustness to arbitrary distribution shift has not been established.

#### Metrics

Metrics are chosen for a probabilistic multiclass classifier whose intended use spans balanced and imbalanced tables.

###### Performance Measures

The fine-tuner scores the fitted model on the validation split, and on `test.csv` when provided, in `evaluate_split` (`tabpfn-classifier-finetuner/train.py`) and writes `accuracy`, `balancedAccuracy`, `logLoss` (where defined), and `rocAuc` for binary targets or `rocAucOvrMacro` for multiclass, with `rocAucError` recorded when AUC is undefined, into `evaluation/report.json` and the `result.json` envelope.

Why these: accuracy captures discrete correctness; balanced accuracy re-weights it by class so that a majority-class predictor cannot look good on an imbalanced table; log loss captures probability quality and penalises confident mistakes, which matters whenever the class probabilities are used operationally; ROC-AUC captures ranking quality independent of any threshold, with the macro one-vs-rest form extending it to many classes. Reading only accuracy hides imbalance and calibration failures, which is why all four are written. The pipeline reports no upstream benchmark number of its own; TabPFN's published rankings are not claimed here.

###### Decision thresholds

The default decision rule is an implicit `argmax`: TabPFN's `predict()` returns the class with the highest predicted probability, and this pipeline ships that rule unchanged. No acceptance threshold on any metric was set during development, because the pipeline is domain-agnostic and the tolerable error rate is a property of the deployment.

No probability cutoff is applied, and none is shipped, because the emitted probabilities are not calibrated for the operator's domain. Calibrating and thresholding are the deployment owner's responsibility: set the operating point from the asymmetric cost of false positives against false negatives and the class prevalence on held-out data, and revisit it when either changes. For a screening use where a missed positive is the expensive error, the threshold on the positive-class probability belongs below 0.5; where a false alarm is expensive, above it.

###### Approaches to uncertainty and variability

The pipeline's reported metrics come from a single validation split (a stratified random holdout when `val.csv` is absent) and, when supplied, a single `test.csv`. No dispersion is reported alongside the point value: one split, one run, no confidence interval. Operators who need one should repeat the run across seeds or use cross-validation on their own side.

Sources of run-to-run variability: the holdout split, the training cap, TabPFN's internal estimator ensembling (`n_estimators_finetune`, `n_estimators_validation`, `n_estimators_final_inference`, defaults 2/2/8), and gradient fine-tuning; all are driven by the DIMER `seed` hyperparameter, which `_seed_everything` propagates to Python, NumPy, and torch (including CUDA) and which is passed as `random_state` to the estimators. Non-deterministic CUDA kernels can still produce small differences. The class probabilities are raw ensemble-averaged outputs and have not been calibrated; a caller who needs calibrated probabilities must fit a calibrator on their own holdout data. A zero-shot fallback run and a fine-tuned run are different estimators and their metrics must not be compared as if from the same procedure — `fineTuneEffective` says which one ran.

#### Ethical considerations and biases

No external ethics board reviewed this pipeline, and no clearance testing with a specific group took place; the subsections record what the developers considered and what the repository actually does.

###### Data

TabPFN's pretraining prior is synthetic; whether the TabPFN-3 generation additionally saw real-world tables is not enumerated in this repository, so the sensitivity of the pretraining data is unknown rather than ruled out. What this repository distributes: the pipeline contract, component manifest, serving loader, tests, and a synthetic example builder (`examples/build_synthetic_dataset.py`); it does **not** distribute the checkpoint — the TabPFN-3 weights are a gated Hugging Face download under `tabpfn-3-license-v1.0`, and the local `weights/` mirror is gitignored.

Fine-tuned weights may encode information derived from the uploaded dataset, so the fitted artifact inherits the dataset's confidentiality. The pipeline does not audit the operator's data for personal, sensitive, or proprietary attributes — the validator checks structure, not content — so the legality, privacy, consent, retention, and governance of downstream data, and the licensing of the fine-tuned model, remain with the application developer and data owner.

###### Human Life

The pipeline is not intended for decisions in health care, physical safety, criminal justice, legal rights, employment, credit, insurance, education access, or public benefits, and it has not been validated for any of them. The only validation performed is the contract and integration testing in `tests/` (CPU CI, which does not exercise GPU fine-tuning or production serving) and the DIMER holdout evaluation on the operator's own table; no clinical, regulatory, or independent domain validation has been carried out by the developers or by any external body.

Where such a use is foreseeable — a triage classifier on a clinical feature table, for example — it would be admissible only with independent domain validation on that operator's population, a human decision-maker between the prediction and the action, subgroup evaluation, and whatever regulatory clearance the domain requires.

###### Mitigations

Implemented in the composed workers, each inspectable in the named code:

- **Supply-chain integrity:** `tabpfn` is pinned to 8.1.0 and the component commits are pinned in `COMPONENTS.json`. The selected generation is explicit (`model_version`) and a mismatch between the DIMER-resolved and requested version raises `MODEL_IDENTITY_MISMATCH`. When DIMER mounts an approved checkpoint through `DIMER_TABPFN_MODEL_PATH`, its SHA-256 is recorded and, if the model config carries `expectedSha256`, a mismatch raises `MODEL_INTEGRITY_FAILED`. Without a mounted checkpoint the package's cached weights are used and only their digest is recorded — that path is not pinned, and the card says so.
- **Input integrity:** the validator enforces the selected generation's row, feature, and class caps and writes `classNames` on every result; `test.csv` is isolated from fine-tuning, early stopping, and checkpoint selection.
- **Statistical mitigations:** stratified splitting and capping keep every class represented; evaluation prediction is chunked to bound memory.
- **Reproducibility:** `seed` propagates to Python, NumPy, torch, and the estimators; the artifact manifest records target column, ordered feature columns, class labels, and per-artifact SHA-256; the dataset fingerprint is recorded.
- **Refusals:** a GPU-less fine-tune request is not silently satisfied — the run falls back to zero-shot and says so in `fineTuneSkippedReason`; the serving layer must load the actual `model.tabpfn_fit` artifact before the pipeline is considered production-ready.

###### Risks and harms

- **Overconfidence outside the training distribution** (model-intrinsic): the probabilities are uncalibrated and carry no out-of-distribution signal; borne by whoever the operator's decision affects; likely under normal use as the deployment drifts.
- **Amplification of input bias** (model-intrinsic): a table whose labels encode a historical disparity yields a classifier that reproduces it; borne by the data subjects in the disadvantaged group; realised whenever such a table is used without subgroup evaluation.
- **Silent zero-shot fallback misread as fine-tuning** (use-context): a run without a GPU still succeeds; an operator who does not read `fineTuneEffective` ships a different model than they think; borne by the operator.
- **Leakage through random splitting** (use-context): temporal or grouped data split at random produces a holdout score that collapses in production; the pipeline does not detect it; borne by the operator and downstream users.
- **Automation bias** (use-context): a numerically precise probability displaces human judgement; borne by the data subject.
- **Licence breach** (use-context): production-deploying or commercially hosting the TabPFN-3 weights without clearance, since the licence's Non-Commercial Purpose excludes both; borne by the operator.

###### Use cases

Distinct from the capability and decision boundaries listed under *Out-of-scope use cases*, the developers consider the following uses prohibited even where the model would produce a numerically plausible label:

- surveillance, biometric or demographic profiling, or social scoring of individuals;
- unlawful discrimination in employment, housing, credit, insurance, education, or healthcare access, including classification on a target that proxies a protected attribute;
- deceptive, manipulative, or predatory applications, including presenting an uncalibrated class probability as a certified risk estimate;
- criminal-justice, medical-diagnosis, or legal-rights determinations without the validation and oversight described under *Human Life*;
- any use outside the terms of the selected model weights — for TabPFN-3, `tabpfn-3-license-v1.0`, whose Non-Commercial Purpose excludes production deployment and revenue generation without a separate agreement — or of the DIMER deployment.

---

## Model family

**TabPFN** (Tabular Prior-data Fitted Network), maintained by Prior Labs.

This pipeline defaults to the TabPFN-3 generation through `tabpfn==8.1.0`. The selected generation is explicit in DIMER configuration (`model_version`) rather than inferred from whatever checkpoint happens to be cached.

## Intended task

Supervised **tabular classification** where each example is one row and the target is categorical. Numeric and categorical feature columns may be mixed, subject to the selected TabPFN generation's supported input characteristics.

This pipeline is not for image, text-generation, object-detection, segmentation, or numeric-target regression tasks.

## Adaptation modes

### Task-specific fine-tuning

The fine-tuner uses Prior Labs' `FinetunedTabPFNClassifier` with explicit train and validation sets. Fine-tuning performs gradient updates to the pretrained model and needs a CUDA GPU.

CPU is the default deployment (GPU is opt-in). When fine-tuning is requested without a usable CUDA device, the run falls back to zero-shot ICL and records the reason in `metrics.fineTuneSkippedReason` (`metrics.fineTuneEffective=false`) rather than failing.

### In-context / zero-shot mode

When `fine_tune=false`, the pipeline uses `TabPFNClassifier` directly. `.fit(X, y)` establishes the labelled in-context task but does not represent the same task-specific gradient update as the fine-tuning wrapper. Results are therefore labelled `zero-shot-icl`.

## Version pinning

Container dependency:

```text
tabpfn==8.1.0
```

Default requested generation:

```text
v3
```

For a production-quality reproducibility chain, DIMER should mount an approved checkpoint and set:

```text
DIMER_TABPFN_MODEL_PATH=/path/to/approved/model.ckpt
```

When a concrete local checkpoint path is available, the fine-tuner records its SHA-256. This is stronger than reporting an arbitrary file found in a package/model cache.

## Saved artifacts

A successful run writes, under `DIMER_OUTPUT_DIR/artifacts/`:

- `model.tabpfn_fit` — fitted estimator state, saved with TabPFN's fitted-model serialization utility.
- `model.ckpt` — TabPFN model weights/checkpoint, saved with TabPFN's model serialization utility.
- `artifact_manifest.json` — target column, ordered feature columns, class labels, and artifact names.

Alongside `artifacts/`, the run also writes `evaluation/report.json`, `logs/run-summary.json`, and `progress/epoch_*.json`, plus the `result.json` envelope (see `CONTRACT.md`). `result.json` declares `artifacts.modelArtifact` (path relative to `/data`) that DIMER's `export-to-repository` resolves.

## Evaluation

Validation and optional post-fit test evaluation report:

- accuracy
- balanced accuracy
- log loss where defined
- binary ROC AUC for two-class problems
- macro one-vs-rest ROC AUC for multiclass problems where defined

A provided `test.csv` is not passed to fine-tuning, early stopping, checkpoint selection, or internal validation.

## Resource considerations

Fine-tuning TabPFN is substantially more memory-intensive than ordinary sklearn-style estimators. Prior Labs' official fine-tuning example recommends an 80 GB CUDA GPU. Treat that as the initial DIMER provisioning profile, then reduce resources only after recording real peak-memory and runtime measurements for representative datasets.

The pipeline exposes estimator/context budgets and caps evaluation prediction batches to reduce avoidable memory spikes, but these controls do not guarantee that every dataset will fit a smaller GPU.

## Licensing — deployment gate

The deployment team must review the **exact model checkpoint license**, not only the Python package source license.

As of this pipeline's initial implementation (August 2026), Prior Labs documentation distinguishes:

- TabPFN code and TabPFN-2 weights: Prior Labs' Apache-derived license with an additional attribution provision.
- TabPFN-2.5, TabPFN-2.6, and TabPFN-3 weights: non-commercial model-weight terms.

Because this pipeline defaults to TabPFN-3, **production or externally offered DIMER service use must be cleared against those model-weight terms before enablement**. If the intended deployment falls outside the permitted scope, use an appropriately licensed checkpoint/model generation or obtain the necessary rights.

Upstream references:

- Prior Labs TabPFN repository: `https://github.com/PriorLabs/TabPFN`
- TabPFN documentation: `https://docs.priorlabs.ai/`
- Model licensing: `https://docs.priorlabs.ai/faq#licensing`

## Data governance

Fine-tuned weights may encode information derived from the uploaded dataset. Dataset ownership, privacy, confidentiality, retention, and downstream-model licensing therefore remain separate deployment considerations from the upstream model license.

Do not upload data to DIMER unless its processing and model-training use are authorized.

## Known limitations

- Performance is dataset-dependent; TabPFN benchmark strength is not a guarantee of task-level superiority.
- High-cardinality categoricals, unusual missingness, distribution shift, leakage, and non-independent rows can invalidate apparently strong holdout scores.
- Random stratified splitting is appropriate for exchangeable classification rows but not for temporal forecasting, grouped/patient-level leakage scenarios, or other datasets requiring group/time-aware validation. In those cases, provide an explicit `val.csv` and `test.csv` constructed with the correct domain split.
- The DIMER serving layer must be tested against the actual `model.tabpfn_fit` artifact before the pipeline is considered production-ready.
