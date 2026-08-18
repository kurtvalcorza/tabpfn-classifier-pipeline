# Model Card — TabPFN Classifier for DIMER

## Model family

**TabPFN** (Tabular Prior-data Fitted Network), maintained by Prior Labs.

This pipeline defaults to the TabPFN-3 generation through `tabpfn==8.1.0`. The selected generation is explicit in DIMER configuration (`model_version`) rather than inferred from whatever checkpoint happens to be cached.

## Intended task

Supervised **tabular classification** where each example is one row and the target is categorical. Numeric and categorical feature columns may be mixed, subject to the selected TabPFN generation's supported input characteristics.

This pipeline is not for image, text-generation, object-detection, segmentation, or numeric-target regression tasks.

## Adaptation modes

### Task-specific fine-tuning

The fine-tuner uses Prior Labs' `FinetunedTabPFNClassifier` with explicit train and validation sets. Fine-tuning performs gradient updates to the pretrained model and requires CUDA in this implementation.

The pipeline fails rather than silently switching modes when a user requests fine-tuning without a usable CUDA device.

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

A successful run writes both:

- `model.tabpfn_fit` — fitted estimator state, saved with TabPFN's fitted-model serialization utility.
- `model.ckpt` — TabPFN model weights/checkpoint, saved with TabPFN's model serialization utility.

The accompanying `artifact_manifest.json` records the target column, ordered feature columns, class labels, and artifact names.

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
