# TabPFN Tabular Classification Dataset Specification

## 1. Package layout

DIMER accepts either a directory or a single ZIP archive containing CSV tables:

```text
dataset.zip
├── train.csv          required
├── val.csv            optional
└── test.csv           optional
```

The files may appear under one ordinary folder prefix, but each logical stem must resolve to **exactly one** table. For example, an archive containing both `a/train.csv` and `b/train.csv` is invalid.

Nested ZIP files, path-traversal members, duplicate normalized archive paths, and archives exceeding configured resource limits are rejected.

## 2. Row semantics

Each row is one supervised classification example.

The target column defaults to:

```text
target
```

and can be changed through the DIMER `target_column` preprocessing field.

All columns other than the target and configured `drop_columns` are features.

## 3. Target requirements

`train.csv` must contain:

- at least 50 rows with a non-null target by default;
- at least two distinct classes;
- at least two usable rows in every class, so a stratified holdout can preserve all classes when `val.csv` is absent.

Rows with missing targets are not counted toward the minimum and are removed before training/evaluation.

The validator does not impose an arbitrary ten-class Mitra-style ceiling. The selected TabPFN generation remains the final authority on model-specific class support.

## 4. Feature requirements

At least one feature must remain after removing the target and `drop_columns`.

The default DIMER validator ceiling is 2,000 feature columns. Override only after confirming the selected TabPFN generation and DIMER hardware profile can support the intended dataset.

Identifier columns, free-form unique strings, raw timestamps, and other columns that should not be predictive inputs should generally be listed in `drop_columns` or transformed upstream.

## 5. Validation split

If `val.csv` is supplied:

- its column set must match `train.csv`;
- it must contain usable target values;
- every validation class must already occur in the training table;
- it is used for fine-tuning validation/early stopping and reported validation metrics.

If `val.csv` is absent, the fine-tuner creates a deterministic **stratified** holdout using `validation_split` and `seed`. The split algorithm ensures, where mathematically possible, that every class remains represented in both training and validation.

Do not rely on the automatic split for datasets that require temporal, grouped, entity-level, geographic, patient-level, or other leakage-aware partitioning. Construct explicit `train.csv`, `val.csv`, and `test.csv` instead.

## 6. Test split

`test.csv` is optional.

When supplied:

- its schema must match `train.csv`;
- its classes must occur in training;
- it is loaded only for final post-fit evaluation;
- it is **not** used for optimization, early stopping, checkpoint selection, or validation.

This separation is intentional: reported test metrics are meant to remain held out from model selection.

## 7. Column compatibility

The three CSVs must have the same column names, although physical column order may differ. The fine-tuner reorders validation/test features to the training feature order before inference.

Example:

```csv
age,income,region,target
34,51000,Region-A,low-risk
49,78000,Region-B,high-risk
27,44000,Region-A,low-risk
```

## 8. Sampling cap

`max_train_rows` defaults to 100,000. If usable training rows exceed the configured cap, the fine-tuner applies deterministic stratified sampling so classes are preserved as far as possible.

This is an operating/resource control, not a claim that 100,000 rows is an absolute TabPFN architectural limit.

## 9. Archive/resource limits

Defaults in both validator and fine-tuner:

| Guard | Default |
|---|---:|
| Top-level ZIP size | 1 GiB |
| Total uncompressed ZIP size | 2 GiB |
| Individual file/member size | 512 MiB |
| Compression ratio per member | 200:1 |
| Dataset file count | 200 |

These values can be changed with DIMER environment variables when platform limits are better defined. Both containers enforce the same resolver rules so validation and training cannot intentionally select different logical CSV files.

## 10. Missing values

Missing **target** values are discarded.

Missing **feature** values are left in the table for TabPFN preprocessing/model handling. Dataset owners should nevertheless inspect whether missingness itself leaks outcomes or differs materially between train and deployment populations.

## 11. Leakage guidance

Examples of common leakage:

- including a post-outcome status or disposition field;
- splitting repeated records for the same person/device across train and validation;
- random splitting of time-indexed forecasting rows;
- fitting preprocessing or target encoders on the full dataset before constructing held-out splits;
- including IDs that encode the label or collection source.

The validator checks structural compatibility; it cannot establish that a dataset is causally or statistically leakage-free.

## 12. Recommended production package

For a serious benchmark or deployment evaluation, provide all three tables explicitly:

```text
train.csv   model adaptation
val.csv     early stopping / model selection
 test.csv    untouched final evaluation
```

Construct these partitions according to the real deployment mechanism rather than convenience.
