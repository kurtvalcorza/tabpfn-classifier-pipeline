"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier) — TASK-INFERENCE.

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline module
(``src/tabpfn_classifier_pipeline/pipeline.py``) and the base-model pin/stage/verify cell are produced by the
generator from repository sources so they cannot drift from the package. Inference only: the pinned TabPFN-3
checkpoint conditions on the training rows in context; the DIMER fine-tuning path (private worker) is not carried.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "tabpfn-classifier-pipeline"
BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/tabpfn_classifier_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Prior--Labs%2Ftabpfn__3-ffcc4d?style=flat",
        "https://huggingface.co/Prior-Labs/tabpfn_3",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-PriorLabs%2FTabPFN-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/PriorLabs/TabPFN",
    ),
    ("arXiv", "https://img.shields.io/badge/arXiv-2605.13986-b31b1b.svg", "https://arxiv.org/abs/2605.13986"),
]

TEMPLATE = {
    "package": "tabpfn_classifier_pipeline",
    "repo_name": REPO,
    "stem": "tabpfn_classifier",
    "notebook_name": "tabpfn_classifier_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "TabPFNClassifierPipeline",
    "weights_key": "tabpfn-3-classifier",
    "runtime_imports": ["torch", "pandas", "sklearn"],
    "title": "TabPFN-3 Classifier — DIMER task-inference tutorial (standalone)",
    "badges": BADGES,
    "capability": "supervised tabular classification by **in-context learning** with the pinned `Prior-Labs/tabpfn_3` classifier checkpoint: the labelled training rows are the support set, no gradient update, `argmax` decision over uncalibrated class probabilities, a `model.tabpfn_fit` + `model.ckpt` + `artifact_manifest.json` bundle reloaded across a fresh boundary",
    "intro": (
        "TabPFN-3 is a Transformer trained on a prior over synthetic tabular tasks so that it performs supervised "
        "classification in a single forward pass: `fit` registers your labelled training rows as the in-context "
        "support and performs **no gradient update**; query rows attend to that support and the head emits class "
        "probabilities. This notebook is **inference-only**: it runs that in-context path through the carried "
        "`tabpfn_classifier_pipeline` module against the pinned, digest-verified TabPFN-3 checkpoint. **The DIMER "
        "fine-tuning path of this pipeline is not carried here**: it lives in the private `tabpfn-classifier-finetuner` "
        "worker, and the TabPFN-3 weights are released under `tabpfn-3-license-v1.0`, whose Non-Commercial Purpose "
        "excludes production deployment — so this tutorial is testing-and-evaluation material and carries no private "
        "code. The carried module owns the pinned snapshot scheme, the input contract, the ICL fit / predict / evaluate "
        "calls, the artifact bundle the serving path consumes (with its pre-load validation) and the evaluation report. "
        "Sample metrics on synthetic data are tutorial sanity evidence only, not benchmark or production evidence; "
        "`predict_proba` returns raw TabPFN ensemble outputs that are **not calibrated probabilities** and the pipeline "
        "ships no acceptance threshold."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried module guarantees, resolve and digest-verify the immutable "
        "TabPFN-3 checkpoint, generate the synthetic sample or supply a train/val/test ZIP through the archive-safety "
        "rules, validate it into an input manifest with a recorded rejection, fit in context and read accuracy, "
        "balanced accuracy, log loss and ROC-AUC against the majority-class baseline, export the artifact bundle and "
        "prove that it reloads across a fresh boundary and reproduces the recorded metric, score genuinely new rows "
        "under the `argmax` rule, and export machine-readable outputs plus provenance."
    ),
    "exclusions": (
        "gradient fine-tuning (the private DIMER worker path, dropped from this standalone notebook), regression, "
        "calibrated probabilities, the v2 / v2.5 / v2.6 generations (only the pinned v3 checkpoint is carried), "
        "temporal or grouped splitting, or any quality claim beyond one holdout of one table."
    ),
    "prerequisites": [
        "- **Runtime:** Python 3.11+ (Google Colab or Jupyter). CPU is sufficient for the default sample; CUDA is used automatically when present. The pinned `torch==2.11.0` install is the largest download of the run; the checkpoint is 203 MiB.",
        "- **Knowledge:** basic Python and pandas, the train/val/test convention, and what accuracy, balanced accuracy and log loss mean.",
        "- **Data:** the default path draws a deterministic synthetic 3-class table (600 rows, 12 features, `train.csv`/`val.csv`/`test.csv`) in code and needs no download and no private data; a gated BYOD path accepts one ZIP with the same layout (or a single `train.csv`, from which a seeded stratified holdout is drawn). Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
        "- **Licence:** the TabPFN-3 weights are non-commercial (`tabpfn-3-license-v1.0`): testing, evaluation and internal benchmarking only. Clear the licence before any production use.",
        "- **Credentials:** none. `Prior-Labs/tabpfn_3` is public and not access-gated.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Prepare the dataset: default synthetic sample or bring your own\n\n"
                "The expected input is one `train.csv` with a declared categorical target column, plus optional "
                "`val.csv` and `test.csv` with identical columns (explicit splits are preserved, never re-split). Column "
                "names must be unique, the target must have no missing values and at least `MIN_CLASSES` classes with "
                "`MIN_ROWS_PER_CLASS` rows each, and numeric features must be finite; every other column is a feature "
                "(numeric or string categorical). With `USE_BYOD = False` the carried module draws the deterministic "
                "synthetic sample (`build_synthetic_dataset`: scikit-learn `make_classification`, seed `SAMPLE_SEED`) "
                "as a ZIP; with `USE_BYOD = True` a ZIP is taken from `BYOD_ZIP_PATH` (an executor places it there) or "
                "uploaded, and read member by member by `read_dataset_zip` with the archive-safety rules (bare file "
                "names only, expanded-size and compression-ratio ceilings; never `extractall`). Without a `val.csv`, "
                "`stratified_holdout` draws a seeded holdout of `VALIDATION_SPLIT` — correct only for independent rows; "
                "temporal, grouped or patient-level data need your own leakage-safe splits. Look for the split sizes "
                "and the class counts."
            ),
            "code": (
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_ZIP_PATH = ''  # @param {{type:\"string\"}}\n"
                "TARGET_COLUMN = 'target'  # @param {{type:\"string\"}}\n"
                "VALIDATION_SPLIT = 0.2  # @param {{type:\"number\"}}\n"
                "SAMPLE_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "WORK = Path('work')\n"
                "shutil.rmtree(WORK, ignore_errors=True)\n"
                "DATASET_DIR = WORK / 'dataset'\n"
                "DATASET_DIR.mkdir(parents=True)\n"
                "if USE_BYOD:\n"
                "    if BYOD_ZIP_PATH:\n"
                "        zip_name, zip_payload = Path(BYOD_ZIP_PATH).name, Path(BYOD_ZIP_PATH).read_bytes()\n"
                "    else:\n"
                "        from google.colab import files\n"
                "        uploaded = files.upload()\n"
                "        if len(uploaded) != 1:\n"
                "            raise RuntimeError('Upload exactly one dataset ZIP.')\n"
                "        zip_name, zip_payload = next(iter(uploaded.items()))\n"
                "    zip_path = DATASET_DIR / Path(zip_name).name\n"
                "    zip_path.write_bytes(zip_payload)\n"
                "    dataset_origin = {{'type': 'user-supplied ZIP (BYOD)', 'zip': zip_path.name}}\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    zip_path = build_synthetic_dataset(DATASET_DIR / 'synthetic.zip', rows=600, features=12, classes=3, seed=SAMPLE_SEED)\n"
                "    dataset_origin = {{'type': 'deterministic synthetic tutorial sample', 'generator': 'tabpfn_classifier_pipeline.build_synthetic_dataset', 'rows': 600, 'features': 12, 'classes': 3, 'seed': SAMPLE_SEED}}\n"
                "    sample_kind = 'synthetic'\n"
                "dataset_sha256 = sha256_file(zip_path)\n"
                "frames = read_dataset_zip(zip_path)\n"
                "train_df = frames['train.csv']\n"
                "val_df, test_df = frames.get('val.csv'), frames.get('test.csv')\n"
                "split_origin = 'explicit train/val/test from the ZIP'\n"
                "if val_df is None:\n"
                "    train_df, val_df = stratified_holdout(train_df, TARGET_COLUMN, VALIDATION_SPLIT, seed=SAMPLE_SEED)\n"
                "    split_origin = f'seeded stratified holdout of {{VALIDATION_SPLIT}} drawn from train.csv (independent rows assumed)'\n"
                "print({{'sample_kind': sample_kind, **dataset_origin, 'zip_sha256': dataset_sha256[:16], 'splits': split_origin}})\n"
                "print({{'train': train_df.shape, 'val': val_df.shape, 'test': None if test_df is None else test_df.shape}})\n"
                "print('class counts (train):', train_df[TARGET_COLUMN].astype(str).value_counts().sort_index().to_dict())"
            ),
        },
        {
            "md": (
                "## 5. Validate the inputs → input manifest\n\n"
                "`validate_inputs` is the module's public validation stage: it applies the input contract (unique columns, "
                "declared target present and complete, class floor and ceilings of the v3 generation — `MAX_TRAIN_ROWS`, "
                "`MAX_FEATURES`, `MAX_CLASSES` — identical schema across splits, no reserved `prediction`/`proba_*` "
                "columns, finite numeric features) and returns an **input manifest** naming the schema, the feature "
                "columns, the classes, the per-split row and class counts, a digest of the training table and the "
                "verdict; warnings (classes present in `val.csv` but absent from `train.csv`, a ≥10:1 class imbalance) "
                "are carried as non-fatal findings. It is written to `outputs/{stem}_input_manifest.json`. To show what "
                "rejection looks like, the cell also validates a probe table whose target column was renamed and records "
                "the structured finding. The ceilings and the decision rule are printed before any model runs."
            ),
            "code": (
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MAX_TRAIN_ROWS': MAX_TRAIN_ROWS, 'MAX_FEATURES': MAX_FEATURES, 'MAX_CLASSES': MAX_CLASSES, 'MIN_CLASSES': MIN_CLASSES, 'MIN_ROWS_PER_CLASS': MIN_ROWS_PER_CLASS}}, 'decision_rule': DECISION_RULE, 'model_version': MODEL_VERSION}})\n"
                "input_manifest = validate_inputs(train_df, TARGET_COLUMN, val=val_df, test=test_df, names=[dataset_origin['type']])\n"
                "try:\n"
                "    validate_inputs(train_df.rename(columns={{TARGET_COLUMN: 'label'}}), TARGET_COLUMN)\n"
                "except InputRejected as exc:\n"
                "    input_manifest['findings'].append({{'input': 'renamed-target-probe', **exc.finding}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False, default=str)\n"
                "FEATURE_COLUMNS = input_manifest['inputs'][0]['feature_columns']\n"
                "CLASSES = input_manifest['inputs'][0]['classes']\n"
                "print(json.dumps({{k: input_manifest['inputs'][0][k] for k in ('id', 'mode', 'target_column', 'classes', 'numeric_features', 'categorical_features', 'splits')}}, indent=2))\n"
                "print('findings:', json.dumps(input_manifest['findings'], indent=2, default=str))"
            ),
        },
        {
            "md": (
                "## 6. Fit in context and evaluate\n\n"
                "`pipe.fit` builds `tabpfn.TabPFNClassifier` on the digest-verified checkpoint of Section 3 "
                "(`model_path` = the staged file, `N_ESTIMATORS` ensemble members, seed `SEED`) and registers the training "
                "rows as the in-context support — **no gradient update happens**; a run takes seconds on CPU for the "
                "sample. `pipe.evaluate` scores the frozen validation split (and `test.csv` when present): `accuracy` is "
                "discrete correctness under the `argmax` rule; `balancedAccuracy` re-weights it per class so a "
                "majority-class predictor cannot look good on an imbalanced table; `logLoss` scores the probabilities and "
                "penalises confident mistakes; `rocAuc` (binary, or one-vs-rest macro) scores ranking quality independent "
                "of any threshold. These are single-split numbers with no dispersion estimate. Look for the metrics and "
                "the reported device."
            ),
            "code": (
                "N_ESTIMATORS = 4  # @param {{type:\"integer\"}}\n"
                "SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "pipe.n_estimators, pipe.random_state = N_ESTIMATORS, SEED\n"
                "pipe.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN], target_column=TARGET_COLUMN)\n"
                "if pipe.classes != CLASSES:\n"
                "    raise RuntimeError('estimator class order differs from the validated class list')\n"
                "validation_metrics = pipe.evaluate(val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN])\n"
                "test_metrics = None if test_df is None else pipe.evaluate(test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN])\n"
                "print(json.dumps({{'mode': 'zero-shot-icl (in-context conditioning, no gradient update)', 'device': pipe.device, 'source': pipe.source, 'n_estimators': N_ESTIMATORS, 'seed': SEED, 'classes': pipe.classes, 'validation': validation_metrics, 'test': test_metrics}}, indent=2))"
            ),
        },
        {
            "md": (
                "## 7. Majority-class baseline → evaluation report\n\n"
                "`majority_class_baseline` always predicts the most frequent training class and is scored on the same "
                "validation rows (its log loss uses the training class frequencies as probabilities), so the comparison "
                "uses identical rows. `evaluation_report` is the module's public evaluation stage: it carries the four "
                "metrics with the verdict `sample-sanity`, the baseline, the split and the fresh-boundary reload check "
                "(filled in by Section 8 and re-written there); without a scored validation split the verdict is "
                "`not-measurable`. It is written to `outputs/{stem}_evaluation_report.json`. On the synthetic sample "
                "these are sanity metrics for the plumbing, not evidence of tabular-classification skill."
            ),
            "code": (
                "baseline = majority_class_baseline(train_df[TARGET_COLUMN], val_df[TARGET_COLUMN], classes=CLASSES)\n"
                "report = evaluation_report(validation_metrics, baseline=baseline, n_validation=len(val_df), class_labels=CLASSES, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{key: report[key] for key in ('verdict', 'reason', 'adaptation', 'decision_rule', 'n_validation', 'metrics', 'baselines')}}, indent=2))\n"
                "if validation_metrics['accuracy'] < baseline['accuracy']:\n"
                "    print('WARNING: in-context TabPFN does not beat the majority baseline on this holdout; inspect the data before drawing any conclusion.')"
            ),
        },
        {
            "md": (
                "## 8. Export the artifact bundle and verify it across a fresh boundary\n\n"
                "The deployable artifact is the pair `model.tabpfn_fit` (fitted estimator state **including the "
                "in-context training rows**) + `model.ckpt` (a byte copy of the verified foundation checkpoint) described "
                "by `artifact_manifest.json` (task, target, ordered feature columns, classes, per-file SHA-256, base-model "
                "identity); the fitted archive alone is not a model. `pipe.save_artifact` writes it and `zip_artifact_bundle` zips it "
                "to `outputs/{stem}_artifact.zip` for the companion artifact-inference notebook. Then it does what a "
                "downstream consumer would do with the bundle and nothing else: copy it to a **fresh location**, "
                "`validate_artifact_bundle` (schema, member names, sizes, digests, archive safety — before any model "
                "state is deserialised), reconstruct through `TabPFNClassifierPipeline.from_artifact` (the fitted archive's "
                "recorded `model_path` is rewritten in a temporary copy to the companion checkpoint; TabPFN's "
                "`load_fitted_tabpfn_model` restores the estimator; no refit, no download) and require the recomputed "
                "validation accuracy to equal the recorded value within `1e-6`. Because the bundle contains training "
                "rows, treat it with the same confidentiality controls as the dataset; loading it executes trusted "
                "serialised Python/torch state."
            ),
            "code": (
                "ARTIFACT_DIR = WORK / 'artifact'\n"
                "artifact_manifest = pipe.save_artifact(ARTIFACT_DIR)\n"
                "bundle_zip_sha256 = zip_artifact_bundle(ARTIFACT_DIR, 'outputs/{stem}_artifact.zip')\n"
                "print({{'artifact': sorted(p.name for p in ARTIFACT_DIR.iterdir()), 'fittedEstimatorSha256': artifact_manifest['fittedEstimatorSha256'][:16], 'foundationCheckpointSha256': artifact_manifest['foundationCheckpointSha256'][:16], 'bundle_zip_sha256': bundle_zip_sha256[:16]}})\n"
                "if artifact_manifest['foundationCheckpointSha256'] != WEIGHTS_SHA256:\n"
                "    raise RuntimeError('the bundled checkpoint is not the pinned foundation checkpoint')\n"
                "FRESH_DIR = WORK / 'fresh-reload'\n"
                "shutil.copytree(ARTIFACT_DIR, FRESH_DIR)\n"
                "checked = validate_artifact_bundle(FRESH_DIR, expected_checkpoint_sha256=WEIGHTS_SHA256)\n"
                "fresh = TabPFNClassifierPipeline.from_artifact(FRESH_DIR, device=pipe.device, expected_checkpoint_sha256=WEIGHTS_SHA256)\n"
                "if fresh.classes != CLASSES or fresh.feature_columns != FEATURE_COLUMNS:\n"
                "    raise RuntimeError('reloaded artifact disagrees with the validated schema')\n"
                "reloaded_metrics = fresh.evaluate(val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN])\n"
                "reload_check = {{'recordedAccuracy': validation_metrics['accuracy'], 'reloadedAccuracy': reloaded_metrics['accuracy'], 'tolerance': 1e-6, 'recordedModelPath': checked['recordedModelPath']}}\n"
                "reload_check['accuracyMatches'] = abs(reload_check['reloadedAccuracy'] - reload_check['recordedAccuracy']) <= reload_check['tolerance']\n"
                "print(json.dumps(reload_check, indent=2))\n"
                "if not reload_check['accuracyMatches']:\n"
                "    raise RuntimeError('The reloaded artifact does not reproduce the recorded validation metric. Do not ship this artifact.')\n"
                "report = evaluation_report(validation_metrics, baseline=baseline, n_validation=len(val_df), class_labels=CLASSES, sample_kind=sample_kind, reload_check=reload_check)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(f'Fresh-boundary verification PASSED on {{len(val_df)}} validation rows.')"
            ),
        },
        {
            "md": (
                "## 9. Score new rows\n\n"
                "Real use means rows the estimator has not seen. The default takes the first eight rows of `test.csv` "
                "(or of `val.csv` when no test split exists) with the target column removed; set `USE_BYOD_ROWS = True` "
                "to supply your own CSV instead (`NEW_DATA_PATH` for an executor, or the upload dialog) with exactly the "
                "artifact's feature columns and no target/`prediction`/`proba_*` columns — `validate_new_rows` rejects "
                "duplicates, missing or extra columns and infinite values rather than silently dropping anything. "
                "`predict` on the **reloaded** estimator returns `prediction` (the `argmax` class) and one `proba_<class>` "
                "column per class in the artifact's stored class order; the probabilities are raw ensemble outputs, "
                "**not calibrated**, and 0.9 does not mean a 90 % chance of being right."
            ),
            "code": (
                "USE_BYOD_ROWS = False  # @param {{type:\"boolean\"}}\n"
                "NEW_DATA_PATH = ''  # @param {{type:\"string\"}}\n\n"
                "if USE_BYOD_ROWS:\n"
                "    if NEW_DATA_PATH:\n"
                "        new_name, new_payload = Path(NEW_DATA_PATH).name, Path(NEW_DATA_PATH).read_bytes()\n"
                "    else:\n"
                "        from google.colab import files\n"
                "        uploaded = files.upload()\n"
                "        if len(uploaded) != 1:\n"
                "            raise RuntimeError('Upload exactly one CSV of new rows.')\n"
                "        new_name, new_payload = next(iter(uploaded.items()))\n"
                "    new_rows = pd.read_csv(io.BytesIO(new_payload))\n"
                "    new_rows_origin = f'user-supplied CSV (BYOD): {{new_name}}'\n"
                "else:\n"
                "    source_df, source_label = (test_df, 'test.csv') if test_df is not None else (val_df, 'val.csv')\n"
                "    new_rows = source_df.drop(columns=[TARGET_COLUMN]).head(8).reset_index(drop=True)\n"
                "    new_rows_origin = f'first 8 rows of {{source_label}} (held out from the in-context support), target removed'\n"
                "new_rows = validate_new_rows(new_rows, FEATURE_COLUMNS, target_column=TARGET_COLUMN, classes=CLASSES)\n"
                "predictions = fresh.predict(new_rows)\n"
                "print('scored rows drawn from:', new_rows_origin)\n"
                "print(predictions.to_string(index=False))"
            ),
        },
        {
            "md": (
                "## 10. Export machine-readable results and provenance\n\n"
                "`outputs/{stem}_predictions.csv` holds one row per scored input (`row_id`, `prediction`, `proba_<class>` "
                "in class order); `outputs/{stem}_result.json` records the validation and test metrics, the majority "
                "baseline, the evaluation report, the fresh-boundary reload check, the input manifest, the dataset identity "
                "(ZIP digest, split origin, row counts), the artifact manifest (with its digests), the inference settings, "
                "the notebook's source (repository, revision, module digest, generator), the pinned model identity, "
                "revision and licence, and the runtime. `outputs/{stem}_artifact.zip` is the bundle for the companion "
                "notebook. No credentials are recorded."
            ),
            "code": (
                "predictions.to_csv('outputs/{stem}_predictions.csv', index=False)\n"
                "payload = {{\n"
                "    'metrics': {{'validation': validation_metrics, 'test': test_metrics}},\n"
                "    'majority_class_baseline': baseline,\n"
                "    'evaluation_report': report,\n"
                "    'fresh_boundary_reload': reload_check,\n"
                "    'input_manifest': input_manifest,\n"
                "    'dataset': {{**dataset_origin, 'sample_kind': sample_kind, 'zip_sha256': dataset_sha256, 'target_column': TARGET_COLUMN, 'feature_columns': FEATURE_COLUMNS, 'classes': CLASSES, 'splits': split_origin, 'rows': {{'train': len(train_df), 'val': len(val_df), 'test': None if test_df is None else len(test_df)}}}},\n"
                "    'artifact': artifact_manifest,\n"
                "    'inference': {{'mode': 'zero-shot-icl', 'adaptation': 'in-context conditioning only; the private-worker fine-tune path is not carried', 'n_estimators': N_ESTIMATORS, 'seed': SEED, 'decision_rule': DECISION_RULE, 'new_rows_origin': new_rows_origin, 'scored_rows': int(len(predictions))}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'model_file': WEIGHTS_FILE,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'tabpfn': importlib.metadata.version('tabpfn'), 'pandas': pd.__version__, 'scikit_learn': importlib.metadata.version('scikit-learn'), 'device': pipe.device}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The predicted class is `argmax` over TabPFN's class probabilities, which are raw ensemble outputs, not "
        "calibrated, with no shipped threshold. The evaluation report's `sample-sanity` verdict names what it is: one "
        "holdout of one table with no dispersion estimate — on the synthetic sample a plumbing check, and even BYOD "
        "metrics must not be generalised to a domain, a population or a class distribution. Random stratified splitting "
        "assumes independent rows; temporal, grouped or patient-level data need leakage-safe splits you supply. "
        "In-context learning is not fine-tuning: the estimator's quality depends entirely on the training rows it "
        "conditions on, and the artifact carries those rows. Digest equality proves the checkpoint bytes are the ones "
        "pinned at the immutable revision; it does not by itself prove who published them.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this "
        "notebook, can acquire and digest-verify the pinned TabPFN-3 checkpoint, validate the demonstrated table into "
        "an input manifest, fit in context and beat a trivial baseline on the synthetic sample, export the artifact "
        "bundle, reload it across a fresh boundary and reproduce the recorded metric, score new rows, and emit the "
        "shown machine-readable outputs in the tested runtime — without the repository being reachable. It does "
        "**not** establish benchmark superiority, generalisation, fairness, robustness, calibration, safety for "
        "high-consequence decisions, production fitness, or anything about the fine-tuned models the private DIMER "
        "worker produces.\n\n"
        "**Next experiments:** raise `N_ESTIMATORS` to 8 and compare log loss; enable `USE_BYOD` with a small table of "
        "your own (a few hundred rows, two to ten classes) and read `balancedAccuracy` against the majority baseline; "
        "drop `val.csv` from the ZIP and watch the seeded holdout take over; hand `outputs/{stem}_artifact.zip` to the "
        "companion artifact-inference notebook in a fresh session.\n\n"
        "## References\n\n"
        f"- Repository README: https://github.com/kurtvalcorza/{REPO}/blob/main/README.md\n"
        f"- Repository model card: https://github.com/kurtvalcorza/{REPO}/blob/main/MODEL_CARD.md\n"
        f"- Weight provenance: https://github.com/kurtvalcorza/{REPO}/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/PriorLabs/TabPFN\n"
        "- TabPFN-3 technical report: https://arxiv.org/abs/2605.13986"
    ),
}
