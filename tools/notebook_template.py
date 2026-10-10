"""Per-repository template for tools/build_notebook.py /3 (NOTEBOOK_SPEC 2.2 §4 standalone, §25.13 isolated environment) — E2E.

The generator writes the infrastructure cells (runtime check, carrier, isolated install + stage runner, checkpoint
staging) from repository files; this template holds the learner-facing prose, the guided layer and the learner
cells. Every learner cell calls ``run_stage(...)``: the carried ``tools/tutorial_stages.py`` runs one stage per process
in an isolated, hash-locked environment, so nothing is installed into the notebook kernel. The ARTIFACT-INFERENCE
companion has its own template, ``tools/notebook_template_artifact_inference.py``.
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



UV = {
    "version": "0.12.15",
    "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "bytes": 20081404,
    "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
}
ENVIRONMENT = {
    "package": "tabpfn_classifier_pipeline",
    "repo_name": REPO,
    "weights_key": "tabpfn-3-classifier",
    "modules": ["__init__.py", "pipeline.py"],
    "entry_module": "pipeline.py",
    "lock": "tutorials/requirements-colab.lock.txt",
    "managed_python": "3.12.12",
    "uv": UV,
    "disk_gib": {"weights": 0.3, "environment": 8.0},
    "runtime_modules": ["torch", "tabpfn", "numpy", "pandas", "scikit-learn"],
    "install_flags": ["--only-binary", ":all:"],
    "license_file": "NOTICE",
}

RUNTIME_PREREQ = (
    "- **Runtime:** a fresh **Linux x86_64** runtime — Google Colab (CPU is enough for the default sample; a T4 GPU is used automatically when present), Kaggle or a Linux Jupyter kernel. The kernel's own Python version does not matter: the notebook installs nothing into it, and runs every stage with CPython 3.12.12 in an isolated environment built from {n_locked} hash-locked packages (`tabpfn` 8.1.0, `torch` 2.11.0 with its CUDA libraries, `numpy` 2.5.3, `pandas` 2.3.2, `scikit-learn` 1.9.0). About 0.3 GB of disk is needed for the checkpoint and about 8 GB for the isolated environment."
)
LICENCE_PREREQ = "- **Licence:** the TabPFN-3 weights are non-commercial (`tabpfn-3-license-v1.0`): testing, evaluation and internal benchmarking only. A bundle carries a byte copy of the checkpoint, so the same terms travel with it. Clear the licence before any production use. No credentials are needed: `Prior-Labs/tabpfn_3` is public and not access-gated."

TEMPLATE = {
    **ENVIRONMENT,
    "stem": "tabpfn_classifier",
    "notebook_name": "tabpfn_classifier_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "stage_runner": "tools/tutorial_stages.py",
    "run_all": (
        "Selecting **Run all** in a fresh Linux x86_64 runtime builds an isolated Python environment from the carried hash-locked requirements without touching the notebook kernel's own packages, then runs each stage below in its own process: it stages and digest-verifies the pinned TabPFN-3 checkpoint (an ungated download; the TabPFN-3 licence's non-commercial terms still apply to what you do with it), draws the deterministic synthetic 3-class table in code (600 rows, 12 numeric features, seeded `train`/`val`/`test` split, no download), validates it into an input manifest, **fits the classifier in context on the training split** (support-row registration on the pinned checkpoint, no gradient update), evaluates on the held-out split against a majority-class baseline and a logistic-regression reference and writes the evaluation report, exports the artifact bundle and reloads it in a fresh process with a probability-equivalence check, scores new rows and exports machine-readable results and provenance. No repository clone, DIMER worker or service, credential, upload dialog, configuration edit or runtime restart is required (NOTEBOOK_SPEC 2.2 §5). No hosted run of this revision has been recorded yet."
    ),
    "byod": (
        "Both optional branches are off by default and never part of the default path. `USE_BYOD = True` in Section 4 takes your own labelled table by `BYOD_PATH` (a ZIP with `train.csv` and optional `val.csv`/`test.csv`, a single `train.csv`, or a directory holding them — paths work in Colab, Kaggle and Jupyter) or, in Colab, an upload dialog; name the label in `TARGET_COLUMN` and list identifier columns in `DROP_COLUMNS` (kept beside the predictions, never given to the model). A missing target, duplicate columns, a class in `val.csv`/`test.csv` that `train.csv` lacks, or a numeric column with a few stray strings stop Section 4 or 5 with a coded message naming the file. `USE_BYOD_ROWS = True` in Section 9 scores your own unlabelled rows. Uploads stay inside this runtime."
    ),
    "title": "TabPFN-3 Classifier — DIMER E2E tabular classification tutorial (standalone)",
    "badges": BADGES,
    "capability": "supervised tabular classification by **in-context learning** with the pinned `Prior-Labs/tabpfn_3` classifier checkpoint: validated support rows, no gradient update, evaluation against trivial and classical baselines, an `argmax` decision over uncalibrated class probabilities, and a `model.tabpfn_fit` + `model.ckpt` + `artifact_manifest.json` bundle reloaded in a fresh process",
    "intro": (
        "TabPFN-3 is a Transformer trained on a prior over synthetic tabular tasks so that it performs supervised "
        "classification in a single forward pass: `fit` registers your labelled training rows as the in-context "
        "support and performs **no gradient update**; query rows attend to that support and the head emits class "
        "probabilities. `n_estimators` averages several passes over differently preprocessed views of the table. The "
        "carried package owns the pinned snapshot scheme, the input contract, the in-context fit / predict / evaluate "
        "calls, the artifact bundle the serving path consumes (with its pre-load validation) and the evaluation report.\n\n"
        "**Adaptation.** The production DIMER pipeline can also fine-tune TabPFN by gradient descent (Prior Labs' "
        "`FinetunedTabPFNClassifier`, default on in `dimer-pipeline.json`, on a large GPU). This tutorial does not: "
        "in-context conditioning is its adaptation — the same path production falls back to without a large GPU. That is "
        "a proposed RUN7 deviation, recorded in the evaluation report and pending the maintainer's approval. The TabPFN-3 "
        "weights are released under `tabpfn-3-license-v1.0`, whose Non-Commercial Purpose excludes production "
        "deployment, so this tutorial is testing-and-evaluation material. Sample metrics on synthetic data are tutorial "
        "sanity evidence only; `predict_proba` returns raw ensemble outputs that are **not calibrated probabilities** and "
        "the pipeline ships no acceptance threshold."
    ),
    "learning_objectives": (
        "by the end of this notebook you will be able to —\n\n"
        "1. **Explain** what in-context learning means for TabPFN: what `fit` does and does not do (Section 6).\n"
        "2. **Diagnose** an invalid table from a coded validation finding, and explain why an identifier column must never "
        "be a feature (Sections 4, 5).\n"
        "3. **Compare** TabPFN with a majority-class baseline and a logistic-regression reference on the same rows, using "
        "error counts and the one-row resolution (Section 7).\n"
        "4. **Interpret** accuracy, balanced accuracy, log loss and ROC-AUC, and say why the probabilities are uncalibrated "
        "(Sections 6, 7).\n"
        "5. **Verify** that the exported bundle rebuilds the same probabilities in a fresh process (Section 8).\n"
        "6. **Apply** the bundle to new rows and keep your identifiers beside the predictions (Section 9).\n"
        "7. **Predict**, run and **explain** the effect of the ensemble size in an optional activity (Section 10).\n"
        "8. **Write** an evidence-based conclusion that names the baselines and the limits of one synthetic table "
        "(Conclusion)."
    ),
    "exclusions": (
        "gradient fine-tuning (see **Adaptation** above), regression, calibrated probabilities, the v2 / v2.5 / v2.6 "
        "generations (only the pinned v3 checkpoint is carried), temporal or grouped splitting, or any quality claim "
        "beyond one holdout of one table."
    ),
    "prerequisites": [
        RUNTIME_PREREQ,
        "- **Knowledge:** basic Python and pandas, the train/val/test convention, and how to read a printed dictionary. The metrics and in-context learning are explained where they are first used, and the glossary collects them.",
        "- **Model file:** `tabpfn-v3-classifier-v3_default.ckpt` (203 MiB) plus the licence, README and config of the snapshot, staged at a fixed revision and digest-verified in Section 3.",
        "- **Data:** the default path draws a deterministic synthetic 3-class table (600 rows, 12 numeric features, `train.csv`/`val.csv`/`test.csv`, plus a `record_id` identifier and a `category` column derived from the row index, both dropped from the features) in code and needs no download and no private data. BYOD (one labelled ZIP, `train.csv` or directory) is off by default. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
        LICENCE_PREREQ,
    ],
    "guided": {
        "opening": [
            (
                "## How to use this notebook\n\n"
                "**Who this notebook is for.** Learners who can run cells in a hosted notebook and read short Python and pandas, "
                "and who want to see a tabular foundation model used end to end: validated data, in-context fitting, honest "
                "evaluation against simple references, and a reusable bundle checked across a fresh boundary. No experience with "
                "transformers is assumed; the glossary below explains every term.\n\n"
                "**Running it.** Choose *Runtime → Run all*. The default path needs no edit, no upload, no account, no token and "
                "no runtime restart. Section 2 builds an isolated environment, which takes the longest. You can also run one cell "
                "at a time with *Shift + Enter*.\n\n"
                "**Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell "
                "calls `run_stage('…')`, which runs one stage of the carried stage runner in its own process with the isolated "
                "environment's Python, streams what it prints, and stops the notebook with the stage's own error message if it "
                "fails. Stages hand results to each other only through files.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–10) are the machine-learning workflow. *Infrastructure cells* "
                "(Sections 1–3) are collapsed and titled **Infrastructure**; you may run them without studying their "
                "implementation.\n\n"
                "**Form controls.** `USE_BYOD`, `BYOD_PATH`, `TARGET_COLUMN`, `DROP_COLUMNS`, `TEXT_COLUMNS` and "
                "`VALIDATION_SPLIT` (Section 4); `N_ESTIMATORS` and `SEED` (Section 6); `USE_BYOD_ROWS` and `NEW_DATA_PATH` "
                "(Section 9); `RUN_ACTIVITY` and `ACTIVITY_N_ESTIMATORS` (Section 10). Leave them at their defaults for the first "
                "run.\n\n"
                "**Section tags.** **[Concept]** — what the model does and why. **[Evaluation practice]** — how the evidence is "
                "produced and how to read it. **[Engineering]** — reproducibility, provenance and packaging.\n\n"
                "**Predict, then check.** Before Sections 6 and 7 a **Predict before running** prompt asks you to commit to an "
                "expectation; **What to notice** follows each stage; a collapsed **Check your reasoning** answer follows each "
                "checkpoint."
            ),
            (
                "## The task: Input → Model → Output\n\n"
                "| Stage | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **Validate** | a labelled table (train / val / test) | `validate_inputs` with coded findings; identifiers set aside | an input manifest; 420 support, 90 validation and 90 test rows |\n"
                "| **Fit in context** | the support rows | TabPFN-3 registering them as context (`n_estimators=4`) | class probabilities for any query row |\n"
                "| **Evaluate** | validation and test rows | `classification_metrics`, majority baseline, logistic regression | metrics, error counts, an evaluation report |\n"
                "| **Package** | fitted state + checkpoint + manifest | `save_artifact`, `validate_artifact_bundle`, `from_artifact` | a bundle that rebuilds the same probabilities |\n"
                "| **Score** | new unlabelled rows (+ identifiers) | the rebuilt estimator | `prediction` + `proba_<class>` per row, identifiers kept |\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens | What you read |\n"
                "|---|---|---|---|\n"
                "| 1. Check the runtime | [Engineering] | Linux x86_64, GPU and disk; a run directory | the machine |\n"
                "| 2. Carry the code, build the environment | [Engineering] | carried files verified; an isolated hash-locked environment | versions |\n"
                "| 3. Pin, stage and verify the model | [Engineering] | the 203 MiB checkpoint downloaded at a fixed revision, digest-checked | identity and digest |\n"
                "| 4. Prepare the dataset | [Concept] | the synthetic table (or your files); identifiers set aside | split sizes, class counts |\n"
                "| 5. Validate | [Evaluation practice] | input manifest; a refusal probe; majority baseline | the manifest |\n"
                "| 6. Fit in context and evaluate | [Concept] | validation and test metrics | the metrics |\n"
                "| 7. Baselines and report | [Evaluation practice] | logistic-regression reference, error counts, the evaluation report | the principal result |\n"
                "| 8. Export and fresh reload | [Engineering] | bundle exported, rebuilt in a new process, probabilities compared | the reload check and the digests |\n"
                "| 9. Score new rows | [Engineering] | eight rows (or your file) scored by the rebuilt bundle | the output contract |\n"
                "| 10. Optional activity | [Concept] | a different ensemble size (off by default) | your comparison |\n"
                "| Troubleshooting | [Engineering] | common failures and what to do | when something fails |\n"
                "| Interpretation and conclusion | [Evaluation practice] | limits and an evidence-based conclusion | your conclusion |\n\n"
                "**Fast path.** Run all, then read Sections 6, 7 and 8 and the conclusion."
            ),
            (
                "<details>\n"
                "<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
                "| Term | Meaning in this notebook |\n"
                "|---|---|\n"
                "| **In-context learning (ICL)** | Predicting from labelled rows shown to the model at prediction time; `fit` only registers them. |\n"
                "| **Support set** | The labelled rows the model reads; here the 420-row training split. |\n"
                "| **Ensemble members (`n_estimators`)** | Passes over differently preprocessed views of the table, averaged into one prediction. |\n"
                "| **Identifier column** | A key such as `record_id` that names a row but carries no signal; it must never be a feature. |\n"
                "| **Coded finding** | A validation result with a stable code (`TARGET_MISSING`, `UNSEEN_CLASSES`, …) and the observed value. |\n"
                "| **Majority-class baseline** | Always predict the most frequent training class. |\n"
                "| **Logistic regression** | A linear classical model on the standardised numeric features: the non-trivial reference. |\n"
                "| **One-row resolution** | 1 / (rows in the split): the smallest possible change in accuracy. 90 rows → 0.0111. |\n"
                "| **Accuracy / balanced accuracy** | Correct rows / all rows; balanced accuracy averages per-class recall. |\n"
                "| **Log loss** | Average negative log probability of the true class; lower is better; punishes confident mistakes. |\n"
                "| **ROC-AUC (one-vs-rest)** | Ranking quality, threshold-free, averaged over one-class-vs-rest problems. |\n"
                "| **Uncalibrated probability** | A score not guaranteed to match observed frequencies. |\n"
                "| **Fitted archive (`model.tabpfn_fit`)** | The fitted estimator state without the foundation weights. |\n"
                "| **Foundation checkpoint (`model.ckpt`)** | A byte copy of the pinned TabPFN-3 checkpoint, bound by its SHA-256. |\n"
                "| **Digest (SHA-256)** | A fingerprint of a file's bytes. |\n"
                "| **Hash-locked environment / stage** | The isolated Python environment every stage runs in; one workflow step run as its own process. |\n"
                "| **BYOD** | Bring Your Own Data. |\n\n"
                "</details>"
            ),
        ],
    },
    "cells": [
        {
            "md": (
                "## 4. Prepare the dataset: the synthetic sample or your own · [Concept]\n\n"
                "From here on, every code cell runs one stage of the carried runner with `run_stage`. The expected input is one "
                "`train.csv` with a declared categorical target column, plus optional `val.csv` and `test.csv` with identical "
                "columns (explicit splits are preserved, never re-split). With `USE_BYOD = False` the carried package draws the "
                "deterministic synthetic sample (`build_synthetic_dataset`: scikit-learn `make_classification`, seed 42). With "
                "`USE_BYOD = True` the stage reads `BYOD_PATH` — a ZIP (member by member with the archive-safety rules: bare file "
                "names, expanded-size and compression-ratio ceilings, never `extractall`), a single `train.csv`, or a directory — "
                "or, in Colab with an empty path, the file you choose in the upload dialog. Without a `val.csv`, "
                "`stratified_holdout` draws a seeded holdout of `VALIDATION_SPLIT` — correct only for independent rows.\n\n"
                "**Identifiers are not features.** The synthetic table carries `record_id` (a unique row key) and `category` "
                "(derived from the row index, so it carries no real signal). `DROP_COLUMNS` (default `['record_id', 'category']`) "
                "keeps them out of the model and beside the predictions in Section 9. A model given a unique key can memorise it; "
                "on new rows the key is meaningless.\n\n"
                "The stage validates the table **before** any split, so a renamed target, duplicate columns or a reserved column "
                "stop here with a coded finding (`TARGET_MISSING`, …) naming the file and the observed value — not a raw "
                "`KeyError` from the splitter. A column that is numeric except for a few stray strings is refused naming the "
                "column and the values (list it in `TEXT_COLUMNS` if it really is categorical). The stage first removes this "
                "notebook's earlier exports from `outputs/`."
            ),
            "code": (
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "TARGET_COLUMN = 'target'  # @param {{type:\"string\"}}\n"
                "DROP_COLUMNS = ['record_id', 'category']  # @param {{type:\"raw\"}}\n"
                "TEXT_COLUMNS = []  # @param {{type:\"raw\"}}\n"
                "VALIDATION_SPLIT = 0.2  # @param {{type:\"number\"}}\n\n"
                "def upload_one(what, field):\n"
                "    try:\n"
                "        from google.colab import files\n"
                "    except ImportError:\n"
                "        raise RuntimeError(f'{{field}} is empty, and the upload dialog exists only in Google Colab: set {{field}} to {{what}} in this runtime.') from None\n"
                "    uploaded = files.upload()\n"
                "    if not uploaded:\n"
                "        raise RuntimeError(f'The upload was cancelled or empty: no file was received. Run this cell again and choose {{what}}, or set {{field}}.')\n"
                "    if len(uploaded) != 1:\n"
                "        raise ValueError(f'Upload exactly one file ({{what}}); got {{sorted(uploaded)}}.')\n"
                "    upload_name, payload = next(iter(uploaded.items()))\n"
                "    path = ROOT / 'inputs' / Path(upload_name).name\n"
                "    path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    path.write_bytes(payload)\n"
                "    return str(path)\n\n"
                "byod_path = ''\n"
                "if USE_BYOD:\n"
                "    byod_path = BYOD_PATH or upload_one('one dataset ZIP or train.csv', 'BYOD_PATH')\n"
                "run_stage('data', use_byod=USE_BYOD, byod_path=byod_path, target_column=TARGET_COLUMN, drop_columns=DROP_COLUMNS, text_columns=TEXT_COLUMNS, validation_split=VALIDATION_SPLIT)"
            ),
        },
        {
            "md": (
                "**What to notice:** `sample_kind: 'synthetic'`, 420 / 90 / 90 rows, balanced class counts (139 / 142 / 139 in "
                "train), the dataset digest, and `drop_columns: ['record_id', 'category']`."
            ),
        },
        {
            "md": (
                "## 5. Validate the inputs → input manifest · [Evaluation practice]\n\n"
                "`validate_inputs` is the package's public validation stage: unique column names, the target present with no "
                "missing values, at least `MIN_CLASSES` classes with `MIN_ROWS_PER_CLASS` rows each, at most `MAX_FEATURES` "
                "features and `MAX_TRAIN_ROWS` rows, finite numeric features, identical schemas across splits. It returns an "
                "**input manifest** (schema, observed classes and counts, numeric/categorical feature counts, split sizes, the "
                "train-table digest, findings), written to `outputs/{stem}_input_manifest.json`. A class that appears in "
                "`val.csv` or `test.csv` but not in `train.csv` can never be predicted and would break log loss later, so it is "
                "**refused** here as `UNSEEN_CLASSES`, naming the split and the classes. To show what rejection looks like, the "
                "stage validates a probe with the target renamed and records the package's own `TARGET_MISSING` finding. The "
                "majority-class baseline is computed on the validation rows."
            ),
            "code": "run_stage('validate')",
        },
        {
            "md": (
                "**What to notice:** 12 numeric features and no `record_id` among them; three classes; the `renamed-target-probe` "
                "finding with code `TARGET_MISSING`; and the majority-class baseline — accuracy 0.3333 on a balanced table.\n\n"
                "**Checkpoint:** why is `record_id` kept out of the features even though the model might score well with it?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "A unique key identifies each training row exactly, so a flexible model can associate it with that row's label — "
                "a pattern that cannot exist for a new row, whose key it has never seen. At best the key is noise; at worst it "
                "leaks information (keys assigned in label order, by date, or by site) and inflates validation scores that will "
                "not survive deployment. Keeping it beside the predictions lets you join results back without ever showing it to "
                "the model.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 6. Fit in context and evaluate · [Concept]\n\n"
                "`fit` registers the 420 support rows as context — no gradient update, the pinned checkpoint is unchanged — and "
                "the stage checks that the estimator's class order equals the validated class list. `evaluate` then scores the "
                "validation and test rows with **accuracy**, **balanced accuracy**, **log loss** and **ROC-AUC** (one-vs-rest), "
                "computed by the package's `classification_metrics` in the fitted class order. Ensemble averaging is set by "
                "`N_ESTIMATORS` (default 4) and the preprocessing randomness by `SEED`.\n\n"
                "**Predict before running:** the majority baseline is 0.33 and a logistic regression reaches 0.91 on the validation "
                "rows (Section 7). Where will TabPFN land — and how many of the 90 rows will it get wrong?"
            ),
            "code": (
                "N_ESTIMATORS = 4  # @param {{type:\"integer\"}}\n"
                "SEED = 42  # @param {{type:\"integer\"}}\n"
                "run_stage('condition', n_estimators=N_ESTIMATORS, seed=SEED)"
            ),
        },
        {
            "md": (
                "**What to notice:** `mode: zero-shot-icl`, the device, the classes, and the validation and test metrics; the "
                "printed adaptation note. No recorded run of this revision exists yet, so read your own numbers against Section 7.\n\n"
                "**Checkpoint:** `fit` took seconds and changed no weight. What, then, did the model learn from your rows?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Nothing was learned in the gradient sense. TabPFN was trained once, on synthetic tasks, to behave like a learning "
                "algorithm: given labelled rows and a query in the same forward pass, it outputs the class distribution a good "
                "learner would. `fit` stores and preprocesses your rows so each prediction can attend to them. That is why the "
                "bundle must carry the fitted state (which holds the support rows) and why quality depends entirely on them.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 7. Baselines → evaluation report · [Evaluation practice]\n\n"
                "Two references on the same rows: the **majority-class baseline** (always the most frequent training class; log "
                "loss from the training frequencies) and a **standardised logistic regression** on the numeric features — a simple "
                "classical model that is already strong on a synthetic linear-ish table. Each is reported with its validation "
                "error count beside the one-row resolution (1/90 = 0.0111). `evaluation_report` is the package's public evaluation "
                "stage: verdict `sample-sanity` (one holdout of one synthetic table, no dispersion estimate), the metric entries, "
                "the baselines, the caveats and the proposed RUN7 deviation, written to `outputs/{stem}_evaluation_report.json`.\n\n"
                "**Predict before running:** if TabPFN and the logistic regression differ by two validation rows, is that a "
                "difference you would report?"
            ),
            "code": "run_stage('report')",
        },
        {
            "md": (
                "**What to notice:** `validation_errors` — the majority baseline gets 60 of 90 wrong, the logistic regression 8 "
                "(validation accuracy 0.9111; test 0.9444, 5 errors), TabPFN from your run; and the interpretation line.\n\n"
                "**Checkpoint:** suppose TabPFN gets 5 validation rows wrong and the logistic regression 8. What can you claim?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Three rows out of 90 is 0.033 of accuracy on one split of one synthetic table. Without repeated splits there is no "
                "dispersion estimate, and the test split is the place to check whether the gap holds. The honest claim is that "
                "TabPFN is at least as good as a strong linear reference here, without tuning — not that it is better in general. "
                "Read balanced accuracy and log loss too: a lower log loss means more confident correct probabilities.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 8. Export the artifact bundle and verify it in a fresh process · [Engineering]\n\n"
                "`save_artifact` writes `model.tabpfn_fit` (the fitted estimator state, without the foundation weights), "
                "`model.ckpt` (a byte copy of the verified checkpoint) and `artifact_manifest.json` (target and feature columns, "
                "classes, both SHA-256 digests, the base-model identity, the ensemble settings, and the feature kinds the "
                "companion uses to refuse text in numeric columns); `zip_artifact_bundle` writes "
                "`outputs/{stem}_artifact.zip`. The `export` stage prints the ZIP and fitted-archive digests — the **trusted "
                "digests** to give the companion notebook.\n\n"
                "The `reload` stage runs in a **fresh process**: it clears and fills `outputs/artifact-reload/`, checks the bundle "
                "with `validate_artifact_bundle` (manifest schema, member names, sizes, digests, the checkpoint equal to the pinned "
                "one), rebuilds the estimator with `from_artifact` (no refit, no download), and requires its probabilities on "
                "**every** validation row to equal the exporting process's within `rtol=1e-5`, `atol=1e-6`, recording the largest "
                "difference. Comparing probabilities is a much stronger check than comparing accuracy, which on 90 rows moves "
                "only in steps of 0.011.\n\n"
                "**Predict before running:** if the reload produced the same labels but probabilities differing by 0.01, would "
                "this check pass?"
            ),
            "code": "run_stage('export')\nrun_stage('reload')",
        },
        {
            "md": (
                "**What to notice:** the three bundle members, the two digests and the ZIP digest; then `maxAbsProbabilityDifference` "
                "(0 or a few 1e-8 on the same device), `labelsIdentical: True` and the `PASSED` line. Re-running this cell alone "
                "works: the reload directory is cleared first.\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "No. A 0.01 difference is far above `atol=1e-6`, so the check fails even though every label agrees — exactly the "
                "silent drift an accuracy-only comparison would miss. A bundle that cannot reproduce its own probabilities should "
                "not be shipped.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 9. Score new rows with the rebuilt bundle · [Engineering]\n\n"
                "By default the `predict` stage takes the first eight rows of `test.csv` (held out from the support), removes the "
                "target, writes them to `outputs/{stem}_new_rows.csv` — the companion notebook's input — and scores them with the "
                "estimator rebuilt from the bundle. Tick `USE_BYOD_ROWS` and set `NEW_DATA_PATH` (or, in Colab, leave it empty to "
                "upload) to score your own unlabelled CSV. `validate_new_rows` requires exactly the fitted feature columns (any "
                "order) and no target, `prediction` or `proba_<class>` column; `DROP_COLUMNS` present in the rows are kept beside "
                "the predictions, so the output joins back on your own key. The output adds `prediction` (the `argmax` label) and "
                "one `proba_<class>` column per class **in the fitted class order**; the probabilities are uncalibrated and no "
                "threshold is shipped. `outputs/{stem}_result.json` records predictions, metrics, baselines, the reload check, the "
                "input manifest, the dataset and bundle digests, the notebook's source, the model identity and licence, and the "
                "runtime."
            ),
            "code": (
                "USE_BYOD_ROWS = False  # @param {{type:\"boolean\"}}\n"
                "NEW_DATA_PATH = ''  # @param {{type:\"string\"}}\n\n"
                "new_data_file = ''\n"
                "if USE_BYOD_ROWS:\n"
                "    new_data_file = NEW_DATA_PATH or upload_one('one unlabelled CSV', 'NEW_DATA_PATH')\n"
                "run_stage('predict', new_data_path=new_data_file)"
            ),
        },
        {
            "md": (
                "**What to notice:** eight rows with `record_id` and `category` first, then `prediction` and three `proba_class_*` "
                "columns summing to 1 per row."
            ),
        },
        {
            "md": (
                "## 10. Optional activity: how much does the ensemble size matter? · [Concept]\n\n"
                "**Predict → Change → Run → Observe → Explain.** **Predict:** with `ACTIVITY_N_ESTIMATORS = 1` instead of 4, will "
                "validation accuracy drop by more than one row (0.0111), and how far will the probabilities move? **Change:** tick "
                "`RUN_ACTIVITY` (try 1, then 8). **Run** this cell. **Observe** the canonical and changed validation metrics, "
                "`labels_changed` and `max_abs_probability_change`. **Explain** what the ensemble buys. The activity refits in "
                "context, writes only to `outputs/activity/`, and stops if any canonical output changed."
            ),
            "code": (
                "RUN_ACTIVITY = False  # @param {{type:\"boolean\"}}\n"
                "ACTIVITY_N_ESTIMATORS = 1  # @param {{type:\"integer\"}}\n"
                "if RUN_ACTIVITY:\n"
                "    run_stage('activity', n_estimators=ACTIVITY_N_ESTIMATORS)\n"
                "else:\n"
                "    print('Optional activity skipped: tick RUN_ACTIVITY to run it. The canonical outputs are complete.')"
            ),
        },
        {
            "md": (
                "**What to notice (if you ran it):** the two settings, both metric sets, `labels_changed` and the largest "
                "probability change; `canonical_outputs_unchanged: True`.\n\n"
                "<details>\n<summary>Check your reasoning (open after running)</summary>\n\n"
                "Each ensemble member sees the table through a different preprocessing and feature order; averaging them steadies "
                "the probabilities. Going from 4 to 1 usually moves log loss more than accuracy: the labels of confident rows stay, "
                "the borderline rows can flip. On 90 rows a change of one or two labels is within the one-row resolution, so the "
                "honest reading is about the probabilities, not the accuracy.\n\n"
                "</details>"
            ),
        },
    ],
    "closing": (
        "## Troubleshooting · [Engineering]\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| Section 1 stops with `This notebook needs a Linux x86_64 runtime` | a local Windows or macOS kernel, or an ARM machine | Use Google Colab, Kaggle, or a Linux x86_64 Jupyter kernel. |\n"
        "| `Not enough free disk` | the isolated environment needs about 8 GB | Start a fresh runtime; an environment built from the same lock is reused. |\n"
        "| `Carried file integrity failure` | a carried file was edited in the notebook | Open a fresh copy from the repository. |\n"
        "| `uv … mismatch`, `URLError`, or `CalledProcessError` from `uv` | network or a transient PyPI error | Re-run the Section 2 install cell. Never remove a pin or a hash. |\n"
        "| `The run directory … has no carried files, or the isolated environment is gone` | Section 1 run with `NEW_RUN_DIRECTORY` ticked | Run Sections 1–3 again, or *Run all*. |\n"
        "| `RuntimeError: Stage '…' failed (exit 1): …` | the stage's own error follows the colon | Find it below; fix the cause and re-run from that cell. |\n"
        "| A Hub download error in Section 3, or `… sha256 … != manifest` | a transient failure or a corrupted download | Re-run Section 3; delete the partial file under `weights/` if the digest fails. Never edit the manifest. |\n"
        "| `[TARGET_MISSING] … target column … not present` | your label column has another name | Set `TARGET_COLUMN`. |\n"
        "| `[UNSEEN_CLASSES] classes absent from train.csv` | a class only in `val.csv` / `test.csv` | Re-split so every class is in `train.csv`, or remove those rows. |\n"
        "| `[DUPLICATE_COLUMNS]`, `[RESERVED_COLUMNS]`, `[SCHEMA_MISMATCH]`, `[RARE_CLASSES]`, `[TOO_FEW_CLASSES]` | the table breaks the input contract | Fix the named columns or classes. |\n"
        "| `column … is numeric except for N value(s)` | stray text in a numeric column | Fix the values, or list the column in `TEXT_COLUMNS`. |\n"
        "| `DROP_COLUMNS … are not in the header` | a typo in `DROP_COLUMNS` | Use the exact column names. |\n"
        "| `BYOD_PATH … does not exist`, `BYOD_PATH is empty, and the upload dialog exists only in Google Colab`, `The upload was cancelled or empty` | no file supplied | Set the path, or run the cell again and choose a file. |\n"
        "| `The reloaded artifact does not reproduce the exporting process's probabilities` | a corrupted bundle or a changed runtime | Re-run Sections 6–8; do not ship the bundle. |\n"
        "| `new rows: [SCHEMA_MISMATCH]` | your rows lack a feature or carry an unknown column | Supply exactly the fitted features; identifiers listed in `DROP_COLUMNS` are allowed. |\n\n"
        "## Interpretation and limits\n\n"
        "The predicted class is `argmax` over TabPFN's class probabilities, which are raw ensemble outputs, not calibrated, "
        "with no shipped threshold. The evaluation report's `sample-sanity` verdict names what it is: one holdout of one "
        "synthetic table with no dispersion estimate — a plumbing check that must not be generalised. On this table a "
        "logistic regression already reaches 0.911 (validation) and 0.944 (test), and one row is 0.011 of accuracy, so "
        "differences of a few rows rank nothing. Random stratified splitting assumes independent rows; temporal, grouped or "
        "patient-level data need leakage-safe splits you supply. In-context learning is not fine-tuning: quality depends "
        "entirely on the support rows, and the bundle carries them. Digest equality proves the checkpoint bytes are the ones "
        "pinned at the immutable revision; it does not by itself prove who published them.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can acquire "
        "and digest-verify the pinned TabPFN-3 checkpoint, validate the demonstrated table into an input manifest with coded "
        "refusals, fit in context, compute sample metrics against a trivial and a classical baseline, export the artifact "
        "bundle, rebuild equivalent probabilities from it in a fresh process, score new rows with identifiers kept, and emit "
        "the shown machine-readable outputs — without the repository being reachable. It does **not** establish benchmark "
        "superiority, generalisation, fairness, robustness, calibration, safety for high-consequence decisions, production "
        "fitness, or anything about fine-tuned TabPFN models.\n\n"
        "## Conclusion · [Evaluation practice]\n\n"
        "Write three to five sentences, using the numbers your run printed:\n\n"
        "1. **Result:** TabPFN's validation and test accuracy, log loss and error count beside the majority baseline and the "
        "logistic regression.\n"
        "2. **Reading:** whether the gap exceeds a few rows, and what the one-row resolution allows you to claim.\n"
        "3. **Reuse:** what the Section 8 reload proved, and why probabilities were compared rather than accuracy.\n"
        "4. **Limits:** the one limitation you would fix first (for example repeated splits, real data, calibration).\n\n"
        "<details>\n<summary>Sample conclusion (open after writing yours)</summary>\n\n"
        "On the 90-row validation and 90-row test splits of the synthetic three-class table, TabPFN-3 conditioned in context "
        "on 420 rows scored the accuracy and log loss printed in Sections 6 and 7, against 0.33 for the majority class and "
        "0.911 / 0.944 for a standardised logistic regression. Any gap between TabPFN and the logistic regression is a few "
        "rows at most on one split, so the run shows that TabPFN matches a strong simple reference without tuning, not that "
        "it is better. The exported bundle rebuilt identical probabilities in a fresh process, a stronger guarantee than "
        "matching accuracy. Before any use I would evaluate on repeated splits of a real, domain-representative table and "
        "calibrate the probabilities — and clear the non-commercial licence.\n\n"
        "</details>\n\n"
        "**Next experiments:** run the activity with 1 and 8 estimators and compare log loss; enable `USE_BYOD` with a small "
        "table of your own and read balanced accuracy against both references; drop `val.csv` from your ZIP and watch the "
        "seeded holdout take over; hand `outputs/{stem}_artifact.zip`, its printed digests and `outputs/{stem}_new_rows.csv` "
        "to the companion artifact-inference notebook in a fresh session.\n\n"
        "## References\n\n"
        f"- Repository README: https://github.com/kurtvalcorza/{REPO}/blob/main/README.md\n"
        f"- Repository model card: https://github.com/kurtvalcorza/{REPO}/blob/main/MODEL_CARD.md\n"
        f"- Weight provenance: https://github.com/kurtvalcorza/{REPO}/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/PriorLabs/TabPFN\n"
        "- TabPFN-3 technical report: https://arxiv.org/abs/2605.13986"
    ),
}
