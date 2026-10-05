"""Companion template for tools/build_notebook.py /3 — ARTIFACT-INFERENCE (NOTEBOOK_SPEC 2.2 §19, §25.13).

Generate with ``python tools/build_notebook.py --template tools/notebook_template_artifact_inference.py``. The notebook
carries the same package and pinned checkpoint manifest as the E2E notebook and its own stage runner
(``tools/tutorial_stages_artifact_inference.py``). When ``examples/sample-bundle/`` exists (manifest, fitted archive, eight
unlabelled rows, produced from a recorded hosted E2E run by ``tools/build_sample_bundle.py``), it is carried too and the
default path uses it with the checkpoint verified in Section 3. A user bundle ZIP (``ARTIFACT_ZIP_PATH`` or an upload) is
checked against ``EXPECTED_ZIP_SHA256`` / ``EXPECTED_FITTED_SHA256``. The notebook never creates a bundle.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_e2e_notebook_template", Path(__file__).with_name("notebook_template.py"))
assert _spec and _spec.loader
_e2e_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e2e_module)
BADGES, REPO, ENVIRONMENT, RUNTIME_PREREQ, LICENCE_PREREQ = _e2e_module.BADGES, _e2e_module.REPO, _e2e_module.ENVIRONMENT, _e2e_module.RUNTIME_PREREQ, _e2e_module.LICENCE_PREREQ
_SAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "sample-bundle"
SAMPLE = json.loads((_SAMPLE_DIR / "SAMPLE_BUNDLE.json").read_text(encoding="utf-8")) if (_SAMPLE_DIR / "SAMPLE_BUNDLE.json").is_file() else None

if SAMPLE:
    _CARRIED_EXTRA = {f"sample-bundle/{n}": f"examples/sample-bundle/{n}" for n in ("artifact_manifest.json", "new_rows.csv", "SAMPLE_BUNDLE.json")}
    _CARRIED_BINARY = {"sample-bundle/model.tabpfn_fit": "examples/sample-bundle/model.tabpfn_fit"}
    _DEFAULT_PATH = (
        "assembles the carried **sample bundle** (`examples/sample-bundle/`, produced from a recorded E2E run; its files are checked against `SAMPLE_BUNDLE.json`) with the checkpoint verified in Section 3 as its `model.ckpt`; validates the bundle; rebuilds the estimator from it without refitting; validates the eight carried unlabelled sample rows into an input manifest; predicts with class probabilities, reports what cannot be measured, and exports outputs. No upload dialog is opened."
    )
    _SAMPLE_PREREQ = "- **Bundle:** by default the carried sample bundle (`examples/sample-bundle/`: `artifact_manifest.json`, `model.tabpfn_fit`; produced from a recorded hosted run of the E2E notebook by `tools/build_sample_bundle.py`) plus the verified checkpoint as its `model.ckpt`; optionally your own ZIP exported by the E2E notebook, by `ARTIFACT_ZIP_PATH` or upload, with the digests it printed."
else:
    _CARRIED_EXTRA, _CARRIED_BINARY = {}, {}
    _DEFAULT_PATH = (
        "stops in Section 4 with a message naming `ARTIFACT_ZIP_PATH`: **this revision carries no pinned sample bundle yet** (a TabPFN fitted archive can only be produced by a real TabPFN fit, and none has been recorded; `tools/build_sample_bundle.py` adds one from a hosted E2E run). With `ARTIFACT_ZIP_PATH` (and `NEW_DATA_PATH`) set to the E2E notebook's exports it validates the bundle, rebuilds the estimator without refitting, validates the rows, predicts with class probabilities, reports what cannot be measured, and exports outputs. No upload dialog opens unless you tick `UPLOAD_ARTIFACT` / `UPLOAD_NEW_DATA`."
    )
    _SAMPLE_PREREQ = "- **Bundle:** the ZIP the E2E notebook exported (`outputs/tabpfn_classifier_artifact.zip`), by `ARTIFACT_ZIP_PATH` or upload, with the digests it printed. No pinned sample bundle is carried by this revision yet (see Run all)."

TEMPLATE = {
    **ENVIRONMENT,
    "stem": "tabpfn_classifier_artifact_inference",
    "notebook_name": "tabpfn_classifier_artifact_inference_colab.ipynb",
    "profile": "ARTIFACT-INFERENCE",
    "mode": "GUIDED",
    "stage_runner": "tools/tutorial_stages_artifact_inference.py",
    "carried_extra": _CARRIED_EXTRA,
    "carried_binary": _CARRIED_BINARY,
    "run_all": (
        "Selecting **Run all** in a fresh Linux x86_64 runtime builds an isolated Python environment from the carried hash-locked requirements without touching the notebook kernel's own packages, stages and digest-verifies the pinned TabPFN-3 checkpoint, then " + _DEFAULT_PATH + " No repository clone, DIMER worker or service, credential, configuration edit or runtime restart is required (NOTEBOOK_SPEC 2.2 §5, §19). No hosted run of this revision has been recorded yet."
    ),
    "byod": (
        "Your own bundle is the `ARTIFACT_ZIP_PATH` / `UPLOAD_ARTIFACT` branch in Section 4: paste the ZIP and fitted-archive digests the E2E notebook printed into `EXPECTED_ZIP_SHA256` / `EXPECTED_FITTED_SHA256`, and any other file is refused before it is loaded. Your own unlabelled rows are the `NEW_DATA_PATH` / `UPLOAD_NEW_DATA` branch in Section 6 (the bundle's feature columns, plus identifier columns you list in `ID_COLUMNS`, which are kept beside the predictions). Paths work in Colab, Kaggle and Jupyter; the upload dialogs exist only in Colab. Uploads stay inside this runtime; do not upload confidential or restricted data unless you are authorised to process it here."
    ),
    "title": "TabPFN-3 Classifier — DIMER artifact inference tutorial (standalone)",
    "badges": [
        badge
        if badge[0] != "Open In Colab"
        else (
            badge[0],
            badge[1],
            f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/tabpfn_classifier_artifact_inference_colab.ipynb",
        )
        for badge in BADGES
    ],
    "capability": "serving-state reconstruction from an externally produced TabPFN-3 classifier bundle (`artifact_manifest.json` + `model.tabpfn_fit` + `model.ckpt`) and class-probability inference on genuinely new rows under an explicit `argmax` rule",
    "intro": (
        "This notebook consumes a bundle produced **outside this execution** — the ZIP the E2E tutorial exports, from a "
        "separate session. It checks the ZIP against trusted digests, refuses any member other than the three expected "
        "ones before extracting, validates the manifest schema, sizes and digests and requires `model.ckpt` to be the pinned "
        "checkpoint verified in Section 3, rebuilds the fitted estimator with `from_artifact` (no refit, no download), "
        "accepts genuinely new unlabelled rows, predicts class labels with class probabilities, and exports results. **No "
        "artifact is created here** and no fitting happens.\n\n"
        "**Trust boundary.** `model.tabpfn_fit` holds joblib-serialised estimator state, which is code-capable, and "
        "`model.ckpt` is a PyTorch checkpoint. The checkpoint is bound by its digest to the published TabPFN-3 file. The "
        "fitted archive has no public reference digest: the only binding is a trusted digest received from the producer "
        "through a separate channel (`EXPECTED_FITTED_SHA256`, or `EXPECTED_ZIP_SHA256` for the whole ZIP). Digests inside "
        "`artifact_manifest.json` establish internal consistency, not sender authenticity. Use only bundles from a trusted "
        "producer."
    ),
    "learning_objectives": (
        "by the end of this notebook you will be able to —\n\n"
        "1. **Explain** what a TabPFN bundle contains and why the fitted archive travels with the checkpoint (Sections 4, 5).\n"
        "2. **Verify** a bundle against trusted digests and **distinguish** what binds the checkpoint from what binds the fitted archive (Section 4).\n"
        "3. **Diagnose** a refused bundle or input table from its message (Sections 4, 6).\n"
        "4. **Apply** the rebuilt estimator to new rows, keep your identifiers, and read class-ordered, uncalibrated probabilities (Section 7).\n"
        "5. **Explain** why the evaluation report says `not-measurable` here (Section 7).\n"
        "6. **Predict**, run and **explain** which check refuses a tampered bundle, in an optional activity (Section 8)."
    ),
    "exclusions": (
        "artifact creation, in-notebook fitting, fine-tuning, regression, calibrated probabilities, or any quality claim: "
        "without labelled rows nothing is measured, and the exported probabilities are uncalibrated ensemble outputs with "
        "no shipped threshold."
    ),
    "prerequisites": [
        RUNTIME_PREREQ,
        "- **Knowledge:** basic pandas and how to read a printed Python dictionary. The E2E notebook explains in-context learning and the metrics; this notebook's glossary repeats the terms it uses.",
        _SAMPLE_PREREQ,
        "- **Data:** one unlabelled CSV with the bundle's feature columns (the E2E notebook writes `outputs/tabpfn_classifier_new_rows.csv`), by `NEW_DATA_PATH` or upload; extra identifier columns are allowed when listed in `ID_COLUMNS`. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
        LICENCE_PREREQ,
    ],
    "guided": {
        "opening": [
            (
                "## How to use this notebook\n\n"
                "**Who this notebook is for.** Learners who have run, or read, the E2E TabPFN tutorial and want to see how a "
                "packaged in-context classifier is reused safely by someone else: checking what was received, rebuilding it, and "
                "scoring new rows. You need to be able to run notebook cells and read short Python; the glossary explains every "
                "term.\n\n"
                "**Running it.** Run the E2E notebook first and keep its `outputs/tabpfn_classifier_artifact.zip`, the digests it "
                "printed and `outputs/tabpfn_classifier_new_rows.csv`. Put them in this runtime, set `ARTIFACT_ZIP_PATH`, the two "
                "digests and `NEW_DATA_PATH`, then *Runtime → Run all*. No account, token or runtime restart is needed.\n\n"
                "**Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell "
                "calls `run_stage('…')`, which runs one stage of the carried stage runner in its own process and stops the "
                "notebook with the stage's own error message if it fails.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–8) are the workflow. *Infrastructure cells* (Sections 1–3) are "
                "collapsed and titled **Infrastructure**; you may run them without studying their implementation.\n\n"
                "**Form controls.** `ARTIFACT_ZIP_PATH`, `UPLOAD_ARTIFACT`, `EXPECTED_ZIP_SHA256` and `EXPECTED_FITTED_SHA256` "
                "(Section 4); `NEW_DATA_PATH`, `UPLOAD_NEW_DATA` and `ID_COLUMNS` (Section 6); `RUN_ACTIVITY` and `TAMPER` "
                "(Section 8).\n\n"
                "**Section tags.** **[Concept]**, **[Evaluation practice]**, **[Engineering]** as in the E2E notebook.\n\n"
                "**Predict, then check.** Before Sections 4 and 7 a **Predict before running** prompt asks you to commit to an "
                "expectation; **What to notice** follows each stage; a collapsed **Check your reasoning** answer follows each "
                "checkpoint."
            ),
            (
                "## The task: Input → Model → Output\n\n"
                "| Stage | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **Verify** | a bundle ZIP with trusted digests | digest checks, member allowlist, `safe_extract_zip`, `validate_artifact_bundle`, checkpoint = pinned | accepted or refused, with the pinned digests printed |\n"
                "| **Rebuild** | the fitted archive + the verified checkpoint | `from_artifact` (no refit, no download) | an estimator with the producer's class order |\n"
                "| **Validate rows** | unlabelled rows with the bundle's features (+ `ID_COLUMNS`) | `validate_new_rows`, numeric-type check | an input manifest; refusals name the column and rule |\n"
                "| **Predict** | the validated rows | TabPFN-3 reading the bundle's fitted support | `prediction` + `proba_<class>` (stored class order), identifiers kept, a `not-measurable` report |\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens | What you read |\n"
                "|---|---|---|---|\n"
                "| 1–3 | [Engineering] | runtime, carried code, isolated environment, verified checkpoint | versions, digest |\n"
                "| 4. Verify the bundle | [Engineering] | trusted digests, member allowlist, extraction, provenance | the trust record |\n"
                "| 5. Rebuild the estimator | [Concept] | `from_artifact` on the verified files | classes, device |\n"
                "| 6. Validate the new rows | [Evaluation practice] | input manifest, refusal probe, numeric check | the manifest |\n"
                "| 7. Predict and export | [Concept] | probabilities, labels, `not-measurable` report, provenance | the outputs |\n"
                "| 8. Optional activity | [Concept] | tamper with a copy of the bundle (off by default) | which check refuses it |\n"
                "| Troubleshooting | [Engineering] | every refusal and what to do | when something fails |\n"
                "| Interpretation and conclusion | [Evaluation practice] | what was and was not shown | your conclusion |"
            ),
            (
                "<details>\n"
                "<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
                "| Term | Meaning in this notebook |\n"
                "|---|---|\n"
                "| **Bundle** | `artifact_manifest.json` + `model.tabpfn_fit` + `model.ckpt`. |\n"
                "| **Fitted archive** | The fitted estimator state (preprocessing and support rows), without the foundation weights. |\n"
                "| **Foundation checkpoint** | A byte copy of the pinned TabPFN-3 checkpoint; it must equal the file Section 3 verified. |\n"
                "| **Trusted digest** | A SHA-256 obtained from the producer through a channel you trust (the E2E export prints them). |\n"
                "| **Internal consistency** | The manifest's recorded digests match its files; a forger who rewrites both still passes. |\n"
                "| **Code-capable file** | A file whose loading can execute code (joblib, PyTorch); load only verified or trusted files. |\n"
                "| **Member allowlist** | The ZIP may hold exactly the three bundle files, each once, at the top level. |\n"
                "| **Identifier column (`ID_COLUMNS`)** | A key kept beside the predictions and never given to the model. |\n"
                "| **`argmax` rule / uncalibrated** | Predict the most probable class; the probabilities are not guaranteed to match observed frequencies. |\n"
                "| **`not-measurable`** | The evaluation verdict when no labels exist. |\n"
                "| **Hash-locked environment / stage** | The isolated Python environment every stage runs in; one workflow step run as its own process. |\n\n"
                "</details>"
            ),
        ],
    },
    "cells": [
        {
            "md": (
                "## 4. Verify the bundle before any model state is loaded · [Engineering]\n\n"
                "The `artifact` stage takes the bundle from `ARTIFACT_ZIP_PATH` (a ZIP already in the runtime), from the upload "
                "dialog (`UPLOAD_ARTIFACT`, Colab), or — with both empty — from the carried pinned sample bundle when this "
                "revision carries one; otherwise it stops with a message naming `ARTIFACT_ZIP_PATH`. For a ZIP it first checks "
                "`EXPECTED_ZIP_SHA256`, then reads the member list and **refuses** anything but `artifact_manifest.json`, "
                "`model.tabpfn_fit` and `model.ckpt`, each once at the top level (an extra `notes.txt`, or a `sub/model.ckpt` that "
                "flattening would let overwrite `model.ckpt`), and only then extracts member by member with `safe_extract_zip`. "
                "`validate_artifact_bundle` checks the manifest schema, the member names and sizes, both digests (the fitted one "
                "against `EXPECTED_FITTED_SHA256` when given) and requires `model.ckpt` to be the pinned checkpoint. The stage "
                "prints which digests were pinned and which were only taken from the manifest.\n\n"
                "**Predict before running:** if you leave both digest fields empty, which of the three files is still bound to "
                "something outside the bundle?"
            ),
            "code": (
                "ARTIFACT_ZIP_PATH = ''  # @param {{type:\"string\"}}\n"
                "UPLOAD_ARTIFACT = False  # @param {{type:\"boolean\"}}\n"
                "EXPECTED_ZIP_SHA256 = ''  # @param {{type:\"string\"}}\n"
                "EXPECTED_FITTED_SHA256 = ''  # @param {{type:\"string\"}}\n\n"
                "def upload_one(what, field):\n"
                "    try:\n"
                "        from google.colab import files\n"
                "    except ImportError:\n"
                "        raise RuntimeError(f'The upload dialog exists only in Google Colab: set {{field}} to {{what}} in this runtime.') from None\n"
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
                "artifact_source, zip_path = 'sample', ''\n"
                "if ARTIFACT_ZIP_PATH:\n"
                "    artifact_source, zip_path = 'path', ARTIFACT_ZIP_PATH\n"
                "elif UPLOAD_ARTIFACT:\n"
                "    artifact_source, zip_path = 'upload', upload_one('the bundle ZIP', 'ARTIFACT_ZIP_PATH')\n"
                "run_stage('artifact', source=artifact_source, zip_path=zip_path, expected_zip_sha256=EXPECTED_ZIP_SHA256, expected_fitted_sha256=EXPECTED_FITTED_SHA256)"
            ),
        },
        {
            "md": (
                "**What to notice:** `trusted_digest` (which fields were pinned), the members, the fitted-archive and checkpoint "
                "digests, and the manifest summary — 12 features, three classes, `mode: zero-shot-icl`.\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "`model.ckpt` is still bound: it must equal the pinned TabPFN-3 checkpoint, verified independently in Section 3. "
                "The fitted archive is not: without `EXPECTED_FITTED_SHA256` or `EXPECTED_ZIP_SHA256`, its only check is the "
                "manifest's own digest, which anyone who edits the archive can rewrite. Since the fitted archive is joblib — "
                "code-capable — that is the file the trusted digest exists for.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 5. Rebuild the estimator from the bundle alone · [Concept]\n\n"
                "`from_artifact` copies the fitted archive while pointing its recorded `model_path` at the verified `model.ckpt` "
                "(the original is untouched), loads it with tabpfn's `load_fitted_tabpfn_model` (no refit, no download), and "
                "checks that the reconstructed classes and feature columns equal the manifest's. This is serving-state "
                "reconstruction, not training."
            ),
            "code": "run_stage('reconstruct')",
        },
        {
            "md": "**What to notice:** `refit: False`, `network_fallback_for_weights: False`, the classes in stored order and the device.",
        },
        {
            "md": (
                "## 6. Supply new unlabelled rows → validate → input manifest · [Evaluation practice]\n\n"
                "Set `NEW_DATA_PATH` (any runtime) or tick `UPLOAD_NEW_DATA` (Colab). With your own bundle and neither set, the "
                "cell opens the upload dialog in Colab and, elsewhere, stops with a message naming `NEW_DATA_PATH`. List identifier "
                "columns in `ID_COLUMNS` (for the E2E rows: `['record_id', 'category']`): they are kept beside the predictions and "
                "never given to the model; an undeclared extra column is refused with a hint. `validate_new_rows` requires exactly "
                "the fitted feature columns (any order), no target, `prediction` or `proba_<class>` column, and finite numbers; then "
                "every feature the estimator saw as numeric must hold numbers — a value such as `abc` is refused naming the column "
                "and the value. The **input manifest** (row count, features, identifiers, missing values, the file digest, and a "
                "recorded refusal probe) is written to `outputs/{stem}_input_manifest.json`."
            ),
            "code": (
                "NEW_DATA_PATH = ''  # @param {{type:\"string\"}}\n"
                "UPLOAD_NEW_DATA = False  # @param {{type:\"boolean\"}}\n"
                "ID_COLUMNS = []  # @param {{type:\"raw\"}}\n\n"
                "rows_source, rows_path = 'sample', ''\n"
                "if NEW_DATA_PATH:\n"
                "    rows_source, rows_path = 'path', NEW_DATA_PATH\n"
                "elif UPLOAD_NEW_DATA or artifact_source != 'sample':\n"
                "    rows_source, rows_path = 'upload', upload_one('one unlabelled CSV for your bundle', 'NEW_DATA_PATH')\n"
                "run_stage('rows', source=rows_source, path=rows_path, id_columns=ID_COLUMNS)"
            ),
        },
        {
            "md": "**What to notice:** the input manifest with 8 rows, the 12 features, `id_columns`, and the `extra-column-probe` finding.",
        },
        {
            "md": (
                "## 7. Predict, report what cannot be measured, and export · [Concept]\n\n"
                "The `predict` stage scores the rows. `prediction` is the `argmax` label and one `proba_<class>` column per class "
                "follows **in the bundle's stored class order**; the probabilities are uncalibrated ensemble outputs and no "
                "threshold is shipped. Identifier columns come first, so the output joins back on your key. `evaluation_report` is "
                "produced even here: with no labelled rows its verdict is `not-measurable`; its `sample_kind` is `sample` for the "
                "pinned sample bundle and rows and `BYOD` otherwise. The result JSON records the bundle identity and its trust "
                "record, the input manifest, the notebook's source, the pinned model identity, revision and licence, and the "
                "runtime.\n\n"
                "**Predict before running:** these are the E2E notebook's eight new rows. Will the probabilities equal its Section 9 "
                "output?"
            ),
            "code": "run_stage('predict')",
        },
        {
            "md": (
                "**What to notice:** eight rows with the identifiers first, `prediction` and three probability columns summing to 1; "
                "`verdict: not-measurable`.\n\n"
                "**Checkpoint:** why is the verdict `not-measurable`, and what would make it measurable?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "The rows arrive without labels, as in deployment, so there is nothing to compare predictions with. A labelled "
                "holdout whose classes all appear in the support rows, scored with `classification_metrics` against "
                "`majority_class_baseline` (what the E2E notebook does), would make it measurable. With the same fitted state, "
                "checkpoint and device the probabilities match the E2E notebook's up to floating-point differences.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 8. Optional activity: which check catches a tampered bundle? · [Concept]\n\n"
                "**Predict → Change → Run → Observe → Explain.** **Predict:** for each `TAMPER` option — one byte of "
                "`model.tabpfn_fit` flipped; the same flip with the manifest digest rewritten to match; an unlisted `notes.txt` "
                "added to the ZIP — which check refuses it, if any? **Change:** tick `RUN_ACTIVITY` and pick `TAMPER`. **Run** "
                "this cell. **Observe** `refused` and the message. **Explain** the role of the trusted digest. The activity "
                "tampers with a copy; the verified bundle and the canonical outputs are untouched."
            ),
            "code": (
                "RUN_ACTIVITY = False  # @param {{type:\"boolean\"}}\n"
                "TAMPER = 'flip one byte of model.tabpfn_fit'  # @param [\"flip one byte of model.tabpfn_fit\", \"rewrite the manifest digest too\", \"add an unlisted member to the ZIP\"]\n"
                "if RUN_ACTIVITY:\n"
                "    run_stage('activity', tamper=TAMPER)\n"
                "else:\n"
                "    print('Optional activity skipped: tick RUN_ACTIVITY to run it. The canonical outputs are complete.')"
            ),
        },
        {
            "md": (
                "**What to notice (if you ran it):** `refused` and the message for each option.\n\n"
                "<details>\n<summary>Check your reasoning (open after running)</summary>\n\n"
                "A flipped byte breaks the manifest's digest, so internal consistency catches it. Rewriting the manifest digest "
                "defeats that check — but the activity validates against the trusted fitted digest recorded in Section 4, so the "
                "change is still refused; without a trusted digest it would pass. An unlisted member is refused by the allowlist "
                "before extraction. The lesson: internal digests catch accidents; only a digest you trust catches a forger.\n\n"
                "</details>"
            ),
        },
    ],
    "closing": (
        "## Troubleshooting · [Engineering]\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| Section 1–3 failures (platform, disk, `uv`, Hub download, digest) | as in the E2E notebook | See its Troubleshooting; never remove a pin, a hash or a manifest digest. |\n"
        "| `No pinned sample bundle is carried by this notebook revision` | the default path without a sample | Set `ARTIFACT_ZIP_PATH` to the E2E notebook's `outputs/tabpfn_classifier_artifact.zip` and paste its digests. |\n"
        "| `Trusted digest mismatch for …` | the file is not the one the digest was issued for | Do not proceed; obtain the bundle and digests from the producer again. |\n"
        "| `EXPECTED_…_SHA256 must be 64 hexadecimal characters` | a truncated digest | Paste the full digest the E2E notebook printed. |\n"
        "| `No trusted digest was supplied` (a warning) | your bundle without a digest | It runs, but only internal consistency is checked. |\n"
        "| `ARTIFACT_ZIP_PATH … is not a file` | a wrong path | Point it at the E2E notebook's ZIP. |\n"
        "| `a bundle holds exactly […] at the top level` / `occur more than once` | an extra, nested or missing member | Do not use it; re-export the bundle with the E2E notebook. |\n"
        "| `unsafe archive member`, `archive expands to …`, `compression ratio …` | a malformed or hostile ZIP | Do not use it. |\n"
        "| `unsupported manifest`, `manifest lacks …`, `… is missing or implausibly small`, `… SHA-256 … != manifest` | the files do not match the manifest | Re-export; never edit the manifest. |\n"
        "| `the bundled model.ckpt is not the pinned TabPFN-3 checkpoint` | another checkpoint | Use a bundle built on the pinned checkpoint. |\n"
        "| `reconstructed estimator disagrees with the manifest` | a corrupted archive or manifest | Re-export the bundle. |\n"
        "| `The pinned sample rows match only the pinned sample bundle` / `upload dialog exists only in Google Colab: set NEW_DATA_PATH` | your bundle with no rows | Set `NEW_DATA_PATH` (or tick `UPLOAD_NEW_DATA` in Colab). |\n"
        "| `[SCHEMA_MISMATCH] … If [...] are identifiers, list them in ID_COLUMNS` | an identifier column not declared | Add it to `ID_COLUMNS`. |\n"
        "| `ID_COLUMNS … are fitted feature columns` / `… are not in the header` | a wrong `ID_COLUMNS` entry | Fix the names. |\n"
        "| `[RESERVED_COLUMNS]` | the rows carry the target or a previous prediction | Supply unscored rows without the target. |\n"
        "| `numeric feature … has N non-numeric value(s)` | text in a numeric column | Fix the values; leave missing cells empty. |\n\n"
        "## Interpretation and limits\n\n"
        "A successful run proves that the bundle passed the member allowlist, the archive-safety rules and the manifest's "
        "schema, size and digest checks, that `model.ckpt` is byte for byte the pinned TabPFN-3 checkpoint, that the fitted "
        "estimator was rebuilt from the bundle alone with its class order intact, and that schema- and type-compatible new "
        "rows were scored with class probabilities under an explicit `argmax` decision rule — without the repository being "
        "reachable. It does **not** authenticate the producer beyond the channel a trusted digest came through, or establish "
        "predictive quality, robustness, calibration, fairness, or production fitness; the evaluation report says "
        "`not-measurable` because no labels exist here, and the exported probabilities must not be thresholded without the "
        "caller's own calibration data. Never bypass a failed archive, manifest, digest or schema check.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can acquire "
        "and digest-verify the pinned checkpoint, validate and reconstruct an external bundle, validate the supplied inference "
        "table, execute the public prediction path and emit the shown machine-readable outputs in the tested runtime. It does "
        "**not** establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production "
        "fitness on an unseen domain.\n\n"
        "## Conclusion · [Evaluation practice]\n\n"
        "Write three sentences: what the digest checks and the member allowlist established about the bundle you used; what "
        "the rebuild and the eight predictions show; and what you would need before trusting these probabilities for a "
        "decision.\n\n"
        "<details>\n<summary>Sample conclusion (open after writing yours)</summary>\n\n"
        "The ZIP matched the digest the E2E notebook printed, held exactly the three expected members, and its `model.ckpt` "
        "was the pinned checkpoint verified in Section 3, so every file I loaded is pinned. The estimator was rebuilt from the "
        "fitted archive without refitting, kept its class order, and scored eight unseen rows with uncalibrated probabilities "
        "while keeping their `record_id`; with no labels the report is correctly `not-measurable`. Before using these "
        "probabilities for a decision I would need a labelled, domain-representative test set and calibration data to choose a "
        "threshold.\n\n"
        "</details>\n\n"
        "**Next experiments:** paste a digest with one character changed and read the refusal; run the activity with each "
        "`TAMPER` option; score your own rows with an identifier column listed in `ID_COLUMNS`.\n\n"
        "## References\n\n"
        f"- Repository README: https://github.com/kurtvalcorza/{REPO}/blob/main/README.md\n"
        f"- Repository model card: https://github.com/kurtvalcorza/{REPO}/blob/main/MODEL_CARD.md\n"
        f"- Weight provenance: https://github.com/kurtvalcorza/{REPO}/blob/main/docs/WEIGHTS.md\n"
        f"- E2E companion (produces bundles): https://github.com/kurtvalcorza/{REPO}/blob/main/tutorials/tabpfn_classifier_colab.ipynb\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/PriorLabs/TabPFN\n"
        "- TabPFN-3 technical report: https://arxiv.org/abs/2605.13986"
    ),
}
