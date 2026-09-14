"""Companion template for tools/build_notebook.py — ARTIFACT-INFERENCE (NOTEBOOK_SPEC 1.1 §3.6, §18).

Generate with ``python tools/build_notebook.py --template tools/notebook_template_artifact_inference.py``. The
notebook carries the same package module and the same pinned snapshot as the task-inference notebook; it consumes
a `model.tabpfn_fit` + `model.ckpt` + `artifact_manifest.json` bundle produced by a *separate* execution (the
task-inference tutorial, or the DIMER worker — same contract) and never creates one.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_task_notebook_template", Path(__file__).with_name("notebook_template.py"))
assert _spec and _spec.loader
_task_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_task_module)
BADGES, REPO, _TASK = _task_module.BADGES, _task_module.REPO, _task_module.TEMPLATE

TEMPLATE = {
    **{k: _TASK[k] for k in ("package", "repo_name", "pipeline_class", "weights_key", "runtime_imports")},
    "stem": "tabpfn_classifier_artifact_inference",
    "notebook_name": "tabpfn_classifier_artifact_inference_colab.ipynb",
    "profile": "ARTIFACT-INFERENCE",
    "mode": "GUIDED",
    "run_all": (
        "**Known NOTEBOOK_SPEC 2.0 gap (§19, SART1/RUN5/RUN2):** the default path does not yet obtain a trusted sample bundle or sample input automatically — with `ARTIFACT_ZIP_PATH` and `NEW_DATA_PATH` empty, Sections 4 and 6 open upload dialogs for a predictor bundle produced by the E2E tutorial and for unlabelled rows; an executor sets both paths to files already in the runtime to skip the dialogs. Until a published sample bundle and sample rows are wired in, this notebook is a `Candidate`, not release-grade. Once they are present, **Run all** installs the pinned dependencies, validates the bundle (path-safe extraction, manifest digests, provenance, pinned model identity) before any deserialisation, reconstructs the serving estimator from the bundle alone (fitted state and checkpoint digests verified against `EXPECTED_FITTED_SHA256`/`EXPECTED_CKPT_SHA256` when supplied; nothing refit), validates the new rows into an input manifest, predicts with class probabilities, reports what cannot be measured, and exports outputs — all inside this kernel, with no DIMER worker or service and no credential."
    ),
    "byod": (
        "New-input BYOD is the `NEW_DATA_PATH`/upload branch in Section 6: your own unlabelled CSV with the bundle's required feature columns passes through the same validation, prediction and export cells. A user-supplied bundle is the separate `ARTIFACT_ZIP_PATH`/upload branch in Section 4 (`EXPECTED_ZIP_SHA256`, `EXPECTED_FITTED_SHA256` and `EXPECTED_CKPT_SHA256` pin it), validated before any state is reconstructed. Uploads stay inside this runtime; do not upload confidential or restricted data unless you are authorised to process it here."
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
    "capability": "serving-state reconstruction from an externally produced DIMER TabPFN classifier bundle (`artifact_manifest.json` + `model.tabpfn_fit` + `model.ckpt`) and class-probability inference on genuinely new rows under an explicit `argmax` rule",
    "intro": (
        "This notebook consumes a DIMER artifact bundle produced **outside this execution** — by the task-inference "
        "tutorial in a separate session, or by the DIMER worker, which writes the same three files: "
        "`artifact_manifest.json`, `model.tabpfn_fit` (fitted estimator state including the in-context training rows) "
        "and `model.ckpt` (foundation weights). It validates the bundle before any model state is deserialised "
        "(manifest schema and task type, member names, sizes, SHA-256 digests, archive safety of the fitted ZIP), "
        "checks the bundled checkpoint against the pinned TabPFN-3 checkpoint carried by this notebook, reconstructs the "
        "estimator through the carried module (`from_artifact`: the fitted archive's recorded `model_path` is rewritten "
        "in a temporary copy to the companion checkpoint; TabPFN's `load_fitted_tabpfn_model` restores the state), "
        "accepts genuinely new unlabelled rows, predicts, and exports results. **No training, fine-tuning or in-context "
        "refitting occurs, and no artifact is created here.**\n\n"
        "**Trust boundary.** Digest checks establish that the three files are internally consistent, not that the "
        "sender is trustworthy. `model.tabpfn_fit` is a ZIP of JSON parameters plus serialised Python/torch estimator "
        "state and `model.ckpt` is a torch checkpoint; loading them executes trusted model state, and the archive "
        "path-safety checks do not change that. The pinned checkpoint of Section 3 is acquired and digest-verified "
        "independently so the bundled `model.ckpt` can be required to equal it byte for byte. Load only bundles from a "
        "producer you trust, and paste the digests you were given out-of-band into the expected-digest fields. The "
        "TabPFN-3 weights are non-commercial (`tabpfn-3-license-v1.0`)."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried module guarantees, resolve and digest-verify the immutable "
        "TabPFN-3 checkpoint, supply an externally produced bundle and validate it before any model state is "
        "reconstructed, require the bundled checkpoint to equal the pinned one, reconstruct the serving estimator from the "
        "bundle alone, validate new unlabelled rows into an input manifest, predict class labels with class probabilities "
        "under an explicit `argmax` rule, produce an evaluation report that is `not-measurable` because no labels exist, "
        "and export machine-readable predictions plus provenance."
    ),
    "exclusions": (
        "artifact creation, in-notebook support fitting, fine-tuning, regression, calibrated probabilities, or any "
        "quality claim: without labelled rows nothing is measured, and the exported probabilities are uncalibrated "
        "ensemble outputs with no shipped threshold."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.11+). The default path runs on CPU and uses CUDA automatically when available.",
        "- **Artifact:** an externally produced bundle ZIP holding `artifact_manifest.json`, `model.tabpfn_fit` and `model.ckpt` (the task-inference tutorial writes `outputs/tabpfn_classifier_artifact.zip`; a DIMER worker's `artifacts/` directory zipped flat works too). Supply it through the upload dialog, or set `ARTIFACT_ZIP_PATH` to a file already present in the runtime for non-interactive execution. Nothing in this notebook manufactures it.",
        "- **Data:** one separate, unlabelled CSV with exactly the bundle's feature columns. It is supplied by upload or by `NEW_DATA_PATH`; no sample is bundled, because scoring self-generated rows would not be external-artifact evidence. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Supply the external bundle and validate it before any model state is reconstructed\n\n"
                "The bundle ZIP comes from `ARTIFACT_ZIP_PATH` (an executor places it there) or from the upload dialog; "
                "an optional `EXPECTED_ZIP_SHA256` and the two member digests `EXPECTED_FITTED_SHA256` / "
                "`EXPECTED_CKPT_SHA256` — pasted from the producer's record — fail closed on mismatch. "
                "`safe_extract_zip` extracts member by member (bare file names only, expanded-size and ratio ceilings; "
                "never `extractall`), then `validate_artifact_bundle` checks the manifest schema and task type, that "
                "both binary members are named by the manifest, exist and are plausibly sized, that their SHA-256 "
                "digests match the manifest (and the expected values), and that the fitted archive is a ZIP whose members "
                "are all safe relative paths and which carries `init_params.json` — all **before** anything is "
                "deserialised. The bundled `model.ckpt` must equal the pinned checkpoint of Section 3 (`WEIGHTS_SHA256`): "
                "a bundle produced on other weights is refused. Look for the feature schema and the classes the "
                "artifact expects."
            ),
            "code": (
                "ARTIFACT_ZIP_PATH = ''  # @param {{type:\"string\"}}\n"
                "EXPECTED_ZIP_SHA256 = ''  # @param {{type:\"string\"}}\n"
                "EXPECTED_FITTED_SHA256 = ''  # @param {{type:\"string\"}}\n"
                "EXPECTED_CKPT_SHA256 = ''  # @param {{type:\"string\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "WORK = Path('work')\n"
                "shutil.rmtree(WORK, ignore_errors=True)\n"
                "WORK.mkdir(parents=True)\n"
                "if ARTIFACT_ZIP_PATH:\n"
                "    zip_name, zip_payload = Path(ARTIFACT_ZIP_PATH).name, Path(ARTIFACT_ZIP_PATH).read_bytes()\n"
                "    artifact_source = f'path: {{ARTIFACT_ZIP_PATH}}'\n"
                "else:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    if len(uploaded) != 1:\n"
                "        raise RuntimeError('Upload exactly one bundle ZIP.')\n"
                "    zip_name, zip_payload = next(iter(uploaded.items()))\n"
                "    artifact_source = 'upload dialog'\n"
                "zip_path = WORK / Path(zip_name).name\n"
                "zip_path.write_bytes(zip_payload)\n"
                "zip_sha256 = sha256_file(zip_path)\n"
                "for label, expected in (('EXPECTED_ZIP_SHA256', EXPECTED_ZIP_SHA256), ('EXPECTED_FITTED_SHA256', EXPECTED_FITTED_SHA256), ('EXPECTED_CKPT_SHA256', EXPECTED_CKPT_SHA256)):\n"
                "    expected = expected.strip().lower()\n"
                "    if expected and (len(expected) != 64 or any(character not in '0123456789abcdef' for character in expected)):\n"
                "        raise ValueError(f'{{label}} must be 64 hex chars')\n"
                "if EXPECTED_ZIP_SHA256 and zip_sha256 != EXPECTED_ZIP_SHA256.strip().lower():\n"
                "    raise RuntimeError('Artifact ZIP SHA-256 mismatch')\n"
                "ARTIFACT_DIR = WORK / 'external-artifact'\n"
                "members = safe_extract_zip(zip_path, ARTIFACT_DIR)\n"
                "required = {{ARTIFACT_MANIFEST_NAME, FITTED_NAME, CHECKPOINT_NAME}}\n"
                "if not required <= set(members):\n"
                "    raise ValueError(f'bundle must carry {{sorted(required)}}; got {{sorted(members)}}')\n"
                "artifact = validate_artifact_bundle(ARTIFACT_DIR, expected_fitted_sha256=EXPECTED_FITTED_SHA256, expected_checkpoint_sha256=EXPECTED_CKPT_SHA256 or WEIGHTS_SHA256)\n"
                "if artifact['verifiedSha256']['foundationCheckpoint'] != WEIGHTS_SHA256:\n"
                "    raise RuntimeError('the bundled model.ckpt is not the pinned TabPFN-3 checkpoint carried by this notebook')\n"
                "FEATURE_COLUMNS = list(artifact['featureColumns'])\n"
                "TARGET_COLUMN = artifact['targetColumn']\n"
                "CLASSES = [str(c) for c in artifact['classes']]\n"
                "print({{'artifact_source': artifact_source, 'zip': zip_name, 'zip_sha256': zip_sha256[:16], 'members': sorted(members), 'mode': artifact.get('mode'), 'portableLoader': artifact.get('portableLoader'), 'baseModel': artifact.get('baseModel')}})\n"
                "print({{'featureColumns': FEATURE_COLUMNS, 'targetColumn': TARGET_COLUMN, 'classes': CLASSES, 'fittedSha256': artifact['verifiedSha256']['fittedEstimator'][:16], 'ckptSha256': artifact['verifiedSha256']['foundationCheckpoint'][:16], 'recordedModelPath': artifact['recordedModelPath']}})"
            ),
        },
        {
            "md": (
                "## 5. Reconstruct the serving estimator from the bundle alone\n\n"
                "`TabPFNClassifierPipeline.from_artifact` re-runs the bundle validation, rewrites the fitted archive's "
                "recorded `model_path` — in a temporary copy, never the original — to point at the companion `model.ckpt` "
                "beside it, and calls TabPFN's `load_fitted_tabpfn_model`. No network download is attempted for the "
                "foundation weights: they come from the bundle (and were proven equal to the pinned checkpoint). The "
                "in-context training rows travel inside the fitted archive; nothing is refit here. The reconstructed "
                "class order must agree with the manifest, else the notebook stops."
            ),
            "code": (
                "fresh = TabPFNClassifierPipeline.from_artifact(ARTIFACT_DIR, device=pipe.device, expected_checkpoint_sha256=WEIGHTS_SHA256)\n"
                "if fresh.classes != CLASSES or fresh.feature_columns != FEATURE_COLUMNS:\n"
                "    raise RuntimeError('reconstructed estimator disagrees with the manifest')\n"
                "print({{'source': fresh.source, 'device': fresh.device, 'classes': fresh.classes, 'n_estimators': fresh.n_estimators, 'random_state': fresh.random_state, 'refit': False, 'network_fallback_for_weights': False}})"
            ),
        },
        {
            "md": (
                "## 6. Supply new unlabelled rows → validate → input manifest\n\n"
                "Upload one CSV (or point `NEW_DATA_PATH` at one) containing exactly the artifact's feature columns and "
                "no target, `prediction` or `proba_*` columns. `validate_new_rows` rejects duplicate, missing or extra "
                "columns and infinite numeric values rather than silently dropping anything; the resulting input "
                "manifest names the schema the artifact imposes, the row count, the column types and the verdict, and "
                "is written to `outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also "
                "validates a probe with an extra column and records the structured finding."
            ),
            "code": (
                "NEW_DATA_PATH = ''  # @param {{type:\"string\"}}\n\n"
                "if NEW_DATA_PATH:\n"
                "    new_name, new_payload = Path(NEW_DATA_PATH).name, Path(NEW_DATA_PATH).read_bytes()\n"
                "else:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    if len(uploaded) != 1:\n"
                "        raise RuntimeError('Upload exactly one CSV of new rows.')\n"
                "    new_name, new_payload = next(iter(uploaded.items()))\n"
                "raw_rows = pd.read_csv(io.BytesIO(new_payload))\n"
                "new_rows = validate_new_rows(raw_rows, FEATURE_COLUMNS, target_column=TARGET_COLUMN, classes=CLASSES)\n"
                "input_manifest = {{\n"
                "    'schema': {{'format': 'CSV of unlabelled rows', 'columns': 'exactly the artifact feature columns, in any order', 'reserved': [TARGET_COLUMN, 'prediction', *[f'proba_{{c}}' for c in CLASSES]], 'numeric': 'finite values'}},\n"
                "    'inputs': [{{'id': new_name, 'mode': 'artifact-inference', 'rows': int(len(new_rows)), 'feature_columns': FEATURE_COLUMNS, 'numeric_features': int(len(new_rows.select_dtypes(include=np.number).columns)), 'missing_values': {{k: int(v) for k, v in new_rows.isna().sum().items() if v}}, 'sha256': sha256_hex(new_payload)}}],\n"
                "    'verdict': 'accepted',\n"
                "    'findings': [],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "}}\n"
                "try:\n"
                "    validate_new_rows(new_rows.assign(unexpected_column=0), FEATURE_COLUMNS, target_column=TARGET_COLUMN, classes=CLASSES)\n"
                "except InputRejected as exc:\n"
                "    input_manifest['findings'].append({{'input': 'extra-column-probe', **exc.finding}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False, default=str)\n"
                "print(json.dumps(input_manifest['inputs'][0], indent=2, default=str))\n"
                "print('findings:', json.dumps(input_manifest['findings'], indent=2, default=str))"
            ),
        },
        {
            "md": (
                "## 7. Predict, report what cannot be measured, and export\n\n"
                "`predict` returns `prediction` (the `argmax` class) and one `proba_<class>` column per class in the "
                "artifact's stored class order; the probabilities are raw ensemble outputs, **not calibrated**, and no "
                "threshold is shipped. Because the rows carry no labels, `evaluation_report` records the verdict "
                "`not-measurable` and states what labelled data would make the task measurable — it does not invent a "
                "score. It is written to `outputs/{stem}_evaluation_report.json`; the predictions go to "
                "`outputs/{stem}_predictions.csv`, and `outputs/{stem}_result.json` records the artifact identity and "
                "verified digests, the input manifest, the notebook's source, the pinned model identity, revision and "
                "licence, and the runtime. No credentials are recorded."
            ),
            "code": (
                "predictions = fresh.predict(new_rows)\n"
                "predictions.to_csv('outputs/{stem}_predictions.csv', index=False)\n"
                "report = evaluation_report(None, n_validation=0, class_labels=CLASSES, sample_kind='BYOD')\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "payload = {{\n"
                "    'artifact': {{k: artifact.get(k) for k in ('schemaVersion', 'taskType', 'targetColumn', 'featureColumns', 'classes', 'fittedEstimator', 'foundationCheckpoint', 'portableLoader', 'baseModel', 'mode', 'nEstimators', 'randomState')}},\n"
                "    'artifact_verified_sha256': artifact['verifiedSha256'],\n"
                "    'artifact_zip': {{'name': zip_name, 'sha256': zip_sha256, 'source': artifact_source}},\n"
                "    'reconstruction': {{'loader': 'tabpfn_classifier_pipeline.TabPFNClassifierPipeline.from_artifact', 'device': fresh.device, 'network_fallback_for_weights': False, 'refit': False, 'decision_rule': DECISION_RULE}},\n"
                "    'input_manifest': input_manifest,\n"
                "    'evaluation_report': report,\n"
                "    'scored_rows': int(len(predictions)),\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'model_file': WEIGHTS_FILE,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'tabpfn': importlib.metadata.version('tabpfn'), 'pandas': pd.__version__, 'device': fresh.device}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)\n"
                "print(predictions.head(8).to_string(index=False))\n"
                "print({{'verdict': report['verdict'], 'needs': report['needs']}})\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A successful run proves that an independently supplied bundle is internally consistent with its manifest and "
        "was produced on the pinned TabPFN-3 checkpoint, that the carried module reconstructs the estimator from the "
        "bundle alone without refitting or downloading weights, and that schema-compatible new rows can be scored and "
        "exported. It does **not** authenticate the producer, make untrusted serialised estimator state safe to load, "
        "or establish predictive quality, calibration, fairness or production fitness — the evaluation report is "
        "`not-measurable` by construction. Never bypass a failed digest, schema or archive-safety check; obtain a "
        "correct bundle from a trusted producer.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this "
        "notebook, can acquire and digest-verify the pinned checkpoint, validate an external bundle before "
        "deserialisation, reconstruct the serving estimator, validate new rows and emit the shown machine-readable "
        "outputs in the tested runtime — without the repository being reachable. It does **not** establish benchmark "
        "superiority or anything about the quality of the artifact's training rows.\n\n"
        "**Next experiments:** paste the producer's digests into the expected-digest fields and watch a tampered "
        "bundle fail closed; supply rows with a missing feature column and read the structured rejection; compare the "
        "probabilities of the same rows scored by the task-inference notebook's in-memory estimator and by this "
        "reconstruction (they should agree to floating-point precision).\n\n"
        "## References\n\n"
        f"- Repository README: https://github.com/kurtvalcorza/{REPO}/blob/main/README.md\n"
        f"- Repository model card: https://github.com/kurtvalcorza/{REPO}/blob/main/MODEL_CARD.md\n"
        f"- Weight provenance: https://github.com/kurtvalcorza/{REPO}/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/PriorLabs/TabPFN\n"
        "- TabPFN-3 technical report: https://arxiv.org/abs/2605.13986"
    ),
}
