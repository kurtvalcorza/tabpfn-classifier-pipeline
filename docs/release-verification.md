# Release verification

`tutorials/tabpfn_classifier_colab.ipynb` (`TASK-INFERENCE`) and `tutorials/tabpfn_classifier_artifact_inference_colab.ipynb` (`ARTIFACT-INFERENCE`) are **release
candidates** until the exact notebook revisions have executed top-to-bottom in a clean supported runtime. Unit tests,
JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 1.1. This file is the durable release-gate record for both
notebooks; the previous notebook pair's execution records (worker-driven, NOTEBOOK_SPEC 1.0) are kept below as history.

## Automatic coverage (static, every pull request)

CI (`.github/workflows/ci.yml`, job `test`) runs `tools/validate_release_assets.py`, which checks, for each notebook:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly the two tutorial notebooks, named in `tutorials/README.md` with their profiles, the notebook-spec version and
  the standalone carrier; `metadata.dimer` declares the profile, spec `1.1`, `standalone: true` and `generated_from`
  (repository, module commit, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install, repository import, worker call
  (`worker.run(`, `worker_cli(`, `subprocess.run([`), token or `GITHUB_TOKEN` on the primary path; exactly one cell
  tagged `embedded_module` equal to `src/tabpfn_classifier_pipeline/pipeline.py` after the generator's documented rewrites; the inline
  `MANIFEST` equal to the committed snapshot manifest and the inline `PINS` equal to the `pyproject.toml` runtime pins;
  each notebook byte-identical to `tools/build_notebook.py` output; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision is a 40-hex immutable commit, and the same
  identity string appears in `README.md`, `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions beyond the
  allowlisted `COMPONENTS.json` worker commits;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`, `TabPFNClassifierPipeline.from_pretrained(weights_dir=...)`,
  `build_synthetic_dataset` / `read_dataset_zip` / `stratified_holdout`, `validate_inputs` with a rejected probe,
  `pipe.fit` / `pipe.evaluate`, the baseline, `evaluation_report`, `save_artifact`, `validate_artifact_bundle`,
  `from_artifact` with the reload equivalence check, `validate_new_rows`, `predict`; in the companion
  `safe_extract_zip`, `validate_artifact_bundle`, `from_artifact`, `validate_new_rows`, `predict` and a
  `not-measurable` report), the ceiling print, the exports, the learner-facing statements (in-context learning, no
  gradient update, private fine-tune path not carried, uncalibrated outputs, single holdout, `sample-sanity` /
  `not-measurable`, trust boundary, non-commercial licence) and the gated-off BYOD defaults; forbidden patterns
  (credential-in-URL, any `git clone` / `github.com` / repository import on the primary path, a mutable
  `revision='main'`, direct `tabpfn` / `sklearn.metrics` / `sklearn.datasets` / `huggingface_hub` / `zipfile.ZipFile` /
  `torch.load` use **outside the carried module cell**, `GITHUB_TOKEN`, `import train as worker`,
  `trust_remote_code=True`, `pickle.load`, `extractall(`); companion-forbidden code (`build_synthetic_dataset(`,
  `save_artifact(`, `pipe.fit(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and the `## Version pinning` provenance section.

CI also runs `ruff check .`, `tools/build_notebook.py --check` for both templates, and the offline unit suite
(`tests/test_role_helpers.py`, `tests/test_notebook_parity.py`, `tests/test_companion_parity.py`,
`tests/test_notebook_spec.py`; no tabpfn, no torch, no weights — the module imports them lazily). The `integration`
job installs the real `tabpfn` package and exercises `serving/load_artifact.py` as before. These are source/provenance
and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorials are written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU or T4 kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (no repository checkout is needed — the notebooks are standalone); the companion run is fed the first run's `outputs/tabpfn_classifier_artifact.zip` through `ARTIFACT_ZIP_PATH` and a separately generated CSV through `NEW_DATA_PATH` |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence. `scripts/execute_notebook_release.py` (bootstrap-stripping harness of the previous pair) does not apply: the standalone notebooks have no bootstrap region |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open the exact task-inference notebook revision in a new runtime (Colab, or the Kaggle executor above) with **no
   repository checkout** and a clean model cache;
3. run it top-to-bottom without editing implementation cells (form parameters at their defaults for the sample path:
   `USE_BYOD = False`, `USE_BYOD_ROWS = False`, `N_ESTIMATORS = 4`, `SEED = 42`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the module commit recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`; the pinned `torch==2.11.0` replaces the Colab-provided wheel and the install guard may require one
   runtime restart);
5. verify every default-path stage completes:
   - the carried module cell executes (defines the pipeline class and helpers) with no import of the repository package;
   - the inline `MANIFEST` is asserted against the module identity and written to `weights/tabpfn-3-classifier/`,
     `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports all four manifest entries on a clean runtime,
     `verify_snapshot` returns the manifest dict, and `from_pretrained(weights_dir=WEIGHTS_DIR)` reports
     `source == 'local-snapshot'`;
   - the synthetic table is drawn in code and `validate_inputs` writes `outputs/tabpfn_classifier_input_manifest.json` (verdict
     `accepted`, one recorded rejection from the renamed-target probe);
   - `pipe.fit` runs in context and `pipe.evaluate` reports `accuracy`, `balancedAccuracy`, `logLoss`, `rocAuc` that beat the majority-class baseline (`majority_class_baseline`) on the sample;
   - `save_artifact` writes the bundle, `outputs/tabpfn_classifier_artifact.zip` is produced, and the fresh-boundary reload prints
     the equivalence check as matching (recomputed validation accuracy equal to the recorded value within `1e-6`);
   - `evaluation_report` writes `outputs/tabpfn_classifier_evaluation_report.json` with verdict `sample-sanity` and the reload check;
   - new rows are scored and `outputs/tabpfn_classifier_predictions.csv` / `outputs/tabpfn_classifier_result.json` are written with
     `NOTEBOOK_SOURCE`, model revision, model licence, runtime versions and device;
6. in a **second fresh runtime**, run the companion with `ARTIFACT_ZIP_PATH` pointing at the first run's bundle and
   `NEW_DATA_PATH` at a separately generated CSV: the bundle validates, the bundled checkpoint equals the pinned one,
   `from_artifact` reconstructs, the rows are scored, the report is `not-measurable`, the exports exist;
7. verify the exports and the interpretation sections match the observed path;
8. record the notebook Git blob ids, commit, runtime (platform, Python, torch, tabpfn, device), model identifier and
   immutable revision, whether the model cache was clean, outcome, produced outputs, and any warning or applicable
   `SHOULD` deviation in the table below;
9. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of the notebook (verify with `git rev-parse <commit>:tutorials/<name>`). Wall
times, when recorded, are the sum of per-cell times reported by the executor and include installs and the model
download; they are measurements for the stated runtime, not general estimates.

### Manual clean-runtime evidence (standalone notebooks)

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| | | | Task inference, default sample path | | pending — queued to the executor lane |
| | | | Artifact inference, bundle from the run above | | pending — queued to the executor lane |

### History: previous (worker-driven, NOTEBOOK_SPEC 1.0) notebook pair

The rows below were recorded for the previous notebook pair, which cloned this repository and the private
`tabpfn-classifier-finetuner` worker and ran `train.run()`. The standalone notebooks share none of that code path (the
worker is not carried), so **none of these rows carry over**; they are kept as provenance for the artifact contract
the companion still consumes.
Each row is evidence for the revision named in its own `pipeline.commit`, and for nothing later. The rows below predate the
secure private-source bootstrap and the `# >>> colab-bootstrap` harness contract, so they are
**superseded**: they do not carry to the current candidate head, and no Colab record exists for any
revision. A fresh `scripts/execute_notebook_release.py` record at the candidate head, and then a
clean Colab record, are both still required before either notebook is marked release-grade.

| Date (UTC) | Notebook revision | Environment | Engine | Outcome |
|---|---|---|---|---|
| 2026-09-11 | `247c4d381d93` (clean tree); worker `8ebcfb6dd56a` | Local host `Kurt-Valcorza`, Windows-11-10.0.26200-SP0; Python 3.12.10; torch 2.11.0+cu128; tabpfn 8.1.0; GPU NVIDIA GeForce RTX 5070 Ti Laptop GPU; `MODEL_VERSION=v2`, default zero-shot | `scripts/execute_notebook_release.py --skip-bootstrap`: nbclient, one fresh kernel per notebook; `/content` → `D:/tabpfn-clf-run/content`; worker cloned from the local finetuner checkout at the pinned commit; `google.colab.files.upload()` served from explicit paths | **PASS** — E2E: `mode=zero-shot-icl`, `fineTuneEffective=false`, validation accuracy 0.9667 / logLoss 0.0796 vs trivial baseline; reload via serving loader reproduced the recorded validation metric within 1e-6. ARTIFACT-INFERENCE (second kernel, bundle + 8 fresh rows supplied externally): reconstructed and scored 8 rows. **Not a Google Colab run** — a Colab record is still required before either notebook is marked release-grade. |
| 2026-09-11 | `247c4d381d93` (clean tree); worker `8ebcfb6dd56a` | Local host `Kurt-Valcorza`, Windows-11-10.0.26200-SP0; Python 3.12.10; torch 2.11.0+cu128; tabpfn 8.1.0; GPU NVIDIA GeForce RTX 5070 Ti Laptop GPU; `MODEL_VERSION=v2`, FINE_TUNE=True | `scripts/execute_notebook_release.py --skip-bootstrap --set FINE_TUNE=True`: nbclient, one fresh kernel per notebook; `/content` → `D:/tabpfn-clf-ft/content`; worker cloned from the local finetuner checkout at the pinned commit; `google.colab.files.upload()` served from explicit paths | **PASS** — E2E: `mode=fine-tune`, `fineTuneEffective=true`, validation accuracy 0.9667 / logLoss 0.0866 vs trivial baseline; reload via serving loader reproduced the recorded validation metric within 1e-6. ARTIFACT-INFERENCE (second kernel, bundle + 8 fresh rows supplied externally): reconstructed and scored 8 rows. **Not a Google Colab run** — a Colab record is still required before either notebook is marked release-grade. |

## Current status

No clean-runtime execution of the standalone notebooks has been recorded yet; the runs are **pending** and queued to the
executor lane. Static validation (`tools/validate_release_assets.py`), the generator parity checks, a `compile()` sweep
over every code cell, and the offline unit suite passed on the tutorial source at the candidate revision, which is
necessary but not sufficient. The registry status remains **Candidate** until a reviewer confirms recorded runs against
the notebook blobs under review and an integrator promotes them; promotion is not performed by the builder. Facts a
reviewer should weigh: the pinned checkpoint `tabpfn-v3-classifier-v3_default.ckpt` has never been staged or loaded locally (the manifest's digest
comes from the Hub API; `verify_snapshot` was exercised on the three small files only); the module's `tabpfn` calls
(`TabPFNClassifier(model_path=...)`, `save_fitted_tabpfn_model`, `load_fitted_tabpfn_model`) were exercised only
through injected fakes in the unit suite — the real-package path runs for the first time in the clean run; the
standalone carrier was validated statically and by a CPU carrier probe (module cells + identity assertion, no fetch).
