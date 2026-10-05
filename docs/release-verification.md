# Release verification

`tutorials/tabpfn_classifier_colab.ipynb` (`E2E`) and `tutorials/tabpfn_classifier_artifact_inference_colab.ipynb` (`ARTIFACT-INFERENCE`) are **release
candidates** until the exact notebook revisions have executed top-to-bottom in a clean supported runtime. Unit tests,
JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.2. This file is the durable release-gate record for both
notebooks; the previous notebook pair's execution records (worker-driven, NOTEBOOK_SPEC 1.0) are kept below as history.

## Automatic coverage (static, every pull request)

CI (`.github/workflows/ci.yml`, job `test`) runs `tools/validate_release_assets.py`, which checks, for each notebook:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or execution
  counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly the two tutorial notebooks, each named in `tutorials/README.md` with its profile, the spec version (`2.2`) and
  the standalone carrier; `metadata.dimer` declares the profile (`E2E`, `ARTIFACT-INFERENCE`), mode `GUIDED`,
  `standalone: true` and `generated_from` (repository, generating commit, package paths and SHA-256, carried-file
  digests, generator `build_notebook.py/3.0-tabular`);
- the standalone carrier and isolated environment (ST1–ST6, PAR1–PAR3, RUN1, RUN10, ENV6): one carrier cell whose carried
  files equal the repository files (`src/tabpfn_classifier_pipeline/{__init__,pipeline}.py`, the stage runner,
  `tutorials/requirements-colab.lock.txt`, the 4-file snapshot manifest, `NOTICE`, and — once it exists — the companion's
  pinned sample bundle) with matching digests; the lock pins every `pyproject.toml` runtime pin with hashes; a pinned
  `uv` builds a managed-CPython environment with `--require-hashes`, reused per lock digest; no in-kernel install and no
  restart instruction; the four Infrastructure cells are titled and collapsed; every learner cell runs a stage; the
  notebook byte-identical to `tools/build_notebook.py` output;
- the stage-runner markers — E2E: validation before any split with coded findings, `UNSEEN_CLASSES` refused,
  identifiers (`DROP_COLUMNS`) kept out of the features, numeric columns with stray strings refused, the majority-class
  baseline and the logistic-regression reference, the proposed RUN7 deviation recorded, `save_artifact` + recorded
  feature kinds, the fresh-process reload comparing probabilities on every validation row (`rtol=1e-5`, `atol=1e-6`);
  companion: trusted ZIP / fitted digests, the member allowlist before extraction, `validate_artifact_bundle` with the
  checkpoint bound to the pinned one, `from_artifact`, `ID_COLUMNS`, the numeric-type check, a `not-measurable` report,
  and no self-production; the form-parameter defaults; no quality `assert`; the forbidden patterns;
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and the `## Version pinning` section.

CI also runs `ruff check .`, both generator `--check` runs, and the offline tests (`tests/`, no tabpfn, no torch, no
weights, no Parquet engine), including `test_notebook_review_fixes.py`, which execs the notebooks' own cells with
stand-ins and runs the model-free stages. These are source and unit checks, **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU or T4 runtime; any kernel Python — the stages run on the isolated environment's CPython 3.12.12 | The runtime the tutorials are written for; a clean one-pass **Run all** here is promotion evidence |
| Kaggle kernel | Kaggle CPU or GPU image, any Python | Reproducible clean-room executor of the same class; the notebook runs verbatim (no checkout needed). For the companion set `ARTIFACT_ZIP_PATH`, `EXPECTED_ZIP_SHA256`, `EXPECTED_FITTED_SHA256`, `NEW_DATA_PATH` and `ID_COLUMNS = ['record_id', 'category']` to the E2E run's exports |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open the exact E2E notebook revision in a fresh runtime with **no repository checkout** and a clean model cache;
3. run it with one **Run all** and no restart, form parameters at their defaults (`USE_BYOD = False`,
   `DROP_COLUMNS = ['record_id', 'category']`, `N_ESTIMATORS = 4`, `SEED = 42`, `USE_BYOD_ROWS = False`,
   `RUN_ACTIVITY = False`), then re-run the export cell once;
4. verify: the carried-file verification and the isolated environment's versions (CPython 3.12.12, `torch` 2.11.0,
   `tabpfn` 8.1.0); the 4-file snapshot staged and verified; 420 / 90 / 90 rows and 12 features (no `record_id`); the
   input manifest with the `renamed-target-probe` finding; TabPFN's validation and test metrics; the majority baseline
   (60 validation errors) and the logistic regression (8 errors, 0.9111; test 0.9444); the evaluation report
   `sample-sanity` with the adaptation deviation; the bundle digests printed; the reload with
   `maxAbsProbabilityDifference` within tolerance; `outputs/tabpfn_classifier_new_rows.csv` and the predictions with
   `record_id` and `category`; the result JSON with source, model identity, licence and runtime;
5. in a **second** clean runtime run the exact companion revision with `ARTIFACT_ZIP_PATH`, both digests, `NEW_DATA_PATH`
   and `ID_COLUMNS` set to the E2E exports, and verify the digest checks, the member allowlist, the checkpoint binding,
   the rebuild, the input manifest, the `not-measurable` report and the predictions; with all fields empty it must stop
   in Section 4 naming `ARTIFACT_ZIP_PATH` until a pinned sample bundle is added with `tools/build_sample_bundle.py`;
6. record the notebook Git blob ids, commit, runtime (platform, Python, PyTorch, tabpfn, device), model identifier and
   immutable revision, whether the model cache was clean, `restarted: false`, outcome, metrics and outputs, and any
   warning or deviation in the table below;
7. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of the notebook (verify with `git rev-parse <commit>:tutorials/<name>`). Wall
times, when recorded, are the sum of per-cell times reported by the executor and include installs and the model
download; they are measurements for the stated runtime, not general estimates.

### Manual clean-runtime evidence (standalone notebooks)

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `1d93609` / `ccdcd20093ed` | Kaggle CPU (`kurtvalcorza/dimer-nb2-tabpfn-classifier` v1) | Default sample path | 203.5 s | **PASSED** — 10/10 ok code cells executed cleanly, 10 files, 213 MB staged |
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