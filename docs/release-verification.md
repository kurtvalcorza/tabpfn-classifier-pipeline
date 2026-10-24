# Release verification

`tutorials/tabpfn_classifier_colab.ipynb` (`E2E`, `TUTORIAL`) is a **release candidate** until the exact notebook
revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell compilation, and
`tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER Notebook
Specification 3.0. This file is the durable release-gate record for the notebook.

**NOTEBOOK_SPEC 3.0 migration (§32.2).** The `ARTIFACT-INFERENCE` companion
(`tutorials/tabpfn_classifier_artifact_inference_colab.ipynb`) is retired (KND11) and folded into the E2E notebook's
closing external-artifact section (Section 10, §19.1). Its sample bundle `sample-bundle-v1` was assembled by
`tools/build_sample_bundle.py` and so does not meet SART9; it is no longer used, and the release is left in place
(SART7). The section's default artifact is the release asset `sample-artifact-synthetic-v1` /
`tabpfn_classifier_artifact.zip` (197,068,508 bytes, SHA-256
`8fe248e2f9841c452316d07ee3e88e5fcde89ec458ff630954028b1c22df1054`, fitted archive
`1f48ecaf118c834633c8c000579e3f68227cf11c9ae0b146b68509f6f3bb9af6`), the export of a recorded hosted Colab T4 run of
the E2E default path at `7fce7f9` (blob `a4c8b2904e71`, 2026-10-22, PASSED; SART8, SART9). The companion's rows below
and the earlier notebook pair's records are kept as history (REL14); none of them is evidence for the migrated
notebook. Its own hosted run (the second pass of §32.2) is the `fcfa6ff` record under *Recorded executions*; what promotion
still needs is under *Current status*.

## Automatic coverage (static, every pull request)

CI (`.github/workflows/ci.yml`, job `test`) runs `tools/validate_release_assets.py`, which checks, for each notebook:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or execution
  counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly the one tutorial notebook, named in `tutorials/README.md` with its profile, the spec version (`3.0`) and
  the standalone carrier; `metadata.dimer` declares the profile `E2E`, kind `TUTORIAL`, mode `GUIDED`,
  `standalone: true` and `generated_from` (repository, generating commit, package paths and SHA-256, carried-file
  digests, generator `build_notebook.py/3.0-tabular`);
- the standalone carrier and isolated environment (ST1–ST6, PAR1–PAR3, RUN1, RUN10, ENV6): one carrier cell whose carried
  files equal the repository files (`src/tabpfn_classifier_pipeline/{__init__,pipeline}.py`, the stage runner,
  `tutorials/requirements-colab.lock.txt`, the 4-file snapshot manifest and `NOTICE`) with matching digests; the lock pins every `pyproject.toml` runtime pin with hashes; a pinned
  `uv` builds a managed-CPython environment with `--require-hashes`, reused per lock digest; no in-kernel install and no
  restart instruction; the four Infrastructure cells are titled and collapsed; every learner cell runs a stage; the
  notebook byte-identical to `tools/build_notebook.py` output;
- the stage-runner markers — E2E: validation before any split with coded findings, `UNSEEN_CLASSES` refused,
  identifiers (`DROP_COLUMNS`) kept out of the features, numeric columns with stray strings refused, the majority-class
  baseline and the logistic-regression reference, the proposed RUN7 deviation recorded, `save_artifact` + recorded
  feature kinds, the fresh-process reload comparing probabilities on every validation row (`rtol=1e-5`, `atol=1e-6`);
  external-artifact section: the exact pinned URL and SHA-256, the digest checked before extraction, the member
  allowlist, `validate_artifact_bundle` with the fitted archive and checkpoint bound to their pinned digests, the
  base-model and SART10 data-contract checks, `from_artifact`, outputs under `outputs/external_artifact/`; ST1 allows
  only that exact URL as a whole token (no prefix match; `..`, `%2e`, `?` and `#` refused); the form-parameter defaults; no quality `assert`; the forbidden patterns;
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and the `## Version pinning` section.

CI also runs `ruff check .`, the generator `--check` run, and the offline tests (`tests/`, no tabpfn, no torch, no
weights, no Parquet engine), including `test_notebook_review_fixes.py` and `test_external_artifact_section.py`, which exec
the notebook's own cells with stand-ins and run the model-free stages (the section against a stand-in artifact). These are source and unit checks, **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU or T4 runtime; any kernel Python — the stages run on the isolated environment's CPython 3.12.12 | The runtime the tutorials are written for; a clean one-pass **Run all** here is promotion evidence |
| Kaggle kernel | Kaggle CPU or GPU image, any Python | Reproducible clean-room executor of the same class; the notebook runs verbatim (no checkout needed). |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open the exact E2E notebook revision in a fresh runtime with **no repository checkout** and a clean model cache;
3. run it with one **Run all** and no restart, form parameters at their defaults (`USE_BYOD = False`,
   `DROP_COLUMNS = ['record_id', 'category']`, `N_ESTIMATORS = 4`, `SEED = 42`, `USE_BYOD_ROWS = False`,
   `RUN_ACTIVITY = False`, `ARTIFACT_SOURCE = 'Pinned sample artifact'`, `USE_OWN_ROWS = False`), then re-run the
   export cell once;
4. verify: the carried-file verification and the isolated environment's versions (CPython 3.12.12, `torch` 2.11.0,
   `tabpfn` 8.1.0); the 4-file snapshot staged and verified; 420 / 90 / 90 rows and 12 features (no `record_id`); the
   input manifest with the `renamed-target-probe` finding; TabPFN's validation and test metrics; the majority baseline
   (60 validation errors) and the logistic regression (8 errors, 0.9111; test 0.9444); the evaluation report
   `sample-sanity` with the adaptation deviation; the bundle digests printed; the reload with
   `maxAbsProbabilityDifference` within tolerance; `outputs/tabpfn_classifier_new_rows.csv` and the predictions with
   `record_id` and `category`; the result JSON with source, model identity, licence and runtime;
5. verify the external-artifact section (Section 10): the provenance printed before the download, the pinned ZIP
   digest verified before extraction, the three members, the fitted-archive and checkpoint digests, the SART10 data
   contract, the rebuild without refit, eight predictions with `record_id` and `category`, and the two files under
   `outputs/external_artifact/`;
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
| 2026-10-22 | `7fce7f9` / `a4c8b2904e71` of `tabpfn_classifier_colab.ipynb` (pre-migration blob); executed file `docs/execution-evidence/2026-10-22-7fce7f9/tabpfn_classifier_colab_7fce7f9_colab-cli-t4.ipynb` with its exec log, run summary and suite log (local paths redacted to `<local-run-dir>/`; captured→committed SHA-256: executed notebook `15e48acf…`→`15e48acf…`, exec log `610f3557…`→`4cc575f9…`, run summary `d9be0f31…`→`d9be0f31…`, suite log `ef13d51b…`→`b430e3c6…`) | Colab CLI 0.7.4 `colab exec -f`, fresh hosted T4 session | E2E default path (synthetic sample, `N_ESTIMATORS=4`) | 116.6 s | **PASSED** — 11/11 code cells, exit 0, fresh-boundary verification PASSED; printed `EXPECTED_ZIP_SHA256 = 8fe248e2…1054` and `EXPECTED_FITTED_SHA256 = 1f48ecaf…9af6`. Its export is the pinned sample artifact of the external-artifact section (release `sample-artifact-synthetic-v1`, SART8). Evidence for the producing blob only, not for the migrated notebook. |
| 2026-10-10 | `0e5a27c` / `a4c8b2904e71` of `tabpfn_classifier_colab.ipynb`, downloaded from GitHub at the PR head and blob-verified before the session; executed file `docs/execution-evidence/2026-10-10/tabpfn_classifier_colab_0e5a27c_colab-cli-t4.ipynb`, sha256 `982a162b6c7ac09b0b02bf74102488e6955d368ae5d8f733ab4bb9573e9dc8d0` (`…_run_summary.json` `0f38bab3a0a2…`, `…_exec.log` `c2874688ec26…` (redacted 2026-10-22, now `427bd2876eef…`)) | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 (15,360 MiB, reported by cell 1), via the workspace `colab-cli-serial-test-suite`; isolated uv CPython 3.12.12 environment from the carried hash lock (63 packages; torch 2.11.0, tabpfn 8.1.0, numpy 2.5.3, pandas 2.3.2, scikit-learn 1.9.0). Not a browser Run all; the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log (1/11–11/11) | E2E, default synthetic sample (420/90/90, 12 features, 3 classes), `N_ESTIMATORS=4`, device cuda; checkpoint fetched from the Hub at revision `24a16a89d245` and digest-verified (`d0d865d5…`) | 140.0 s session wall | **PASSED — one pass, no restart, 0 errors, 11/11 code cells.** Renamed-target probe rejected (`TARGET_MISSING`); majority-class val accuracy 0.3333; logistic regression val 0.9111 / test 0.9444; TabPFN val accuracy 0.9778, log loss 0.0658, ROC AUC 0.9994; test 0.9778 / 0.0794 / 0.9978 (the CPU lock-only drive below gave log loss 0.0668 / 0.0804 and test AUC 0.9972: GPU vs CPU, rounding-level; accuracies equal). Fresh-boundary reload PASSED, max abs probability difference 0.0 on 90 rows; 8 new rows scored. Two scikit-learn `y_prob values do not sum to one` UserWarnings in the condition stage. Carried provenance names `ea1d63e` (the generation-time HEAD, not `0e5a27c`); follow-up. BYOD and the optional activity not exercised. Status stays **Candidate** |
| 2026-10-10 | `0e5a27c` / `6d7b0040df03` of `tabpfn_classifier_artifact_inference_colab.ipynb`, downloaded from GitHub at the PR head and blob-verified before the session; executed file `docs/execution-evidence/2026-10-10/tabpfn_classifier_artifact_inference_colab_0e5a27c_colab-cli-t4.ipynb`, sha256 `1fd43670e1ab8fc598eaebe2ad8706fc347db78eae92af72e50ec61e93685d54` (`…_run_summary.json` `d7d0c1e0f051…`, `…_exec.log` `918a69fdec3a…` (redacted 2026-10-22, now `136f147c3d69…`)) | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4, via the workspace `colab-cli-serial-test-suite`; same 63-package isolated uv environment, built fresh. Not a browser Run all; order from the CLI log (1/9–9/9) | Artifact inference, default pinned sample: the `sample-bundle-v1` release asset was **downloaded for real** and verified before extraction; checkpoint fetched from the Hub and digest-verified | 104.6 s session wall | **PASSED — one pass, no restart, 0 errors, 9/9 code cells.** Release download exercised: `tabpfn_classifier_sample_bundle.zip` 169,881 bytes, SHA-256 `d61f1c2642154c2ab0d7a81954ec6522a908e690177f9723563ab44054fde0ba` verified before extraction, sample files verified against `SAMPLE_BUNDLE.json`; reconstructed from the artifact on cuda (no refit, no network fallback for weights); extra-column probe refused (`SCHEMA_MISMATCH`); predictions on the eight rows identical to the E2E run's; report `not-measurable` (no labels). **The tamper activity was not exercised** (`RUN_ACTIVITY` off by default; cell 9 skipped), so the only refusal run was `SCHEMA_MISMATCH`. Carried provenance reads `uncommitted` (git rev-parse failed during regeneration, `tools/build_notebook.py`); provenance-only, follow-up. BYOD not exercised. Status stays **Candidate** |
| 2026-09-14 | `1d93609` / `ccdcd20093ed` | Kaggle CPU (`kurtvalcorza/dimer-nb2-tabpfn-classifier` v1; image torch 2.10.0+cpu) | E2E, default sample path | 203.5 s | **PASSED** — 10/10 ok code cells, `restarted`: no (none recorded), 10 files, 213 MB staged. Metrics and package versions were not captured by that executor's ledger. A Kaggle CPU image, not Colab, and an earlier blob than the current E2E notebook, so it is not evidence for the current candidate |

**First hosted run of the migrated notebook (2026-10-22, `fcfa6ff` / blob `d766d7b710ec`, Notebook Spec 3.0; the second pass of
§32.2).** Colab CLI 0.7.4 `colab exec -f` on a fresh Colab Tesla T4, default path of `tutorials/tabpfn_classifier_colab.ipynb` on
PR #24, started 2026-10-22T04:00:12Z, wall 161.8 s, exit 0, input unchanged; executed cell sources equal the blob (44 cells);
verdict PASS. Evidence: `docs/execution-evidence/2026-10-22-fcfa6ff/`. The fresh-boundary reload passed (probabilities equivalent
on 90 validation rows). **Section 10 ran end to end on the default path with real tabpfn:** it downloaded the 197,068,508-byte
release asset of tag `sample-artifact-synthetic-v1`, verified both pinned digests before deserialisation (ZIP
`8fe248e2f9841c452316d07ee3e88e5fcde89ec458ff630954028b1c22df1054`, fitted estimator
`1f48ecaf118c834633c8c000579e3f68227cf11c9ae0b146b68509f6f3bb9af6`), rebuilt the classifier from the files alone and wrote
`outputs/external_artifact/tabpfn_classifier_external_predictions.csv` and `..._external_result.json` (printed in the executed
notebook; the files stayed on the released VM). This run's own export printed fitted-estimator SHA-256
`9b27a525ca46870329918f997381e6c1f5fb702a73a916b2cd23cf75e3b32fad` and bundle ZIP `247d3c633b6c8b61af034c037de3078545f28231edefec06bb0b9448e2ddb0ae`:
the fitted `.tabpfn_fit` file is not byte-reproducible across runs, so a re-export does not reproduce the pinned digests even
with the deterministic ZIP writer. The pin stays valid because it names one recorded run's export (SART8/SART9); a re-pin
(SART11) publishes a new tag. Local paths were replaced by `<local-run-dir>/` before the first commit:

| File (under `execution-evidence/2026-10-22-fcfa6ff/`) | Captured | Committed |
|---|---|---|
| `tabpfn_classifier_colab_fcfa6ff_colab-cli-t4.ipynb` | `dd781497b1d4aebc05352e8bb51f3abdd950b732f7406b8fe6efda240992d62d` | unchanged (no local path) |
| `tabpfn_classifier_colab_fcfa6ff_colab-cli-t4_run_summary.json` | `2da1ca66ad616bf1b24a0e3e9402703e9a4e36af0c570ca42e5472f4e4b5286b` | unchanged (no local path) |
| `tabpfn_classifier_colab_fcfa6ff_colab-cli-t4_exec.log` | `a6dc465fcc8efc8f5918f38f82be418579172a9c36a001f4f8556820a9782668` | `e827b1f4dca093669219a5c4f4a42e78932842724f894d7bfcc01cf81145c366` |
| `tabpfn_classifier_colab_fcfa6ff_colab-cli-t4_suite.log` | `3c833899455cf92060665079b6e41e4d7e4e61575a087c1e1f2cda0f6722a292` | `ca55c93e17860557f1214414717fb15e576d7f3159a1f06213aad15b0d4c8147` |

**Hosted default-path run of blob `7d8051f712ad` (2026-10-22, `197d45b`; the current blob until `c6e880b`, now history).** Colab CLI 0.7.5 `colab exec -f`
(a scripted run, not a browser Run all) on a fresh Colab Tesla T4, default path of `tutorials/tabpfn_classifier_colab.ipynb`
at `197d45b`, executed as committed, started 2026-10-22T13:21:05Z, wall 220.9 s, exit 0, sequential, input unchanged, 14/14
code cells, 0 errors, verdict PASS. Evidence: `docs/execution-evidence/2026-10-22-197d45b/`. Printed: validation accuracy 0.9778 / log loss
0.0658 / ROC AUC 0.9994, test accuracy 0.9778 / ROC AUC 0.9978 (cuda, `n_estimators` 4); "Fresh-boundary verification PASSED:
probabilities equivalent on 90 validation rows"; Section 10 reported `trusted_digest: verified (pinned SAMPLE_ARTIFACT_SHA256 and
SAMPLE_ARTIFACT_FITTED_SHA256)` for `8fe248e2…`; the activity cell printed "Optional activity skipped". Local paths were replaced
by `<local-run-dir>` before the first commit:

| File (under `execution-evidence/2026-10-22-197d45b/`) | Captured | Committed |
|---|---|---|
| `tabpfn_classifier_colab_197d45b_colab-cli-t4.ipynb` | `cdf29a508424e6c0dfa9a5bd1e6780d1e5ee2461e8a5fb7a01cfdbb7943b3dae` | unchanged (no local path) |
| `tabpfn_classifier_colab_197d45b_colab-cli-t4_exec.log` | `1f4e670511a9ed976d06c9dc76693269ca8dd8afe2b812392a276d8c7bcdcdef` | `dd55e5f6aabac33c34f9ffea5e207b6d73a166672c98155cbe995e4fe0ba4395` (2 replacements) |
| `tabpfn_classifier_colab_197d45b_colab-cli-t4_run_summary.json` | `6137e1353f5bd320a53075bca09213e947fdc3a7c4f09895a89b7f7009836e6c` | unchanged (no local path) |

**Optional-activity run of blob `7d8051f712ad` (2026-10-22, `197d45b`, `RUN_ACTIVITY = True`; history).** A run copy of
committed blob `7d8051f7` with one override, `RUN_ACTIVITY = False` → `True` (run copy blob `586d5f40fe54`, SHA-256
`1d36ec2ecbff48c3991f0a58a7a2da2e243093a20f1a12d59e595d6dc5d08d47`; it differs from the committed blob only in that assignment
literal and is committed beside the evidence as `…_activity_run_copy.ipynb`), run by Colab CLI 0.7.5 `colab exec -f` (scripted,
not a browser Run all) on a fresh Colab Tesla T4, started 2026-10-22T13:38:45Z, wall 316.5 s, exit 0, sequential, input (the run
copy) unchanged, 14/14 code cells, 0 errors, verdict PASS. Evidence: `docs/execution-evidence/2026-10-22-197d45b-activity/`. The Section 11 activity
(`n_estimators` 4 → 1) printed validation accuracy 0.9778 → 0.9778, log loss 0.0658 → 0.0607, ROC AUC 0.9994 → 0.9998 on 90
validation rows (`one_row_resolution` 0.0111), `max_abs_probability_change` 0.1001, `labels_changed` 0, and
`{'canonical_outputs_unchanged': True}`. Local paths were replaced by `<local-run-dir>` before the first commit:

| File (under `execution-evidence/2026-10-22-197d45b-activity/`) | Captured | Committed |
|---|---|---|
| `tabpfn_classifier_colab_197d45b_colab-cli-t4_activity.ipynb` | `fb79400bde4e29d89bde744b458334e0922ac3a83aed247f780430982ffb7643` | unchanged (no local path) |
| `tabpfn_classifier_colab_197d45b_colab-cli-t4_activity_exec.log` | `f6edb1da7dcb269f82b736280f772138027392e250cdf2fc1500cdfe31cb037b` | `b4a591f8e3c1ae0cef4d3eaa8f4e5a9485b24e95c7aab3d88fc9b44795665531` (2 replacements) |
| `tabpfn_classifier_colab_197d45b_colab-cli-t4_activity_run_summary.json` | `9655dc7351df3e0bdec272ce7647b477e1ce8f5862a47cea4acf0fadf207d468` | unchanged (no local path) |
| `tabpfn_classifier_colab_197d45b_colab-cli-t4_activity_run_copy.ipynb` | `1d36ec2ecbff48c3991f0a58a7a2da2e243093a20f1a12d59e595d6dc5d08d47` | unchanged (no local path) |

**Hosted default-path run of blob `e705db6eb176` (2026-10-23, `9e69f88`; the current blob).** Colab CLI `colab exec -f`
(a scripted run, not a browser Run all) on a fresh Colab Tesla T4, default path of `tutorials/tabpfn_classifier_colab.ipynb`
at `9e69f88`, executed as committed, started 2026-10-23T08:53:20Z, wall 142.0 s, exit 0, sequential, input unchanged, 14/14
code cells, 0 errors, verdict PASS. Evidence: `docs/execution-evidence/2026-10-23-9e69f88/`. Printed: majority-class baseline
validation accuracy 0.3333; TabPFN validation accuracy 0.9778 / log loss 0.0658 / ROC AUC 0.9994, test accuracy 0.9778 / log
loss 0.0794 / ROC AUC 0.9978; logistic-regression reference 0.9111 (validation) / 0.9444 (test); "Fresh-boundary verification
PASSED: probabilities equivalent on 90 validation rows (rtol=1e-05, atol=1e-06)"; Section 10 reported `trusted_digest: verified
(pinned SAMPLE_ARTIFACT_SHA256 and SAMPLE_ARTIFACT_FITTED_SHA256)` for `8fe248e2…`; the activity cell printed "Optional activity
skipped". The default path supplies no refused user input, so the new `Refused:` display was not exercised by this run. Local
paths were replaced by `<local-run-dir>` before the first commit:

| File (under `execution-evidence/2026-10-23-9e69f88/`) | Captured | Committed |
|---|---|---|
| `tabpfn_classifier_colab_9e69f88_colab-cli-t4.ipynb` | `2d27ba305f0d1c17b660ed585a1ca2f89ef8ba050c74d27dc20a1114f48f7e19` | unchanged (no local path) |
| `tabpfn_classifier_colab_9e69f88_colab-cli-t4_exec.log` | `c305728676a76dbc4a888cdb4ebb0f7424fab8e3513e86d26b8da0f7425d53a2` | `1f0b18e82f6338d9b65de157eedebd511e1d7c6e8ffb69eb438322eae48d4de7` (2 replacements) |
| `tabpfn_classifier_colab_9e69f88_colab-cli-t4_run_summary.json` | `3648cc0fd260d98aba47625f67238fe72ab109058b88cbf3c66b48f25c0b1a5e` | unchanged (no local path) |

**Optional-activity run of blob `e705db6eb176` (2026-10-23, `9e69f88`, `RUN_ACTIVITY = True`; the current blob).** A run copy
of committed blob `e705db6e` with one override, `RUN_ACTIVITY = False` → `True` (run copy blob `9cb5e29298ac`, SHA-256
`e9d59aef2f2a891f18ba616d5d97b9f48e34ae75e66c224225b7385ee22564ab`; it differs from the committed blob only in that assignment
literal and is committed beside the evidence as `…_activity_run_copy.ipynb`), run by Colab CLI `colab exec -f` (scripted, not a
browser Run all) on a fresh Colab Tesla T4, started 2026-10-23T08:53:45Z, wall 319.7 s, exit 0, sequential, input (the run copy)
unchanged, 14/14 code cells, 0 errors, verdict PASS. Evidence: `docs/execution-evidence/2026-10-23-9e69f88-activity/`. The
canonical stages printed the same metrics as the default-path run above. The Section 11 activity (`n_estimators` 4 → 1)
printed validation accuracy 0.9778 → 0.9778, log loss 0.0658 → 0.0607, ROC AUC 0.9994 → 0.9998 on 90 validation rows,
`max_abs_probability_change` 0.1001, `labels_changed` 0, and `{'canonical_outputs_unchanged': True}`. Local paths were
replaced by `<local-run-dir>` before the first commit:

| File (under `execution-evidence/2026-10-23-9e69f88-activity/`) | Captured | Committed |
|---|---|---|
| `tabpfn_classifier_colab_9e69f88_colab-cli-t4_activity.ipynb` | `d506a341411a616e3c6830f21c96a7e020d005ce6226253b16d70fa7c1164ff8` | unchanged (no local path) |
| `tabpfn_classifier_colab_9e69f88_colab-cli-t4_activity_exec.log` | `027769edf6e3e6a40ae1cfdada2ecbde230faa37e9da0fcfd9bdb432ec07ae59` | `9a95a32f96c128eb2bff901211cb589e0046d6262784391bb318182db81824f9` (2 replacements) |
| `tabpfn_classifier_colab_9e69f88_colab-cli-t4_activity_run_copy.ipynb` | `e9d59aef2f2a891f18ba616d5d97b9f48e34ae75e66c224225b7385ee22564ab` | unchanged (no local path) |
| `tabpfn_classifier_colab_9e69f88_colab-cli-t4_activity_run_summary.json` | `3babac27e56c93db90bb29ba08b8a1f1a31fe2c4a8c814a559913bf74be85a2c` | unchanged (no local path) |

### Serialization scan of the pinned sample artifact (2026-10-23, retroactive)

ARTIFACT_SPEC 2.0 (ml-worker#90): SER9 (b) for `model.ckpt`, a byte copy of the upstream code-capable checkpoint kept so the bundle can be checked against the pinned upstream identity; SER9 (a) for `model.tabpfn_fit`, TabPFN's own fitted-state format, which has no faithful non-code-executing equivalent. Because `model.ckpt` is a plain tensor state dict, a faithful `safetensors` conversion exists, so under SER9 (b) the next release must publish that conversion beside the byte copy; this asset does not have it yet. Both files need a recorded trust note (SER6: "Artifact contract and trust boundary" in [`tutorials/README.md`](../tutorials/README.md)) and a scan with an acceptable outcome (SER13). SER13 asks for the scan **before** publication; this asset was published on
2026-10-22, before 2.0 existed, so this is a **retroactive** scan of the published bytes, not the pre-publication control. The
next sample artifact is to be scanned before upload and published only at the digest that passed.

The release asset `sample-artifact-synthetic-v1/tabpfn_classifier_artifact.zip` was downloaded on 2026-10-23; its SHA-256 (`8fe248e2f9841c452316d07ee3e88e5fcde89ec458ff630954028b1c22df1054`) matches the pinned value. `scan.py`
(picklescan 1.0.5, joblib 1.6.0, numpy 2.5.4, Python 3.13) extracted it with member-path checks, reached every nested archive,
and judged each pickle file on its own: **dangerous** if any global is dangerous, **incomplete** if no recorded method read the
whole file, otherwise **clean**. Content decides what is a pickle, not the file name, and every archive (PyTorch zips
included) is unpacked so each member gets its own verdict.

The two joblib files interleave raw numpy bytes with pickle instructions, so picklescan cannot read them to the end. For those, `scan.py` runs a **restricted load**: joblib's own reader with `find_class` replaced, so only numpy's array and scalar rebuilders, `numpy.dtype` and joblib's `NumpyArrayWrapper` are real and run; every other global is an inert stub, recorded, and classified with picklescan's own tables. Payloads of `torch.storage._load_from_bytes` are saved as files and scanned like any other file, never loaded.

| File | Kind | Verdict |
|---|---|---|
| `model.tabpfn_fit` | archive | extracted |
| `model.ckpt` | pytorch-zip | extracted |
| `model.tabpfn_fit!/fitted_attrs.joblib` | joblib | clean |
| `model.tabpfn_fit!/executor_state.joblib` | joblib | clean |
| `model.ckpt!/archive/data.pkl` | pickle | clean |

- **Result:** 3 pickle files; clean=3 dangerous=0 incomplete=0; `scan_status=0`. Dangerous globals: none.
- **Globals (distinct, all files):** 43, by top-level package: tabpfn (28), numpy (5), sklearn (5), torch (2), builtins (1), collections (1), joblib (1). The per-file lists with picklescan's
  classification are in the report.
- **Evidence:** `docs/execution-evidence/2026-10-23/ser13-sample-artifact-scan/` holds `scan-report.txt` exactly as `scan.py`
  wrote it (it records only paths inside the asset), `scan.py`, and `test_scan.py`, its regression tests (dangerous globals
  behind joblib buffers or beside joblib files, protocol-0 pickles under other names, pickle/zip polyglots, every PyTorch zip
  member, object-dtype .npy files, torch payloads embedded in complete pickles, compressed (including zlib of any
  window size) and truncated pickles, nested zip and tar archives that try to overwrite queued members, duplicate,
  escaping and malformed members).
  `scan.py` exits 0 clean, 1 dangerous, 2 incomplete, 3 could not scan.

### Local lock-only drives (not clean-runtime evidence)

Code cells driven in order by a plain-CPython stand-in kernel on Linux x86_64 (WSL2), CPU only (`CUDA_VISIBLE_DEVICES=-1`), inside each notebook's own isolated hash-locked uv environment (CPython 3.12.12; torch 2.11.0, tabpfn 8.1.0, numpy 2.5.3, pandas 2.3.2, scikit-learn 1.9.0; 63 locked packages), default form values. The checkpoint was copied in from a local checkout and digest-verified (`d0d865d5…`), so the Hub download was not exercised; no notebook kernel, no Colab.

| Date (UTC) | Commit / notebook blob | Path exercised | Wall | Outcome |
|---|---|---|---|---|
| 2026-10-10 | `08dbf95` / `a4c8b2904e71` | E2E, default synthetic sample, `N_ESTIMATORS=4` | 210.6 s | 11/11 code cells ok, `restarted`: no. Majority-class baseline val accuracy 0.3333; logistic regression val 0.9111 / test 0.9444; TabPFN val accuracy 0.9778, log loss 0.0668, ROC AUC 0.9994; test 0.9778 / 0.0804 / 0.9972. Fresh-boundary reload PASSED (rtol 1e-5, atol 1e-6). Its exports are the producer of `sample-bundle-v1` |
| 2026-10-10 | working tree on `08dbf95` / `6d7b0040df03` | Artifact inference, default pinned sample (pre-seeded, so the release download was not exercised), then the tamper activity | 108.1 s | 9/9 code cells ok, `restarted`: no. Sample verified before extraction (169,881 bytes, `d61f1c2642154c2a…`); predictions on the eight rows identical to the E2E run's; the tampered `model.tabpfn_fit` was refused before loading |

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

**Current status (2026-10-23): Candidate.** The current E2E notebook blob is `e705db6e` (`tutorials/tabpfn_classifier_colab.ipynb`, regenerated from code commit `c6e880b` and recorded at commit `9e69f88` on `fix/refusal-display`: a refused user input — the BYOD table, Section 9 rows, your own Section 10 artifact ZIP or rows — is shown as one `Refused: <file>: <reason>` line with no traceback, names the uploaded file rather than `train.csv`, and Run all stops at that cell; NOTEBOOK_SPEC 3.0 DAT19/DAT20). **Default path covered:** hosted Colab T4 run at `9e69f88` on 2026-10-23 (PASS, 14/14 code cells, 0 errors). **Optional activity covered:** a `RUN_ACTIVITY = True` run copy of the same blob at `9e69f88` on 2026-10-23 (PASS, 14/14 code cells, 0 errors). Both records are under *Recorded executions*. The `197d45b` default-path and optional-activity records cover the previous blob `7d8051f7` and are history, as is the earlier `fcfa6ff` record of blob `d766d7b7` (REL14). The notebook stays **Candidate** pending only these two:

- **REL12:** a BYOD run in which one representative input is accepted through the downstream stages and one incompatible input is refused;
- **REL2:** a browser run of the interactive journeys (BYOD upload dialog, optional-activity toggle, own-artifact ZIP); the scripted CLI runs count for the default path and the activity only.

Promotion is an integrator decision; the builder does not change the status.

**History, 2026-10-10 (blobs `a4c8b2904e71` and `6d7b0040df03`, both superseded; not evidence for the current blob, REL14) — one-pass hosted runs of both then-current blobs were recorded** (first two rows of the clean-runtime table): `tabpfn_classifier_colab.ipynb` (`a4c8b2904e71`) and `tabpfn_classifier_artifact_inference_colab.ipynb` (`6d7b0040df03`) at `0e5a27c`, each by Colab CLI 0.7.4 sequential execution on a fresh Colab Tesla T4 (not a browser Run all), no restart, 0 error outputs. Notes for the reviewer: (a) the companion's carried revision label reads `uncommitted` and the E2E's reads `ea1d63e` (generation-time HEAD, not `0e5a27c`) — a provenance follow-up; (b) the tamper activity was not exercised (`RUN_ACTIVITY` off by default), only the `SCHEMA_MISMATCH` refusal ran; (c) the `sample-bundle-v1` release download and its SHA-256 verification were exercised for real. Both notebooks stayed **Candidate**; the gates then remaining were reviewer confirmation of these runs, the BYOD journeys, and the optional activities (including the tamper refusal on a hosted runtime).

Before 2026-10-10: no clean-runtime execution of the current standalone notebook blobs had been recorded; the only clean-room run was the
Kaggle CPU run of an earlier E2E blob above. Static validation (`tools/validate_release_assets.py`), the generator parity checks, a `compile()` sweep
over every code cell, and the offline unit suite passed on the tutorial source at the candidate revision, which is
necessary but not sufficient. The registry status remains **Candidate** until a reviewer confirms recorded runs against
the notebook blobs under review and an integrator promotes them; promotion is not performed by the builder. Facts a
reviewer should weigh: the pinned checkpoint `tabpfn-v3-classifier-v3_default.ckpt` was staged by the Kaggle run and
staged, digest-verified and loaded in the 2026-10-10 local lock-only drives; the module's `tabpfn` calls
(`TabPFNClassifier(model_path=...)`, `save_fitted_tabpfn_model`, `load_fitted_tabpfn_model`) ran on the real package in
those drives, and only through injected fakes in the unit suite; the
standalone carrier was validated statically and by a CPU carrier probe (module cells + identity assertion, no fetch).

## Local-path redaction of earlier evidence (2026-10-22)

The files below were committed with local absolute paths in them. Each local run-directory prefix was replaced with `<local-run-dir>/`, the convention the evidence committed from 2026-10-22 on already uses; the local run tree named in `PHASE2_ACCEPTANCE_REPORT.md` became `<local-run-dir>`. No other byte changed and line endings are unchanged. Git history still holds the original files. "Before" is the SHA-256 of the file as committed on `main` at `b6dd9d0`.

| File | Replacements | SHA-256 before | SHA-256 after |
| --- | --- | --- | --- |
| `PHASE2_ACCEPTANCE_REPORT.md` | 1 | `4371ee2f18b2597a4c15e6cf6365fe5f21e70dbf781dc673ea3c321a7de478ad` | `db3339cfbc3758a59a61878cb2b4df8e1a0cc72ebeaeb88e650d0f844db896a2` |
| `docs/execution-evidence/2026-10-10/tabpfn_classifier_artifact_inference_colab_0e5a27c_colab-cli-t4_exec.log` | 2 | `918a69fdec3aa730050f3fb8340558409acbbca956effb658a230428557dc67a` | `136f147c3d69d6c4a824b19fb327e983377708d152fbd2c90c8ccede4ce418c1` |
| `docs/execution-evidence/2026-10-10/tabpfn_classifier_colab_0e5a27c_colab-cli-t4_exec.log` | 2 | `c2874688ec26e323e88a5e62f50f534f5d24304726ca177edc7ae9c540245bdb` | `427bd2876eefc9560235f2c0d93aba12fd23b9f81d1ec3afb3b3e9f951d0e79c` |

The Recorded executions table above quotes the pre-redaction hash of 2 of these logs as the hash of the committed file. Those quotes are kept as recorded, and the post-redaction hash is noted beside each one.
