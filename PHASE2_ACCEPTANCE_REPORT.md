# Phase 2 — Local GPU Acceptance Report

**Date:** 2026-08-19
**Runtime host:** Windows 11 + WSL2 `nvidia-docker` distro, Docker 29.6.2
**GPU:** RTX 5070 Ti Laptop, 12 GB VRAM, compute cap 12.0 (Blackwell / sm_120), driver 610.88
**Stack:** `tabpfn==8.1.0`, `torch 2.11.0+cu128` (CUDA 12.8), base image `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`
**Components under test (pinned in `COMPONENTS.json`):** validator `13c2ff0a`, finetuner `1a94cf4c`

## Verdict

The CPU→GPU→artifact→clean-reload chain is **proven end-to-end for both `v2` and `v3`**. Two real, previously-unproven defects were found, fixed, and verified (now on PR [#2](https://github.com/kurtvalcorza/tabpfn-classifier-finetuner/pull/2)). `v3` (TabPFN-3, gated non-commercial) required accepting the license via the **HuggingFace gate** and a manual weight download (`DIMER_TABPFN_MODEL_PATH`), because the Prior Labs portal license-acceptance flow never registered against the API token's account (see §v3). DIMER serving (gate 8) is not yet exercised.

## Gate results (`DEPLOYMENT.md §2`)

| # | Gate | Result | Evidence |
|---|------|--------|----------|
| 1 | Validator CI | PASS (also re-run locally) | 18/18 checks green on synthetic ZIP |
| 2 | Finetuner unit/config CI | PASS (per prior CI) | — |
| 3 | **Finetuner CUDA image builds** | **FAIL as-is → PASS with 1-line fix** | bare `pip install` dies on PEP 668 in the 2.11 base |
| 4 | TabPFN checkpoint licensing reviewed | **RESOLVED (HF gate)** | v3 license accepted via HF gated repo; portal flow was broken (§v3) |
| 5 | Approved model-distribution path | PASS | v2 auto-download (Apache); v3 manual HF download + `DIMER_TABPFN_MODEL_PATH` |
| 6 | **Real GPU fine-tuning smoke test** | **PASS (v2 + v3)** | exit 0, artifacts written, metrics below |
| 7 | **Artifact reload in clean runtime** | **PASS (v2 + v3)** | CUDA reload exact; CPU reload within tol |
| 8 | DIMER serving E2E inference | NOT RUN | needs DIMER PoC wiring |

Validator negatives (all fail-closed, exit ≠ 0): `dup_train`, `unseen_class`, `too_few_rows`, `traversal`, `zip_bomb`.

## v2 fine-tune smoke test (gate 6)

Config: `fine_tune=true, model_version=v2, epochs=2, n_estimators 1/1/2, n_finetune_ctx_plus_query_samples=4000, seed=0`.
Dataset: synthetic 600-row / 3-class → 420 train / 90 val / 90 test, 14 feature columns.

- mode `fine-tune`, device `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- **validation:** acc 0.9778, balAcc 0.9778, logLoss 0.0848, rocAucOvrMacro 0.9956
- **test:** acc 0.9778, balAcc 0.9778, logLoss 0.0952, rocAucOvrMacro 0.9972
- artifacts: `model.tabpfn_fit` (194,514 B, sha256 `dc374f41…fbad7`), `model.ckpt` (29,030,255 B, sha256 `435c7861…9d28b`), `artifact_manifest.json`
- **peak GPU memory (host-total): 5,403 MiB** — vs the 80 GB reference profile in `DEPLOYMENT.md §3`
- wall-clock: 8 s

## v2 artifact round-trip (gate 7)

Reloaded `model.tabpfn_fit` in a **fresh container** via `serving/load_artifact.py` (rewrites `init_params.json:model_path` to the companion `model.ckpt`), re-derived the test metrics, compared to training `result.json`:

- **CUDA reload:** all metric diffs `0.0` (exact reproduction) — PASS at tol 1e-6
- **CPU reload:** acc/balAcc/rocAuc exact; logLoss diff `2.59e-4` (device float noise) — PASS at tol 1e-2

This confirms the portable loader reconstructs the identical fitted estimator after relocation, on GPU and CPU.

## Defects found (previously unproven; CI could not catch either)

### D1 — Finetuner Docker image does not build (gate 3)
`pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime` ships a **PEP 668 externally-managed** system Python, so `Dockerfile:12` `RUN pip install --no-cache-dir -r requirements.txt` fails with `error: externally-managed-environment`.

**Proposed fix (finetuner `Dockerfile`):**
```dockerfile
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt
```
(equivalently `ENV PIP_BREAK_SYSTEM_PACKAGES=1`). Safe in a single-purpose container.

### D2 — Fine-tuned artifact cannot be saved on GPU (blocks gates 6/7/8)
TabPFN 8.1.0's `save_fitted_tabpfn_model` (`model_loading.py:1059-1061`) str-coerces `torch.dtype` **but no other non-JSON init param**, then `json.dump(params)`. The fine-tuned inference estimator's `get_params()` returns `device` (a **tuple**) and `model_path` (a **`ClassifierModelSpecs`**), so the save raises `TypeError: Object of type … is not JSON serializable` and **no artifact is written**.

**Proposed fix (finetuner `train.py`, in `save_artifacts` immediately before `save_fitted_tabpfn_model`):**
```python
# Work around TabPFN 8.1.0 save_fitted_tabpfn_model: it str-coerces torch.dtype
# but not other non-JSON init params (device tuple, ClassifierModelSpecs).
for _k, _v in model.get_params(deep=False).items():
    try:
        json.dumps(_v)
    except (TypeError, ValueError):
        setattr(model, _k, str(_v))
```
Safe because evaluation runs before saving, the foundation weights are persisted separately via `save_tabpfn_model(model.ckpt)`, and `serving/load_artifact.py` overwrites `model_path` (and load supplies `device`) at load time — verified by the passing round-trip above. Both fixes were applied **locally only** (patched Dockerfile via `-f`; a mounted monkeypatch wrapper); the pinned repos were not modified.

## v3 (TabPFN-3) — gated, resolved via HuggingFace {#v3}

Same config as v2 but `model_version=v3`, base weights supplied by `DIMER_TABPFN_MODEL_PATH`.

- mode `fine-tune`, device `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- **validation:** acc 0.9778, balAcc 0.9778, logLoss 0.0665, rocAucOvrMacro 0.9994
- **test:** acc 0.9778, balAcc 0.9778, logLoss 0.0822, rocAucOvrMacro 0.9985
- artifacts: `model.tabpfn_fit` (131,268 B, sha256 `1c13fc04…8a81`), `model.ckpt` (212,825,211 B, sha256 `5616fd8b…e942`)
- base checkpoint `Prior-Labs/tabpfn_3 / tabpfn-v3-classifier-v3_default.ckpt` (212,804,803 B, sha256 `d0d865d5…3988`)
- **peak GPU memory: 7,316 MiB**, wall-clock 18 s
- round-trip: CUDA diffs all `0.0`; CPU logLoss diff `6.80e-5` — both PASS

### License path (important operational finding)

The Prior Labs OSS package gates v3 behind a `TabPFNLicenseError` unless its API (`api.priorlabs.ai/account/license/?version=v3`) returns `accepted:true`. In this environment that endpoint **stayed `{"accepted":false}` across every attempt** — accepting on the ux.priorlabs.ai Licenses tab (generic and repo-specific `?hf_repo_id=tabpfn_3` URL), with a freshly issued, `verify_token:true` API key. The portal acceptance never propagated to the token's account (likely account-split or an unfilled professional/VAT profile gate; possibly a Prior Labs backend issue).

**Resolution:** the license was accepted through the **HuggingFace gate** on `Prior-Labs/tabpfn_3` (a valid acceptance channel — the repo ships the same non-commercial `LICENSE`). The weights were then downloaded with the owner's `HF_TOKEN` and passed to the fine-tuner via `DIMER_TABPFN_MODEL_PATH`, which loads the checkpoint directly and does **not** invoke the portal license check. This is the offline path TabPFN's own docs endorse; it is legitimate because the license was accepted (HF gate) and the use is local non-commercial evaluation, which the license expressly permits.

Two minor follow-ups (not blocking): (a) with `DIMER_TABPFN_MODEL_PATH` set, `result.json` `provenance.model.baseModelSha256` is `null` because `model_provenance` reads the estimator's resolved `ClassifierModelSpecs` object rather than the configured checkpoint path — it could hash `config.model_path` directly. (b) The portal-vs-HF acceptance discrepancy is worth raising with Prior Labs.

## What remains

1. **D1 + D2 repo fixes** — committed on PR [#2](https://github.com/kurtvalcorza/tabpfn-classifier-finetuner/pull/2); pending dual-bot review + merge, then re-pin the finetuner commit in `COMPONENTS.json`.
2. **Gate 8 (DIMER serving E2E)** — wire the artifact into the DIMER PoC serving layer and issue a real inference request.
3. **Resource profile** — measured peak 5.4 GB (v2) / 7.3 GB (v3) on this smoke dataset; keep `DEPLOYMENT.md §3` at the 80 GB starting point until measured on representative data.
4. **Minor** — provenance `baseModelSha256` under `DIMER_TABPFN_MODEL_PATH` (§v3); raise the portal license-acceptance discrepancy with Prior Labs.
5. **v3 licensing for production** — the v3 weights remain **non-commercial**; DIMER production/external enablement still requires a commercial license from Prior Labs regardless of this local eval.

## Reproducibility

All harness scripts and the isolated run tree live under the session scratchpad and `/home/kurt/phase2` in the `nvidia-docker` distro (dataset, negatives, `output-v2`, `output-v3`, `results`, `model-cache`). Cached weights: v2 `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` (29,009,539 B) and v3 `model-cache/v3/tabpfn-v3-classifier-v3_default.ckpt` (212,804,803 B). The v3 base was obtained under the owner's accepted HuggingFace gate for non-commercial use.
