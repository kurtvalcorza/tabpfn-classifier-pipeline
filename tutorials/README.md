# TabPFN Classification Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/tabpfn-classifier-pipeline)
[![Upstream](https://img.shields.io/badge/Upstream-PriorLabs%2FTabPFN-181717?style=flat&logo=github&logoColor=white)](https://github.com/PriorLabs/TabPFN)

**DIMER Notebook Specification:** 1.0 (2026-09-10)

These notebooks exercise the repository's production code path — the DIMER fine-tuner worker (`tabpfn-classifier-finetuner`, cloned at the commit pinned in [`COMPONENTS.json`](../COMPONENTS.json)) and the serving loader ([`serving/load_artifact.py`](../serving/load_artifact.py)). Static validation runs in CI; a notebook is promoted from release candidate to release-grade only after clean-runtime execution evidence is recorded for the exact release revision.

## Notebook registry

| Notebook | Profile | Capability | Default runtime | Release status |
|---|---|---|---|---|
| [`tabpfn_classifier_colab.ipynb`](tabpfn_classifier_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tabpfn-classifier-pipeline/blob/main/tutorials/tabpfn_classifier_colab.ipynb) | `E2E` | Pinned worker revision check, synthetic sample or BYOD ZIP, `train.run()` exactly as DIMER invokes it (zero-shot ICL default; GPU fine-tune opt-in with explicit fallback reporting), trivial baseline, fresh reload through the serving loader with recorded-metric reproduction, new-data inference, CSV/JSON exports and provenance | Python 3.11+; CPU sufficient for the zero-shot default; CUDA required for effective fine-tuning; `MODEL_VERSION=v2` default (Apache-derived licence) | **release candidate** — clean-runtime execution evidence required |
| [`tabpfn_classifier_artifact_inference_colab.ipynb`](tabpfn_classifier_artifact_inference_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tabpfn-classifier-pipeline/blob/main/tutorials/tabpfn_classifier_artifact_inference_colab.ipynb) | `ARTIFACT-INFERENCE` | Validate an externally supplied `artifact_manifest.json` + `model.tabpfn_fit` + `model.ckpt` bundle (digests, schema, archive safety) before deserialization, reconstruct the estimator via the serving loader without refit or weight download, validate/score new rows, export predictions/provenance | Python 3.11+; CPU sufficient | **release candidate** — external-artifact clean-runtime evidence required |

## Runtime and model contract

- Dependencies are pinned in [`requirements-colab.txt`](requirements-colab.txt): `torch==2.11.0` (the finetuner's Docker base image version) plus the finetuner's own `requirements.txt` pins (`tabpfn==8.1.0`, `pandas==2.3.2`, `scikit-learn==1.9.0`, `requests`, `boto3`).
- The worker is pinned by immutable commit in `COMPONENTS.json`; the E2E notebook refuses to continue if the clone is not at that commit.
- Both notebooks pin the **pipeline** revision too: `TUTORIAL_REF` (default `main`, overridden by the `DIMER_TUTORIAL_REF` environment variable) is fetched and checked out detached, a 40-character ref is verified against the resolved commit, and the resolved SHA is printed and written to provenance as `pipeline.commit`. `scripts/execute_notebook_release.py` sets it to the commit it staged. Without this an execution record names a worker commit but not the repository revision whose loader and example builder actually ran.
- `tabpfn-classifier-finetuner` is private. The E2E notebook needs a `GITHUB_TOKEN` with read access, supplied as a Colab Secret or an environment variable, and fails closed without one. The token reaches Git only through an ephemeral `http.<url>.extraheader` in the clone subprocess environment — never a credential-bearing URL, never persisted `.git/config`, never printed or recorded in provenance.
- Every line that clones or installs sits inside a `# >>> colab-bootstrap` … `# <<< colab-bootstrap` region. `scripts/execute_notebook_release.py --skip-bootstrap` drops exactly those regions, fails closed if a notebook has none, and CI rejects environment setup that escapes one — so a notebook run against a pre-staged checkout can never silently re-clone over it.
- Model generation is explicit (`MODEL_VERSION`). The tutorial defaults to `v2` because its weights carry Prior Labs' Apache-derived licence; DIMER's default `v3` is selectable but its weights are under `tabpfn-3-license-v1.0`, whose Non-Commercial Purpose excludes production deployment. As checked on 2026-09-11 neither `Prior-Labs/TabPFN-v2-*` nor `Prior-Labs/tabpfn_3` is access-gated on Hugging Face.
- Weights are resolved by the `tabpfn` package for the selected generation; the worker records the resolved base-model path and a SHA-256 only when DIMER mounts a checkpoint through `DIMER_TABPFN_MODEL_PATH`.

## Fine-tuning semantics

`FINE_TUNE=False` (default) runs zero-shot in-context learning: `fit()` registers the training rows and performs no gradient update. `FINE_TUNE=True` requests Prior Labs' gradient fine-tuning wrapper, which is effective only on a CUDA device; without one the worker **does not fail** — it falls back to zero-shot and records `fineTuneEffective=false` and `fineTuneSkippedReason`, which the notebook prints. A zero-shot run and a fine-tuned run are different estimators; never compare their metrics without `fineTuneEffective` in view.

## Artifact contract and trust boundary

The deployable artifact is `model.tabpfn_fit` (fitted estimator state including the in-context training rows) + `model.ckpt` (foundation weights) + `artifact_manifest.json` (task type, target, ordered feature columns, classes, file names and SHA-256 digests). The fitted archive intentionally omits the weights and records a `model_path`; `serving/load_artifact.py` rewrites that path in a temporary copy to the companion checkpoint and never modifies the original. Because the artifact contains training rows, treat it with the same confidentiality, licensing, and retention controls as the source dataset.

Digest checks establish internal consistency, not sender authenticity. `model.tabpfn_fit` and `model.ckpt` contain serialized Python/torch state: loading them executes trusted model state, and the archive path-safety checks do not change that.

## Release verification

CI covers notebook JSON/source structure, Python-cell compilation, profile metadata, runtime-floor markers, worker-invocation and reload-equivalence markers, and ordinary repository tests. These are not evidence that the current Colab runtime, model host, and worker revision execute together.

`scripts/execute_notebook_release.py` needs `nbclient`, `nbformat`, and `ipykernel` in addition to the lock set; it runs both notebooks through real IPython kernels — the E2E notebook first, then the artifact-inference notebook in a second fresh kernel fed with the first run's bundle and separately generated rows — and writes an evidence JSON. Record each such run below with the notebook commit, environment, and outcome. Until a clean-runtime record exists for the candidate commit, the correct status is **release candidate**.

### Execution records

Each row is evidence for the revision named in its own `pipeline.commit`, and for nothing later. The rows below predate the
secure private-source bootstrap and the `# >>> colab-bootstrap` harness contract, so they are
**superseded**: they do not carry to the current candidate head, and no Colab record exists for any
revision. A fresh `scripts/execute_notebook_release.py` record at the candidate head, and then a
clean Colab record, are both still required before either notebook is marked release-grade.

| Date (UTC) | Notebook revision | Environment | Engine | Outcome |
|---|---|---|---|---|
| 2026-09-11 | `247c4d381d93` (clean tree); worker `8ebcfb6dd56a` | Local host `Kurt-Valcorza`, Windows-11-10.0.26200-SP0; Python 3.12.10; torch 2.11.0+cu128; tabpfn 8.1.0; GPU NVIDIA GeForce RTX 5070 Ti Laptop GPU; `MODEL_VERSION=v2`, default zero-shot | `scripts/execute_notebook_release.py --skip-bootstrap`: nbclient, one fresh kernel per notebook; `/content` → `D:/tabpfn-clf-run/content`; worker cloned from the local finetuner checkout at the pinned commit; `google.colab.files.upload()` served from explicit paths | **PASS** — E2E: `mode=zero-shot-icl`, `fineTuneEffective=false`, validation accuracy 0.9667 / logLoss 0.0796 vs trivial baseline; reload via serving loader reproduced the recorded validation metric within 1e-6. ARTIFACT-INFERENCE (second kernel, bundle + 8 fresh rows supplied externally): reconstructed and scored 8 rows. **Not a Google Colab run** — a Colab record is still required before either notebook is marked release-grade. |
| 2026-09-11 | `247c4d381d93` (clean tree); worker `8ebcfb6dd56a` | Local host `Kurt-Valcorza`, Windows-11-10.0.26200-SP0; Python 3.12.10; torch 2.11.0+cu128; tabpfn 8.1.0; GPU NVIDIA GeForce RTX 5070 Ti Laptop GPU; `MODEL_VERSION=v2`, FINE_TUNE=True | `scripts/execute_notebook_release.py --skip-bootstrap --set FINE_TUNE=True`: nbclient, one fresh kernel per notebook; `/content` → `D:/tabpfn-clf-ft/content`; worker cloned from the local finetuner checkout at the pinned commit; `google.colab.files.upload()` served from explicit paths | **PASS** — E2E: `mode=fine-tune`, `fineTuneEffective=true`, validation accuracy 0.9667 / logLoss 0.0866 vs trivial baseline; reload via serving loader reproduced the recorded validation metric within 1e-6. ARTIFACT-INFERENCE (second kernel, bundle + 8 fresh rows supplied externally): reconstructed and scored 8 rows. **Not a Google Colab run** — a Colab record is still required before either notebook is marked release-grade. |

