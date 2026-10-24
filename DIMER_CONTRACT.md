# DIMER contract: TabPFN artifact bundles

The worker envelope (`result.json`) is in `CONTRACT.md`; this file records the artifact bundle format.

## Bundle format v1 → v2

This section is identical in kurtvalcorza/tabpfn-classifier-pipeline, kurtvalcorza/tabpfn-regressor-pipeline,
kurtvalcorza/tabpfn-classifier-finetuner and kurtvalcorza/tabpfn-regressor-finetuner. The format logic lives in
`safetensors_weights.py` (canonical in the classifier pipeline; byte-identical copies elsewhere, each pinned by SHA-256).

**v2 (current; written by every producer).** An artifact directory holds exactly:

| File | Content |
|---|---|
| `artifact_manifest.json` | `schemaVersion` 2, `bundleFormat` `tabpfn-dimer-classifier-v2` or `tabpfn-dimer-regressor-v2` (by `taskType`), `fittedEstimator` `model.tabpfn_fit`, `foundationCheckpoint` = `modelPath` = `model.safetensors`, `payloadFiles` `["model.safetensors", "model.tabpfn_fit"]`, `foundationCheckpointSha256` / `foundationCheckpointBytes` (that file's digest and size), `fittedEstimatorSha256`, and the task fields (target, features, classes or target statistics). |
| `model.safetensors` | The network weights in tabpfn 8.1.0's own safetensors layout: the state dict as tensors, every other checkpoint field as sorted JSON header metadata, a `dimer_conversion` provenance field, a sorted header (byte-reproducible). tabpfn opens it with `safetensors.safe_open`; no `torch.load`. |
| `model.tabpfn_fit` | tabpfn's fitted-estimator archive (unchanged): `init_params.json` plus joblib-pickled fitted state, which can carry code. Its `model_path` is rewritten at load time to `model.safetensors`. |

- A **pretrained** (zero-shot) v2 bundle is bound to the pinned conversion of the base checkpoint (`CONVERTED_SHA256`). A
  fine-tuned bundle carries the fine-tuned weights; the worker's `result.json` names them in `artifacts.modelArtifact`
  (with `sha256`) and `artifacts.foundationCheckpoint`.
- A worker reads the `.ckpt` that tabpfn wrote (`save_tabpfn_model`) once with `torch.load(weights_only=True)`, converts
  it, deletes it and every training checkpoint pickle, and fails the run if any `.ckpt` / `.pt` / `.pth` / `.bin` /
  `.pkl` file is left in its output.

**v1 (deprecated; read-only).** `schemaVersion` 1, `foundationCheckpoint` `model.ckpt` (a pickle), none of the v2-only
fields. Still accepted by `from_artifact` and `serving/load_artifact.py`: the pickle is read once with
`torch.load(weights_only=True)` into a temporary safetensors file and a `DeprecationWarning` is emitted; tabpfn never
unpickles it. A pretrained v1 bundle stays bound to the source checkpoint (`WEIGHTS_SHA256`). No producer writes v1.

**Refusals, before any weights are read.** A v2 manifest naming `model.ckpt` or missing `bundleFormat` / `modelPath` /
`payloadFiles` / `foundationCheckpointBytes`; a v1 manifest naming `model.safetensors` or carrying v2-only fields; any
other `schemaVersion`; a directory or ZIP holding both `model.safetensors` and `model.ckpt`; a size or digest mismatch.

**Sample artifacts.** `sample-artifact-synthetic-v1` (v1) stays the notebooks' default. Its v2 re-issue
`sample-artifact-synthetic-v2` (fitted archive byte-identical, weights = the pinned conversion, a `reissue` block in the
manifest) is staged and pinned in code, not published; the default moves to it once it is.
