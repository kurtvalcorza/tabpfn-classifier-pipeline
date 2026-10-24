# Serving the TabPFN artifact

A DIMER training run produces two complementary files:

- `model.tabpfn_fit` — fitted estimator/preprocessing/in-context state;
- `model.safetensors` — the foundation/fine-tuned neural-network weights (bundle format v2; a v1 bundle carries `model.ckpt` instead, deprecated).

TabPFN's upstream fitted-state format intentionally omits foundation weights. Its normal loader reconstructs the estimator using the `model_path` recorded when the fitted state was saved. That path may refer to the training container's cache and therefore may not exist after DIMER relocates the artifact.

Use [`load_artifact.py`](load_artifact.py) for DIMER serving. It:

1. locates `model.tabpfn_fit` and the weights in the deployed artifact directory: `model.safetensors` (v2), or a v1 `model.ckpt`, which it reads once with `torch.load(weights_only=True)` into a temporary safetensors file and warns about (`DeprecationWarning`); a directory with both, or a manifest that disagrees with the files, is refused;
2. creates a temporary copy of the fitted-state archive;
3. rewrites only `init_params.json:model_path` to the safetensors weights;
4. calls TabPFN's official `load_fitted_tabpfn_model` on the temporary archive;
5. leaves the persisted artifact unchanged.

Example:

```python
from serving.load_artifact import load_dimer_tabpfn_artifact

model = load_dimer_tabpfn_artifact("/models/current", device="cuda")
probabilities = model.predict_proba(frame)
predictions = model.predict(frame)
```

The serving image should pin a TabPFN version compatible with the training artifact (`8.1.0` for the initial pipeline) and should verify the checkpoint/artifact hashes from the training result before loading them.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
