# Serving the TabPFN artifact

A DIMER training run produces two complementary files:

- `model.tabpfn_fit` — fitted estimator/preprocessing/in-context state;
- `model.ckpt` — the foundation/fine-tuned neural-network weights.

TabPFN's upstream fitted-state format intentionally omits foundation weights. Its normal loader reconstructs the estimator using the `model_path` recorded when the fitted state was saved. That path may refer to the training container's cache and therefore may not exist after DIMER relocates the artifact.

Use [`load_artifact.py`](load_artifact.py) for DIMER serving. It:

1. locates `model.tabpfn_fit` and `model.ckpt` in the deployed artifact directory;
2. creates a temporary copy of the fitted-state archive;
3. rewrites only `init_params.json:model_path` to the deployed companion checkpoint;
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
