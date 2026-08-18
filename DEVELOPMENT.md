# Development

The umbrella repository owns the DIMER integration contract, documentation, examples, and serving helper. It intentionally does not duplicate validator/fine-tuner source.

Local checks:

```bash
pip install -r requirements-test.txt
python -m py_compile examples/build_synthetic_dataset.py serving/load_artifact.py
pytest -q
```

When a component changes, review its CI and update `COMPONENTS.json` to the exact approved commit. Production acceptance additionally requires a real CUDA fine-tuning run and an end-to-end DIMER artifact-serving test.
