# Development

The umbrella repository owns the DIMER integration contract, documentation, examples, and serving helper. It intentionally does not duplicate validator/fine-tuner source.

Local checks:

```bash
pip install -r requirements-test.txt
python -m py_compile examples/build_synthetic_dataset.py serving/load_artifact.py
pytest -q -m "not integration"
ruff check .
python tools/validate_release_assets.py
python tools/build_notebook.py --check
python tools/build_notebook.py --check --template tools/notebook_template_artifact_inference.py
```

The standalone tutorials are generated from `tools/notebook_template*.py` and `src/tabpfn_classifier_pipeline/pipeline.py`; never edit the `.ipynb` files by hand.

When a component changes, review its CI and update `COMPONENTS.json` to the exact approved commit. Production acceptance additionally requires a real CUDA fine-tuning run and an end-to-end DIMER artifact-serving test.
