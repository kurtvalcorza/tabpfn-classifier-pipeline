---
license: other
license_name: tabpfn-3-license-v1.0
license_link: LICENSE
extra_gated_fields:
  Organization: text
  Role:
    type: select
    options:
    - Field practitioners
    - Researcher
    - Student
  Use-case: text
  May we contact you about future updates?: checkbox
extra_gated_button_content: Agree to license terms and send request to access repo.
extra_gated_description: "Model weights released under `tabpfn-3-license-v1.0`. This license is designed to be permissive for research and internal evaluation. It *explicitly allows* testing, evaluation, and internal benchmarking, so an organization can download the model and run preliminary assessments on its own datasets.\nThe key restriction is that the model, its derivatives, and its outputs cannot be used for any commercial or production purpose. This includes, but is not limited to, revenue-generating products, competitive benchmarking for procurement, client deliverables, or using the model’s results for internal commercial decision-making.\nFor all production use cases, we offer a *Commercial Enterprise License*. This provides access to our proprietary high-speed inference engine, dedicated support, integration tooling, and other internal models. Please contact us at sales@priorlabs.ai for commercial licensing inquiries."
pipeline_tag: tabular-classification
tags:
- chemistry
- biology
- finance
- legal
- climate
- medical
---
### Model Overview
TabPFN-3 is a transformer-based foundation model that uses in-context-learning to solve tabular prediction problems in a forward pass. 
Inference code can be found at [https://github.com/PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN).
More details can be found in the [Model Report](https://arxiv.org/pdf/2605.13986).

### Getting started
First, install the inference package:
```bash
pip install tabpfn
```

Fitting a classifier and predicting looks like this:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

clf = TabPFNClassifier()
clf.fit(X_train, y_train)
prediction_probabilities = clf.predict_proba(X_test)
predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))
```

For more examples (e.g. how to train a regressor), see the github repo: [https://github.com/PriorLabs/tabPFN](https://github.com/PriorLabs/tabPFN)!

### Specialized checkpoints

TabPFN-3 ships with default classification and regression checkpoints, plus a few experimental specialized variants. We recommend starting with the defaults — the variants can be useful in ensembling or HPO setups, or tried manually in the regime they were trained for. Their name suffixes indicate what we expect them to be good at.

The following specialized checkpoints are available:

| Checkpoint | Task | Specialization |
| --- | --- | --- |
| [`tabpfn-v3-classifier-v3_20260417_binary.ckpt`](https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/tabpfn-v3-classifier-v3_20260417_binary.ckpt) | Classification | Specialized for binary classification for datasets with <200k rows |
| [`tabpfn-v3-classifier-v3_20260417_multiclass.ckpt`](https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt) | Classification | Specialized for multiclass classification for datasets with <200k rows |
| [`tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt`](https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt) | Regression | Specialized for regression for datasets with <100k rows and with alternative preprocessing |
| [`tabpfn-v3-regressor-v3_20260506_timeseries.ckpt`](https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/tabpfn-v3-regressor-v3_20260506_timeseries.ckpt) | Regression / Time-series forecasting | Fine-tuned on synthetic time-series data; used by default in TabPFN-TS-3 |
| [`tabpfn-v3-classifier-v3_20260506_ood.ckpt`](https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/tabpfn-v3-classifier-v3_20260506_ood.ckpt) | Classification | Useful when test inputs may fall outside the training distribution and you want the model to extrapolate. See Figure 26 in the [model report](https://arxiv.org/pdf/2605.13986). Bundles OOD-robust preprocessors (`squashing_scaler_max10` + `none`), similar to default otherwise. |
| [`tabpfn-v3-regressor-v3_20260506_ood.ckpt`](https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/tabpfn-v3-regressor-v3_20260506_ood.ckpt) | Regression | Useful when test inputs may fall outside the training distribution and you want the model to extrapolate. See Figure 26 in the [model report](https://arxiv.org/pdf/2605.13986). Bundles OOD-robust preprocessors (`quantile_uni_extrapolate` + `squashing_scaler_max10`), similar to default otherwise. |

To use one of these checkpoints, pass its filename via `model_path`:

```python
from tabpfn import TabPFNClassifier

clf = TabPFNClassifier(model_path="tabpfn-v3-classifier-v3_20260417_binary.ckpt")
```

### Developers & Affiliations
Developed by Prior Labs.

### Intended Use
Regression and classification tasks with up to 1M samples and ≤2000 features in structured tabular format.

### Not Intended Use
- Not suitable for unstructured data (text, images); use API version for textual features.
- Performance not guaranteed above 1M train samples or >2000 features.

### Model Architecture
Multi-stage transformer-based architecture with 24 main layers.

### Training Data and Priors
TabPFN-3 is trained purely on synthetic tabular tasks.

### Performance Benchmarks
Evaluated on public benchmarks such as TabArena and TALENT, the model yields SOTA results.
On proprietary benchmark collections, it yields SOTA results for datasets with <100k rows and datasets with 100k-1M rows and <200 features.


### Ethical Considerations
Having been trained purely on synthetic datasets, TabPFN-3 is free from dataset leakage from the pretraining stage.
However, like for any other tabular prediction method, when applied to high-risk use cases, users should ensure that the labelled data is free of biases.

### Limitations
Performance can degrade when applied to >1M data points and/or >2000 features.

### Licensing
Model weights released under tabpfn-3-license-v1.0.

The license is designed to be permissive for research and limited internal evaluation. It *explicitly allows* testing, evaluation, and internal benchmarking, so an organization can download the model and run preliminary assessments on its own datasets.
The key restriction is that the model, its derivatives, and its outputs cannot be used for any commercial or production purpose. This includes, but is not limited to, revenue-generating products, competitive benchmarking for procurement, client deliverables, or using the model’s results for internal commercial decision-making.
For all production use cases, we offer a *Commercial Enterprise License*. This provides access to our proprietary high-speed inference engine, dedicated support, integration tooling, and other internal models.
Please contact us at sales@priorlabs.ai for commercial licensing inquiries.

### Version
v1.0: initial release.

### Citation
```
@misc{grinsztajn2026tabpfn3technicalreport,
      title={TabPFN-3: Technical Report}, 
      author={Léo Grinsztajn and Klemens Flöge and Oscar Key and Felix Birkel and Philipp Jund and Brendan Roof and Mihir Manium and Shi Bin Hoo and Magnus Bühler and Anurag Garg and Dominik Safaric and Jake Robertson and Benjamin Jäger and Simone Alessi and Adrian Hayler and Vladyslav Moroshan and Lennart Purucker and Philipp Singer and Alan Arazi and Julien Siems and Jan Hendrik Metzen and Georg Grab and Nick Erickson and Siyuan Guo and Eliott Kalfon and Simon Bing and David Salinas and Clara Cornu and Lilly Charlotte Wehrhahn and Diana Kriuchkova and Kursat Kaya and Lydia Sidhoum and Marie Salmon and Jerry Chen and Madelon Hulsebos and Yann LeCun and Samuel Müller and Bernhard Schölkopf and Sauraj Gambhir and Noah Hollmann and Frank Hutter},
      year={2026},
      eprint={2605.13986},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.13986}, 
}
```