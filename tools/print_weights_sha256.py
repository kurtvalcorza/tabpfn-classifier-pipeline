"""Print the SHA-256 of the canonical safetensors_weights.py, the value every vendored copy's test pins.

The fine-tune workers (kurtvalcorza/tabpfn-classifier-finetuner, kurtvalcorza/tabpfn-regressor-finetuner)
vendor the file byte for byte as tabpfn_safetensors_weights.py; kurtvalcorza/tabpfn-regressor-pipeline carries
an identical copy.
CI prints this value on every run so a re-sync can be checked against it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "src" / "tabpfn_classifier_pipeline" / "safetensors_weights.py"

if __name__ == "__main__":
    print(f"{hashlib.sha256(PATH.read_bytes()).hexdigest()}  {PATH.relative_to(PATH.parents[2]).as_posix()}")
