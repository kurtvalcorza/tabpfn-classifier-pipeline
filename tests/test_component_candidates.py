from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_component_candidate_conformance_offline(monkeypatch):
    monkeypatch.delenv("DIMER_COMPONENT_TOKEN", raising=False)
    subprocess.run(
        [sys.executable, "scripts/check_component_contracts.py"],
        check=True,
        env={**os.environ, "DIMER_COMPONENT_TOKEN": ""},
    )


def test_candidate_pins_do_not_mutate_release_components():
    candidates = json.loads(Path("contract/component-candidates.json").read_text())
    released = json.loads(Path("COMPONENTS.json").read_text())
    assert candidates["releasePinsAreSeparate"] is True
    for role in ("validator", "finetuner"):
        assert len(candidates["components"][role]["commit"]) == 40
        assert candidates["components"][role]["commit"] != released["components"][role]["commit"]
