#!/usr/bin/env python3
"""Statically validate the DIMER Colab tutorials against NOTEBOOK_SPEC 1.0 source contracts.

This validator does not claim runtime execution evidence. It checks notebook JSON, profile
metadata, Python-cell syntax, source hygiene, and profile-specific markers that are falsifiable
without running the worker or downloading weights.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS_DIR = ROOT / "tutorials"
ALLOWED_PROFILES = {"E2E", "ARTIFACT-INFERENCE", "TASK-INFERENCE", "MULTI-CAPABILITY", "SMOKE"}
EXPECTED_PROFILES = {
    "tabpfn_classifier_colab.ipynb": "E2E",
    "tabpfn_classifier_artifact_inference_colab.ipynb": "ARTIFACT-INFERENCE",
}
ABSOLUTE_PATH_PATTERNS = [re.compile(r"[a-zA-Z]:[\\/]"), re.compile(r"/(?:Users|home|root)/")]
PLACEHOLDER_PATTERN = re.compile(r"\b(?:TODO|TBD|FIXME)\b", re.IGNORECASE)
CREDENTIAL_IN_URL = re.compile(r"https://[^\s\"']*(?:GITHUB_TOKEN|x-access-token|token)[^\s\"']*@github\.com", re.IGNORECASE)
E2E_REQUIRED = {
    "BYOD path": "USE_BYOD",
    "pinned worker commit check": "!= pinned",
    "worker invocation": "worker.run(",
    "trivial baseline": "trivial baseline",
    "fallback visibility": "fineTuneSkippedReason",
    "serving reload": "load_dimer_tabpfn_artifact(",
    "reload equivalence": "reproduces the recorded validation metric",
    "new-data inference": ".predict(",
    "machine-readable predictions": "to_csv(",
    "provenance": "provenance",
    "licence statement": "tabpfn-3-license-v1.0",
    "class probabilities": ".predict_proba(",
    "decision rule": "argmax",
    "probability semantics": "calibrated",
    "private-source secret": "GITHUB_TOKEN",
    "ephemeral git header": "GIT_CONFIG_KEY_0",
    "git extra header": ".extraheader",
}
AI_REQUIRED = {
    "external upload": "files.upload(",
    "digest check": "Sha256",
    "archive safety": "unsafe member",
    "serving reload": "load_dimer_tabpfn_artifact(",
    "no network fallback": "networkFallbackForWeights",
    "new-data inference": ".predict(",
    "machine-readable predictions": "to_csv(",
    "provenance": "provenance",
    "trust boundary": "Trust boundary",
    "schema check": "Feature schema mismatch",
}
AI_FORBIDDEN = {
    "worker invocation": "worker.run(",
    "synthetic sample": "build_synthetic_dataset",
    "in-notebook fitting": ".fit(",
}
COMMON_REQUIRED = {"supported Python floor": "Python 3.11+", "runtime Python guard": "sys.version_info < (3, 11)"}


def clean_code_for_ast(code: str) -> str:
    return "\n".join(f"# {line}" if line.strip().startswith(("%", "!")) else line for line in code.splitlines())


def check_absolute_paths(code: str, filename: str, cell_idx: int) -> None:
    for line in code.splitlines():
        if "http://" in line or "https://" in line:
            continue
        for pattern in ABSOLUTE_PATH_PATTERNS:
            if pattern.search(line):
                raise AssertionError(f"{filename} (cell {cell_idx}): Absolute filesystem path detected: {line.strip()}")


def _source_text(nb: dict) -> str:
    parts = []
    for cell in nb.get("cells", []):
        source = cell.get("source", "")
        parts.append("".join(source) if isinstance(source, list) else str(source))
    return "\n".join(parts)


def _require_markers(name: str, profile: str, text: str, required: dict[str, str]) -> None:
    for label, marker in required.items():
        if marker not in text:
            raise AssertionError(f"{name}: {profile} contract missing {label} ({marker!r})")


def validate_notebook(nb_path: Path) -> None:
    if not nb_path.exists():
        raise AssertionError(f"Missing notebook: {nb_path}")
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{nb_path.name}: invalid notebook JSON: {exc}") from exc
    if nb.get("nbformat") != 4 or not nb.get("cells"):
        raise AssertionError(f"{nb_path.name}: must be nbformat 4 with cells")
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("execution_count") is not None or cell.get("outputs") not in (None, []):
            raise AssertionError(f"{nb_path.name} (cell {idx}): execution state must be cleared")
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        code = "".join(src) if isinstance(src, list) else str(src)
        if not code.strip():
            continue
        check_absolute_paths(code, nb_path.name, idx)
        try:
            ast.parse(clean_code_for_ast(code), filename=f"{nb_path.name}:cell_{idx}")
        except SyntaxError as exc:
            raise AssertionError(f"Syntax error in {nb_path.name} (cell {idx}): {exc}") from exc

    text = _source_text(nb)
    dimer = nb.get("metadata", {}).get("dimer", {})
    profile = dimer.get("notebook_profile")
    if profile not in ALLOWED_PROFILES or dimer.get("notebook_spec") != "1.0":
        raise AssertionError(f"{nb_path.name}: missing/invalid metadata.dimer profile or spec")
    expected = EXPECTED_PROFILES.get(nb_path.name)
    if expected and profile != expected:
        raise AssertionError(f"{nb_path.name}: expected profile {expected}, got {profile}")
    if f"**Profile:** `{profile}`" not in text:
        raise AssertionError(f"{nb_path.name}: profile line missing from notebook text")
    if PLACEHOLDER_PATTERN.search(text):
        raise AssertionError(f"{nb_path.name}: unresolved TODO/TBD/FIXME placeholder found")
    _require_markers(nb_path.name, profile, text, COMMON_REQUIRED)
    if profile == "E2E":
        _require_markers(nb_path.name, profile, text, E2E_REQUIRED)
        if CREDENTIAL_IN_URL.search(text):
            raise AssertionError(f"{nb_path.name}: credential-bearing GitHub clone URL is forbidden")
        if "git config --global" in text or "git config --local" in text:
            raise AssertionError(f"{nb_path.name}: persistent Git credential configuration is forbidden")
        if "smoke tutorial" in text.lower():
            raise AssertionError(f"{nb_path.name}: E2E notebook must not identify itself as smoke")
    if profile == "ARTIFACT-INFERENCE":
        _require_markers(nb_path.name, profile, text, AI_REQUIRED)
        for label, marker in AI_FORBIDDEN.items():
            if marker in text:
                raise AssertionError(f"{nb_path.name}: ARTIFACT-INFERENCE must not perform {label} ({marker!r})")
    print(f"[PASS] Validated {nb_path.name}")


def main() -> None:
    notebooks = sorted(TUTORIALS_DIR.glob("*.ipynb"))
    if not notebooks:
        raise AssertionError(f"No notebooks found in {TUTORIALS_DIR}")
    for nb_path in notebooks:
        validate_notebook(nb_path)
    print(f"All {len(notebooks)} tutorial notebooks passed notebook-spec v1 static validation.")
    print("NOTE: static validation is not clean-runtime execution evidence.")


if __name__ == "__main__":
    main()
