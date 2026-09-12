#!/usr/bin/env python3
"""Execute the DIMER tutorial pair through real IPython kernels and record release evidence.

Runs `tabpfn_classifier_colab.ipynb` (E2E) in one fresh kernel, then
`tabpfn_classifier_artifact_inference_colab.ipynb` (ARTIFACT-INFERENCE) in a second fresh kernel whose
artifact bundle and new-data CSV come from the first run and a separately generated file — the
same external-artifact boundary an interactive Colab user crosses with the upload dialog.

Disclosed substitutions, recorded in the evidence JSON:
1. `/content/...` paths are rewritten to `<work>/content/...`.
2. With `--skip-bootstrap`, every `# >>> colab-bootstrap` ... `# <<< colab-bootstrap` region is
   dropped, IPython `!`/`%` lines inside one included; lines outside a region are left alone.
   The executing
   interpreter must already provide the lock set, and the worker is cloned by this script from
   `--finetuner-source` (a local path or URL) at the commit pinned in COMPONENTS.json, so the
   notebook's own pinned-commit check still runs against a real checkout.
3. `DIMER_TUTORIAL_REF` is set to the staged pipeline commit, so the notebook's own revision guard
   checks the checkout this script prepared and provenance records that exact SHA.
4. `google.colab.files.upload()` is served by a shim that returns the files named in
   `DIMER_UPLOAD_FILES` (`;`-separated batches, `|`-separated files) instead of a dialog.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import re
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_colab_tutorial import strip_bootstrap  # noqa: E402  (shared --skip-bootstrap rule)

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "tutorials/tabpfn_classifier_colab.ipynb"
INFERENCE = ROOT / "tutorials/tabpfn_classifier_artifact_inference_colab.ipynb"
SHIM = textwrap.dedent(
    '''
    import os
    from pathlib import Path
    _QUEUE = [p for p in os.environ.get("DIMER_UPLOAD_FILES", "").split(";") if p]
    def upload():
        if not _QUEUE:
            raise RuntimeError("harness upload queue is empty; set DIMER_UPLOAD_FILES")
        return {Path(p).name: Path(p).read_bytes() for p in _QUEUE.pop(0).split("|")}
    '''
)


def _install_shim(work: Path) -> Path:
    site = work / "harness-site"
    (site / "google" / "colab").mkdir(parents=True, exist_ok=True)
    (site / "google" / "__init__.py").write_text("", encoding="utf-8")
    (site / "google" / "colab" / "__init__.py").write_text("from . import files\n", encoding="utf-8")
    (site / "google" / "colab" / "files.py").write_text(SHIM, encoding="utf-8")
    return site


def _prepare(source: Path, work: Path, skip_bootstrap: bool, overrides: dict[str, str]) -> nbformat.NotebookNode:
    nb = nbformat.read(source, as_version=4)
    content_root = (work / "content").as_posix()
    regions = 0
    applied: set[str] = set()
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for name, value in overrides.items():  # only `NAME = ...  # @param` form lines are overridable
            cell.source, hits = re.subn(rf"^{name} = .*?(  # @param.*)$", lambda m, n=name, v=value: f"{n} = {v}{m.group(1)}", cell.source, flags=re.M)
            if hits:
                applied.add(name)
        if skip_bootstrap:
            stripped, found = strip_bootstrap(cell.source)
            if found:
                cell.source = "# bootstrap region(s) dropped by execute_notebook_release.py --skip-bootstrap\n" + stripped
                regions += found
        cell.source = cell.source.replace('"/content/', f'"{content_root}/')
    if skip_bootstrap and not regions:
        raise RuntimeError(f"{source.name}: --skip-bootstrap found no '# >>> colab-bootstrap' region; "
                           "the notebook would re-clone and re-install over the staged checkout")
    missed = sorted(set(overrides) - applied)
    if missed:
        raise RuntimeError(f"{source.name}: --set {missed} matched no `# @param` line; the run would silently record evidence for the unoverridden value")
    return nb


def _execute(nb, executed: Path, cwd: Path, env: dict[str, str], timeout: int) -> None:
    os.environ.update(env)
    NotebookClient(nb, timeout=timeout, kernel_name="python3", allow_errors=False,
                   resources={"metadata": {"path": str(cwd)}}).execute()
    nbformat.write(nb, executed)


def _stage_repos(content: Path, finetuner_source: str) -> tuple[str, str]:
    """Provide the pipeline repo (this checkout) and the worker at its pinned commit under <content>."""
    pipe = content / "tabpfn-classifier-pipeline"
    if not pipe.exists():
        subprocess.run(["git", "clone", "-q", str(ROOT), str(pipe)], check=True)
        subprocess.run(["git", "-C", str(pipe), "checkout", "-q", subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()], check=True)
    components = json.loads((ROOT / "COMPONENTS.json").read_text(encoding="utf-8"))["components"]["finetuner"]
    ft = content / "tabpfn-classifier-finetuner"
    if not ft.exists():
        subprocess.run(["git", "clone", "-q", finetuner_source, str(ft)], check=True)
        subprocess.run(["git", "-C", str(ft), "checkout", "-q", components["commit"]], check=True)
    return components["repository"], components["commit"]


def _parse_overrides(items: list[str]) -> dict[str, str]:
    """Turn NAME=VALUE into a substitutable Python literal.

    The value is spliced into the notebook verbatim, so a bare word such as `v3` would reach the
    kernel as a name and raise NameError mid-run. Anything that is not already a Python literal is
    quoted as a string instead.
    """
    overrides: dict[str, str] = {}
    for item in items:
        name, sep, raw = item.partition("=")
        if not sep or not name.isidentifier():
            raise SystemExit(f"--set expects NAME=VALUE with NAME a Python identifier, got {item!r}")
        try:
            ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raw = repr(raw)
        overrides[name] = raw
    return overrides


def make_genuinely_new_rows(path: Path, feature_columns: list[str], reference_csv: Path) -> pd.DataFrame:
    ref = pd.read_csv(reference_csv)
    numeric = ref[feature_columns].select_dtypes(include=np.number)
    fresh = ref[feature_columns].head(8).copy()
    for c in numeric.columns:
        fresh[c] = numeric[c].mean() + np.linspace(-0.37, 0.37, len(fresh)) * numeric[c].std()
    if fresh.equals(ref[feature_columns].head(8)):
        raise RuntimeError("fresh rows must differ from the producer sample")
    fresh.to_csv(path, index=False)
    return fresh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, default=Path("/content"))
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--finetuner-source", default="https://github.com/kurtvalcorza/tabpfn-classifier-finetuner.git",
                        help="git URL or local path of the worker repository (checked out at the COMPONENTS.json commit)")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--evidence", type=Path, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="override a `# @param` form value, e.g. --set FINE_TUNE=True or --set MODEL_VERSION=v3. "
                             "A value that is not a Python literal is substituted as a string.")
    args = parser.parse_args()
    overrides = _parse_overrides(args.set)

    work = args.work.resolve(); content = work / "content"; content.mkdir(parents=True, exist_ok=True)
    site = _install_shim(work)
    env = {"PYTHONPATH": str(site) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    ft_repo, ft_commit = _stage_repos(content, args.finetuner_source)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    env["DIMER_TUTORIAL_REF"] = commit  # the notebook guard then checks the checkout staged above
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()

    main_executed = work / "tabpfn_classifier_colab.executed.ipynb"
    _execute(_prepare(MAIN, work, args.skip_bootstrap, overrides), main_executed, content, env, args.timeout)
    art = content / "dimer" / "output" / "artifacts"
    manifest_path = art / "artifact_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("E2E notebook did not produce the artifact bundle")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads((content / "tabpfn-tutorial-output" / "tabpfn_classification_metrics.json").read_text(encoding="utf-8"))

    import zipfile
    zips = sorted((content / "dimer" / "dataset").glob("*.zip"))
    with zipfile.ZipFile(zips[0]) as z:
        member = next(n for n in z.namelist() if n.endswith("test.csv"))
        reference = work / "reference_test.csv"; reference.write_bytes(z.read(member))
    fresh_rows = work / "tabpfn_classifier_release_fresh_rows.csv"
    fresh = make_genuinely_new_rows(fresh_rows, manifest["featureColumns"], reference)
    env["DIMER_UPLOAD_FILES"] = f"{manifest_path}|{art / manifest['fittedEstimator']}|{art / manifest['foundationCheckpoint']};{fresh_rows}"
    inference_executed = work / "tabpfn_classifier_artifact_inference_colab.executed.ipynb"
    _execute(_prepare(INFERENCE, work, args.skip_bootstrap, {}), inference_executed, content, env, args.timeout)

    predictions = content / "tabpfn-artifact-inference-output" / "tabpfn_classification_predictions.csv"
    if not predictions.exists():
        raise RuntimeError("Artifact-inference notebook did not write the predictions CSV")
    result = pd.read_csv(predictions)
    if len(result) != len(fresh) or "prediction" not in result.columns:
        raise RuntimeError("Artifact-inference prediction output contract failed")

    import importlib.metadata as md
    import torch
    evidence = {
        "result": "PASS", "recordedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repositoryCommit": commit, "workingTreeDirty": bool(dirty), "worker": {"repository": ft_repo, "commit": ft_commit, "source": args.finetuner_source},
        "notebooks": {"e2e": MAIN.name, "artifactInference": INFERENCE.name},
        "engine": "nbclient / IPython kernel, one fresh kernel per notebook",
        "environment": {"host": platform.node(), "os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
                        "tabpfn": md.version("tabpfn"), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "substitutions": {"contentRoot": str(content), "bootstrapSkipped": args.skip_bootstrap, "paramOverrides": overrides,
                          "tutorialRef": commit, "uploadShim": "google.colab.files.upload() served from DIMER_UPLOAD_FILES"},
        "notColab": "Clean local-kernel execution, not a Google Colab run; a Colab record is still required for a Colab claim.",
        "e2e": {"executedNotebook": str(main_executed), "metrics": metrics},
        "artifactInference": {"executedNotebook": str(inference_executed), "externalArtifact": str(manifest_path), "freshRows": str(fresh_rows), "rowsScored": int(len(result))},
    }
    evidence_path = args.evidence or (work / "release-evidence.json")
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
