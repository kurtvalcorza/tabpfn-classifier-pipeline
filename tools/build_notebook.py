#!/usr/bin/env python3
"""Generate a STANDALONE DIMER tutorial notebook (NOTEBOOK_SPEC 2.2 §4) from repository sources — /3 (tabular).

/3 replaces the in-kernel ``pip install`` of /2 (and its restart-on-stale-import guard) with the isolated, hash-locked
environment of the NOTEBOOK_SPEC 2.2 reference notebooks (§25.13; the fleet pattern of
``mediapipe-face-landmarker-pipeline`` and ``nafnet-deblurring-pipeline``). Nothing is pip-installed into the
notebook kernel, so a hosted runtime's preloaded packages are never replaced and no restart is ever needed
(RUN1, RUN10, ENV6). The notebook

1. checks the runtime (Linux x86_64, disk; a CUDA GPU is recommended, not required) and creates a run directory
   ``ROOT`` — re-running that cell in the same session keeps ``ROOT``, so later cells are never stranded;
2. writes the carried files — the repository's package under ``src/``, the stage runner, the hash-locked
   requirements, the pinned snapshot manifest, any pinned sample files, the licence and ``source.json`` — to
   ``ROOT`` and verifies each against ``CARRIED_HASHES`` (the carried text stays visible in that cell: ST5, SRC12);
3. downloads a pinned ``uv`` wheel (URL + size + SHA-256), builds a managed-Python virtual environment keyed on the
   lock's SHA-256 — a later **Run all** in the same runtime reuses a matching environment instead of rebuilding it —
   installs the lock with ``--require-hashes`` into it, and defines ``run_stage``, which runs one stage of the
   carried runner per process (``MPLBACKEND=Agg``; no ``PYTHONPATH``/``PYTHONHOME``/``PYTHONSTARTUP`` and no
   credentials are inherited) and re-raises a failed stage's own error message in the kernel;
4. runs the template's learner cells, which call ``run_stage(...)`` and read the files it writes.

The carried files are the repository's files byte for byte (read as UTF-8 text, newlines normalised to LF; binary
sample files as base64); the parity tests fail whenever the carrier and the repository diverge. This file is vendored
per repository (identical across the six tabular in-context pipelines).

Usage (from the repository root, or with --repo):
    python tools/build_notebook.py                      # write tutorials/<notebook_name>
    python tools/build_notebook.py --check              # exit 1 if the committed notebook differs (PAR3)
    python tools/build_notebook.py --template tools/notebook_template_artifact_inference.py [--check]
    python tools/build_notebook.py --out PATH           # write elsewhere (review copies)
"""
# ruff: noqa: E501  -- learner-facing prose and generated code are kept on single lines so they render readably
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

GENERATOR_VERSION = "build_notebook.py/3.0-tabular"
NOTEBOOK_SPEC = "2.2"
SOURCE_RECORD = "source.json"
LOCK_DEST = "requirements.txt"
RUNNER_DEST = "tutorial_stages.py"


def template_contract() -> dict[str, str]:
    """Keys ``TEMPLATE`` must define (documentation for template authors). Optional keys are marked."""
    return {
        "package": "import name of the repository package",
        "repo_name": "GitHub repository name",
        "stem": "output file stem (cell ids, run directory, export names)",
        "notebook_name": "tutorials/<notebook_name>",
        "profile": "TASK-INFERENCE | MULTI-CAPABILITY | E2E | ARTIFACT-INFERENCE",
        "mode": "REFERENCE | GUIDED | WORKSHOP",
        "run_all": "the Run-all declaration (§28)",
        "byod": "the BYOD declaration (§28)",
        "title": "H1 text",
        "badges": "list of (alt, image_url, link_url)",
        "capability": "one-line capability statement",
        "intro": "markdown paragraphs after the header block (no heading)",
        "learning_objectives": "markdown after the bold label",
        "exclusions": "markdown after the bold label",
        "prerequisites": "list of markdown bullets; the generator appends the External access bullet",
        "weights_key": "MODEL_KEY value (weights/<key>/dimer-base-manifest.json)",
        "modules": "package files carried under src/<package>/ (e.g. ['__init__.py', 'pipeline.py'])",
        "stage_runner": "repository-relative stage runner, carried as tutorial_stages.py",
        "lock": "repository-relative hash-locked requirements compiled from the pyproject pins (uv pip compile --generate-hashes)",
        "managed_python": "exact CPython version uv installs for the isolated environment",
        "uv": "{'version', 'url', 'bytes', 'sha256'} of the pinned manylinux x86_64 uv wheel",
        "disk_gib": "{'weights', 'environment'} free-space needs in GiB",
        "runtime_modules": "distributions whose versions the isolated environment prints (installed metadata), e.g. ['torch', 'numpy']",
        "install_flags": "uv pip install flags after --require-hashes, e.g. ['--only-binary', ':all:']",
        "guided": "{'opening': [markdown cells after the header], 'infrastructure': {cell: markdown after it}}",
        "cells": "list of {'md': str, 'code': str (optional)} learner cells; may use {stem}, {MODEL_ID}, {MODEL_REVISION}",
        "closing": "markdown: Troubleshooting, Interpretation and limits, conclusion scaffold and References",
        # optional:
        "package_dir": "OPTIONAL repository-relative package directory (default 'src/<package>')",
        "entry_module": "OPTIONAL module defining the identity constants (default 'pipeline.py')",
        "identity_names": "OPTIONAL {fleet name: package constant} when the package spells an identity constant differently",
        "carried_extra": "OPTIONAL {destination: repository-relative source} further UTF-8 text files",
        "carried_binary": "OPTIONAL {destination: repository-relative source} binary files, carried as base64",
        "model_host": "OPTIONAL {name, reference_url, revision_label, hosts} for a non-Hub checkpoint host",
        "license_file": "OPTIONAL repository-root licence/notice file carried beside the code (default 'LICENSE')",
    }


REQUIRED_KEYS = [k for k, v in template_contract().items() if not v.startswith("OPTIONAL")]
MODES = ("REFERENCE", "GUIDED", "WORKSHOP")
IDENTITY = ("MODEL_ID", "MODEL_REVISION", "MODEL_LICENSE", "MODEL_KEY")


def load_template(path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    template = module.TEMPLATE
    missing = [k for k in REQUIRED_KEYS if k not in template]
    if missing:
        raise SystemExit(f"template missing keys: {missing}")
    if template["mode"] not in MODES:
        raise SystemExit(f"template mode {template['mode']!r} is not one of {MODES}")
    return template


def _pins(repo: Path) -> list[str]:
    """The `==` runtime pins of pyproject.toml (ENV2); the lock is compiled from exactly these."""
    text = (repo / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.M | re.S)
    if not block:
        raise SystemExit("pyproject.toml: dependencies block not found")
    pins = re.findall(r'"([^"]+)"', block.group(1))
    bad = [p for p in pins if "==" not in p]
    if bad:
        raise SystemExit(f"unpinned runtime dependency (ENV2): {bad}")
    return pins


def lock_packages(lock_text: str) -> dict[str, str]:
    """`{name: version}` of every requirement in a uv/pip-compile hash lock."""
    return {m.group(1).lower(): m.group(2) for m in re.finditer(r"^([A-Za-z0-9._-]+)==([^\s\\]+)", lock_text, re.M)}


def check_lock(pins: list[str], lock_text: str) -> None:
    """Every direct pin must appear in the lock at the same version, and every lock entry must carry a hash."""
    locked = lock_packages(lock_text)
    for pin in pins:
        name, version = pin.split("==", 1)
        if locked.get(name.lower()) != version:
            raise SystemExit(f"lock does not pin {pin} (found {locked.get(name.lower())}); recompile the lock")
    blocks = re.split(r"\n(?=[A-Za-z0-9])", lock_text)
    unhashed = [b.split("==", 1)[0] for b in blocks if re.match(r"[A-Za-z0-9._-]+==", b) and "--hash=sha256:" not in b]
    if unhashed:
        raise SystemExit(f"lock entries without --hash: {unhashed}")


def _head_revision(repo: Path) -> str:
    """HEAD at generation time: a provenance label only; parity is anchored on file content (see ``--check``)."""
    try:
        out = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        out = ""
    return out or "uncommitted"


def recorded_revision(notebook_path: Path) -> str | None:
    """`generated_from.revision` of an existing notebook, or None."""
    if not notebook_path.exists():
        return None
    try:
        meta = json.loads(notebook_path.read_text(encoding="utf-8"))["metadata"]["dimer"]["generated_from"]
        return str(meta["revision"])
    except (KeyError, ValueError, TypeError):
        return None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def carried_sources(repo: Path, template: dict[str, Any]) -> dict[str, str]:
    """`{destination under ROOT: repository-relative source}` of every carried text file, in carrier order."""
    pkg_rel = template.get("package_dir", f"src/{template['package']}")
    sources = {f"src/{template['package']}/{m}": f"{pkg_rel}/{m}" for m in template["modules"]}
    sources[RUNNER_DEST] = template["stage_runner"]
    sources[LOCK_DEST] = template["lock"]
    key = template["weights_key"]
    sources[f"weights/{key}/dimer-base-manifest.json"] = f"weights/{key}/dimer-base-manifest.json"
    sources.update(template.get("carried_extra", {}))
    licence = template.get("license_file", "LICENSE")
    sources[licence] = licence
    return sources


def carried_files(repo: Path, template: dict[str, Any], revision: str) -> tuple[dict[str, str], dict[str, str]]:
    """(`{destination: text}`, `{destination: base64}`) of every carried file; `source.json` is generated last."""
    files: dict[str, str] = {}
    sources = carried_sources(repo, template)
    for dest, source in sources.items():
        path = repo / source
        if not path.is_file():
            raise SystemExit(f"carried source missing: {source}")
        files[dest] = _text(path)
    binary: dict[str, str] = {}
    for dest, source in template.get("carried_binary", {}).items():
        path = repo / source
        if not path.is_file():
            raise SystemExit(f"carried binary source missing: {source}")
        binary[dest] = base64.b64encode(path.read_bytes()).decode("ascii")
    check_lock(_pins(repo), files[LOCK_DEST])
    record = {
        "repository": f"kurtvalcorza/{template['repo_name']}",
        "revision": revision,
        "generator": GENERATOR_VERSION,
        "notebook_spec": NOTEBOOK_SPEC,
        "files": {dest: sha256_text(text) for dest, text in files.items()} | {dest: sha256_bytes(base64.b64decode(b)) for dest, b in binary.items()},
        "sources": {**sources, **template.get("carried_binary", {})},
    }
    files[SOURCE_RECORD] = json.dumps(record, indent=2) + "\n"
    return files, binary


def load_context(repo: Path, template: dict[str, Any], revision: str | None = None) -> dict[str, Any]:
    pkg = template["package"]
    pkg_rel = template.get("package_dir", f"src/{pkg}")
    revision = revision or _head_revision(repo)
    files, binary = carried_files(repo, template, revision)
    entry = template.get("entry_module", "pipeline.py")
    entry_text = files[f"src/{pkg}/{entry}"]
    names = {**{k: k for k in IDENTITY}, **template.get("identity_names", {})}
    ident: dict[str, str] = {}
    for key, const in names.items():
        m = re.search(rf'^{const} = "([^"]+)"$', entry_text, re.M)
        if not m:
            raise SystemExit(f"{entry}: {const} not found as a top-level string constant")
        ident[key] = m.group(1)
    manifest = json.loads(files[f"weights/{template['weights_key']}/dimer-base-manifest.json"])
    if (manifest["modelId"], manifest["revision"]) != (ident["MODEL_ID"], ident["MODEL_REVISION"]):
        raise SystemExit("manifest identity != module identity")
    if template["weights_key"] != ident["MODEL_KEY"]:
        raise SystemExit("template weights_key != MODEL_KEY")
    modules = [f"{pkg_rel}/{m}" for m in template["modules"] if m.endswith(".py")]
    hashes = {d: sha256_text(t) for d, t in files.items()} | {d: sha256_bytes(base64.b64decode(b)) for d, b in binary.items()}
    return {
        "pkg": pkg,
        "pkg_rel": pkg_rel,
        "revision": revision,
        "files": files,
        "binary": binary,
        "hashes": hashes,
        "modules": modules,
        "module_sha256": sha256_text("".join(_text(repo / m) for m in modules)),
        "manifest": manifest,
        "lock_sha256": hashes[LOCK_DEST],
        "lock_packages": len(lock_packages(files[LOCK_DEST])),
        "pins": _pins(repo),
        "host": {
            "name": "the Hugging Face Hub",
            "reference_url": f"https://huggingface.co/{ident['MODEL_ID']}",
            "revision_label": "revision",
            "hosts": "`huggingface.co` (no account or token)",
            **template.get("model_host", {}),
        },
        **ident,
    }


def _md(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "id": "", "metadata": {}, "source": source.rstrip("\n")}


def _code(source: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "id": "", "metadata": metadata or {}, "outputs": [], "source": source.rstrip("\n")}


# GDL11: collapsed-by-default metadata for infrastructure cells (Colab form view; Jupyter source_hidden).
INFRASTRUCTURE_METADATA: dict[str, Any] = {"cellView": "form", "jupyter": {"source_hidden": True}}

# ---- infrastructure cell sources -------------------------------------------------------------------------------------

CHECK_CELL = """# @title Infrastructure: check the runtime, accelerator and disk; create (or keep) the run directory
NEW_RUN_DIRECTORY = False  # @param {{type:"boolean"}}
import base64
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

SESSION_START = time.perf_counter()
if platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise RuntimeError('This notebook needs a Linux x86_64 runtime (Google Colab, Kaggle, or a Linux x86_64 Jupyter kernel): its locked environment is built for manylinux x86_64 wheels. A GPU is recommended, not required.')
try:
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True)
    accelerator = gpu.stdout.strip() if gpu.returncode == 0 else ''
except FileNotFoundError:
    accelerator = ''
if not accelerator:
    accelerator = 'none (CPU only)'
    print('No CUDA GPU detected. The notebook still runs on the CPU, more slowly (see Prerequisites). For the documented runtime choose Runtime > Change runtime type > T4 GPU before Run all.')
STEM = {stem!r}
# Re-running this cell keeps the run directory of this session, so the cells after it keep working; tick
# NEW_RUN_DIRECTORY (then run the cells below again) for a fresh one.
if NEW_RUN_DIRECTORY or not isinstance(globals().get('ROOT'), Path) or not globals()['ROOT'].is_dir():
    ROOT = Path.cwd() / 'dimer_runs' / STEM / uuid.uuid4().hex[:12]
ROOT.mkdir(parents=True, exist_ok=True)
OUTPUTS = Path.cwd() / 'outputs'
OUTPUTS.mkdir(exist_ok=True)
WEIGHTS = Path.cwd() / 'weights'
WEIGHTS.mkdir(exist_ok=True)
ENV_ROOT = Path(tempfile.gettempdir()) / 'dimer_uv'
staged_gib = sum(p.stat().st_size for p in WEIGHTS.rglob('*') if p.is_file()) / 1024**3
need = {{'weights': max(0.0, {weights_gib} - staged_gib), 'environment': {env_gib}}}
free = {{'weights': shutil.disk_usage(WEIGHTS).free / 1024**3, 'environment': shutil.disk_usage(tempfile.gettempdir()).free / 1024**3}}
if os.stat(WEIGHTS).st_dev == os.stat(tempfile.gettempdir()).st_dev:
    short = free['weights'] < need['weights'] + need['environment']
else:
    short = free['weights'] < need['weights'] or free['environment'] < need['environment']
if short:
    raise RuntimeError(f'Not enough free disk: need about {{need}} GiB, free {{free}} GiB. Start a fresh runtime (see Troubleshooting).')
print({{'accelerator': accelerator, 'kernel_python': platform.python_version(), 'run_directory': str(ROOT), 'outputs': str(OUTPUTS), 'weights': str(WEIGHTS), 'environment_root': str(ENV_ROOT), 'free_gib': {{k: round(v, 1) for k, v in free.items()}}}})"""

CARRIER_CELL = """# @title Infrastructure: write and verify the carried package, stage runner, lock, manifest and samples
CARRIED_FILES = {files}
CARRIED_BINARY = {binary}
CARRIED_HASHES = {hashes}
for name, payload in [*((n, t.encode('utf-8')) for n, t in CARRIED_FILES.items()), *((n, base64.b64decode(b)) for n, b in CARRIED_BINARY.items())]:
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    if hashlib.sha256(path.read_bytes()).hexdigest() != CARRIED_HASHES[name]:
        raise RuntimeError('Carried file integrity failure: ' + name + '. Do not edit this cell; regenerate the notebook from the repository.')
NOTEBOOK_SOURCE = json.loads((ROOT / {source!r}).read_text(encoding='utf-8'))
print({{'carried_files': len(CARRIED_HASHES), 'verified': True, 'repository': NOTEBOOK_SOURCE['repository'], 'revision': NOTEBOOK_SOURCE['revision'], 'generator': NOTEBOOK_SOURCE['generator']}})"""

INSTALL_CELL = """# @title Infrastructure: build (or reuse) the isolated locked environment and define the stage runner
import io
import urllib.error
import urllib.request
import zipfile

LOCK_SHA256 = {lock_sha256!r}
UV_URL = {uv_url!r}
UV_BYTES = {uv_bytes}
UV_SHA256 = {uv_sha256!r}
ENV_ROOT.mkdir(parents=True, exist_ok=True)
UV = ENV_ROOT / 'uv-{uv_version}'
UV_MARK = ENV_ROOT / 'uv-{uv_version}.sha256'  # digest of the binary extracted from the verified wheel
if not (UV.is_file() and UV_MARK.is_file() and hashlib.sha256(UV.read_bytes()).hexdigest() == UV_MARK.read_text().strip()):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(UV_URL, timeout=90) as response:
                wheel = response.read(UV_BYTES + 1)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))
    if len(wheel) != UV_BYTES or hashlib.sha256(wheel).hexdigest() != UV_SHA256:
        raise RuntimeError('uv {uv_version} wheel size/hash mismatch: refusing to run it')
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        member = next(n for n in archive.namelist() if n.endswith('.data/scripts/uv'))
        UV.write_bytes(archive.read(member))
    UV.chmod(0o700)
    UV_MARK.write_text(hashlib.sha256(UV.read_bytes()).hexdigest())
# Stage processes inherit no kernel Python path or start-up file and no credentials (every download is public). The
# kernel may export an inline matplotlib backend that the isolated environment cannot import; stages write files.
ENV = dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1', UV_CACHE_DIR=str(ENV_ROOT / 'cache'), MPLBACKEND='Agg', PYTHONNOUSERSITE='1')
for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'GOOGLE_APPLICATION_CREDENTIALS'):
    ENV.pop(name, None)
# One environment per lock digest: a second Run all (or a re-run of this cell) reuses a complete matching environment.
VENV = ENV_ROOT / ('venv-' + LOCK_SHA256[:16])
READY = VENV / '.dimer-ready'
PYTHON = VENV / 'bin' / 'python'
environment_reused = READY.is_file() and READY.read_text().strip() == LOCK_SHA256 and PYTHON.is_file()
if not environment_reused:
    shutil.rmtree(VENV, ignore_errors=True)
    subprocess.run([str(UV), 'venv', '--managed-python', '--python', {python!r}, str(VENV)], env=ENV, check=True)
    subprocess.run([str(UV), 'pip', 'install', '--python', str(PYTHON), '--require-hashes', *{install_flags!r}, '--index-url', 'https://pypi.org/simple', '-r', str(ROOT / {lock!r})], env=ENV, check=True)
    READY.write_text(LOCK_SHA256)
probe = subprocess.run([str(PYTHON), '-c', {probe!r}], env=ENV, check=True, capture_output=True, text=True)
RUNTIME = json.loads(probe.stdout.strip().splitlines()[-1])
print({{'notebook_source': NOTEBOOK_SOURCE['revision'], **RUNTIME, 'locked_packages': {n_locked}, 'environment': str(VENV), 'environment_reused': environment_reused, 'setup_seconds': round(time.perf_counter() - SESSION_START)}})
if not RUNTIME['cuda']:
    print('The isolated environment sees no CUDA GPU: every stage runs on the CPU (slower; see Prerequisites and Troubleshooting).')


def run_stage(stage, **options):
    \"\"\"Run one stage of the carried runner in its own process with the isolated interpreter; stream its output.\"\"\"
    if not (ROOT / {runner!r}).is_file() or not PYTHON.is_file():
        raise RuntimeError(f'The run directory {{ROOT}} has no carried files, or the isolated environment is gone: run the Infrastructure cells again in order (Sections 1, 2 and 3), or choose Runtime > Run all.')
    log = ROOT / 'logs' / (stage + '.log')
    log.parent.mkdir(exist_ok=True)
    error_file = ROOT / 'state' / (stage + '.error.json')
    error_file.unlink(missing_ok=True)
    command = [str(PYTHON), '-u', str(ROOT / {runner!r}), '--root', str(ROOT), '--outputs', str(OUTPUTS), '--weights', str(WEIGHTS), '--stage', stage, '--options', json.dumps(options)]
    print('Running stage', repr(stage), 'in the isolated environment; log:', log, flush=True)
    with log.open('w', encoding='utf-8') as output:
        process = subprocess.Popen(command, env=ENV, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            output.write(line)
            output.flush()
            if line.strip() and len(line) < 4000:
                print(line.rstrip(), flush=True)
        process.wait()
    if process.returncode:
        detail = 'see the log above'
        if error_file.is_file():
            error = json.loads(error_file.read_text(encoding='utf-8'))
            detail = error['type'] + ': ' + error['message']
        raise RuntimeError(f'Stage {{stage!r}} failed (exit {{process.returncode}}): {{detail}}')


def load_record(name):
    \"\"\"A JSON record a stage wrote to OUTPUTS.\"\"\"
    return json.loads((OUTPUTS / name).read_text(encoding='utf-8'))"""

WEIGHTS_CELL = """# @title Infrastructure: stage and digest-verify the pinned checkpoint
run_stage('weights')"""


def _probe(distributions: list[str]) -> str:
    """Versions from installed distribution metadata (a module's ``__version__`` is optional: ``tabdpt`` has none)."""
    names = ", ".join(repr(d) for d in distributions)
    return f"import importlib.metadata, json, platform, torch; print(json.dumps({{'python': platform.python_version(), **{{d: importlib.metadata.version(d) for d in [{names}]}}, 'cuda': torch.cuda.is_available()}}))"


def _infrastructure_code(key: str, ctx: dict[str, Any], template: dict[str, Any]) -> str:
    if key == "check":
        disk = template["disk_gib"]
        return CHECK_CELL.format(stem=template["stem"], weights_gib=float(disk["weights"]), env_gib=float(disk["environment"]))
    if key == "carrier":
        return CARRIER_CELL.format(files=repr(ctx["files"]), binary=repr(ctx["binary"]), hashes=repr(ctx["hashes"]), source=SOURCE_RECORD)
    if key == "install":
        uv = template["uv"]
        return INSTALL_CELL.format(
            lock_sha256=ctx["lock_sha256"],
            uv_url=uv["url"],
            uv_bytes=int(uv["bytes"]),
            uv_sha256=uv["sha256"],
            uv_version=uv["version"],
            python=template["managed_python"],
            install_flags=list(template["install_flags"]),
            lock=LOCK_DEST,
            probe=_probe(template["runtime_modules"]),
            n_locked=ctx["lock_packages"],
            runner=RUNNER_DEST,
        )
    if key == "weights":
        return WEIGHTS_CELL
    raise SystemExit(f"unknown infrastructure cell {key!r}")


def _infrastructure_md(key: str, ctx: dict[str, Any], template: dict[str, Any]) -> str:
    total_mb = ctx["manifest"]["totalBytes"] / 1e6
    n_mod = len(ctx["modules"])
    extras = sorted(set(template.get("carried_extra", {})) | set(template.get("carried_binary", {})))
    extra_note = f", the pinned sample files ({', '.join(f'`{e}`' for e in extras)})" if extras else ""
    if key == "check":
        return (
            "## 1. Check the runtime · [Engineering]\n\n"
            "> **Infrastructure.** The code cells in Sections 1–3 are collapsed and titled **Infrastructure**. You may run them "
            "without studying their implementation; they exist for reproducibility and provenance, not as prerequisite "
            "machine-learning knowledge. The learning activities start in Section 4.\n\n"
            "**Input:** a fresh hosted runtime. **System:** checks that it is Linux x86_64 with enough free disk, looks for a "
            "CUDA GPU, and creates a run directory under `dimer_runs/{stem}/` for the carried code, the stage logs and the "
            "hand-off state between stages. Learner-facing files are written to `outputs/`. **Output:** one dictionary naming "
            "the accelerator, the kernel's Python version and the directories. Running this cell again in the same session "
            "keeps the run directory, so the cells after it keep working; tick `NEW_RUN_DIRECTORY` for a fresh one and run "
            "Sections 2 and 3 again. The verified checkpoint is kept in `weights/` and reused."
        ).format(stem=template["stem"])
    if key == "carrier":
        return (
            "## 2. Carry the code and build the isolated environment · [Engineering]\n\n"
            "> **Infrastructure.** The next two code cells are collapsed. The first **is** the repository's code, carried so "
            "that this notebook works on its own; the second builds the environment every stage runs in.\n\n"
            f"The first cell holds, as text, the files the workflow needs: the package's {n_mod} modules under "
            f"`src/{ctx['pkg']}/` (at revision `{ctx['revision'][:12]}`), the stage runner `{RUNNER_DEST}`, the hash-locked "
            f"`{LOCK_DEST}` ({ctx['lock_packages']} packages), the pinned checkpoint manifest{extra_note} and the licence. It "
            "writes each file into the run directory and checks its SHA-256 against `CARRIED_HASHES`, stopping on any "
            "mismatch. The text is the repository's files byte for byte; the repository's parity tests fail whenever the two "
            "diverge, so what runs here is what the repository tests. Nothing in this cell runs a model.\n\n"
            "**Expected result:** `carried_files`, `verified: True`, and the repository revision the notebook was generated from."
        )
    if key == "install":
        return (
            "**Infrastructure: the isolated environment.** This cell installs **nothing into the notebook kernel**. It downloads "
            f"one pinned file — the `uv` {template['uv']['version']} installer wheel, refused unless its size and SHA-256 match — "
            f"creates a separate virtual environment with its own CPython {template['managed_python']}, and installs "
            f"`{LOCK_DEST}` into it with `--require-hashes`: every package must be the locked version and match a locked "
            "digest. The hosted runtime's own packages (Colab preloads its own NumPy, pandas and PyTorch) are never replaced, "
            "which is why no restart is needed. The environment is keyed on the lock's SHA-256: a second **Run all** in the "
            "same runtime reuses it and prints `environment_reused: True`. The first build downloads PyTorch with its CUDA "
            "libraries (a few GB) and typically takes a few minutes. The cell also defines `run_stage` and `load_record`, the "
            "two helpers the learner cells use: each learner cell runs one stage of the carried runner in its own process, "
            "streams what it prints, and stops with the stage's own error message if it fails.\n\n"
            "**Expected result:** one dictionary with the generating revision, the isolated environment's Python, the "
            + ", ".join(f"`{m}`" for m in template["runtime_modules"])
            + " versions, whether CUDA is visible, the number of locked packages, whether an existing environment was reused, "
            "and the setup time. A failed download or a hash mismatch stops the cell; never remove a pin or a hash to get past one."
        )
    if key == "weights":
        return (
            "## 3. Pin, stage and verify the model · [Engineering]\n\n"
            "> **Infrastructure.** The next code cell is collapsed. It downloads one file and checks it; you may run it without "
            "studying its implementation.\n\n"
            f"The model identity is carried twice — `MODEL_ID`/`MODEL_REVISION` in the carried package and the "
            f"{len(ctx['manifest']['files'])}-file manifest (path, byte size, SHA-256) — and the `weights` stage first checks "
            f"that they agree. `stage_missing_files(..., allow_download=True)` then fetches exactly the absent entries from "
            f"{ctx['host']['name']} **at {ctx['host']['revision_label']} `{ctx['MODEL_REVISION'][:12]}…`** (never `main`, "
            f"~{total_mb:.0f} MB), and `verify_snapshot` re-hashes every file and raises on the first size or digest mismatch. "
            "There is no fallback to a different download and no remote model code is executed.\n\n"
            "**What to notice:** the model id, revision, licence and file count; `fetched` lists the checkpoint on a first run "
            "and is empty on a rerun (staging only fetches a missing file); then `verified_files`. A mismatch stops the cell "
            "with an error naming the file — see **Troubleshooting**, and never edit a manifest to get past one."
        )
    raise SystemExit(f"unknown infrastructure cell {key!r}")


def render(repo: Path, template: dict[str, Any], revision: str | None = None) -> dict[str, Any]:
    ctx = load_context(repo, template, revision)
    stem = template["stem"]
    fmt = {"stem": stem, "n_locked": ctx["lock_packages"], **{k: ctx[k] for k in IDENTITY}}
    cells: list[dict[str, Any]] = []

    def add(cell: dict[str, Any]) -> None:
        cell["id"] = f"{stem}-{len(cells):02d}"
        cells.append(cell)

    badges = " ".join(f"[![{alt}]({img})]({link})" for alt, img, link in template["badges"])
    total_mb = ctx["manifest"]["totalBytes"] / 1e6
    header = (
        f"# {template['title']}\n\n{badges}\n\n"
        f"**Profile:** `{template['profile']}`  \n"
        f"**Mode:** `{template['mode']}`  \n"
        f"**Notebook specification:** DIMER Notebook Specification {NOTEBOOK_SPEC} — **standalone** (§4)  \n"
        f"**Capability:** {template['capability']}\n\n"
        f"**This notebook is standalone.** Section 2 carries the repository's package ({len(ctx['modules'])} modules under "
        f"`{ctx['pkg_rel']}/`, at revision `{ctx['revision'][:12]}`), the stage runner, the hash-locked requirements "
        f"({ctx['lock_packages']} packages) and the pinned checkpoint manifest, and verifies every carried file against its "
        "SHA-256 before use, so the notebook keeps working after export even if the repository changes or disappears. "
        "Nothing is installed into the notebook kernel: a pinned `uv` (checked by size and SHA-256) builds an isolated Python "
        "environment from the lock with `--require-hashes`, and every stage runs there in its own process, so the hosted "
        "runtime's own packages are never replaced and no restart is needed. Its only external dependencies are PyPI, the "
        f"managed CPython build that `uv` downloads, and {ctx['host']['name']} at the immutable {ctx['host']['revision_label']} "
        f"`{ctx['MODEL_REVISION']}` (~{total_mb:.0f} MB, digest-verified before loading). It was generated by "
        f"`tools/build_notebook.py` ({GENERATOR_VERSION}); edit the repository and regenerate rather than editing cells.\n\n"
        f"**Run all:** {template['run_all'].strip()}\n\n"
        f"**Bring Your Own Data:** {template['byod'].strip()}\n\n"
        f"{template['intro'].strip()}\n\n"
        f"**Learning objectives:** {template['learning_objectives'].strip()}\n\n"
        f"**This notebook does not demonstrate:** {template['exclusions'].strip()}"
    )
    add(_md(header.replace("{MODEL_ID}", ctx["MODEL_ID"])))
    for opening in template["guided"].get("opening", []):
        add(_md(opening.format(**fmt)))

    prereq = list(template["prerequisites"]) + [
        f"- **External access:** {ctx['host']['hosts']}, to fetch the pinned `{ctx['MODEL_ID']}` checkpoint (~{total_mb:.0f} MB) "
        f"at {ctx['host']['revision_label']} `{ctx['MODEL_REVISION'][:12]}…`, accepted only at the pinned size and SHA-256; "
        f"PyPI (`pypi.org`, `files.pythonhosted.org`), for the pinned `uv` wheel and the {ctx['lock_packages']} hash-locked "
        "packages; and the managed CPython build (python-build-standalone) that `uv` downloads for the isolated environment. "
        "No repository clone and no credentials are required; nothing is installed from this repository."
    ]
    add(_md("## Prerequisites\n\n" + "\n".join(p.format(**fmt) for p in prereq)))

    after = template["guided"].get("infrastructure", {})
    for key in ("check", "carrier", "install", "weights"):
        add(_md(_infrastructure_md(key, ctx, template)))
        metadata: dict[str, Any] = json.loads(json.dumps(INFRASTRUCTURE_METADATA))
        if key == "carrier":
            metadata["dimer"] = {"embedded_sources": True, "files": dict(ctx["hashes"])}
        add(_code(_infrastructure_code(key, ctx, template), metadata))
        if after.get(key):
            add(_md(after[key].format(**fmt)))

    for stage in template["cells"]:
        add(_md(stage["md"].format(**fmt)))
        if stage.get("code"):
            add(_code(stage["code"].format(**fmt)))
    add(_md(template["closing"].format(**fmt)))

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",  # recommended; the notebook also runs on a CPU-only runtime
            "colab": {"gpuType": "T4", "name": template["notebook_name"], "provenance": []},
            "dimer": {
                "notebook_profile": template["profile"],
                "notebook_mode": template["mode"],
                "notebook_spec": NOTEBOOK_SPEC,
                "standalone": True,
                "requires_dimer_worker": False,
                "environment": "isolated hash-locked uv environment; nothing installed into the kernel",
                "generated_from": {
                    "repository": template["repo_name"],
                    "revision": ctx["revision"],
                    "module": f"{ctx['pkg_rel']}/{template.get('entry_module', 'pipeline.py')}",
                    "modules": ctx["modules"],
                    "module_sha256": ctx["module_sha256"],
                    "files": dict(ctx["hashes"]),
                    "generator": GENERATOR_VERSION,
                },
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def to_bytes(notebook: dict[str, Any]) -> bytes:
    return (json.dumps(notebook, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--template", type=Path, default=None, help="default: <repo>/tools/notebook_template.py")
    parser.add_argument("--out", type=Path, default=None, help="default: <repo>/tutorials/<notebook_name>")
    parser.add_argument("--check", action="store_true", help="exit 1 if the existing notebook differs (PAR3)")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    template = load_template(args.template or repo / "tools" / "notebook_template.py")
    out = args.out or repo / "tutorials" / template["notebook_name"]
    if args.check:
        # The recorded revision is a provenance label carried through the check; drift is caught by content — a changed
        # carried file changes the carrier cell and its hashes, so the byte comparison fails regardless of the label.
        rendered = to_bytes(render(repo, template, recorded_revision(out)))
        current = out.read_bytes().replace(b"\r\n", b"\n") if out.exists() else b""
        if current != rendered:
            print(f"STALE: {out} differs from the generator output; run tools/build_notebook.py", file=sys.stderr)
            return 1
        print(f"OK: {out} is up to date ({len(rendered)} bytes)")
        return 0
    rendered = to_bytes(render(repo, template))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(rendered)
    print(f"wrote {out} ({len(rendered)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
