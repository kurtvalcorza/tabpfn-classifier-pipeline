"""Static release-asset validation for the TabPFN-3 classifier DIMER pipeline.

Checks the two STANDALONE tutorial notebooks (DIMER Notebook Specification 2.0 §4) — the `E2E`
tutorial and its `ARTIFACT-INFERENCE` companion — the tutorial registry, model card, README, STATUS.md
and weight documentation for source conformance and cross-document identity consistency, and runs the
generator parity checks (PAR1–PAR3) for every notebook.

This is source validation only. A PASS here is NOT clean-runtime execution evidence;
the release gate is defined in docs/release-verification.md.

Two-notebook variant of the fleet validator (snapshot resnet50 @ 6c77f84, tooling updates 2026-09-13 15:40 + 17:10):
the constants block declares NOTEBOOKS (one entry per generated notebook: template, profile, gates, outputs,
markers), PACKAGE_DIR, IDENTITY_DOCS and EXPECTED_CARD_SPEC; the shared block is the snapshot's (per-module
PAR1 through build.load_context, joined-module digest, own-repo and mutable-git-dependency rules) with the
per-notebook spec threaded through validate_notebooks().
"""
# ruff: noqa: E501  -- rule messages name the file and requirement in full; they are kept on one line
from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "tabpfn_classifier_pipeline"
PACKAGE_DIR = "src/tabpfn_classifier_pipeline"  # the template's package_dir (default src/<package>)
REPO_NAME = "tabpfn-classifier-pipeline"
EXPECTED_MODEL_ID = "Prior-Labs/tabpfn_3"
PIPELINE_CLASS = "TabPFNClassifierPipeline"
# INF1: the exact load expression the E2E stage runner must use for the pinned checkpoint.
MODEL_LOAD_EXPR = f"{PIPELINE_CLASS}.from_pretrained(weights_dir=run.weights / P.MODEL_KEY, n_estimators=n, random_state=seed)"
# Additional 40-hex revisions a document may legitimately cite: the pinned validator / finetuner worker
# commits of COMPONENTS.json (the private worker path the notebooks do not take).
KNOWN_SHAS: frozenset[str] = frozenset(
    (
        "4333bb8c1a18320e27bab6b4801cbdbde322fb53",  # tabpfn-classifier-dataset-validator (COMPONENTS.json)
        "8ebcfb6dd56a0d8860c778843f2776ad42ebe030",  # tabpfn-classifier-finetuner (COMPONENTS.json)
    )
)
# Documents that must name the model id and the immutable revision.
IDENTITY_DOCS = ("README.md", "MODEL_CARD.md", "docs/WEIGHTS.md")
# MODEL_CARD_SPEC version the card on this branch declares.
EXPECTED_CARD_SPEC = "1.1"
# Heading of the card section that carries the immutable provenance.
PROVENANCE_HEADING = "## Version pinning"
# Direct-library use that must stay inside the carried package and stage runner (G2: the notebook's own cells call
# `run_stage`, they do not reimplement the pipeline). Checked on every kernel code cell except the carrier. The install
# cell opens the pinned uv wheel with zipfile (member read, never extractall), so zipfile itself is not listed.
FORBIDDEN_OUTSIDE_MODULE = (
    "from huggingface_hub import",
    "import huggingface_hub",
    "hf_hub_download(",
    "from tabpfn import",
    "from tabpfn.",
    "import tabpfn",
    "TabPFNClassifier(",
    "FinetunedTabPFNClassifier",
    "load_fitted_tabpfn_model(",
    "save_fitted_tabpfn_model(",
    "from sklearn",
    "import sklearn",
    "import torch",
    "accuracy_score(",
    "log_loss(",
    "roc_auc_score(",
    "balanced_accuracy_score(",
    "torch.load(",
    "GITHUB_TOKEN",
    "userdata.get(",
    "import train as worker",
    "DIMER_TABPFN_MODEL_PATH",
)
# One entry per generated notebook: its template module (tools/<template>.py), profile, Colab form gates that must
# default to the non-interactive path, the machine-readable artifacts its stage runner must write (OUT1-OUT3, DAT24,
# EVAL21), the stage-runner and kernel-cell markers, and the learner-facing markers.
NOTEBOOKS = {
    "tabpfn_classifier_colab.ipynb": {
        "template": "notebook_template",
        "profile": "E2E",
        "byod_gates": {"USE_BYOD": False, "BYOD_PATH": "", "USE_BYOD_ROWS": False, "NEW_DATA_PATH": "", "RUN_ACTIVITY": False},
        "stages": ("weights", "data", "validate", "condition", "report", "export", "reload", "predict", "activity"),
        "expected_outputs": (
            "{STEM}_input_manifest.json",
            "{STEM}_evaluation_report.json",
            "{STEM}_result.json",
            "{STEM}_predictions.csv",
            "{STEM}_new_rows.csv",
        ),
        "runner_markers": (
            'STEM = "tabpfn_classifier"',
            'DEFAULT_DROP_COLUMNS = ["record_id", "category"]',
            "zip_path = P.build_synthetic_dataset(run.state / \"synthetic.zip\", rows=600, features=12, classes=3, seed=SEED)",
            "P.validate_inputs(features_only(train), target, val=features_only(val), test=features_only(test), names=[origin[\"type\"]])",
            "train, val = P.stratified_holdout(train, target, split, seed=SEED)",
            "check_numeric_like(frame, name, [target, *drop, *text_columns])",
            "[UNSEEN_CLASSES] classes absent from train.csv",
            "P.validate_inputs(strip(train).rename(columns={target: \"label\"}), target)",
            "baseline = P.majority_class_baseline(train[target], val[target], classes=classes)",
            "metrics = pipe.evaluate(val[v[\"features\"]], val[target])",
            "estimator class order differs from the validated class list",
            "standardised_logistic_regression",
            "report = P.evaluation_report(cond[\"validation\"], baseline=v[\"baseline\"], n_validation=len(val), class_labels=v[\"classes\"], sample_kind=data[\"sample_kind\"])",
            "RUN7_DEVIATION",
            "manifest = pipe.save_artifact(art)",
            "zip_sha = P.zip_artifact_bundle(art, zip_path)",
            'manifest["featureKinds"]',
            "if manifest[\"foundationCheckpointSha256\"] != P.WEIGHTS_SHA256:",
            "shutil.rmtree(fresh_dir, ignore_errors=True)",
            "checked = P.validate_artifact_bundle(fresh_dir, expected_checkpoint_sha256=P.WEIGHTS_SHA256)",
            "fresh = P.TabPFNClassifierPipeline.from_artifact(fresh_dir, device=reference[\"device\"], expected_checkpoint_sha256=P.WEIGHTS_SHA256)",
            "np.testing.assert_allclose(proba, expected, rtol=RELOAD_RTOL, atol=RELOAD_ATOL",
            '"maxAbsProbabilityDifference": diff',
            "features = P.validate_new_rows(rows.drop(columns=passthrough), v[\"features\"], target_column=data[\"target\"], classes=v[\"classes\"])",
            '"model_revision": P.MODEL_REVISION',
            '"model_license": P.MODEL_LICENSE',
            'importlib.metadata.version("tabpfn")',
        ),
        "kernel_markers": (
            "USE_BYOD = False  # @param",
            "BYOD_PATH = ''  # @param",
            "TARGET_COLUMN = 'target'  # @param",
            "DROP_COLUMNS = ['record_id', 'category']  # @param",
            "TEXT_COLUMNS = []  # @param",
            "N_ESTIMATORS = 4  # @param",
            "USE_BYOD_ROWS = False  # @param",
            "RUN_ACTIVITY = False  # @param",
            "ACTIVITY_N_ESTIMATORS = 1  # @param",
            "from google.colab import files",
            "files.upload()",
            "run_stage('data'",
            "run_stage('validate')",
            "run_stage('condition'",
            "run_stage('report')",
            "run_stage('export')",
            "run_stage('reload')",
            "run_stage('predict'",
            "run_stage('activity'",
        ),
        "markdown_markers": (
            "**Capability:** supervised tabular classification by **in-context learning**",
            "**no gradient update**",
            "**not calibrated probabilities**",
            "**Adaptation.**",
            "proposed RUN7 deviation",
            "**Identifiers are not features.**",
            "`tabpfn-3-license-v1.0`",
            "majority-class baseline",
            "standardised logistic regression",
            "`sample-sanity`",
        ),
        "forbidden_runner": (),
    },
    "tabpfn_classifier_artifact_inference_colab.ipynb": {
        "template": "notebook_template_artifact_inference",
        "profile": "ARTIFACT-INFERENCE",
        "byod_gates": {"ARTIFACT_ZIP_PATH": "", "UPLOAD_ARTIFACT": False, "EXPECTED_ZIP_SHA256": "", "EXPECTED_FITTED_SHA256": "", "NEW_DATA_PATH": "", "UPLOAD_NEW_DATA": False, "RUN_ACTIVITY": False},
        "stages": ("weights", "artifact", "reconstruct", "rows", "predict", "activity"),
        "expected_outputs": (
            "{STEM}_input_manifest.json",
            "{STEM}_evaluation_report.json",
            "{STEM}_result.json",
            "{STEM}_predictions.csv",
        ),
        "runner_markers": (
            'STEM = "tabpfn_classifier_artifact_inference"',
            'zip_sha = check_trusted_digest(zip_path, opts.get("expected_zip_sha256", ""), "EXPECTED_ZIP_SHA256")',
            "members = check_members(P, zip_path)",
            "Trusted digest mismatch",
            "P.safe_extract_zip(zip_path, bundle)",
            "artifact = P.validate_artifact_bundle(bundle, expected_fitted_sha256=expected_fitted, expected_checkpoint_sha256=P.WEIGHTS_SHA256)",
            "the bundled model.ckpt is not the pinned TabPFN-3 checkpoint carried by this notebook",
            "fresh = P.TabPFNClassifierPipeline.from_artifact(Path(art[\"bundle\"]), device=device, expected_checkpoint_sha256=P.WEIGHTS_SHA256)",
            "frame = P.validate_new_rows(raw.drop(columns=id_columns), features, target_column=target, classes=classes)",
            "frame = check_numeric_features(path.name, frame, feature_kinds(art, frame))",
            "P.validate_new_rows(frame.assign(unexpected_column=0), features, target_column=target, classes=classes)",
            "report = P.evaluation_report(None, n_validation=0, class_labels=fresh.classes, sample_kind=rows_state[\"sample_kind\"])",
            "NO_SAMPLE",
            '"model_revision": P.MODEL_REVISION',
            '"model_license": P.MODEL_LICENSE',
        ),
        "kernel_markers": (
            "ARTIFACT_ZIP_PATH = ''  # @param",
            "UPLOAD_ARTIFACT = False  # @param",
            "EXPECTED_ZIP_SHA256 = ''  # @param",
            "EXPECTED_FITTED_SHA256 = ''  # @param",
            "NEW_DATA_PATH = ''  # @param",
            "UPLOAD_NEW_DATA = False  # @param",
            "ID_COLUMNS = []  # @param",
            "from google.colab import files",
            "files.upload()",
            "run_stage('artifact'",
            "run_stage('reconstruct')",
            "run_stage('rows'",
            "run_stage('predict')",
        ),
        "markdown_markers": (
            "**Capability:** serving-state reconstruction from an externally produced TabPFN-3 classifier bundle",
            "produced **outside this execution**",
            "**No artifact is created here**",
            "**Trust boundary.**",
            "EXPECTED_FITTED_SHA256",
            "verdict is `not-measurable`",
            "artifact creation, in-notebook fitting, fine-tuning, regression, calibrated probabilities",
        ),
        "forbidden_runner": (
            "build_synthetic_dataset(",
            "stratified_holdout(",
            "majority_class_baseline(",
            "save_artifact(",
            "zip_artifact_bundle(",
            ".fit(",
        ),
    },
}

# ---------------------------------------------------------------------------
# Shared checks. Everything below is source/structure validation only. Passing
# these checks is NOT clean-runtime execution evidence under DIMER Notebook
# Specification 2.0; see docs/release-verification.md for the release gate.
# ---------------------------------------------------------------------------

NOTEBOOK_SPEC = "2.2"
ALLOWED_PROFILES = {"E2E", "ARTIFACT-INFERENCE", "TASK-INFERENCE", "MULTI-CAPABILITY", "SMOKE"}
STATUS_TOKENS = ("Candidate", "Release-grade")
PLACEHOLDER = re.compile(r"\b(TODO|TBD|FIXME)\b|Insert text here|Tooltip:", re.I)
SHA40 = re.compile(r"^[0-9a-f]{40}$")
IDENTITY_NAMES = ("MODEL_ID", "MODEL_REVISION", "MODEL_LICENSE", "MODEL_KEY")
UNSUPPORTED_CLAIMS = re.compile(
    r"\b(production[- ]ready|battle[- ]tested|state[- ]of[- ]the[- ]art results (were|are) reproduced"
    r"|benchmark superiority (is|was) (shown|established)|is release-grade|now release-grade)\b",
    re.I,
)
REQUIRED_CARD_HEADINGS = [
    (4, "Description"),
    (4, "Intended Use and Limitations"),
    (6, "Primary Intended Uses"),
    (6, "Primary Intended Users"),
    (6, "Out-of-scope use cases"),
    (4, "Factors"),
    (6, "Groups"),
    (6, "Instrumentation"),
    (6, "Environment"),
    (4, "Metrics"),
    (6, "Performance Measures"),
    (6, "Decision thresholds"),
    (6, "Approaches to uncertainty and variability"),
    (4, "Ethical considerations and biases"),
    (6, "Data"),
    (6, "Human Life"),
    (6, "Mitigations"),
    (6, "Risks and harms"),
    (6, "Use cases"),
]
# Learner-facing markers every standalone DIMER tutorial in this fleet must carry (guided layer: GDL1–GDL15).
COMMON_MARKDOWN_MARKERS = (
    f"**Notebook specification:** DIMER Notebook Specification {NOTEBOOK_SPEC} — **standalone** (§4)",
    "**Mode:** `GUIDED`",
    "**Run all:**",
    "**Bring Your Own Data:**",
    "**This notebook is standalone.**",
    "**Learning objectives:**",
    "## How to use this notebook",
    "**Who this notebook is for.**",
    "## The task: Input → Model → Output",
    "## Roadmap",
    "<summary><strong>Glossary</strong>",
    "## Prerequisites",
    "Do not upload confidential or restricted",
    "- **External access:**",
    "## 1. Check the runtime",
    "## 2. Carry the code and build the isolated environment",
    "## 3. Pin, stage and verify the model",
    "> **Infrastructure.**",
    "Predict before running",
    "**What to notice:**",
    "Check your reasoning",
    "## Troubleshooting",
    "## Interpretation and limits",
    "## Conclusion",
    "Successful execution proves that the recorded repository revision",
    "without the repository being",
    "It does **not** establish benchmark superiority",
    "## References",
    f"- Repository model card: https://github.com/kurtvalcorza/{REPO_NAME}/blob/main/MODEL_CARD.md",
)
# Isolated-runtime contract (RUN1, RUN10, ENV6; TDC-M1/TDCA-M2): markers the generator-owned install cell carries.
ISOLATION_MARKERS = (
    "--require-hashes",
    "'--managed-python'",
    "READY = VENV / '.dimer-ready'",
    "environment_reused = READY.is_file() and READY.read_text().strip() == LOCK_SHA256",
    "MPLBACKEND='Agg'",
    "for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP'",
    "def run_stage(stage, **options):",
)
# Patterns that must never appear in tutorial code (comment-stripped), in any kernel cell or the stage runner.
FORBIDDEN_PATTERNS = (
    ("credential in clone URL", re.compile(r"https://[^/'\"\s]*@github\.com/|x-access-token:")),
    ("repository clone (ST1)", re.compile(r"\bgit\b[^\n]*\bclone\b|github\.com/kurtvalcorza")),
    ("mutable git dependency (MOD14)", re.compile(r"git\+https?://(?![^\n]*@[0-9a-f]{40}\b)")),
    ("editable self-install", re.compile(r"""['"](?:-e|--editable)['"]|pip install (?:-e|--editable)\b""")),
    ("mutable model reference (MOD14)", re.compile(r"revision\s*=\s*['\"](?:main|latest)['\"]")),
    ("trust_remote_code enabled", re.compile(r"trust_remote_code\s*[=:]\s*True")),
    (
        "unsafe deserialization",
        re.compile(r"\bpickle\.load|\btorch\.load\s*\(|getattr\(\s*torch\s*,\s*['\"]load['\"]"),
    ),
    ("archive extractall", re.compile(r"\.extractall\s*\(")),
    ("notebook magic or shell escape", re.compile(r"(?m)^\s*[%!]|get_ipython\(\)")),
)
# In-kernel installs (TDC-M1): no pip into the notebook kernel, and no restart instruction anywhere.
KERNEL_INSTALL = re.compile(r"['\"]-m['\"]\s*,\s*['\"]pip['\"]|^\s*[%!]\s*pip\b|['\"]pip install\b|\bpip\.main\(", re.M)


class ValidationError(AssertionError):
    """Raised for any release-asset defect; the message names the file and rule."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _cell_source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def _strip_comments(source: str) -> str:
    """Return the source without comment tokens (string contents are preserved)."""
    out: list[str] = []
    last_row, last_col = 1, 0
    lines = source.splitlines(keepends=True)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return source
    for token in tokens:
        (srow, scol), (erow, ecol) = token.start, token.end
        if srow > last_row:
            out.append(lines[last_row - 1][last_col:] if last_row - 1 < len(lines) else "")
            for row in range(last_row, srow - 1):
                out.append(lines[row])
            last_row, last_col = srow, 0
        if srow - 1 < len(lines):
            out.append(lines[srow - 1][last_col:scol])
        if token.type != tokenize.COMMENT:
            out.append(token.string)
        last_row, last_col = erow, ecol
    return "".join(out)


def _assignment_targets(node: ast.AST):
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign | ast.AugAssign | ast.NamedExpr | ast.For | ast.comprehension):
        targets = [node.target]
    elif isinstance(node, ast.withitem) and node.optional_vars is not None:
        targets = [node.optional_vars]
    else:
        return []
    names = []
    for target in targets:
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name):
                names.append(sub.id)
    return names


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    _check(spec is not None and spec.loader is not None, f"tools/{name}.py is required")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _entry_module(template: dict) -> str:
    return template.get("entry_module", "pipeline.py")


def _modules(template: dict) -> list[str]:
    return list(template.get("modules", ["pipeline.py"]))


def _package_identity(template: dict) -> tuple[str, str]:
    """Read MODEL_ID / MODEL_REVISION from the entry module source without importing torch."""
    text = _read(ROOT / PACKAGE_DIR / _entry_module(template))
    model_id = re.search(r'^MODEL_ID = "([^"]+)"$', text, re.M)
    revision = re.search(r'^MODEL_REVISION = "([^"]+)"$', text, re.M)
    _check(
        model_id is not None and revision is not None,
        f"{_entry_module(template)} must define MODEL_ID and MODEL_REVISION",
    )
    _check(SHA40.match(revision.group(1)) is not None, "MODEL_REVISION must be a 40-hex immutable commit")
    _check(model_id.group(1) == EXPECTED_MODEL_ID, f"MODEL_ID drifted from {EXPECTED_MODEL_ID}")
    return model_id.group(1), revision.group(1)


def _primary_template() -> dict:
    return _load_tool(next(iter(NOTEBOOKS.values()))["template"]).TEMPLATE


def validate_model_card() -> None:
    path = ROOT / "MODEL_CARD.md"
    text = _read(path)
    _check(text.startswith("---\n"), "MODEL_CARD.md must start with YAML front matter")
    front = text.split("---", 2)[1]
    for key in ("license:", "model_card_spec:", "base_model:"):
        _check(key in front, f"MODEL_CARD.md missing front-matter field: {key}")
    _check(f'model_card_spec: "{EXPECTED_CARD_SPEC}"' in front, f"MODEL_CARD.md model_card_spec must be {EXPECTED_CARD_SPEC}")
    _check(f"base_model: {EXPECTED_MODEL_ID}" in front, "MODEL_CARD.md base_model must equal MODEL_ID")
    _check(not PLACEHOLDER.search(text), "MODEL_CARD.md contains placeholder/scaffolding text")
    _check(not UNSUPPORTED_CLAIMS.search(text), "MODEL_CARD.md makes an unsupported release/benchmark claim")
    h1 = re.findall(r"(?m)^# (?!#)(.+)$", text)
    _check(len(h1) == 1, f"MODEL_CARD.md must contain exactly one H1, got {len(h1)}")
    found = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            found.append((len(match.group(1)), match.group(2).strip()))
    positions = []
    for heading in REQUIRED_CARD_HEADINGS:
        matches = [
            index
            for index, item in enumerate(found)
            if item[0] == heading[0] and item[1].casefold() == heading[1].casefold()
        ]
        _check(len(matches) == 1, f"required model-card heading missing/duplicated: {heading}")
        positions.append(matches[0])
    _check(positions == sorted(positions), "required model-card headings are out of order")
    _check(PROVENANCE_HEADING in text, f"MODEL_CARD.md must carry a {PROVENANCE_HEADING!r} section")


def validate_identity_consistency() -> None:
    """The immutable upstream identity must be the same string in every document."""
    model_id, revision = _package_identity(_primary_template())
    for name in IDENTITY_DOCS:
        text = _read(ROOT / name)
        _check(model_id in text, f"{name} must name the upstream model `{model_id}`")
        _check(revision in text, f"{name} must cite the immutable revision {revision}")
        other = re.findall(r"\b[0-9a-f]{40}\b", text)
        stray = sorted({sha for sha in other if sha != revision and sha not in KNOWN_SHAS})
        _check(not stray, f"{name} cites an unexpected 40-hex revision: {stray}")


# --- weight-facts check (fleet rollout 2026-09-24) ---
# Every SHA-256 digest and byte count quoted in the weight prose must come from a committed
# weights/*/dimer-base-manifest.json, or be declared below with a label saying what it describes
# (dataset files, upstream files that are not staged, origin checkpoints, totals). Declared entries
# that no document cites any more are rejected, so the allowlist cannot go stale.
WEIGHT_DOCS = ("README.md", "MODEL_CARD.md", "docs/WEIGHTS.md")
EXTERNAL_WEIGHT_BYTES: dict[int, str] = {
    212_818_107: "untracked weights/model.ckpt, a re-serialised TabPFN-3 checkpoint copy",
}
EXTERNAL_WEIGHT_DIGESTS: dict[str, str] = {}
_DIGEST = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{64}(?![0-9a-fA-F])")
_GROUPED = r"(\d{1,3}(?:[,\u202f\u00a0 ]\d{3})+|\d+)"
_BYTE_COUNT = re.compile(r"(?<![\d,\-])" + _GROUPED + r"\s*bytes\b|totalBytes`?\s*" + _GROUPED)


def _manifest_facts(root: Path = ROOT) -> tuple[set[str], set[int]]:
    digests: set[str] = set()
    sizes: set[int] = set()
    for path in sorted(root.glob("weights/*/dimer-base-manifest.json")):
        manifest = json.loads(_read(path))
        sizes.add(manifest["totalBytes"])
        for entry in manifest["files"]:
            digests.add(entry["sha256"])
            sizes.add(entry["bytes"])
    return digests, sizes


def validate_weight_facts(root: Path = ROOT) -> None:
    """Every SHA-256 and byte count quoted in the weight prose must come from a manifest or a labelled allowlist entry."""
    digests, sizes = _manifest_facts(root)
    _check(bool(digests), "no weights/*/dimer-base-manifest.json found to check weight facts against")
    cited_digests: set[str] = set()
    cited_sizes: set[int] = set()
    for name in WEIGHT_DOCS:
        path = root / name
        if not path.exists():
            continue
        text = _read(path)
        found_digests = set(_DIGEST.findall(text))
        found_sizes = {int(re.sub(r"[,\u202f\u00a0 ]", "", m.group(1) or m.group(2))) for m in _BYTE_COUNT.finditer(text)}
        cited_digests |= found_digests
        cited_sizes |= found_sizes
        bad_digests = sorted(found_digests - digests - set(EXTERNAL_WEIGHT_DIGESTS))
        _check(not bad_digests, f"{name} cites SHA-256 digests absent from every manifest and from EXTERNAL_WEIGHT_DIGESTS: {bad_digests}")
        bad_sizes = sorted(found_sizes - sizes - set(EXTERNAL_WEIGHT_BYTES))
        _check(not bad_sizes, f"{name} cites byte counts absent from every manifest and from EXTERNAL_WEIGHT_BYTES: {bad_sizes}")
    stale = sorted(set(EXTERNAL_WEIGHT_BYTES) - cited_sizes) + sorted(set(EXTERNAL_WEIGHT_DIGESTS) - cited_digests)
    _check(not stale, f"EXTERNAL_WEIGHT_* entries no weight document cites any more: {stale}")


# --- end weight-facts check ---

def validate_release_status() -> None:
    """STATUS.md, README.md and tutorials/README.md must agree on one status token."""
    status = _read(ROOT / "STATUS.md")
    match = re.search(r"Current status: \*\*(Candidate|Release-grade)\b", status)
    _check(match is not None, "STATUS.md must declare 'Current status: **Candidate**' or '**Release-grade**'")
    token = match.group(1)
    readme = _read(ROOT / "README.md")
    _check("## Release status" in readme, "README.md must have a '## Release status' section")
    section = readme.split("## Release status", 1)[1]
    _check(section.lstrip().startswith(f"**{token}"), f"README.md release status must open with **{token}**")
    registry = _read(ROOT / "tutorials" / "README.md").replace("**", "")
    _check(f"| {token}" in registry, f"tutorials/README.md must record the {token} status")
    other = [t for t in STATUS_TOKENS if t != token]
    for name, text in (("README.md", section.replace("**", "")), ("tutorials/README.md", registry)):
        for stale in other:
            _check(f"| {stale}" not in text, f"{name} carries a conflicting status token")
    if token == "Candidate":
        _check(
            "docs/release-verification.md" in registry or "release-verification" in registry,
            "tutorials/README.md must point Candidate notebooks at docs/release-verification.md",
        )
    for name in ("README.md", "STATUS.md", "tutorials/README.md", "docs/release-verification.md"):
        text = _read(ROOT / name)
        _check(not PLACEHOLDER.search(text), f"{name} contains placeholder text")
        _check(not UNSUPPORTED_CLAIMS.search(text), f"{name} makes an unsupported release/benchmark claim")
    verification = _read(ROOT / "docs" / "release-verification.md")
    _check(
        "## Recorded executions" in verification,
        "docs/release-verification.md must have '## Recorded executions'",
    )


def _validate_notebook_structure(path: Path, notebook: dict, spec: dict, template: dict, build) -> tuple[list[tuple[int, str, ast.Module]], str]:
    _check(notebook.get("nbformat") == 4, f"{path.name}: nbformat must be 4")
    dimer = notebook.get("metadata", {}).get("dimer")
    _check(isinstance(dimer, dict), f"{path.name}: metadata.dimer block is required")
    profile = dimer.get("notebook_profile")
    _check(profile in ALLOWED_PROFILES, f"{path.name}: invalid metadata.dimer.notebook_profile {profile!r}")
    _check(profile == spec["profile"], f"{path.name}: profile {profile!r} != declared {spec['profile']!r}")
    _check(dimer.get("notebook_spec") == NOTEBOOK_SPEC, f"{path.name}: metadata.dimer must declare notebook spec version '{NOTEBOOK_SPEC}'")
    _check(dimer.get("notebook_mode") == "GUIDED", f"{path.name}: metadata.dimer.notebook_mode must be GUIDED")
    _check(dimer.get("standalone") is True, f"{path.name}: metadata.dimer.standalone must be true (ST6)")
    generated = dimer.get("generated_from")
    _check(isinstance(generated, dict), f"{path.name}: metadata.dimer.generated_from is required (ST5)")
    _check(generated.get("repository") == REPO_NAME, f"{path.name}: generated_from.repository must be {REPO_NAME}")
    entry_rel = f"{PACKAGE_DIR}/{_entry_module(template)}"
    _check(generated.get("module") == entry_rel, f"{path.name}: generated_from.module must be {entry_rel}")
    module_rels = [f"{PACKAGE_DIR}/{m}" for m in _modules(template) if m.endswith(".py")]
    _check(generated.get("modules") == module_rels, f"{path.name}: generated_from.modules must be {module_rels}")
    module_sha = hashlib.sha256("".join(_read(ROOT / m) for m in module_rels).encode("utf-8")).hexdigest()
    _check(generated.get("module_sha256") == module_sha, f"{path.name}: generated_from.module_sha256 does not match {PACKAGE_DIR}/ (PAR4: regenerate the notebook)")
    _check(generated.get("generator", "").startswith("build_notebook.py/3"), f"{path.name}: generated_from.generator must be build_notebook.py/3 (isolated environment)")
    cells = notebook.get("cells", [])
    _check(bool(cells) and cells[0].get("cell_type") == "markdown", f"{path.name}: first cell must be markdown")
    code_cells: list[tuple[int, str, ast.Module]] = []
    markdown_parts: list[str] = []
    for index, cell in enumerate(cells):
        source = _cell_source(cell)
        if cell.get("cell_type") == "markdown":
            markdown_parts.append(source)
            _check("{{" not in source and "@P:" not in source and "{MODEL_ID}" not in source, f"{path.name}: unresolved template placeholder in markdown cell {index}")
            continue
        _check(cell.get("cell_type") == "code", f"{path.name}: unexpected cell type at {index}")
        _check(cell.get("execution_count") is None, f"{path.name}: code cell {index} has execution_count")
        _check(not cell.get("outputs"), f"{path.name}: code cell {index} persists outputs")
        _check(index > 0 and cells[index - 1].get("cell_type") == "markdown", f"{path.name}: code cell {index} lacks a preceding explanatory markdown cell")
        for line in source.splitlines():
            _check(not line.lstrip().startswith(("%", "!")), f"{path.name}: cell {index} uses a magic")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise ValidationError(f"{path.name}: code cell {index} does not compile: {exc}") from exc
        code_cells.append((index, source, tree))
    markdown = "\n".join(markdown_parts)
    raw_code = "\n".join(source for _, source, _ in code_cells)
    _check(not PLACEHOLDER.search(raw_code + markdown), f"{path.name}: placeholder text found")
    _check(not UNSUPPORTED_CLAIMS.search(markdown), f"{path.name}: unsupported release/benchmark claim")
    return code_cells, markdown


def _carrier(path: Path, notebook: dict) -> tuple[int, dict[str, str], dict[str, str], dict[str, str]]:
    """The single carrier cell (metadata.dimer.embedded_sources) and its CARRIED_FILES / CARRIED_BINARY / CARRIED_HASHES literals."""
    tagged = [(i, c) for i, c in enumerate(notebook["cells"]) if c.get("cell_type") == "code" and c.get("metadata", {}).get("dimer", {}).get("embedded_sources")]
    _check(len(tagged) == 1, f"{path.name}: exactly one carrier cell (metadata.dimer.embedded_sources) is required, found {len(tagged)}")
    index, cell = tagged[0]
    values: dict[str, dict] = {}
    for node in ast.parse(_cell_source(cell)).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("CARRIED_FILES", "CARRIED_BINARY", "CARRIED_HASHES"):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    _check(set(values) == {"CARRIED_FILES", "CARRIED_BINARY", "CARRIED_HASHES"}, f"{path.name}: the carrier cell must assign CARRIED_FILES, CARRIED_BINARY and CARRIED_HASHES literals")
    return index, values["CARRIED_FILES"], values["CARRIED_BINARY"], values["CARRIED_HASHES"]


def carried_runner(notebook_path: Path) -> str:
    """The stage runner text carried by a notebook (used by scripts/validate_colab_tutorial.py and the tests)."""
    _, files, _, _ = _carrier(notebook_path, json.loads(_read(notebook_path)))
    return files["tutorial_stages.py"]


def _validate_carrier(path: Path, notebook: dict, build, template: dict) -> tuple[int, str]:
    """PAR1/PAR2: every carried file equals the repository file it was generated from; hashes match the bytes."""
    import base64

    index, files, binary, hashes = _carrier(path, notebook)
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    context = build.load_context(ROOT, template, recorded)
    _check(files == context["files"], f"{path.name}: carried files differ from the repository (PAR1); regenerate the notebook")
    _check(binary == context["binary"], f"{path.name}: carried binary files differ from the repository (PAR1); regenerate the notebook")
    for name, text in files.items():
        _check(hashes.get(name) == hashlib.sha256(text.encode("utf-8")).hexdigest(), f"{path.name}: CARRIED_HASHES[{name!r}] does not match its text")
    for name, data in binary.items():
        _check(hashes.get(name) == hashlib.sha256(base64.b64decode(data)).hexdigest(), f"{path.name}: CARRIED_HASHES[{name!r}] does not match its bytes")
    _check(json.loads(files[f"weights/{template['weights_key']}/dimer-base-manifest.json"]) == json.loads(_read(ROOT / "weights" / template["weights_key"] / "dimer-base-manifest.json")), f"{path.name}: carried manifest != committed manifest (PAR2)")
    _check(files["tutorial_stages.py"] == _read(ROOT / template["stage_runner"]).replace("\r\n", "\n"), f"{path.name}: carried stage runner != {template['stage_runner']}")
    lock = files["requirements.txt"]
    build.check_lock(build._pins(ROOT), lock)
    return index, files["tutorial_stages.py"]


def _validate_parity(path: Path, notebook: dict, build, template: dict) -> None:
    """PAR3: the generator reproduces the committed file byte for byte."""
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    rendered = build.to_bytes(build.render(ROOT, template, recorded))
    current = path.read_bytes().replace(b"\r\n", b"\n")  # autocrlf checkouts are CRLF
    _check(current == rendered, f"{path.name}: differs from tools/build_notebook.py output (PAR3); regenerate")


def _validate_gates(path: Path, code_cells: list[tuple[int, str, ast.Module]], gates: dict[str, object]) -> None:
    """Each optional-input gate is assigned exactly once, to its non-interactive default, on a Colab form line; google.colab is
    imported only inside a conditional branch (never on the default path)."""
    for gate, default in gates.items():
        assignments = []
        for index, source, tree in code_cells:
            lines = source.splitlines()
            for node in tree.body:
                if gate in _assignment_targets(node):
                    assignments.append((index, node, lines[node.lineno - 1]))
        _check(len(assignments) == 1, f"{path.name}: {gate} must be assigned exactly once at cell top level, found {len(assignments)}")
        index, node, line = assignments[0]
        ok = isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and node.value.value == default and type(node.value.value) is type(default)
        _check(ok, f"{path.name}: {gate} must default to {default!r} (cell {index})")
        _check("# @param" in line, f"{path.name}: {gate} must be a Colab form parameter (`# @param`)")
    for index, _source, tree in code_cells:
        for node in tree.body:
            if isinstance(node, ast.Import | ast.ImportFrom):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                _check(not any(n.startswith("google.colab") for n in names), f"{path.name}: google.colab must only be imported inside an optional branch (cell {index})")


def _validate_isolation(path: Path, notebook: dict, code_cells: list[tuple[int, str, ast.Module]], carrier_index: int) -> None:
    """RUN1/RUN10/ENV6 (TDC-M1): nothing is pip-installed into the kernel, no restart is ever requested, the infrastructure
    cells are collapsed and titled, and every learner cell runs a stage."""
    text = json.dumps(notebook, ensure_ascii=False)
    _check("Restart the runtime" not in text and "restart the runtime" not in text, f"{path.name}: must not instruct a runtime restart")
    install = [src for _, src, _ in code_cells if src.startswith("# @title Infrastructure: build (or reuse) the isolated")]
    _check(len(install) == 1, f"{path.name}: exactly one isolated-environment install cell is required")
    missing = [m for m in ISOLATION_MARKERS if m not in install[0]]
    _check(not missing, f"{path.name}: install cell lacks isolated-environment markers: {missing}")
    for index, source, _tree in code_cells:
        if index == carrier_index:
            continue
        _check(not KERNEL_INSTALL.search(_strip_comments(source)), f"{path.name}: cell {index} installs into the notebook kernel (RUN10)")
    cells = notebook["cells"]
    infra = [i for i, c in enumerate(cells) if c.get("cell_type") == "code" and _cell_source(c).startswith("# @title Infrastructure:")]
    _check(len(infra) == 4, f"{path.name}: four Infrastructure cells (check, carrier, install, weights) are required, found {len(infra)}")
    for i in infra:
        _check(cells[i].get("metadata", {}).get("cellView") == "form", f"{path.name}: Infrastructure cell {i} must be collapsed (cellView: form, GDL11)")
    learner = [i for i, src, _ in code_cells if i not in infra]
    _check(bool(learner), f"{path.name}: no learner cells")
    for i in learner:
        src = next(s for j, s, _ in code_cells if j == i)
        _check("run_stage(" in src, f"{path.name}: learner cell {i} must run a stage (run_stage)")


def _validate_notebook_content(path: Path, code_cells: list[tuple[int, str, ast.Module]], markdown: str, carrier_index: int, runner: str, spec: dict, template: dict) -> None:
    model_id, revision = _package_identity(template)
    kernel = "\n".join(_strip_comments(source) for index, source, _ in code_cells if index != carrier_index)
    runner_code = _strip_comments(runner)
    stem = template["stem"]
    missing = [m for m in spec["runner_markers"] if m not in runner_code]
    _check(not missing, f"{path.name}: stage runner lacks required markers: {missing}")
    kernel_raw = "\n".join(source for index, source, _ in code_cells if index != carrier_index)
    missing = [m for m in spec["kernel_markers"] if m not in kernel_raw]
    _check(not missing, f"{path.name}: notebook cells lack required markers: {missing}")
    for stage in spec["stages"]:
        _check(f'"{stage}": stage_{stage}' in runner_code, f"{path.name}: stage runner must define stage {stage!r}")
    for name in spec["expected_outputs"]:
        _check('f"' + name in runner_code, f"{path.name}: stage runner must export {name.format(STEM=stem)}")
    present = [label for label, pattern in FORBIDDEN_PATTERNS if pattern.search(kernel) or pattern.search(runner_code)]
    _check(not present, f"{path.name}: forbidden/insecure source: {present}")
    leaked = [m for m in FORBIDDEN_OUTSIDE_MODULE if m in kernel]
    _check(not leaked, f"{path.name}: direct library use in the notebook's own cells (G2): {leaked}")
    _check(revision not in kernel, f"{path.name}: the model revision may appear only in the carried files")
    forbidden = [m for m in spec["forbidden_runner"] if m in runner_code]
    _check(not forbidden, f"{path.name}: profile-forbidden code in the stage runner: {forbidden}")
    if spec["profile"] == "E2E":
        _check(MODEL_LOAD_EXPR in runner_code, f"{path.name}: the stage runner must load through {MODEL_LOAD_EXPR} (INF1)")
    _check(not re.search(r"^\s*assert\b", runner_code, re.M), f"{path.name}: the stage runner must report verdicts, not assert (quality asserts abort BYOD runs)")
    _validate_gates(path, code_cells, spec["byod_gates"])
    missing_md = [m for m in COMMON_MARKDOWN_MARKERS + spec["markdown_markers"] if m not in markdown]
    _check(not missing_md, f"{path.name}: missing learner-facing markers: {missing_md}")
    _check(f"**Profile:** `{spec['profile']}`" in markdown, f"{path.name}: markdown must state the profile")
    ref = template.get("model_host", {}).get("reference_url", f"https://huggingface.co/{model_id}")
    _check(ref in markdown, f"{path.name}: references must link {ref}")


def validate_notebooks() -> None:
    tutorials = ROOT / "tutorials"
    notebooks = sorted(tutorials.glob("*.ipynb"))
    names = sorted(NOTEBOOKS)
    _check([p.name for p in notebooks] == names, f"tutorial notebooks must be exactly {names}, found {[p.name for p in notebooks]}")
    build = _load_tool("build_notebook")
    registry = _read(tutorials / "README.md")
    for path in notebooks:
        spec = NOTEBOOKS[path.name]
        template = _load_tool(spec["template"]).TEMPLATE
        _check(template["notebook_name"] == path.name, f"tools/{spec['template']}.py must name {path.name}")
        _check(template["profile"] == spec["profile"], f"tools/{spec['template']}.py profile must be {spec['profile']}")
        notebook = json.loads(_read(path))
        code_cells, markdown = _validate_notebook_structure(path, notebook, spec, template, build)
        carrier_index, runner = _validate_carrier(path, notebook, build, template)
        _validate_parity(path, notebook, build, template)
        _validate_isolation(path, notebook, code_cells, carrier_index)
        _validate_notebook_content(path, code_cells, markdown, carrier_index, runner, spec, template)
        _check(f"`{path.name}`" in registry, f"{path.name} missing from tutorials/README.md")
        _check(f"`{spec['profile']}`" in registry, f"tutorials/README.md must record `{spec['profile']}`")
    _check(f"DIMER Notebook Specification {NOTEBOOK_SPEC}" in registry, "tutorials/README.md must name the notebook spec version")
    _check("standalone" in registry.lower(), "tutorials/README.md must record that the notebooks are standalone")


def validate_all() -> list[str]:
    validate_model_card()
    validate_identity_consistency()
    validate_weight_facts()
    validate_release_status()
    validate_notebooks()
    return ["model-card", "identity-consistency", "weight-facts", "release-status", "notebooks+parity"]


def main() -> int:
    passed = validate_all()
    print(f"release asset validation: PASS ({', '.join(passed)})")
    print("NOTE: static source validation only; not clean-runtime execution evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
