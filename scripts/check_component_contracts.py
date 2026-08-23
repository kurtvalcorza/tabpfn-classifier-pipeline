from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "contract" / "component-candidates.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return value


def _git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def _canonical_semantics(data: bytes) -> Any:
    return json.loads(data.decode("utf-8"))


def _fetch_exact(repository: str, commit: str, path: str, token: str) -> tuple[str, bytes]:
    url = f"https://api.github.com/repos/{repository}/contents/{path}?ref={commit}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "tabpfn-contract-conformance",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"cannot fetch exact candidate {repository}@{commit}:{path}: HTTP {exc.code}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("type") != "file":
        raise SystemExit(f"unexpected GitHub response for {repository}@{commit}:{path}")
    encoded = payload.get("content")
    if not isinstance(encoded, str):
        raise SystemExit(f"missing file content for {repository}@{commit}:{path}")
    return str(payload.get("sha") or ""), base64.b64decode(encoded)


def main() -> int:
    candidates = _load_json(CANDIDATES)
    if candidates.get("schemaVersion") != 1:
        raise SystemExit("component-candidates.json schemaVersion must be 1")
    if candidates.get("releasePinsAreSeparate") is not True:
        raise SystemExit("candidate verification must remain separate from release COMPONENTS.json")

    canonical_files = candidates.get("canonicalFiles")
    components = candidates.get("components")
    if not isinstance(canonical_files, dict) or not isinstance(components, dict):
        raise SystemExit("candidate manifest is incomplete")

    canonical_bytes: dict[str, bytes] = {}
    for remote_name, local_name in canonical_files.items():
        local = ROOT / str(local_name)
        data = local.read_bytes()
        _canonical_semantics(data)
        canonical_bytes[str(remote_name)] = data

    token = os.getenv("DIMER_COMPONENT_TOKEN", "").strip()
    for role in ("validator", "finetuner"):
        item = components.get(role)
        if not isinstance(item, dict):
            raise SystemExit(f"missing {role} candidate")
        repository = str(item.get("repository") or "")
        commit = str(item.get("commit") or "")
        files = item.get("files")
        if not repository.startswith("kurtvalcorza/"):
            raise SystemExit(f"{role} repository is outside the expected owner")
        if not HEX40.fullmatch(commit):
            raise SystemExit(f"{role} candidate commit must be an immutable 40-char SHA")
        if not isinstance(files, dict) or not files:
            raise SystemExit(f"{role} candidate has no pinned files")

        for path, expected_sha in files.items():
            path = str(path)
            expected_sha = str(expected_sha)
            if not HEX40.fullmatch(expected_sha):
                raise SystemExit(f"{role}:{path} blob SHA must be a 40-char Git SHA")
            if path in canonical_bytes:
                local = canonical_bytes[path]
                local_sha = _git_blob_sha(local)
                if local_sha != expected_sha:
                    raise SystemExit(
                        f"{role}:{path} expected blob {expected_sha} does not match "
                        f"canonical pipeline bytes {local_sha}"
                    )

            if token:
                actual_sha, actual_bytes = _fetch_exact(repository, commit, path, token)
                if actual_sha != expected_sha:
                    raise SystemExit(
                        f"{role}:{path} exact candidate blob changed: "
                        f"expected {expected_sha}, got {actual_sha}"
                    )
                if _git_blob_sha(actual_bytes) != expected_sha:
                    raise SystemExit(f"{role}:{path} downloaded bytes do not match Git blob SHA")
                if path in canonical_bytes and _canonical_semantics(actual_bytes) != _canonical_semantics(
                    canonical_bytes[path]
                ):
                    raise SystemExit(f"{role}:{path} semantics differ from pipeline authority")

    if token:
        print("Exact component commit/blob conformance verified.")
    else:
        print(
            "Offline candidate conformance verified; set DIMER_COMPONENT_TOKEN "
            "to verify the exact private component commits through GitHub."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
