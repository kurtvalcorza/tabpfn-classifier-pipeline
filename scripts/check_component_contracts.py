#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = json.loads((ROOT / "contract/contract-v1.json").read_text(encoding="utf-8"))
CANDIDATES = json.loads((ROOT / "contract/component-candidates.json").read_text(encoding="utf-8"))
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def fetch_exact(repo: str, commit: str, token: str) -> bytes:
    url = f"https://api.github.com/repos/{repo}/contents/contract-v1.json?ref={commit}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "User-Agent": "dimer-contract-conformance"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)
    return base64.b64decode(payload["content"])


def main() -> None:
    token = os.getenv("DIMER_COMPONENT_TOKEN", "").strip()
    for role in ("validator", "finetuner"):
        item = CANDIDATES["components"][role]
        assert HEX40.fullmatch(item["commit"]), f"{role}: immutable 40-char commit required"
        assert HEX40.fullmatch(item["contractBlobSha"]), f"{role}: contract blob SHA required"
        snapshot = (ROOT / item["snapshot"]).read_bytes()
        assert json.loads(snapshot) == CANONICAL, f"{role}: vendored contract differs semantically from canonical"
        if token:
            remote = fetch_exact(item["repository"], item["commit"], token)
            assert git_blob_sha(remote) == item["contractBlobSha"], f"{role}: exact commit contract blob mismatch"
            assert json.loads(remote) == CANONICAL, f"{role}: exact commit contract differs from canonical"
    print("component contract conformance: OK" + (" (online exact-commit + blob verified)" if token else " (vendored snapshots verified; online check unavailable)"))


if __name__ == "__main__":
    main()
