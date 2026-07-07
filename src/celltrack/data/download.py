"""Download competition data via the Kaggle API, working around corporate SSL.

The Kaggle Python client fails TLS verification under Netskope + OpenSSL 3.x
(``Basic Constraints of CA cert not marked critical``) because it uses its own
bundled certifi store rather than the corporate CA. We instead call the Kaggle
REST API with ``curl``, which honours the corporate CA bundle from the
environment.

Rules acceptance: the competition rules must be accepted once on the Kaggle
website before downloads are authorized; otherwise the API returns 403.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

COMPETITION = "biohub-cell-tracking-during-development"
_API = "https://www.kaggle.com/api/v1"


def _credentials() -> tuple[str, str]:
    path = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if not path.exists():
        raise FileNotFoundError(f"Kaggle credentials not found at {path}")
    creds = json.loads(path.read_text())
    return creds["username"], creds["key"]


def _ca_bundle() -> str | None:
    for var in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
        val = os.environ.get(var)
        if val and Path(val).exists():
            return val
    return None


def download_competition_data(
    dest: str | Path = "data",
    competition: str = COMPETITION,
    *,
    insecure: bool = False,
) -> Path:
    """Download and return the path to the competition data archive.

    Set ``insecure=True`` only as a last resort if the corporate CA bundle still
    fails verification (``curl -k``).
    """
    username, key = _credentials()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / f"{competition}.zip"

    cmd = [
        "curl",
        "--fail",
        "--location",
        "-u",
        f"{username}:{key}",
        "-o",
        str(archive),
        f"{_API}/competitions/data/download-all/{competition}",
    ]
    ca = _ca_bundle()
    if insecure:
        cmd.insert(1, "-k")
    elif ca:
        cmd.extend(["--cacert", ca])

    subprocess.run(cmd, check=True)
    return archive
