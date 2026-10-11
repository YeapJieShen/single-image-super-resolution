"""Fetch the 5 calibration images the NIQE/PI parity test scores.

They are IQA-PyTorch's own calibration set (5 TID2013 distortions), whose
``results_official.csv`` records the MATLAB outputs the parity test compares
against. Each file is checked against a pinned SHA-256 and never redistributed:
``data/`` is gitignored.

Source: https://github.com/chaofengc/IQA-PyTorch, ``ResultsCalibra/dist_dir/`` at
commit 6e48c5ca9e3e58bd632af52f06a2ca976ccf2722 (the images are unchanged since
2022).

Usage (from the repo root, sisr env active)::

    PYTHONPATH=$PWD python tests/reference/fetch_niqe_calibration.py

Populates data/reference/niqe_calibration/{I03,I04,I06,I08,I19}.bmp. Idempotent:
files already present with the right hash are left alone.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

COMMIT = "6e48c5ca9e3e58bd632af52f06a2ca976ccf2722"
BASE_URL = (
    f"https://raw.githubusercontent.com/chaofengc/IQA-PyTorch/{COMMIT}/ResultsCalibra/dist_dir"
)
SHA256 = {
    "I03.bmp": "bb817e7fef614dcc12b3a99af33ce3d201bc4d56c4827254c0bcc4fa6ea11151",
    "I04.bmp": "f3d685983aacd76b2c131579ea60cee345205a1169a4bd2ff5f3039ecf81729e",
    "I06.bmp": "726e043e99d8e07235c2535107d4bb6fc85bad0cb3cc7d6ba3ecf3364ac97539",
    "I08.bmp": "8ad39042d7e17d6c96ecfd8c4b9f0e98c803b303a92d368ef57bc9c3c40f5330",
    "I19.bmp": "e0c6000652d270d1c39b7094761d01e14695fc5aa7cb76b945be6391327ee886",
}
DEST = Path(__file__).resolve().parents[2] / "data" / "reference" / "niqe_calibration"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, expected in SHA256.items():
        path = DEST / name
        if path.exists() and _digest(path) == expected:
            continue
        print(f"downloading {name} ...")
        with urllib.request.urlopen(f"{BASE_URL}/{name}") as response:
            payload = response.read()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected:
            sys.exit(f"{name}: checksum mismatch: got {digest}, expected {expected}")
        path.write_bytes(payload)
    print(f"{DEST} holds all {len(SHA256)} calibration images")


if __name__ == "__main__":
    main()
