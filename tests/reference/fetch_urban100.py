"""Verify-only fetcher for the Urban100 benchmark subtree (issue #281).

Downloads the EDSR authors' own benchmark distribution, checks it against
the published SHA-256 already pinned in ``tests/utils/test_imresize.py``
for this same archive, extracts only the ``Urban100`` subtree, and deletes
the archive afterward. Never keeps or redistributes the tar itself --
``data/`` is gitignored and this project does not mirror third-party
benchmark archives, only checks a copy against a pinned hash (see
docs/reproduction.md's Comparability section and the README's
redistribution table).

Source: https://cv.snu.ac.kr/research/EDSR/benchmark.tar
SHA-256: 80c21c333bbf6ceb5308b7243761f8284478274413a97b96f1d63e9045fd93e8
(250112000-byte archive)

Usage (from the repo root, sisr env active)::

    PYTHONPATH=$PWD python tests/reference/fetch_urban100.py

Populates data/reference/Urban100/{HR,LR_bicubic/{X2,X3,X4}}, the same
layout Set5/Set14/B100 already use under data/reference/. Idempotent: does
nothing and exits 0 if that directory already looks populated.
"""

import hashlib
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://cv.snu.ac.kr/research/EDSR/benchmark.tar"
SHA256 = "80c21c333bbf6ceb5308b7243761f8284478274413a97b96f1d63e9045fd93e8"
DEST = Path(__file__).resolve().parents[2] / "data" / "reference" / "Urban100"


def main() -> None:
    if (DEST / "HR").is_dir() and any((DEST / "HR").glob("*.png")):
        print(f"{DEST} already populated, nothing to do.")
        return

    tar_path = DEST.parent / "benchmark.tar"
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, tar_path)

        digest = hashlib.sha256(tar_path.read_bytes()).hexdigest()
        if digest != SHA256:
            sys.exit(
                f"checksum mismatch: got {digest}, expected {SHA256} -- "
                "refusing to extract a tar that does not match the pinned hash."
            )

        print("checksum verified, extracting Urban100 ...")
        with tarfile.open(tar_path) as tar:
            members = [m for m in tar.getmembers() if m.name.startswith("benchmark/Urban100/")]
            tar.extractall(DEST.parent, members=members, filter="data")
        shutil.move(str(DEST.parent / "benchmark" / "Urban100"), str(DEST))
        shutil.rmtree(DEST.parent / "benchmark", ignore_errors=True)
    finally:
        tar_path.unlink(missing_ok=True)

    n = len(list((DEST / "HR").glob("*.png")))
    print(f"wrote {DEST} ({n} HR images)")


if __name__ == "__main__":
    main()
