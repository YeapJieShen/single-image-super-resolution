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

Populates data/reference/Urban100/{HR,LR_bicubic/{X2,X3,X4}} (for byte-equality
testing) and data/Urban100_HR (for validation/test datasets), using the same
layout Set5/Set14/B100 already use. Idempotent: does nothing and exits 0 if
both directories already look populated.

The download and extraction happen in a temporary directory under
data/reference/ that is removed afterwards, so nothing else under data/reference/
(a manually downloaded benchmark.tar or benchmark/ extraction) is touched. The
server often resets the first connection, so the download is tried 3 times.

Note: once data/ exists,
tests/metrics/test_ssim.py::test_real_image_matches_daala_c_reference expects the
Set5/Set14/BSD100 HR dirs listed in tests/reference/daala_ssim_cases.py REAL_SETS
(data/Set5_HR, data/Set14_HR, data/BSD100_HR) and fails on a checkout holding
only this script's output.
"""

import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

URL = "https://cv.snu.ac.kr/research/EDSR/benchmark.tar"
SHA256 = "80c21c333bbf6ceb5308b7243761f8284478274413a97b96f1d63e9045fd93e8"
DEST_REFERENCE = Path(__file__).resolve().parents[2] / "data" / "reference" / "Urban100"
DEST_HR = Path(__file__).resolve().parents[2] / "data" / "Urban100_HR"
DOWNLOAD_ATTEMPTS = 3


def main() -> None:
    reference_populated = (DEST_REFERENCE / "HR").is_dir() and any(
        (DEST_REFERENCE / "HR").glob("*.png")
    )
    hr_populated = DEST_HR.is_dir() and any(DEST_HR.glob("*.png"))

    if reference_populated and hr_populated:
        print(f"{DEST_REFERENCE} and {DEST_HR} already populated, nothing to do.")
        return

    if not reference_populated:
        DEST_REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=DEST_REFERENCE.parent) as tmp:
            tar_path = Path(tmp) / "benchmark.tar"
            for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
                print(f"downloading {URL} (attempt {attempt}/{DOWNLOAD_ATTEMPTS}) ...")
                try:
                    urllib.request.urlretrieve(URL, tar_path)
                    break
                except OSError as err:
                    if attempt == DOWNLOAD_ATTEMPTS:
                        raise
                    print(f"download failed ({err}), retrying")

            with open(tar_path, "rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            if digest != SHA256:
                sys.exit(
                    f"checksum mismatch: got {digest}, expected {SHA256} -- "
                    "refusing to extract a tar that does not match the pinned hash."
                )

            print("checksum verified, extracting Urban100 ...")
            with tarfile.open(tar_path) as tar:
                members = [m for m in tar.getmembers() if m.name.startswith("benchmark/Urban100/")]
                tar.extractall(tmp, members=members, filter="data")
            shutil.rmtree(DEST_REFERENCE, ignore_errors=True)
            shutil.move(str(Path(tmp) / "benchmark" / "Urban100"), str(DEST_REFERENCE))

        n = len(list((DEST_REFERENCE / "HR").glob("*.png")))
        print(f"wrote {DEST_REFERENCE} ({n} HR images)")

    if not hr_populated:
        DEST_HR.mkdir(parents=True, exist_ok=True)
        for src in (DEST_REFERENCE / "HR").glob("*.png"):
            shutil.copy2(src, DEST_HR / src.name)
        n = len(list(DEST_HR.glob("*.png")))
        print(f"wrote {DEST_HR} ({n} HR images)")


if __name__ == "__main__":
    main()
