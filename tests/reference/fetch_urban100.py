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

Note: the daala_c_reference test (tests/metrics/test_ssim.py::test_real_image_
matches_daala_c_reference) expects Set5/Set14/BSD100 data in data/reference/ and
will fail with an AssertionError if only Urban100 is present. This is expected
when fetching Urban100 in isolation — fetch all reference sets separately if you
plan to run the full test suite.
"""

import hashlib
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://cv.snu.ac.kr/research/EDSR/benchmark.tar"
SHA256 = "80c21c333bbf6ceb5308b7243761f8284478274413a97b96f1d63e9045fd93e8"
DEST_REFERENCE = Path(__file__).resolve().parents[2] / "data" / "reference" / "Urban100"
DEST_HR = Path(__file__).resolve().parents[2] / "data" / "Urban100_HR"


def main() -> None:
    reference_populated = (DEST_REFERENCE / "HR").is_dir() and any(
        (DEST_REFERENCE / "HR").glob("*.png")
    )
    hr_populated = DEST_HR.is_dir() and any(DEST_HR.glob("*.png"))

    if reference_populated and hr_populated:
        print(f"{DEST_REFERENCE} and {DEST_HR} already populated, nothing to do.")
        return

    tar_path = DEST_REFERENCE.parent / "benchmark.tar"
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not reference_populated:
            print(f"downloading {URL} ...")
            urllib.request.urlretrieve(URL, tar_path)

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
                tar.extractall(DEST_REFERENCE.parent, members=members, filter="data")
            shutil.rmtree(DEST_REFERENCE, ignore_errors=True)
            shutil.move(str(DEST_REFERENCE.parent / "benchmark" / "Urban100"), str(DEST_REFERENCE))
            shutil.rmtree(DEST_REFERENCE.parent / "benchmark", ignore_errors=True)

            n = len(list((DEST_REFERENCE / "HR").glob("*.png")))
            print(f"wrote {DEST_REFERENCE} ({n} HR images)")

        if not hr_populated:
            DEST_HR.mkdir(parents=True, exist_ok=True)
            for src in (DEST_REFERENCE / "HR").glob("*.png"):
                shutil.copy2(src, DEST_HR / src.name)
            n = len(list(DEST_HR.glob("*.png")))
            print(f"wrote {DEST_HR} ({n} HR images)")
    finally:
        tar_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
