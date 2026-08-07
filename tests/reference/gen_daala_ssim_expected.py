"""Regenerate ``daala_ssim_expected.json`` from the real daala C reference.

Developer-only: needs a C compiler in the ``sisr`` env (see the plan). The test
suite and CI read the committed JSON and never run this.

Usage (from the repo root, PowerShell, with the ``sisr`` conda env activated --
activation puts the env's ``Library\\bin``, hence the compiler, on PATH):
    $env:PYTHONPATH = $PWD.Path
    python tests/reference/gen_daala_ssim_expected.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tests.reference.daala_ssim_cases import (
    CASES,
    discover_real_cases,
    make_planes,
    make_real_planes,
)

HERE = Path(__file__).resolve().parent


def build(workdir: Path) -> Path:
    """Compile the vendored reference and return the executable path."""
    # conda-forge's Windows gcc/clang packages only ship a target-prefixed
    # driver (e.g. x86_64-w64-mingw32-gcc.exe) -- no plain gcc/clang alias --
    # so that name is tried too, after the plain ones.
    cc = shutil.which("gcc") or shutil.which("clang") or shutil.which("x86_64-w64-mingw32-gcc")
    if cc is None:
        sys.exit(
            "No C compiler on PATH. If m2w64-gcc is already installed in the sisr env, "
            "activate that env (or add its Library\\bin to PATH) so "
            "x86_64-w64-mingw32-gcc.exe resolves. Otherwise install it: "
            "conda install -n sisr -c conda-forge m2w64-gcc"
        )
    exe = workdir / ("daala_ssim.exe" if sys.platform == "win32" else "daala_ssim")
    subprocess.run(
        [cc, "-O2", "-std=c99", str(HERE / "daala_ssim.c"), "-o", str(exe), "-lm"],
        check=True,
    )
    return exe


def score(exe: Path, workdir: Path, a, b) -> float:
    """Run the reference on one plane pair."""
    h, w = a.shape
    pa, pb = workdir / "a.raw", workdir / "b.raw"
    pa.write_bytes(a.tobytes())
    pb.write_bytes(b.tobytes())
    out = subprocess.run(
        [str(exe), str(w), str(h), str(pa), str(pb)], check=True, capture_output=True, text=True
    )
    return float(out.stdout.strip())


def main() -> None:
    real_cases = discover_real_cases()
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        exe = build(workdir)
        expected = {c["name"]: score(exe, workdir, *make_planes(c)) for c in CASES}
        expected.update({c["name"]: score(exe, workdir, *make_real_planes(c)) for c in real_cases})
    (HERE / "daala_ssim_expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(expected)} cases ({len(CASES)} synthetic, {len(real_cases)} real)")


if __name__ == "__main__":
    main()
