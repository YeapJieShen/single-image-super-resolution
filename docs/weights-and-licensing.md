# Weights and licensing

**No pretrained checkpoint is released yet.** This page is the policy that will apply when
one is. Parity figures in [models.md](models.md) are measured against the *authors'* released
weights, which are fetched by the user and never redistributed here.

**Licence: MIT, matching the code**, with the caveat below stated alongside the download
rather than buried.

**The caveat.** Training data such as DIV2K carries an *"academic research purpose only"*
restriction. Whether that kind of training-data restriction legally propagates to a
derivative model's weights is a **genuinely unsettled question** in ML/copyright practice
today. A prospective commercial user of a future checkpoint should treat it as an open
question, not a resolved one — this project states the input and declines to assert the
conclusion in either direction.

**What this repository redistributes, now and at release.**

| artifact | redistributed? |
|---|---|
| Source code | Yes — MIT. |
| Training data (DIV2K) | **No.** Fetched by the user; `data/` is not tracked. |
| Benchmark data (Set5, Set14, BSD100, Urban100) | **No.** Fetched by the user from the EDSR authors' archive, whose SHA-256 is pinned under [Comparability](reproduction.md#comparability) so the copy can be checked without being shipped. Urban100 is fetched via [`tests/reference/fetch_urban100.py`](../tests/reference/fetch_urban100.py). |
| Generated LR / `Bicubic_up` reference pairs | **No** — they are derivatives of the above and inherit whatever those carry. |
| Trained weights | Not yet. MIT when released, under the caveat above. |
| Benchmark output images (SR reconstructions) | Not yet, and **not without a per-set check first**. |

**Benchmark sets carry real and differing licensing friction**, which is why the last row is
not a blanket yes. Set5, Set14, BSD100, and Urban100 originate from separate publications
with separate terms, and "we redistribute nothing" is what currently makes that moot. Any
release that ships reconstructed images — a qualitative figure, a demo gallery — stops it
being moot and needs the terms of that specific set checked first. Pinning the archive hash
rather than mirroring the archive is a deliberate choice in the same direction.

**The tests are hermetic for this reason too.** Every test that needs benchmark data skips
cleanly when it is absent, so CI never requires a copy and a contributor without the archive
still gets a green suite. **A skip means the case was not exercised, never that it passed**
— check the skip count, not just the exit code.
