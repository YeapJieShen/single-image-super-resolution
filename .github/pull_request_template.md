<!--
Delete any section that does not apply. A short PR should have a short body.
Conventions: CONTRIBUTING.md → "Making a change".
-->

## What changed

<!-- The effect, not the mechanism. One or two sentences. -->

## Why

<!-- What made this necessary. The diff already shows what you did. -->

## Evidence

<!--
Keep whichever applies; delete the rest. Assertions without evidence are the
thing this section exists to prevent.
-->

**Bug fix** — the test that catches it:

- Test: `tests/...::test_...`
- Confirmed **RED** against the unfixed code, **GREEN** after.
  <!-- A test that passes both ways documents nothing. Say how you proved it fails. -->

**Performance** — before/after on real data:

| | before | after |
| --- | --- | --- |
| <!-- ms/step, ops/s --> | | |

- Medians with warm-up excluded; ___ iterations discarded.
  <!-- A mean over a short run hides worker spawn / cache warm / autotune and can invert the result. -->
- Measured on: <!-- which checkout, which dataset, which power state -->

**Touches `sisr/imresize.py`** — byte-equality against the MATLAB reference set re-proven:

- <!-- result, e.g. 357/357, max|diff| = 0 -->

## Checklist

- [ ] `pytest` passes locally, and I have said which tests **skipped** (a skip is not a pass)
- [ ] `ruff check` and `ruff format --check` are clean
- [ ] Commit subjects use a Conventional Commit type and describe the **effect**
- [ ] Docs changes are in their own `docs:` commit, not mixed into a code commit
- [ ] Breaking changes carry `!` and a `BREAKING CHANGE:` footer
- [ ] No internal tracker identifiers in the code or commit messages

## Merge strategy

<!-- Pick one. This is not cosmetic: squashing a mixed PR collapses the docs
     commit back into the code commit and defeats the separation rule. -->

- [ ] **Squash** — single-purpose PR
- [ ] **Rebase** — carries both code and docs commits

---

<sub>**Note on `codecov/patch`:** it is expected to fail and is not a required check. It
reports on paths that cannot execute on CPU-only runners (cache-lock liveness). The real
coverage gate is `--cov-fail-under=90` inside `test`. No need to
investigate it.</sub>
