"""Facts-only regression tests for public docs (issue #285).

Each test pins one factual claim a public doc makes about the code, so a future doc
drift (or a reverted fix) is caught the same way a code regression would be. These are
not style or rewrite checks — every assertion is either "this string that names a fact
no longer true is absent" or "a link/example the doc gives actually resolves/works".
"""

import re
from pathlib import Path

from sisr.models.srcnn.model import SRCNN

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def test_readme_step_axis_table_puts_checkpoint_filenames_on_batch_axis() -> None:
    """Catches: the table listing checkpoint `{step}` under "global steps" when
    SRCheckpoint stamps _logger_step, the batch axis, into the filename.
    Mutation: move "the `{step}` in checkpoint filenames" back into the global-steps
    column -> fails.
    """
    content = _read("README.md")
    start = content.index("| Counted in global steps | Counted in batches |")
    data_row = content[start:].splitlines()[2]
    global_col, batches_col = data_row.split("|")[1], data_row.split("|")[2]
    assert "checkpoint filenames" not in global_col
    assert "checkpoint filenames" in batches_col


def test_readme_does_not_claim_a_factor_of_two_between_curve_and_checkpoint() -> None:
    """Catches: the stale "a curve and the checkpoint pulled off it are a factor of 2
    apart" claim, which stopped being true once checkpoint filenames moved onto the
    batch axis (matching the curve's own axis, off by one).
    Mutation: reinstate the literal "factor of 2 apart" sentence -> fails.
    """
    content = _read("README.md")
    assert "factor of 2 apart" not in content
    assert "not a factor of 2" in content


def test_readme_srgan_comment_pointer_names_a_public_resource() -> None:
    """Catches: README pointing at a resource readers cannot access (the golden config
    is gitignored and not distributed). The public mechanics reference is in
    docs/configuration.md.
    Mutation: delete the docs/configuration.md link or break it to a different target
    -> fails.
    """
    content = _read("README.md")
    # Pin the specific link in the "mechanics in" phrase
    m = re.search(r"mechanics in \[[^\]]+\]\(([^)]+)\)", content)
    assert m and m.group(1) == "docs/configuration.md"
    # Verify the link resolves from REPO_ROOT (README lives there)
    assert (REPO_ROOT / m.group(1)).is_file()
    # Verify the gitignored config is not referenced
    assert "config.srgan.golden.yaml" not in content


def test_readme_mentions_srcnn_ram_floor() -> None:
    """Catches: the ~16 GB SRCNN RAM floor living only in a source docstring, with no
    mention in any public doc a reader would see before hitting the slowdown.
    Mutation: delete the added README sentence -> fails.
    """
    content = _read("README.md")
    assert "16 GB" in content
    assert "sisr/datasets/srcnn.py" in content


def test_configuration_md_artifact_filename_example_matches_naming_scheme() -> None:
    """Catches: the worked example `sr-weights-10000.safetensors`, which matches
    neither the prefix convention nor the `_s{step}` separator the doc's own later
    section shows.
    Mutation: put back "sr-weights-10000.safetensors" -> fails.
    """
    content = _read("docs/configuration.md")
    assert "sr-weights-10000.safetensors" not in content
    assert re.search(r"[A-Za-z0-9_]+_s10000\.safetensors", content)


def test_configuration_md_tables_include_srprogressbar_and_ycbcrprocessor() -> None:
    """Catches: the callbacks table omitting SRProgressBar and the processor table
    omitting YCbCrProcessor, both real shipped classes.
    Mutation: remove either row -> fails.
    """
    content = _read("docs/configuration.md")
    assert "SRProgressBar" in content
    assert "YCbCrProcessor" in content


def test_reproduction_md_relative_links_resolve() -> None:
    """Catches: links in docs/reproduction.md written root-relative (e.g.
    `sisr/utils/imresize.py`) that resolve to a nonexistent docs/sisr/... path from a
    file that actually lives in docs/.
    Mutation: strip the `../` prefix from any one link -> fails.
    """
    content = _read("docs/reproduction.md")
    doc_dir = REPO_ROOT / "docs"
    links = re.findall(r"\]\(([^)]+)\)", content)
    checked = 0
    for target in links:
        if target.startswith("http") or target.startswith("#"):
            continue
        checked += 1
        assert (doc_dir / target).resolve().exists(), f"broken link target: {target}"
    assert checked >= 8


def test_reproduction_md_does_not_cite_removed_mean_psnr() -> None:
    """Catches: docs/reproduction.md citing `SRLightning._mean_psnr`, which does not
    exist anywhere in lightning_module.py, plus drifted line-number citations for the
    PSNR averaging it was trying to describe.
    Mutation: reinstate "_mean_psnr" anywhere in the file -> fails.
    """
    content = _read("docs/reproduction.md")
    assert "_mean_psnr" not in content
    assert "callbacks.py:365-370" not in content
    assert "callbacks.py:440-442" not in content
    assert "lightning_module.py:504-515" not in content
    assert "SRScorer.psnr" in content


def test_reproduction_md_checkpoint_filename_carries_no_metric_value() -> None:
    """Catches: reverting the :244 fix back to the stale
    `sr-{step}-ssim_val_RGB=...ckpt` filename example, or the :255-256 LPIPS bullet
    regressing to the same stale claim that the checkpoint filename carries a metric
    value (it doesn't -- `SRCheckpoint` builds `{prefix}_s{step}.ckpt`).
    Mutation: reinstate either stale phrase -> fails.
    """
    content = _read("docs/reproduction.md")
    assert "ssim_val_RGB=" not in content
    assert "checkpoint filename carry the bare value" not in content


def test_reproduction_md_mentions_external_degradation_verification() -> None:
    """Catches: docs/reproduction.md, the project's reproduction-record doc, never
    mentioning the most recent (and largest) degradation-order verification.
    Mutation: delete the added sentence -> fails.
    """
    content = _read("docs/reproduction.md")
    assert "357/357" in content
    assert "800/800" in content


def test_srcnn_docstring_example_is_constructible() -> None:
    """Catches: SRCNN's own docstring giving `num_filters=(64, 32, 1)` alongside
    `kernel_sizes=(9, 1, 5)` as "the original architecture" -- `_check_architecture`
    requires `len(num_filters) + 1 == len(kernel_sizes)`, so constructing SRCNN with the
    docstring's own example raises ValueError.
    Mutation: revert the docstring's num_filters example back to `(64, 32, 1)` -> fails.
    """
    doc = SRCNN.__doc__
    assert doc is not None
    filters_match = re.search(r"num_filters:.*?``\(([\d,\s]+)\)``", doc, re.DOTALL)
    kernels_match = re.search(r"kernel_sizes:.*?``\(([\d,\s]+)\)``", doc, re.DOTALL)
    assert filters_match and kernels_match
    num_filters = tuple(int(x) for x in filters_match.group(1).split(","))
    kernel_sizes = tuple(int(x) for x in kernels_match.group(1).split(","))
    SRCNN(num_channels=1, num_filters=num_filters, kernel_sizes=kernel_sizes)


def test_contributing_architecture_step_names_required_hooks() -> None:
    """Catches: CONTRIBUTING's "add a new architecture" step 1 omitting
    `input_contract` and `variant_tag` (both abstract on SRModel, enforced by
    sisr/models/base.py) and `as_int_tuple` (the normalization helper every existing
    architecture uses).
    Mutation: remove any one of the three names from CONTRIBUTING.md -> fails.
    """
    content = _read("CONTRIBUTING.md")
    assert "input_contract" in content
    assert "variant_tag" in content
    assert "as_int_tuple" in content


def test_contributing_pr_rule_matches_gh_stack_practice() -> None:
    """Catches: CONTRIBUTING's "Stacked PRs must be retargeted to `main`" rule, which
    describes a workflow this project no longer uses (current practice is `gh stack`,
    sequential per-PR bases, no retarget).
    Mutation: reinstate "retargeted to `main`" -> fails.
    """
    content = _read("CONTRIBUTING.md")
    assert "retargeted to `main`" not in content
    assert "gh stack" in content


def test_changelog_mentions_recent_merged_work() -> None:
    """Catches: CHANGELOG's Unreleased section stopping short of ~25 merged PRs
    (#244-#270) -- a degradation-order fix, a dependency-comparability fix, and an
    artifact-rebuild fix among them.
    Mutation: remove any one of these phrases -> fails.
    """
    content = _read("CHANGELOG.md")
    assert "SRProgressBar" in content
    assert "modcropped" in content
    assert "357/357" in content
    assert "rebuilds from its own artifact header" in content
