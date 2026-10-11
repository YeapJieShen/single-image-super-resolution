"""The native NIQE port: each piece of the MATLAB release, pinned on its own.

Nothing here needs SciPy, pyiqa or the downloaded pristine model except the loader
tests (``importorskip`` SciPy) -- the pipeline tests inject a synthetic model.
"""

import hashlib
import io
import math
import zipfile

import numpy as np
import pytest
import torch

from sisr.metrics import niqe as niqe_mod
from sisr.metrics.niqe import niqe, niqe_features, to_uint8_luma
from sisr.utils.imresize import matlab_imresize


def _identity_model() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(36, dtype=torch.float64), torch.eye(36, dtype=torch.float64)


def _textured(h: int, w: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(1, 3, h // 8, w // 8, generator=g)
    smooth = torch.nn.functional.interpolate(base, size=(h, w), mode="bilinear")
    return (smooth + 0.1 * torch.rand(1, 3, h, w, generator=g)).clamp(0, 1)


# --- luma ------------------------------------------------------------------------


def test_gray_luma_uses_matlab_rgb2gray_coefficients():
    """(189, 0, 0) is 56.4989 under MATLAB's 0.298936... and 56.511 under 0.299."""
    img = torch.tensor([189.0, 0.0, 0.0]).reshape(1, 3, 1, 1) / 255

    assert to_uint8_luma(img, "gray").item() == 56


def test_ycbcr_luma_is_matlab_studio_range_y():
    img = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]).T.reshape(1, 3, 1, 2)

    assert to_uint8_luma(img, "ycbcr").flatten().tolist() == [16.0, 235.0]


def test_luma_is_computed_on_8bit_input():
    """The release runs on imread output: R = 189.4 is read as 189, whose luma is
    56.4989 -> 56. Unquantised it would be 56.618 -> 57."""
    img = torch.tensor([189.4, 0.0, 0.0]).reshape(1, 3, 1, 1) / 255

    assert to_uint8_luma(img.to(torch.float64), "gray").item() == 56


def test_unknown_luma_is_rejected():
    with pytest.raises(ValueError, match="luma"):
        to_uint8_luma(torch.zeros(1, 3, 1, 1), "yiq")  # type: ignore[arg-type]


# --- building blocks ---------------------------------------------------------------


def test_gaussian_window_is_fspecial_7_sigma_7_over_6():
    w = niqe_mod._gaussian_window()

    assert w.shape == (7, 7)
    assert w.sum().item() == pytest.approx(1.0, abs=1e-15)
    assert (w[3, 3] / w[3, 4]).item() == pytest.approx(math.exp(1 / (2 * (7 / 6) ** 2)))


def test_mscn_of_a_constant_plane_is_zero():
    """Replicate padding: no border artefact on a flat image."""
    flat = torch.full((20, 20), 80.0, dtype=torch.float64)

    torch.testing.assert_close(niqe_mod._mscn(flat), torch.zeros_like(flat), rtol=0, atol=1e-12)


def test_half_scale_is_matlab_bicubic_without_requantising():
    """Rounded, it agrees with the byte-exact uint8 imresize to within its one
    extra rounding step; a bilinear or non-antialiased resize does not."""
    rng = np.random.default_rng(0)
    plane = rng.integers(60, 196, size=(96, 144)).astype(np.uint8)  # no clipping

    ours = niqe_mod._imresize_half(torch.from_numpy(plane.astype(np.float64)))
    reference = matlab_imresize(plane, (48, 72)).astype(np.float64)

    assert ours.shape == (48, 72)
    assert np.abs(ours.numpy().round() - reference).max() <= 1
    assert not np.array_equal(ours.numpy(), ours.numpy().round())  # doubles, not levels


def _aggd_samples(alpha: float, beta_l: float, beta_r: float, n: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    magnitude = torch.distributions.Gamma(1 / alpha, 1.0).sample((n,)) ** (1 / alpha)
    left = torch.rand(n, generator=g) < beta_l / (beta_l + beta_r)
    return torch.where(left, -beta_l * magnitude, beta_r * magnitude).to(torch.float64)


@pytest.mark.parametrize(("alpha", "beta_l", "beta_r"), [(0.8, 0.5, 1.2), (2.0, 1.0, 1.0)])
def test_aggd_fit_recovers_known_parameters(alpha, beta_l, beta_r):
    torch.manual_seed(0)
    fitted = niqe_mod._aggd(_aggd_samples(alpha, beta_l, beta_r, 400_000)[None])

    assert fitted[0].item() == pytest.approx(alpha, abs=0.03)
    assert fitted[1].item() == pytest.approx(beta_l, rel=0.02)
    assert fitted[2].item() == pytest.approx(beta_r, rel=0.02)


def test_aggd_grid_is_matlabs_colon_range():
    """gam = 0.2:0.001:10 -- 9801 points, both ends exact."""
    gam, r_gam = niqe_mod._aggd_grid()

    assert gam.shape == r_gam.shape == (9801,)
    assert gam[0].item() == 0.2 and gam[-1].item() == 10.0
    assert gam[1].item() == pytest.approx(0.201, abs=1e-15)
    expected = math.gamma(2 / 0.201) ** 2 / (math.gamma(1 / 0.201) * math.gamma(3 / 0.201))
    assert r_gam[1].item() == pytest.approx(expected, rel=1e-12)


def test_aggd_of_a_one_sided_sample_is_nan_like_matlab():
    """sqrt(mean([])) is NaN in MATLAB, and that NaN must reach the features."""
    _, beta_l, _ = niqe_mod._aggd(torch.rand(1, 50, dtype=torch.float64))

    assert beta_l.isnan().item()


def test_features_are_per_96x96_block_at_two_scales():
    luma = to_uint8_luma(_textured(200, 300))[0]

    assert niqe_features(luma).shape == (2 * 3, 36)


def test_pair_products_shift_within_each_block():
    """circshift runs inside blkproc, so a block's features ignore its neighbours."""
    g = torch.Generator().manual_seed(1)
    structdis = torch.randn(96, 192, generator=g, dtype=torch.float64)

    both = niqe_mod._block_features(structdis, 96)
    alone = niqe_mod._block_features(structdis[:, :96].clone(), 96)

    torch.testing.assert_close(both[0], alone[0], rtol=0, atol=0)


def test_mvg_distance_is_zero_when_features_centre_on_the_model_mean():
    mu = torch.linspace(0.1, 1.0, 36, dtype=torch.float64)
    delta = torch.linspace(-1, 1, 36, dtype=torch.float64)
    features = torch.stack([mu + delta, mu - delta])

    assert niqe_mod._mvg_distance(
        features, (mu, torch.eye(36, dtype=torch.float64))
    ).item() == pytest.approx(0.0, abs=1e-12)


def test_a_single_block_has_zero_covariance_like_matlab_cov():
    """MATLAB normalises a one-observation covariance by 1, giving zeros, so a
    96x96 image scores sqrt(d' pinv(cov_p / 2) d)."""
    features = niqe_features(to_uint8_luma(_textured(96, 96))[0])
    mu_p, cov_p = _identity_model()

    expected = torch.sqrt(2 * ((mu_p - features[0]) ** 2).sum())

    torch.testing.assert_close(niqe_mod._mvg_distance(features, (mu_p, cov_p)), expected)


def test_mvg_distance_uses_the_pseudo_inverse():
    """Identical blocks give a zero covariance; with a zero pristine covariance too,
    only pinv (as the release uses) is defined."""
    mu = torch.zeros(36, dtype=torch.float64)
    features = torch.ones(3, 36, dtype=torch.float64)

    distance = niqe_mod._mvg_distance(features, (mu, torch.zeros(36, 36, dtype=torch.float64)))

    assert distance.item() == 0.0


# --- the whole score ----------------------------------------------------------------


def test_niqe_scores_each_image_of_a_batch():
    batch = torch.cat([_textured(96, 96, seed=0), _textured(96, 96, seed=1)])

    scores = niqe(batch, model=_identity_model())
    singles = [niqe(img[None], model=_identity_model()) for img in batch]

    assert scores.shape == (2,)
    torch.testing.assert_close(scores, torch.cat(singles))


def test_a_constant_image_scores_nan():
    """No block of a flat image has measurable statistics; the release's pinv raises
    on the resulting NaN covariance, this reports NaN."""
    assert niqe(torch.full((1, 3, 96, 96), 0.5), model=_identity_model()).isnan().item()


def test_shave_is_applied_before_blocking():
    img = _textured(100, 100)

    with pytest.raises(ValueError, match="96x96"):
        niqe(img, shave=4, model=_identity_model())
    niqe(img, shave=2, model=_identity_model())


# --- pristine model loader -----------------------------------------------------------


@pytest.fixture
def fake_release(tmp_path, monkeypatch):
    scipy_io = pytest.importorskip("scipy.io")
    mu = np.arange(36, dtype=np.float64)[None]
    cov = np.eye(36) * 2.0
    buf = io.BytesIO()
    scipy_io.savemat(buf, {"mu_prisparam": mu, "cov_prisparam": cov})
    mat = buf.getvalue()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("niqe_release/modelparameters.mat", mat)
        zf.writestr("niqe_release/image1.bmp", b"not wanted")
    payload = archive.getvalue()

    calls = []

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(url):
        calls.append(url)
        return _Response(payload)

    monkeypatch.setattr(niqe_mod.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(niqe_mod, "RELEASE_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(niqe_mod, "MODEL_SHA256", hashlib.sha256(mat).hexdigest())
    monkeypatch.setattr(niqe_mod, "_cache_path", lambda: tmp_path / "hub" / "modelparameters.mat")
    niqe_mod.pristine_model.cache_clear()
    yield calls, tmp_path / "hub"
    niqe_mod.pristine_model.cache_clear()


def test_pristine_model_is_fetched_once_and_only_the_model_is_kept(fake_release):
    calls, cache_dir = fake_release

    mu, cov = niqe_mod.pristine_model()
    niqe_mod.pristine_model.cache_clear()
    niqe_mod.pristine_model()

    assert calls == [niqe_mod.RELEASE_URL]
    assert sorted(p.name for p in cache_dir.iterdir()) == ["modelparameters.mat"]
    assert mu.shape == (36,) and mu[5].item() == 5.0
    assert cov.shape == (36, 36) and cov[0, 0].item() == 2.0


def test_a_tampered_archive_is_refused(fake_release, monkeypatch):
    monkeypatch.setattr(niqe_mod, "RELEASE_SHA256", "0" * 64)

    with pytest.raises(RuntimeError, match="checksum"):
        niqe_mod.pristine_model()


def test_a_tampered_cached_model_is_refused(fake_release):
    _, cache_dir = fake_release
    cache_dir.mkdir(parents=True)
    (cache_dir / "modelparameters.mat").write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="checksum"):
        niqe_mod.pristine_model()


def test_the_pinned_release_hashes_are_the_authors_2012_release():
    """Spelled out so a change to either pin is a visible, deliberate diff."""
    assert niqe_mod.RELEASE_URL == "http://live.ece.utexas.edu/research/quality/niqe_release.zip"
    assert (
        niqe_mod.RELEASE_SHA256
        == "3d613b077a1b08273c79999f2e1643d5b2d9b2d0d3f3aa622236fc9592ea857c"
    )
    assert (
        niqe_mod.MODEL_SHA256 == "fa2aa4d12dd9595c3fb6bf8b4e1a15c79d5d1469238d0901c88dead195be578d"
    )
