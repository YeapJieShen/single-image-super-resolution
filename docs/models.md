# Models

One row per architecture: what its LR input is, what it does, which template runs it, and **what
is verified against its paper**. Parity figures are measured by data-gated tests against the
authors' released weights and scored under the paper's own protocol; the measurements, protocols
and their caveats live in [`reproduction.md`](reproduction.md), and this page links to them
rather than restating tables.

**Parity status** is one of: *reproduced* (every set and scale within 0.05 dB of the paper's
PSNR-Y), *reproduced, difference named* (within 0.20 dB, with the cause stated), or *not yet
reproduced*; where a result is within 0.20 dB but not claimed as a reproduction, the row says what
was measured. "Ported" means the authors' released weights load into this implementation; the
weights themselves are not redistributed ([licensing](weights-and-licensing.md)).

| Model | Paper | LR input | Template | Parity status |
|---|---|---|---|---|
| SRCNN | [Dong et al. 2015](https://arxiv.org/pdf/1501.00092) | pre-upsampled | `config.srcnn.template.yaml` | not yet reproduced: no publishable run ([details](reproduction.md#srcnn-has-no-publishable-row)) |
| SRResNet | [Ledig et al. 2017](https://arxiv.org/pdf/1609.04802) | native LR | `config.srresnet.template.yaml` | reproduced, difference named: within about 0.1 dB on Set5, Set14 and BSD100, trained on DIV2K-800 rather than the paper's 350k ImageNet images ([details](reproduction.md#srresnet-reproduces-ledig-et-al)) |
| SRGAN | [Ledig et al. 2017](https://arxiv.org/pdf/1609.04802) | native LR | `config.srgan.template.yaml` | not reproduced: one constant is isolated ([details](reproduction.md#srgan-vgg54-does-not-and-the-cause-is-isolated)) |
| ESPCN | [Shi et al. 2016](https://arxiv.org/pdf/1609.05158) | native LR | `config.espcn.template.yaml` | not yet reproduced: built and smoke-tested |
| DRCN | [Kim et al. 2016](https://arxiv.org/abs/1511.04491) | pre-upsampled | `config.drcn.template.yaml` | reproduced (official weights ported) |
| VDSR | [Kim et al. 2016](https://arxiv.org/abs/1511.04587) | pre-upsampled | `config.vdsr.template.yaml` | reproduced (official weights ported) |
| ESRGAN | [Wang et al. 2018](https://arxiv.org/abs/1809.00219) | native LR | `config.esrgan.template.yaml`, `config.esrgan_psnr.template.yaml` | PSNR stage reproduced (official weights ported); GAN stage: a perceptual-index baseline is recorded (#289), deliberately not compared pass/fail against the paper |
| EDSR, EDSR+ | [Lim et al. 2017](https://arxiv.org/abs/1707.02921) | native LR | `config.edsr.template.yaml` | reproduced (six released checkpoints ported) |
| MDSR, MDSR+ | [Lim et al. 2017](https://arxiv.org/abs/1707.02921) | native LR | `config.mdsr.template.yaml` | within 0.16 dB of the paper with the gaps named; not claimed as reproduced (released 80-block checkpoint ported) |

*Native LR* means the dataset serves a true low-resolution image and the network upsamples it
(sub-pixel convolution or nearest-neighbour). *Pre-upsampled* means the dataset bicubic-upsamples
LR back to HR size and the network refines it at that size. The choice is the model's
`input_contract`, and the datamodule checks the dataset against it.

## Per model

**SRCNN** is Y-channel only, same-resolution, on the full-range Y that its training
([`YChannelProcessor`](configuration.md#model-the-network-and-processor)) uses. It degrades whole
images and caches the LR plane, upsampling each training patch at load; on DIV2K-800 at x3 with 16
workers the process tree holds about 9.5 GiB, so a 16 GB host fits it. With much less RAM a
shuffling loader goes I/O-bound **without any error** (low GPU utilisation, high iowait). Training
scales must be odd or a power of two.

**SRResNet and SRGAN** share one network; SRGAN differs in how it is trained
([SRGAN specifics](configuration.md#srgan-specifics)). SRResNet scores SSIM with the `daala`
convention by default, as its paper did ([why](reproduction.md#comparability)).

**ESPCN** is sub-pixel convolution on native LR.

**DRCN and VDSR** were trained on MATLAB's studio-range luma, so their templates use
`StudioYChannelProcessor` rather than the full-range processor. Both ported-weights tests score
PSNR-Y on Set5, Set14 and BSD100 at x2/x3/x4 within 0.05 dB of the paper under the authors' own
evaluation. Through the pipeline, the templates set the evaluation switches that those scripts
imply (8-bit output rounding, float LR degradation, input-luma clamp, grey-plane scoring for
single-channel images; see [Paper protocols](configuration.md#eval_config-and-the-two-ssim-conventions)),
which keeps all 18 cells within 0.025 dB. DRCN is 16 shared recursions with recursive
supervision; VDSR is 20 layers with a global residual, one model for x2/x3/x4.

**DRCN follows the authors' released network, which differs from the paper's equations.** The
paper (Section 3, eqs. 4-10) describes a recursion `H_d = max(0, W * H_{d-1} + b)`, a two-layer
reconstruction net with a ReLU after each layer that only reads `H_d`, and a skip inside every
prediction: `y_d = x + f3(H_d)`, `y = sum_d w_d * y_d`. The released network
([DRCN_code.zip](https://cv.snu.ac.kr/research/DRCN/)) differs in three places, visible in the
layer graph stored in `DRCN_sf{2,3,4}.mat`:

| | Paper | Released network |
|---|---|---|
| Recursion step | shared conv + ReLU | shared conv (`filters_share`) + ReLU, then the first reconstruction conv (`filters19`), whose output is the next step's input |
| Reconstruction ReLUs | after both layers | none (`conv_outh` and `conv_out` are linear) |
| Input skip | inside each prediction, so the output is `(sum_d w_d) * x + sum_d w_d * f3(H_d)` | added once after the ensemble (`runDRCN.m`: `impred = imlow + impred`), so the output is `x + sum_d w_d * r_d` |

The release also makes 17 predictions (one before the first recursion, then one per recursion),
and combines them with a bias-free 1x1 conv whose raw weights sum to 1.65 (x2), 1.28 (x3) and
1.48 (x4). The paper's skip placement would therefore scale the input by those factors.

`sisr.models.drcn.DRCN` implements the released wiring, because it is the one that reproduces
the paper's numbers. The ported official weights score within 0.025 dB of the paper's table on
every set and scale, and within 0.016 dB of the authors' own evaluation script (#326). Training
this model from scratch therefore trains the released architecture. A re-implementation from the
paper's text alone is a different network, and one public re-implementation reports PSNR 0.5 to
1.0 dB below the paper
([deeply-recursive-cnn-tf](https://github.com/jiny2001/deeply-recursive-cnn-tf)).

**ESRGAN** is the RRDB generator (x4, nearest-neighbour upsampling), a relativistic GAN
discriminator and VGG54 features before activation. The PSNR-stage official weights score
within 0.05 dB of the paper's DF2K row on Set5, Set14 and BSD100. The GAN stage's perceptual
index (PI, from `niqe` and `ma`) is recorded as a baseline on the benchmark sets and the paper's Fig. 7 images, with no pass/fail against the paper, whose figures are for the competition variant.

**EDSR and EDSR+** are the residual trunk without BatchNorm, x2/x3/x4 with sub-pixel
convolution; EDSR+ is the same weights with `eval_config.self_ensemble: true`. Under the paper's
MATLAB 8-bit-luma protocol every set and scale is within 0.05 dB of the EDSR and EDSR+ rows (this
pipeline's unrounded float Y reads 0.01 to 0.10 dB higher).

**MDSR and MDSR+** are one model for x2/x3/x4: scale-specific pre-processing and upsampling around
a shared trunk; MDSR+ is `self_ensemble`. The released 80-block checkpoint is within 0.05 dB of the
paper on BSD100 and within 0.16 dB on Set5 and Set14. The Set14 gap is a protocol difference: the
paper's `evaluation.m` scores a single-channel image (Set14's `bridge`) on its raw plane, which
moves the Set14 mean by 0.094 dB (the template sets `grey_plane_scoring`). The Set5 gap is weights
provenance: the released PyTorch MDSR is a different network from the authors' Torch7 paper-release model
(`model_paper.tar`), which has no scale-specific pre-processing; that model too misses Set5 x4 by
0.066 dB.

## Adding a model

See [CONTRIBUTING.md](../CONTRIBUTING.md#adding-a-new-architecture). A new row here needs a
template and, for a parity claim, a data-gated test against released weights.
