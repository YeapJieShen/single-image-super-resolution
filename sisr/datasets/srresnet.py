"""SRResNet-style data pipeline.

LR is the bicubic downscale of HR by ``scale`` (no upsample round-trip);
the model is responsible for the ×``scale`` upsampling. :class:`TrainDataset`
serves random ``hr_crop_size`` crops without caching (random crops aren't
cacheable); :class:`ValidationDataset` serves full images cropped to a
multiple of ``scale``.
"""
import cv2
import numpy as np
import torch
import torchvision  # still used by ValidationDataset; removed in Task 5
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pathlib import Path


class TrainDataset(torch.utils.data.Dataset):
    """Random-crop HR/LR pairs for SRResNet-style training (AlbumentationsX backend).

    Each ``__getitem__`` takes a random ``hr_crop_size`` square crop from an
    HR image via :class:`albumentations.RandomCrop` and bicubic-downsamples
    it by ``scale`` (via :class:`albumentations.Resize` with
    ``cv2.INTER_CUBIC``) to form the LR input. Unlike
    :class:`sisr.datasets.srcnn.TrainDataset` there is **no
    blur+downsample+upsample round-trip** and the LR is *not* upsampled back —
    the model is responsible for the ×``scale`` upsampling, so the LR tensor is
    ``hr_crop_size // scale`` on a side.

    Crops are generated on the fly (so they differ every epoch); there is no
    LMDB cache because random crops are not cacheable. HR is always served as
    RGB; Y/YCbCr selection happens downstream in :class:`SRLightning`.

    Reference:
        Photo-Realistic Single Image Super-Resolution Using a Generative
        Adversarial Network (https://arxiv.org/pdf/1609.04802)

    Args:
        img_dir (str | Path): Directory containing the high-resolution images.
        scale (int): Upscaling factor. ``hr_crop_size`` must be divisible by it.
        hr_crop_size (int): Side length of the square HR crop.
        crops_per_image (int): Number of random crops drawn per image per
            epoch (the dataset length is ``len(images) * crops_per_image``).
            Defaults to ``1``.

    Raises:
        ValueError: If no images are found, or ``hr_crop_size`` is not
            divisible by ``scale``.
    """

    def __init__(
        self,
        img_dir: str | Path,
        scale: int,
        hr_crop_size: int,
        crops_per_image: int = 1,
    ):
        super().__init__()

        if hr_crop_size % scale != 0:
            raise ValueError(
                f"hr_crop_size ({hr_crop_size}) must be divisible by scale ({scale})."
            )

        self.img_dir = Path(img_dir)
        self.scale = scale
        self.hr_crop_size = hr_crop_size
        self.crops_per_image = crops_per_image

        self.img_paths = sorted(
            [p for p in self.img_dir.glob('*.*') if p.is_file()])
        if not self.img_paths:
            raise ValueError(f"No images found in {img_dir}")

        lr_size = hr_crop_size // scale
        self._hr_crop = A.Compose([A.RandomCrop(hr_crop_size, hr_crop_size)])
        self._lr_pipeline = A.Compose([
            A.Resize(lr_size, lr_size, interpolation=cv2.INTER_CUBIC),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ])
        self._hr_to_tensor = A.Compose([
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.img_paths) * self.crops_per_image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a ``(lr_tensor, hr_tensor)`` pair where ``hr_tensor`` is a random crop.

        ``hr_tensor`` is a random ``hr_crop_size`` square crop and
        ``lr_tensor`` is its bicubic downscale by ``scale`` (side
        ``hr_crop_size // scale``). Both are ``float32`` in ``[0, 1]``
        with shape ``(3, H, W)``.
        """
        path = self.img_paths[idx % len(self.img_paths)]
        arr = np.array(Image.open(path).convert('RGB'))  # HWC uint8 RGB

        h, w = arr.shape[:2]
        if w < self.hr_crop_size or h < self.hr_crop_size:
            raise ValueError(
                f"Image {path.name} ({w}x{h}) is smaller than hr_crop_size {self.hr_crop_size}."
            )

        hr_arr = self._hr_crop(image=arr)['image']  # HWC uint8

        lr_tensor = self._lr_pipeline(image=hr_arr)['image']
        hr_tensor = self._hr_to_tensor(image=hr_arr)['image']

        return lr_tensor, hr_tensor


class ValidationDataset(torch.utils.data.Dataset):
    """Full-image HR with bicubic-downsampled LR for SRResNet validation/test.

    Each item is a full image pair. The HR image is cropped to a multiple of
    ``scale`` so the model's ×``scale`` output lands exactly on the HR size;
    the LR is the bicubic downscale by ``scale`` (no upsample round-trip).
    HR is always served as RGB.

    Args:
        img_dir (str | Path): Directory containing the high-resolution images.
        scale (int): Upscaling factor.

    Raises:
        ValueError: If no images are found in ``img_dir``.
    """

    def __init__(self, img_dir: str | Path, scale: int):
        super().__init__()

        self.img_dir = Path(img_dir)
        self.scale = scale

        self.img_paths = sorted(
            [p for p in self.img_dir.glob('*.*') if p.is_file()])
        if not self.img_paths:
            raise ValueError(f"No images found in {img_dir}")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a ``(lr_tensor, hr_tensor)`` pair for the image at ``idx``.

        ``hr_tensor`` is the HR image cropped to a multiple of ``scale``,
        and ``lr_tensor`` is its bicubic downscale by ``scale``. Both are
        ``float32`` in ``[0, 1]``.
        """
        path = self.img_paths[idx]
        hr_img = Image.open(path).convert('RGB')

        w, h = hr_img.size
        w_crop = w - (w % self.scale)
        h_crop = h - (h % self.scale)
        hr_img = hr_img.crop((0, 0, w_crop, h_crop))

        lr_img = hr_img.resize(
            (w_crop // self.scale, h_crop // self.scale), resample=Image.BICUBIC)

        hr_tensor = torchvision.transforms.functional.to_tensor(hr_img)
        lr_tensor = torchvision.transforms.functional.to_tensor(lr_img)

        return lr_tensor, hr_tensor
