import torch
import torchvision
import hashlib
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
from ..utils import LMDBCache, LMDBCacheBuildContext


def _process_subimages(
    path: Path,
    sub_img_size: int,
    stride: int,
    scale: int,
    blur_sigma: float,
    base_idx: int,
) -> list[tuple[str, bytes]]:
    """
    Extracts all LR/HR sub-image pairs from a single image and returns
    them as keyed pairs ready for LMDB insertion.

    This is a top-level function (not a method) so that it can be
    pickled by ``ProcessPoolExecutor`` across all platforms.

    For each sliding-window position the function:
      1. Crops the HR sub-image.
      2. Applies Gaussian blur, downsamples, then upsamples to create
         the LR sub-image.
      3. Converts both sub-images to ``(C, H, W)`` uint8 layout. HR is
         always served as RGB; Y/YCbCr selection happens downstream in
         ``SRLightning`` per ``model_colorspace``.
      4. Assigns LMDB keys using ``base_idx`` as the starting offset.

    Args:
        path (Path): File path of the high-resolution image.
        sub_img_size (int): Spatial size of the square sub-images to
            extract.
        stride (int): Step size of the sliding window.
        scale (int): Downscaling factor for LR generation.
        blur_sigma (float): Radius for the Gaussian blur applied
            before downsampling.
        base_idx (int): Starting LMDB key index for this image's
            sub-images.

    Returns:
        A flat list of ``(key_string, value_bytes)`` pairs where keys
        follow the pattern ``'lr_{idx:08d}'`` / ``'hr_{idx:08d}'``.
    """
    img = Image.open(path).convert('RGB')
    w, h = img.size
    w_crop = w - (w % scale)
    h_crop = h - (h % scale)
    img = img.crop((0, 0, w_crop, h_crop))

    keyed_pairs = []
    lr_size = sub_img_size // scale
    idx = base_idx

    for top in range(0, h_crop - sub_img_size + 1, stride):
        for left in range(0, w_crop - sub_img_size + 1, stride):
            hr_subimg = img.crop(
                (left, top, left + sub_img_size, top + sub_img_size))

            lr_patch = hr_subimg.filter(
                ImageFilter.GaussianBlur(radius=blur_sigma))
            lr_patch = lr_patch.resize(
                (lr_size, lr_size), resample=Image.BICUBIC)
            lr_patch = lr_patch.resize(
                (sub_img_size, sub_img_size), resample=Image.BICUBIC)

            hr_arr = np.array(hr_subimg)
            lr_arr = np.array(lr_patch)

            hr_arr = hr_arr.transpose(2, 0, 1)
            lr_arr = lr_arr.transpose(2, 0, 1)

            keyed_pairs.append((f'lr_{idx:08d}', lr_arr.tobytes()))
            keyed_pairs.append((f'hr_{idx:08d}', hr_arr.tobytes()))
            idx += 1

    return keyed_pairs


class TrainDataset(torch.utils.data.Dataset):
    """
    Dataset that serves precomputed LR/HR sub-image pairs from an LMDB cache.

    On first instantiation with a given set of parameters the dataset
    extracts every sliding-window sub-image from every image, generates the
    corresponding low-resolution version, and persists both as uint8
    arrays in an LMDB database.  A SHA-256 checksum over the file
    manifest and all extraction parameters is stored inside the LMDB so
    that subsequent runs with identical settings skip the build entirely.

    Reference:
        Image Super-Resolution Using Deep Convolutional Networks
        https://arxiv.org/pdf/1501.00092

    Args:
        img_dir (str | Path): Directory containing the high-resolution
            images.
        subimg_size (int): Spatial size of the square sub-images to extract.
        stride (int): Step size of the sliding window used for sub-image
            extraction.
        scale (int): Downscaling factor for generating low-resolution
            sub-images.
        blur_sigma (float): Radius for the Gaussian blur applied before
            downsampling.  Defaults to ``1.0``.
        use_tqdm (bool): Whether to display a progress bar during the LMDB
            build.  Defaults to ``False``.
        cache_dir (str | Path | None): Directory in which to store the
            LMDB cache.  Defaults to ``img_dir / '.lmdb_cache'``.

    Raises:
        ValueError: If no image files are found in ``img_dir``.
    """

    def __init__(
        self,
        img_dir: str | Path,
        subimg_size: int,
        stride: int,
        scale: int,
        blur_sigma: float = 1.0,
        use_tqdm: bool = False,
        cache_dir: str | Path | None = None,
    ):
        super().__init__()

        self.img_dir = Path(img_dir)
        self.sub_img_size = subimg_size
        self.stride = stride
        self.scale = scale
        self.blur_sigma = blur_sigma
        self._num_channels = 3

        self.img_paths = sorted(
            [p for p in self.img_dir.glob('*.*') if p.is_file()])
        if not self.img_paths:
            raise ValueError(f"No images found in {img_dir}")

        cache_dir = Path(cache_dir) if cache_dir else self.img_dir / '.lmdb_cache'
        checksum = self._compute_checksum()

        self._img_offsets, total_patches = self._compute_offsets()

        patch_bytes = self._num_channels * self.sub_img_size * self.sub_img_size
        map_size = max(total_patches * 2 * patch_bytes * 2, 512 * 1024 * 1024)

        self._cache = LMDBCache(
            cache_dir=cache_dir,
            name='srcnn_patches',
            checksum=checksum,
            length=total_patches,
            map_size=map_size,
            metadata={
                'channels': str(self._num_channels),
                'subimg_size': str(self.sub_img_size),
            },
            build_fn=self._build,
            use_tqdm=use_tqdm,
        )

    def _compute_checksum(self) -> str:
        """
        Computes a SHA-256 checksum over the file manifest and dataset
        parameters.

        The manifest includes every file name and its size (via ``stat``),
        making it fast (no content hashing) while still detecting added,
        removed, or resized files.

        Returns:
            A hex-encoded SHA-256 digest string.
        """
        file_manifest = ','.join(
            f'{p.name}:{p.stat().st_size}' for p in self.img_paths
        )
        canonical = '|'.join([
            file_manifest,
            str(self.sub_img_size),
            str(self.stride),
            str(self.scale),
            str(self.blur_sigma),
        ])
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    def _compute_offsets(self) -> tuple[list[int], int]:
        """
        Reads image dimensions (without decoding pixels) to compute
        per-image cumulative sub-image offsets.

        Returns:
            A tuple of ``(offsets, total_patches)`` where *offsets* is
            a list with one starting index per image and *total_patches*
            is the grand total.
        """
        offsets = []
        offset = 0
        for path in self.img_paths:
            img = Image.open(path)
            w, h = img.size
            img.close()
            w_crop = w - (w % self.scale)
            h_crop = h - (h % self.scale)
            n_h = max(0, (h_crop - self.sub_img_size) // self.stride + 1)
            n_w = max(0, (w_crop - self.sub_img_size) // self.stride + 1)
            offsets.append(offset)
            offset += n_h * n_w
        return offsets, offset

    def _build(self, ctx: LMDBCacheBuildContext) -> None:
        """
        Populates the LMDB cache using parallel image processing.

        Each image is submitted to a worker process which extracts
        sub-images and returns keyed byte pairs.  Results are written
        to LMDB as they arrive using precomputed offsets.

        Args:
            ctx (LMDBCacheBuildContext): Build context provided by
                :class:`LMDBCache`.
        """
        process_args = [
            (self.sub_img_size, self.stride, self.scale,
             self.blur_sigma, self._img_offsets[i])
            for i in range(len(self.img_paths))
        ]

        ctx.parallel_build(
            items=self.img_paths,
            process_fn=_process_subimages,
            process_args=process_args,
            num_workers=8,
            desc="Building LMDB cache",
        )

    def __len__(self) -> int:
        return self._cache.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the LR/HR sub-image pair at the given index.

        Reads the raw uint8 bytes from LMDB, reshapes them into
        ``(C, H, W)`` arrays, and converts to ``float32`` tensors
        normalised to ``[0, 1]``.

        Args:
            idx (int): Zero-based sub-image index.

        Returns:
            A ``(lr_tensor, hr_tensor)`` tuple of ``float32`` tensors
            with shape ``(C, H, W)`` and values in ``[0, 1]``.
        """
        C = self._num_channels
        H = W = self.sub_img_size

        lr_key = f'lr_{idx:08d}'
        hr_key = f'hr_{idx:08d}'
        env = self._cache.get_env()
        with env.begin(write=False, buffers=True) as txn:
            lr_buf = txn.get(lr_key.encode())
            hr_buf = txn.get(hr_key.encode())
            # A missing key means a corrupt/incomplete cache; surface the key
            # rather than letting None flow into np.frombuffer (cryptic TypeError).
            if lr_buf is None:
                raise KeyError(lr_key)
            if hr_buf is None:
                raise KeyError(hr_key)
            lr_arr = np.frombuffer(
                lr_buf, dtype=np.uint8).reshape(C, H, W).copy()
            hr_arr = np.frombuffer(
                hr_buf, dtype=np.uint8).reshape(C, H, W).copy()

        lr_tensor = torch.tensor(lr_arr, dtype=torch.float32).div_(255.0)
        hr_tensor = torch.tensor(hr_arr, dtype=torch.float32).div_(255.0)

        return lr_tensor, hr_tensor


class ValidationDataset(torch.utils.data.Dataset):
    """
    Dataset that serves full-image LR/HR pairs for validation.

    Unlike :class:`TrainDataset` this dataset does not extract sub-images.
    Each item is a full image pair where the low-resolution version is
    produced by applying Gaussian blur followed by downsampling and
    upsampling back to the original size.

    Args:
        img_dir (str | Path): Directory containing the high-resolution
            images.
        scale (int): Downscaling factor for generating low-resolution images.
        blur_sigma (float): Radius for the Gaussian blur applied before
            downsampling.  Must match :class:`TrainDataset` to keep train/val
            LR generation consistent.  Defaults to ``1.0``.

    Raises:
        ValueError: If no image files are found in ``img_dir``.
    """

    def __init__(
        self,
        img_dir: str | Path,
        scale: int,
        blur_sigma: float = 1.0,
    ):
        super().__init__()

        self.img_dir = Path(img_dir)
        self.scale = scale
        self.blur_sigma = blur_sigma

        self.img_paths = sorted(
            [p for p in self.img_dir.glob('*.*') if p.is_file()])
        if not self.img_paths:
            raise ValueError(f"No images found in {img_dir}")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the LR/HR image pair at the given index.

        Args:
            idx (int): Zero-based image index.

        Returns:
            A ``(lr_tensor, hr_tensor)`` tuple of ``float32`` tensors
            with shape ``(C, H, W)`` and values in ``[0, 1]``.
        """
        path = self.img_paths[idx]

        hr_img = Image.open(path).convert('RGB')

        lr_img = hr_img.filter(ImageFilter.GaussianBlur(radius=self.blur_sigma))
        lr_size = (hr_img.width // self.scale, hr_img.height // self.scale)
        lr_img = lr_img.resize(lr_size, resample=Image.BICUBIC)
        lr_img = lr_img.resize(hr_img.size, resample=Image.BICUBIC)

        # HR is always served as RGB; Y/YCbCr selection happens downstream in
        # SRLightning per model_colorspace.
        hr_tensor = torchvision.transforms.functional.to_tensor(hr_img)
        lr_tensor = torchvision.transforms.functional.to_tensor(lr_img)

        return lr_tensor, hr_tensor
