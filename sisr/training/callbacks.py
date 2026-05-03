import math
import torch
import torch.nn.functional
import torchmetrics.functional
import torchvision
import lightning
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from typing import Any, Dict, List, Optional


class BenchmarkImageLogger(Callback):
    """
    Log per-image LR|SR|HR composites and scalar PSNR/SSIM to TensorBoard
    for one or more benchmark validation dataloaders.

    When using multiple validation dataloaders with
    ``trainer.fit(val_dataloaders=[main_dl, set5_dl, set14_dl])``, this
    callback selectively captures outputs from the benchmark dataloaders
    (identified by their index) and logs each image individually.

    Each image is logged as a horizontal **LR | SR | HR** strip.  When the
    model uses ``padding='valid'`` the SR output is zero-padded (black
    border) so that all three panels share the same spatial size.

    Per-image PSNR and SSIM scalars are logged under
    ``"{name}_psnr/{idx}"`` and ``"{name}_ssim/{idx}"``.

    Logging frequency is controlled by ``log_every_n_val_runs`` which
    counts how many times validation has been triggered (works correctly
    with both epoch-based and step-based validation schedules).

    Args:
        dataloader_mapping (Dict[int, str]): Mapping from validation
            dataloader index to a human-readable dataset name used as
            the TensorBoard tag prefix (e.g. ``{1: "Set5", 2: "Set14"}``).
        log_every_n_val_runs (int): Only log images every *n*-th
            validation run.  Defaults to ``5``.  With step-based training
            (e.g. ``val_check_interval=1000``, ``max_steps=100_000``)
            validation fires ~100 times, so logging every 5 runs gives
            ~20 image snapshots — enough to track visual progress without
            flooding TensorBoard storage.
    """

    def __init__(
        self,
        dataloader_mapping: Optional[Dict[int, str]] = None,
        log_every_n_val_runs: int = 5,
        crop_border: int = 0,
    ):
        super().__init__()
        self.dataloader_mapping = dataloader_mapping
        self.log_every_n_val_runs = log_every_n_val_runs
        self.crop_border = crop_border

        # Internal counter for validation runs
        self._val_run_count = 0

        # Buffer: {dataset_name: [(filename, lr|None, sr|None, hr|None, psnr_dict, ssim), ...]}
        self._buffer: Dict[str, List[tuple]] = {}

    def setup(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        stage: str,
    ):
        """Auto-discover ``dataloader_mapping`` from the attached datamodule.

        If ``dataloader_mapping`` was not supplied at construction time, this
        callback inspects ``trainer.datamodule.benchmark_names`` (a list of
        benchmark dataset names exposed by :class:`SRDataModule`) and builds
        the mapping ``{i + 1: name}`` — index 0 is reserved for the primary
        validation loader, benchmarks start at 1.
        """
        if self.dataloader_mapping is not None:
            return
        dm = getattr(trainer, "datamodule", None)
        names = getattr(dm, "benchmark_names", None) if dm is not None else None
        self.dataloader_mapping = {
            i + 1: name for i, name in enumerate(names or [])
        }

    def on_validation_epoch_start(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ):
        """
        Clear the buffers and bump the validation-run counter.

        Args:
            trainer (lightning.Trainer): The trainer instance.
            pl_module (lightning.LightningModule): The model being trained.
        """
        self._val_run_count += 1
        self._buffer = {name: [] for name in self.dataloader_mapping.values()}

    def on_validation_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """
        Collect LR/SR/HR tensors from benchmark dataloaders.

        Metrics (PSNR/SSIM) are always computed and accumulated so that
        per-dataset means are available every validation epoch.  Image
        tensors are only stored when this run falls on the logging interval.

        Args:
            trainer (lightning.Trainer): The trainer instance.
            pl_module (lightning.LightningModule): The model being trained.
            outputs: The outputs from the validation step (ignored).
            batch: The input batch, expected to be a tuple of (LR, HR) tensors.
            batch_idx: Index of the batch within the current dataloader.
            dataloader_idx: Index of the dataloader (0-based).  Only batches
                from dataloaders in *dataloader_mapping* are collected.
        """
        if dataloader_idx not in self.dataloader_mapping:
            return

        dataset_name = self.dataloader_mapping[dataloader_idx]
        lr_img, hr_img = batch

        with torch.no_grad():
            sr = pl_module(lr_img)

        hr_cropped = torchvision.transforms.functional.center_crop(hr_img, sr.shape[-2:])

        # Resolve filenames from the underlying dataset for use as TB tags.
        dataset = trainer.val_dataloaders[dataloader_idx].dataset
        batch_size = lr_img.size(0)

        # Only store image tensors when this run falls on the image-logging interval
        should_log_images = (self._val_run_count % self.log_every_n_val_runs == 0)

        for i in range(batch_size):
            global_idx = batch_idx * batch_size + i
            filename = dataset.img_paths[global_idx].stem

            sr_4d = sr[i].unsqueeze(0).cpu()
            hr_4d = hr_cropped[i].unsqueeze(0).cpu()
            if self.crop_border > 0:
                n = self.crop_border
                sr_4d = sr_4d[..., n:-n, n:-n]
                hr_4d = hr_4d[..., n:-n, n:-n]
            psnr_tensors = pl_module._build_psnr_tensors(sr_4d, hr_4d)
            psnr_dict = {
                key: torchmetrics.functional.image.peak_signal_noise_ratio(
                    sr_t, hr_t, data_range=1.0
                ).item()
                for key, (sr_t, hr_t) in psnr_tensors.items()
            }
            ssim = torchmetrics.functional.image.structural_similarity_index_measure(
                sr_4d, hr_4d, data_range=1.0
            )
            self._buffer[dataset_name].append((
                filename,
                lr_img[i].cpu() if should_log_images else None,
                sr[i].cpu() if should_log_images else None,
                hr_img[i].cpu() if should_log_images else None,
                psnr_dict,
                ssim.item(),
            ))

    def on_validation_epoch_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ):
        """
        Log benchmark metrics and (on interval) image strips to TensorBoard.

        Mean PSNR/SSIM are logged every validation epoch via ``pl_module.log``.
        Per-image scalars and LR|SR|HR image strips are only written when this
        run falls on the logging interval.

        Args:
            trainer (lightning.Trainer): The trainer instance.
            pl_module (lightning.LightningModule): The model being trained.
        """
        step = trainer.global_step
        should_log_images = (self._val_run_count % self.log_every_n_val_runs == 0)

        for dataset_name, samples in self._buffer.items():
            if not samples:
                continue

            psnr_keys = samples[0][4].keys()
            mean_ssim = sum(s for *_, s in samples) / len(samples)

            for key in psnr_keys:
                mean_psnr = sum(s[4][key] for s in samples) / len(samples)
                pl_module.log(f"{dataset_name}_psnr({key})", mean_psnr, add_dataloader_idx=False)
            pl_module.log(f"{dataset_name}_ssim", mean_ssim, add_dataloader_idx=False)

            if should_log_images:
                tb_logger = next(
                    (l for l in trainer.loggers
                     if isinstance(l, lightning.pytorch.loggers.TensorBoardLogger)),
                    None,
                )
                if tb_logger is None:
                    continue

                experiment = tb_logger.experiment
                for filename, lr, sr, hr, psnr_dict, ssim in samples:
                    for key, psnr_val in psnr_dict.items():
                        experiment.add_scalar(f"{dataset_name}_psnr({key})/{filename}", psnr_val, global_step=step)
                    experiment.add_scalar(f"{dataset_name}_ssim/{filename}", ssim, global_step=step)

                    sr_padded = self._pad_to_match(sr, lr.shape[-2:])
                    strip = torchvision.utils.make_grid(
                        [lr, sr_padded, hr], nrow=3, padding=2, pad_value=0.5
                    )
                    experiment.add_image(f"{dataset_name}/{filename}", strip, global_step=step)

        self._buffer.clear()

    @staticmethod
    def _pad_to_match(img: torch.Tensor, target_hw: tuple) -> torch.Tensor:
        """
        Zero-pad a ``(C, H, W)`` tensor to *target_hw* spatial size.


        Args:
            img (torch.Tensor): Image tensor of shape ``(C, H, W)``.
            target_hw (tuple): Target ``(H, W)`` spatial dimensions.

        Returns:
            torch.Tensor: Padded tensor of shape ``(C, target_H, target_W)``.
        """
        target_h, target_w = target_hw
        _, h, w = img.shape

        # Add symmetric padding to left/right & top/bottom
        pad_lr = (target_w - w) // 2
        pad_ud = (target_h - h) // 2
        img = torch.nn.functional.pad(img, (pad_lr, pad_lr, pad_ud, pad_ud), value=0.0)

        if img.shape[1] != target_h and img.shape[2] != target_w:
            # If target size is odd and img is even (or vice versa), add one more pixel of padding to the right/bottom
            pad_right = target_w - img.shape[2]
            pad_bottom = target_h - img.shape[1]
            img = torch.nn.functional.pad(img, (0, pad_right, 0, pad_bottom), value=0.0)

        return img


class GradNormLogger(Callback):
    """
    Log the total gradient L2 norm to TensorBoard periodically.

    Computes ``sqrt(sum(p.grad.norm() ** 2))`` across all model parameters
    after the backward pass and logs it as the ``"grad_norm"`` scalar.

    Args:
        log_every_n_steps (int): Compute and log every *n* training steps.
            Defaults to ``100``.
    """

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_after_backward(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ):
        """
        Compute and log gradient norm if on the right step cadence.
        
        Args:
            trainer (lightning.Trainer): The trainer instance.
            pl_module (lightning.LightningModule): The model being trained.
        """
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        total_norm_sq = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.norm(2).item() ** 2
        total_norm = math.sqrt(total_norm_sq)

        pl_module.log('grad_norm', total_norm, on_step=True, on_epoch=False)


class WeightHistogramLogger(Callback):
    """
    Log the weights of the model as histograms to TensorBoard periodically,
    grouped by parameter prefixes (e.g., model.feat, model.mapping, model.recon).

    Args:
        log_every_n_steps (int): Log histograms every *n* training steps.
            Defaults to ``100``.
    """

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule, outputs: Any, batch: Any, batch_idx: int
    ):
        """
        Log grouped weights as histograms if on the right step cadence.

        Args:
            trainer (lightning.Trainer): The trainer instance.
            pl_module (lightning.LightningModule): The model being trained.
            outputs (Any): The outputs of the model.
            batch (Any): The current batch.
            batch_idx (int): The index of the batch.
        """
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        tb_logger = next(
            (l for l in trainer.loggers if isinstance(l, lightning.pytorch.loggers.TensorBoardLogger)),
            None,
        )
        if tb_logger is None:
            return

        experiment = tb_logger.experiment

        for name, param in pl_module.named_parameters():
            if param.requires_grad and name.startswith("model."):
                parts = name.split('.', 2)
                tb_name = parts[0] + "." + "/".join(parts[1:])
                experiment.add_histogram(tb_name, param, global_step=trainer.global_step)


class SRCheckpoint(ModelCheckpoint):
    """
    Model checkpoint that monitors a super-resolution quality metric.

    A thin convenience wrapper around
    :class:`~lightning.pytorch.callbacks.ModelCheckpoint` that
    automatically sets ``mode='max'`` (both PSNR and SSIM are
    higher-is-better) and builds a descriptive filename pattern.

    Args:
        monitor_metric (str): The validation metric to monitor.
            Defaults to ``"val_psnr(RGB)"``.  Use any ``val_psnr({key})``
            logged by the lightning module (e.g. ``"val_psnr(Y)"``,
            ``"val_psnr(YCbCr)"``) or ``"val_ssim"``.
        save_top_k (int): Number of best checkpoints to keep.
            Defaults to ``3``.
        dirpath (Optional[str]): Directory to save checkpoints.
        filename_prefix (str): Prefix for checkpoint filenames.
            Defaults to ``"srcnn"``.
        **kwargs: Extra keyword arguments forwarded to
            :class:`~lightning.pytorch.callbacks.ModelCheckpoint`.
    """

    def __init__(
        self,
        monitor_metric: str = 'val_psnr(RGB)',
        save_top_k: int = 3,
        dirpath: Optional[str] = None,
        filename_prefix: str = 'srcnn',
        mode: str = 'max',
        **kwargs: Any,
    ):
        filename = f"{filename_prefix}-{{step}}-{{{monitor_metric}:.4f}}"
        super().__init__(
            monitor=monitor_metric,
            mode=mode,
            save_top_k=save_top_k,
            dirpath=dirpath,
            filename=filename,
            **kwargs,
        )
