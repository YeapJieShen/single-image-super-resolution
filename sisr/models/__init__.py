"""Per-architecture network implementations.

Each subpackage exports its top-level ``nn.Module`` and any per-architecture config
dataclasses for use under ``model.model.class_path`` and ``model.training_config.class_path``
in experiment YAMLs. In architecture order: ``srcnn``, ``espcn``, ``vdsr``, ``drcn``
(pre-upsampled or same-resolution Y-channel nets), ``srresnet``, ``srgan`` (the SRGAN
discriminator and configs; its generator is SRResNet), ``esrgan``, ``edsr``, ``mdsr``.
"""
