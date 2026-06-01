"""Per-architecture network implementations.

Each subpackage (``srcnn``, ``srresnet``) exports its top-level
``nn.Module`` and any per-architecture config dataclasses for use under
``model.model.class_path`` and ``model.training_config.class_path`` in
experiment YAMLs.
"""
