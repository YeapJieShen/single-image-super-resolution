"""Abstract base class for single-image super-resolution architectures."""

import abc
import dataclasses
from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import torch
import torch.nn as nn


def as_int_tuple(name: str, value: Sequence[int]) -> tuple[int, ...]:
    """Normalise a per-layer sequence hyperparameter to a tuple.

    Provenance metadata records hparams tuple-free (JSON has no tuple), so a
    list must rebuild the same model a tuple built. Element values are left to
    the caller's own checks.

    Args:
        name: The parameter's name, for the error message.
        value: A list or tuple.

    Returns:
        ``value`` as a tuple.

    Raises:
        ValueError: If ``value`` is a ``str`` (a sequence, but of characters)
            or is not a sequence at all.
    """
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a list or tuple. Got {type(value)}.")
    return tuple(value)


@dataclasses.dataclass(frozen=True)
class SRModelOutput:
    """A model's output when it has more to expose than one SR tensor.

    Recursive-supervision architectures (e.g. DRCN) need named
    intermediates downstream (a loss that opts in) while every existing
    consumer -- metrics, predict, export -- must keep seeing exactly what it
    sees today. A model returning this instead of a bare tensor is the one
    place that fork happens; nothing downstream re-derives it.

    Args:
        primary: The model's SR output -- exactly what ``forward`` would have
            returned as a bare tensor. Every consumer that has not opted in
            to the full record reads this field, via :func:`unwrap_primary`.
        extras: Named intermediate tensors (e.g. per-recursion predictions).
            Only a loss that opts in (``wants_model_output``, see
            :class:`~sisr.losses.base.SRLoss`) ever sees this.
    """

    primary: torch.Tensor
    extras: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)


def unwrap_primary(output: torch.Tensor | SRModelOutput) -> torch.Tensor:
    """Return a model's primary SR tensor, whichever return shape it used.

    Every consumer that does not opt in to a model's extras -- metric
    scoring, ``predict``/``predict_rgb``, ONNX export -- calls this instead
    of assuming ``output`` is already a tensor, so ``SRModel.forward``
    returning :class:`SRModelOutput` needs no per-consumer special-casing.

    Args:
        output: What ``SRModel.forward`` returned: a bare tensor (today's
            contract, unchanged) or an :class:`SRModelOutput`.

    Returns:
        ``output`` itself if it is already a tensor, else ``output.primary``.
    """
    return output.primary if isinstance(output, SRModelOutput) else output


class SRModel(nn.Module, abc.ABC):
    """Abstract base for SR architectures.

    Subclasses must populate ``self._hparams`` in ``__init__``, implement
    ``forward``, and declare ``input_contract``. ``reset_parameters`` is a
    no-op by default; override it for paper-faithful weight init schemes.
    """

    _hparams: dict

    #: How the model expects its LR input: ``'pre_upsampled'`` if LR arrives
    #: already on the HR grid (SRCNN), ``'native_lr'`` if the model upsamples
    #: internally (SRResNet). Declared, never inferred -- a rule like
    #: ``'scale' in hparams`` holds for both current architectures and would
    #: silently break on a third.
    input_contract: ClassVar[Literal["pre_upsampled", "native_lr"]]

    @property
    def hparams(self) -> dict:
        """Architecture hyperparameters dict for the Lightning HParams merge."""
        return self._hparams

    @property
    @abc.abstractmethod
    def variant_tag(self) -> str:
        """Short token distinguishing this configuration from siblings of the same class.

        Appears in artifact filenames, so a directory of weights reads without
        opening anything: ``SRCNN_x2_Y_915``, ``SRResNet_x4_RGB_16B64F``.

        Abstract, never defaulted, for the same reason ``input_contract`` is:
        no rule over ``hparams`` yields a readable tag for every architecture,
        and an inherited default would label two configurations identically.
        Keep it short, stable and filename-safe.
        """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor | SRModelOutput:
        """Run the model on input ``x`` and return the SR output.

        Returns a bare tensor (the historical, still-default contract) or,
        for an architecture with named intermediates a downstream loss might
        want (e.g. DRCN's recursive supervision), an
        :class:`SRModelOutput` wrapping ``primary`` plus those ``extras``.
        Every consumer that has not opted in reads only ``primary`` -- via
        :func:`unwrap_primary` -- so SRCNN/SRResNet are unaffected either way.
        """

    def reset_parameters(self, **kwargs: Any) -> None:
        """Optional paper-style weight init. Default: no-op.

        ``**kwargs`` lets ``SRLightning`` pass paper-init options
        polymorphically; subclasses declare what they read (``SRCNN``'s
        ``mean``/``std``), and models that do not override absorb them.
        """
        pass
