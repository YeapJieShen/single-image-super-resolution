"""CUDA-graph capture of the training step — :class:`CUDAGraphStep`.

An SRCNN training step is kernel-launch-bound, not compute-bound: measured 78
``cudaLaunchKernel`` calls per step, with the CPU's launch time equal to the
step's wall time and the GPU at under 30% utilisation. Replaying one captured
graph runs the identical kernels as a single ``cudaGraphLaunch`` at near-full
utilisation, cutting device-side time roughly in half and the whole Lightning
step by ~2.6x — see the measured figures in ``templates/config.srcnn.template.yaml``,
which is the one place they are quoted and kept current.

Only ``{zero_grad, forward, loss, backward}`` is captured — ``optimizer.step()``
stays eager and stays Lightning's. Capturing it too saves a further 0.07 ms/step
(measured) but would bake the learning rate in as a graph constant, silently
no-opping every LR scheduler: ``torch.optim.SGD`` has no ``capturable``
parameter to opt out of that. 0.07 ms is not worth a silent-corruption class.
"""

from collections.abc import Callable, Sequence

import torch
from lightning.pytorch.utilities import rank_zero_info, rank_zero_warn


class CUDAGraphStep:
    """One captured ``{zero_grad, forward, loss, backward}`` for a fixed batch shape.

    Capture is lazy — the first :meth:`run` call warms up and captures using
    that batch's shapes, so the caller never has to predict them. Later batches
    whose shapes differ (the partial last batch of an epoch) are rejected with
    ``None`` so the caller can fall back to an eager step.

    The warm-up deliberately omits ``optimizer.step()``, so no parameter and no
    optimizer state is written and none has to be snapshotted. What the warm-up
    *does* mutate is buffers — ``BatchNorm``'s ``running_mean`` /
    ``running_var`` / ``num_batches_tracked`` advance once per forward, and
    SRResNet's residual blocks have BatchNorm — so every buffer is snapshotted
    and copied back in place once capture finishes. Together those two facts
    make capture observationally inert: the module is left exactly as it
    arrived. The warm-up still runs ``backward``, which is what allocates the
    ``.grad`` tensors the graph bakes addresses for and what lets
    ``cudnn.benchmark`` finish autotuning this shape before anything is recorded.

    Args:
        loss_fn: Maps a batch to the scalar loss to backpropagate. Called with
            the internal static buffers, never the caller's batch.
        module: The module whose buffers the warm-up would otherwise advance.
            Only ``named_buffers`` is used; parameters are untouched.
        optimizer: Optimizer whose ``zero_grad`` is captured. Its ``step`` is
            never called here.
        warmup_iters: Warm-up iterations run on a side stream before capture.
            Three is the CUDA-graph documented minimum for the caching
            allocator to settle.
        fallback_warn_after: Warn once after this many consecutive shape
            rejections. A run that never matches would otherwise silently pay
            eager cost with the flag on and look like a 0% speedup.
    """

    def __init__(
        self,
        loss_fn: Callable[[Sequence[torch.Tensor]], torch.Tensor],
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int = 3,
        fallback_warn_after: int = 10,
    ):
        self._loss_fn = loss_fn
        self._module = module
        self._optimizer = optimizer
        self._warmup_iters = warmup_iters
        self._fallback_warn_after = fallback_warn_after
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_batch: tuple[torch.Tensor, ...] | None = None
        self._static_loss: torch.Tensor | None = None
        self._consecutive_fallbacks = 0
        self._fallback_warned = False

    @property
    def captured(self) -> bool:
        """Whether a graph has been captured yet."""
        return self._graph is not None

    def _buffer_snapshot(self) -> dict[str, torch.Tensor]:
        """Clone every module buffer, so the warm-up's forwards can be undone.

        Returns:
            Buffer name -> detached clone. Empty for buffer-free architectures
            like SRCNN, making the restore a no-op there.
        """
        return {name: buf.detach().clone() for name, buf in self._module.named_buffers()}

    def _restore_buffers(self, snapshot: dict[str, torch.Tensor]) -> None:
        """Copy snapshotted buffer values back, in place.

        In place is required, not incidental: the captured graph holds the
        buffers' addresses, so rebinding them would invalidate every replay.

        Args:
            snapshot: Output of :meth:`_buffer_snapshot`.
        """
        with torch.no_grad():
            for name, buf in self._module.named_buffers():
                buf.copy_(snapshot[name])

    def capture(self, batch: Sequence[torch.Tensor]) -> None:
        """Warm up on a side stream, then capture the step for ``batch``'s shapes.

        Args:
            batch: Sequence of device tensors. Cloned into the static buffers
                every later :meth:`run` copies into.
        """
        self._static_batch = tuple(t.detach().clone() for t in batch)
        buffers = self._buffer_snapshot()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self._warmup_iters):
                self._optimizer.zero_grad(set_to_none=False)
                # Not kept: a live reference to a previous iteration's autograd
                # graph keeps its AccumulateGrad nodes alive and breaks capture.
                self._loss_fn(self._static_batch).backward()
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._optimizer.zero_grad(set_to_none=False)
            loss = self._loss_fn(self._static_batch)
            loss.backward()
        # Detached: keeping the graph-built loss itself alive keeps its whole
        # autograd graph alive, including AccumulateGrad nodes bound to the
        # capture stream. A later eager backward on the same parameters then
        # hits them on the default stream and warns about the mismatch. Replay
        # needs only the value buffer, which detach shares.
        self._graph, self._static_loss = graph, loss.detach()
        self._restore_buffers(buffers)
        rank_zero_info(
            f"CUDA graph captured for batch shapes "
            f"{[tuple(t.shape) for t in self._static_batch]} — the training step now "
            f"replays as one graph launch."
        )

    def run(self, batch: Sequence[torch.Tensor]) -> torch.Tensor | None:
        """Replay the captured step for ``batch``, capturing on the first call.

        Args:
            batch: Sequence of device tensors, same arity and shapes as the
                batch that was captured.

        Returns:
            A clone of the loss — the static buffer itself is overwritten in
            place by the next replay, so anything retaining it (Lightning's
            progress-bar metric cache) would silently report a later step's
            value. ``None`` if ``batch`` doesn't match the captured shapes,
            meaning the caller must run an eager step instead.
        """
        if self._graph is None:
            self.capture(batch)
        if not self._matches(batch):
            self._note_fallback(batch)
            return None
        self._consecutive_fallbacks = 0
        for static, incoming in zip(self._static_batch, batch, strict=True):
            static.copy_(incoming, non_blocking=True)
        self._graph.replay()
        return self._static_loss.clone()

    def _matches(self, batch: Sequence[torch.Tensor]) -> bool:
        """Whether ``batch``'s arity and shapes are the captured ones."""
        if len(batch) != len(self._static_batch):
            return False
        return all(t.shape == s.shape for t, s in zip(batch, self._static_batch, strict=True))

    def _note_fallback(self, batch: Sequence[torch.Tensor]) -> None:
        """Count a rejected batch, warning once if they stop being occasional.

        One rejection per epoch is normal and expected — that is the partial
        last batch. A *run* of them means the captured shape never recurs, so
        the flag is on and buying nothing, which is otherwise invisible.

        Args:
            batch: The rejected batch, for the warning's shape report.
        """
        self._consecutive_fallbacks += 1
        if self._fallback_warned or self._consecutive_fallbacks < self._fallback_warn_after:
            return
        self._fallback_warned = True
        rank_zero_warn(
            f"training_config.cuda_graph is on, but the last {self._consecutive_fallbacks} "
            f"batches all fell back to an eager step: shapes "
            f"{[tuple(t.shape) for t in batch]} do not match the captured "
            f"{[tuple(t.shape) for t in self._static_batch]}. The graph is buying nothing. "
            f"Give the train loader a constant batch shape (e.g. drop_last=True) or set "
            f"cuda_graph to false."
        )
