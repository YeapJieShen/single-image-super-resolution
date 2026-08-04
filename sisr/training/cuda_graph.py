"""CUDA-graph capture of the training step — :class:`CUDAGraphStep`.

An SRCNN training step is kernel-launch-bound, not compute-bound: measured 78
``cudaLaunchKernel`` calls per step, with the CPU's launch time equal to the
step's wall time (3.8 ms of each) and the GPU at 27% utilisation. Replaying one
captured graph runs the identical kernels as a single ``cudaGraphLaunch`` at 99%
utilisation, cutting device-side time 1.93 -> 0.90 ms and the whole Lightning
step 6.21 -> 2.21 ms (b64, 33x33, 60 W cap, RTX 5060 Laptop).

Only ``{zero_grad, forward, loss, backward}`` is captured — ``optimizer.step()``
stays eager and stays Lightning's. Capturing it too saves a further 0.07 ms/step
(measured) but would bake the learning rate in as a graph constant, silently
no-opping every LR scheduler: ``torch.optim.SGD`` has no ``capturable``
parameter to opt out of that. 0.07 ms is not worth a silent-corruption class.
"""

from collections.abc import Callable, Sequence

import torch


class CUDAGraphStep:
    """One captured ``{zero_grad, forward, loss, backward}`` for a fixed batch shape.

    Capture is lazy — the first :meth:`run` call warms up and captures using
    that batch's shapes, so the caller never has to predict them. Later batches
    whose shapes differ (the partial last batch of an epoch) are rejected with
    ``None`` so the caller can fall back to an eager step.

    The warm-up deliberately omits ``optimizer.step()``: nothing in the captured
    region writes to a parameter, so capture leaves the model bit-identical to
    how it arrived and no weight/momentum snapshot-and-rewind is needed. It does
    run ``backward``, which is what allocates the ``.grad`` tensors the graph
    then bakes addresses for, and what lets ``cudnn.benchmark`` finish
    autotuning this shape before anything is recorded.

    Args:
        loss_fn: Maps a batch to the scalar loss to backpropagate. Called with
            the internal static buffers, never the caller's batch.
        optimizer: Optimizer whose ``zero_grad`` is captured. Its ``step`` is
            never called here.
        warmup_iters: Warm-up iterations run on a side stream before capture.
            Three is the CUDA-graph documented minimum for the caching
            allocator to settle.
    """

    def __init__(
        self,
        loss_fn: Callable[[Sequence[torch.Tensor]], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        warmup_iters: int = 3,
    ):
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._warmup_iters = warmup_iters
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_batch: tuple[torch.Tensor, ...] | None = None
        self._static_loss: torch.Tensor | None = None

    @property
    def captured(self) -> bool:
        """Whether a graph has been captured yet."""
        return self._graph is not None

    def capture(self, batch: Sequence[torch.Tensor]) -> None:
        """Warm up on a side stream, then capture the step for ``batch``'s shapes.

        Args:
            batch: Sequence of device tensors. Cloned into the static buffers
                every later :meth:`run` copies into.
        """
        self._static_batch = tuple(t.detach().clone() for t in batch)

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
        if len(batch) != len(self._static_batch):
            return None
        if any(t.shape != s.shape for t, s in zip(batch, self._static_batch, strict=True)):
            return None
        for static, incoming in zip(self._static_batch, batch, strict=True):
            static.copy_(incoming, non_blocking=True)
        self._graph.replay()
        return self._static_loss.clone()
