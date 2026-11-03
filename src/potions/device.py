"""Device and numerical-precision policy for the batched model.

Known policy (user-confirmed): **float32 on CUDA, float64 on CPU** by default.
Both are overridable by mutating the module-level ``DEFAULT_DTYPE`` /
``DEFAULT_DEVICE`` constants or by passing an explicit ``dtype``/``device``
argument to any top-level entry point (functions default to these constants).

To avoid noisy ``UserWarning``s from other installed GPUs (e.g. a TITAN V with
compute capability 7.0 that this torch build has no kernels for), only device 0
is exposed when ``CUDA_VISIBLE_DEVICES`` is unset; an explicitly-set variable
is respected.

The package evaluates every kernel with **eager** torch ops: there is no
``torch.compile`` anywhere (no tracing, no graph caching, no fusion dispatch).
Convergence bars, tolerances and control flow are plain Python, and every
residual / Jacobian / solver kernel runs directly through torch's eager
dispatchers on whichever device/dtype the policy (or an explicit argument)
selects.

Functions here are pure helpers; nothing stateful lives on this module besides
the two policy constants.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any, TypeAlias

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

DEFAULT_DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DTYPE: torch.dtype = torch.float32 if DEFAULT_DEVICE.type == "cuda" else torch.float64

Tensor: TypeAlias = torch.Tensor


def get_device() -> torch.device:
    """Return the currently configured default device (CUDA if available else CPU)."""
    return DEFAULT_DEVICE


def get_dtype() -> torch.dtype:
    """Return the currently configured default tensor dtype.

    Prefer the explicit policy via :func:`default_dtype` over raw access to
    ``DEFAULT_DTYPE`` so callers stay consistent with the device policy.
    """
    return DEFAULT_DTYPE


def default_dtype(device: torch.device | None = None) -> torch.dtype:
    """Choose a dtype given ``device`` (CUDA -> float32, otherwise float64).

    When ``device`` is ``None`` the current :data:`DEFAULT_DEVICE` is used, and
    :data:`DEFAULT_DTYPE` reflects an explicit override if one was set.
    """
    if device is None:
        return DEFAULT_DTYPE
    return torch.float32 if device.type == "cuda" else torch.float64


def as_typed(
    value: Any,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convert ``value`` to a torch tensor on ``device``/``dtype``.

    Both arguments default to the module-level policy when omitted.
    """
    return torch.as_tensor(
        value,
        device=DEFAULT_DEVICE if device is None else device,
        dtype=DEFAULT_DTYPE if dtype is None else dtype,
    )


def chunk_ranges(total: int, chunk: int) -> Iterator[tuple[int, int]]:
    """Yield inclusive-exclusive ``(start, stop)`` slices of ``total`` of size ≤ ``chunk``.

    Used to stream very large batches (e.g. 1,000,000 parameter sets) through the
    GPU without exceeding VRAM.
    """
    for start in range(0, total, chunk):
        yield start, min(start + chunk, total)


def peak_tensor_bytes(shape: tuple[int, ...], dtype: torch.dtype = DEFAULT_DTYPE) -> int:
    """Estimated contiguous byte footprint of a tensor with ``shape`` and ``dtype``."""
    return int(torch.empty(1, dtype=dtype).element_size()) * int(
        torch.Size(shape).numel()
    )
