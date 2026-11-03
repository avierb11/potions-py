"""Tests for :mod:`potions.device` (device/dtype policy + tensor helpers)."""

from __future__ import annotations

import torch

from potions import device


class TestDtypePolicy:
    def test_default_dtype_follows_device(self) -> None:
        assert device.default_dtype(torch.device("cpu")) is torch.float64
        assert device.default_dtype(torch.device("cuda")) is torch.float32

    def test_default_dtype_none_matches_module_constant(self) -> None:
        assert device.default_dtype() is device.DEFAULT_DTYPE

    def test_get_device_and_dtype_are_consistent(self) -> None:
        assert device.get_device() is device.DEFAULT_DEVICE
        assert device.get_dtype() is device.DEFAULT_DTYPE

    def test_module_dtype_matches_device(self) -> None:
        if device.DEFAULT_DEVICE.type == "cuda":
            assert device.DEFAULT_DTYPE is torch.float32
        else:
            assert device.DEFAULT_DTYPE is torch.float64


class TestAsTyped:
    def test_explicit_device_and_dtype(self) -> None:
        out: torch.Tensor = device.as_typed(
            [1, 2, 3], device=torch.device("cpu"), dtype=torch.float32
        )
        assert out.dtype is torch.float32
        assert out.device.type == "cpu"
        assert out.tolist() == [1.0, 2.0, 3.0]

    def test_defaults_use_policy(self) -> None:
        out: torch.Tensor = device.as_typed([1, 2])
        assert out.dtype is device.DEFAULT_DTYPE
        assert out.device.type == device.DEFAULT_DEVICE.type


class TestChunkRanges:
    def test_exact_division(self) -> None:
        assert list(device.chunk_ranges(6, 3)) == [(0, 3), (3, 6)]

    def test_remainder(self) -> None:
        assert list(device.chunk_ranges(7, 3)) == [(0, 3), (3, 6), (6, 7)]

    def test_empty(self) -> None:
        assert list(device.chunk_ranges(0, 4)) == []

    def test_chunk_larger_than_total(self) -> None:
        assert list(device.chunk_ranges(2, 10)) == [(0, 2)]

    def test_partitions_exactly(self) -> None:
        total: int = 100
        covered: list[int] = [
            i for lo, hi in device.chunk_ranges(total, 7) for i in range(lo, hi)
        ]
        assert covered == list(range(total))


class TestPeakTensorBytes:
    def test_float64_footprint(self) -> None:
        assert device.peak_tensor_bytes((2, 3), torch.float64) == 2 * 3 * 8

    def test_float32_footprint(self) -> None:
        assert device.peak_tensor_bytes((4,), torch.float32) == 4 * 4

    def test_matches_tensor_numel(self) -> None:
        t: torch.Tensor = torch.empty(5, 7, dtype=torch.float64)
        assert device.peak_tensor_bytes((5, 7), torch.float64) == t.numel() * 8
