"""Tests for the batched root solvers in :mod:`potions.solvers`.

These are the numerical core of the reactive-transport step, so the suite checks
the contract that matters for the Rust-parity guarantee:

* the analytic Jacobian path reproduces the true root of a small linear system,
* both the fused and the split (``f``/``jf``) entry points agree,
* every batch lane is solved independently (no cross-lane contamination),
* failed lanes are surfaced either as NaN or as a
  ``BatchedRootFindingError`` (a ``RuntimeError`` subclass),
* the dtype-aware tolerance helper keeps the float64 constants intact
  and clamps reduced-precision (f32) bars above the ulp floor with a
  two-term ``atol + rtol*max|x|`` convergence bar.
"""

from __future__ import annotations

import torch
import pytest

from potions import solvers


def _linear_system(
    dim: int, batch: int, dtype: torch.dtype = torch.float64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(J, b, x_true)`` for a well-conditioned SPD system ``J x = b``.

    ``J`` is built as ``A^T A + I`` so it is symmetric positive definite and the
    Newton step is guaranteed to be a descent direction for the residual norm.
    """
    gen = torch.Generator().manual_seed(0)
    a: torch.Tensor = torch.randn(batch, dim, dim, dtype=dtype, generator=gen)
    jac: torch.Tensor = a.transpose(-1, -2) @ a + torch.eye(dim, dtype=dtype)
    x_true: torch.Tensor = torch.randn(batch, dim, dtype=dtype, generator=gen)
    b: torch.Tensor = (jac @ x_true.unsqueeze(-1)).squeeze(-1)
    return jac, b, x_true


class TestSolverTols:
    def test_float64_keeps_reference_constants(self) -> None:
        atol, rtol, tol_step, stagn = solvers._solver_tols(torch.float64)
        assert atol == solvers.MULTI_TOL
        assert rtol == 0.0
        assert tol_step == solvers.MULTI_TOL_STEP
        assert stagn == solvers.STAGNATION_REL_TOL

    def test_float32_clamps_above_ulp(self) -> None:
        eps: float = float(torch.finfo(torch.float32).eps)
        atol, rtol, tol_step, stagn = solvers._solver_tols(torch.float32)
        assert atol >= solvers.MULTI_TOL
        assert atol >= 64.0 * eps
        assert rtol == solvers.MULTI_RTOL_F32
        assert tol_step >= eps
        assert stagn >= eps


class TestSolvesLu:
    def test_matches_torch_linalg_solve(self) -> None:
        a: torch.Tensor = torch.tensor(
            [[[3.0, 1.0], [1.0, 2.0]]], dtype=torch.float64
        )
        b: torch.Tensor = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        x: torch.Tensor = solvers._solves_lu(a, b)
        expected: torch.Tensor = torch.linalg.solve(a, b.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(x, expected)


class TestNewtonAnalyticThenLm:
    @pytest.mark.parametrize("dim", [1, 2, 5])
    @pytest.mark.parametrize("batch", [1, 4])
    def test_solves_linear_system(self, dim: int, batch: int) -> None:
        jac, b, x_true = _linear_system(dim, batch)

        def f(x: torch.Tensor) -> torch.Tensor:
            return (jac @ x.unsqueeze(-1)).squeeze(-1) - b

        def jf(x: torch.Tensor) -> torch.Tensor:
            return jac.clone()

        x_0: torch.Tensor = torch.zeros(batch, dim, dtype=torch.float64)
        root: torch.Tensor = solvers.newton_analytic_then_lm(f, jf, x_0, raise_on_failure=True)
        torch.testing.assert_close(root, x_true, atol=1e-8, rtol=1e-8)

    def test_accepts_1d_batch(self) -> None:
        jac, b, x_true = _linear_system(3, 1)
        f = lambda x: (jac @ x.unsqueeze(-1)).squeeze(-1) - b
        jf = lambda x: jac.clone()
        root = solvers.newton_analytic_then_lm(
            f, jf, torch.zeros(1, 3, dtype=torch.float64), raise_on_failure=True
        )
        assert root.shape == (1, 3)
        torch.testing.assert_close(root, x_true, atol=1e-8, rtol=1e-8)


class TestNewtonFusedThenLm:
    def test_matches_split_entry_point(self) -> None:
        jac, b, x_true = _linear_system(4, 3)

        def f(x: torch.Tensor) -> torch.Tensor:
            return (jac @ x.unsqueeze(-1)).squeeze(-1) - b

        def jf(x: torch.Tensor) -> torch.Tensor:
            return jac.clone()

        x_0: torch.Tensor = torch.zeros(3, 4, dtype=torch.float64)
        split: torch.Tensor = solvers.newton_analytic_then_lm(f, jf, x_0)
        fused: torch.Tensor = solvers.newton_fused_then_lm(
            lambda x: (f(x), jf(x)), x_0
        )
        torch.testing.assert_close(fused, split)
        torch.testing.assert_close(fused, x_true, atol=1e-8, rtol=1e-8)

    def test_batched_lanes_are_independent(self) -> None:
        jac, b, x_true = _linear_system(2, 6)
        fused = lambda x: (
            (jac @ x.unsqueeze(-1)).squeeze(-1) - b,
            jac.clone(),
        )
        root: torch.Tensor = solvers.newton_fused_then_lm(
            fused, torch.zeros(6, 2, dtype=torch.float64), raise_on_failure=True
        )
        torch.testing.assert_close(root, x_true, atol=1e-8, rtol=1e-8)

    def test_zero_dimensional_system_is_done_immediately(self) -> None:
        x_0: torch.Tensor = torch.zeros(3, 0, dtype=torch.float64)
        calls: list[int] = []

        def eval_fn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            calls.append(1)
            return torch.zeros(3, 0, dtype=torch.float64), torch.zeros(
                3, 0, 0, dtype=torch.float64
            )

        root: torch.Tensor = solvers.newton_fused_then_lm(eval_fn, x_0)
        assert root.shape == (3, 0)
        assert calls == [1]

    def test_failure_returns_nan_without_raise(self) -> None:
        def eval_fn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            residual: torch.Tensor = torch.ones_like(x[:, :1]) + 1.0
            jacobian: torch.Tensor = torch.zeros(
                x.shape[0], 1, x.shape[1], dtype=x.dtype
            )
            return residual, jacobian

        root: torch.Tensor = solvers.newton_fused_then_lm(
            eval_fn, torch.zeros(1, 1, dtype=torch.float64)
        )
        assert torch.isnan(root).all()

    def test_failure_raises_when_requested(self) -> None:
        def eval_fn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            residual: torch.Tensor = torch.ones_like(x[:, :1]) + 1.0
            jacobian: torch.Tensor = torch.zeros(
                x.shape[0], 1, x.shape[1], dtype=x.dtype
            )
            return residual, jacobian

        with pytest.raises(solvers.BatchedRootFindingError, match="failed"):
            solvers.newton_fused_then_lm(
                eval_fn,
                torch.zeros(1, 1, dtype=torch.float64),
                raise_on_failure=True,
            )


class TestArmijoSufficientDecrease:
    def test_decrease_accepted(self) -> None:
        f_x: torch.Tensor = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        f_new: torch.Tensor = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        accepted: torch.Tensor = solvers._satisfies_sufficient_decrease(
            f_new, f_x, torch.tensor([1.0], dtype=torch.float64)
        )
        assert bool(accepted.all())

    def test_increase_rejected(self) -> None:
        f_x: torch.Tensor = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        f_new: torch.Tensor = torch.tensor([[2.0, 2.0]], dtype=torch.float64)
        accepted: torch.Tensor = solvers._satisfies_sufficient_decrease(
            f_new, f_x, torch.tensor([1.0], dtype=torch.float64)
        )
        assert not bool(accepted.any())

    def test_nan_rejected(self) -> None:
        f_x: torch.Tensor = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        f_new: torch.Tensor = torch.tensor([[float("nan"), 0.0]], dtype=torch.float64)
        accepted: torch.Tensor = solvers._satisfies_sufficient_decrease(
            f_new, f_x, torch.tensor([1.0], dtype=torch.float64)
        )
        assert not bool(accepted.any())


class TestMaxAbs:
    def test_componentwise_max(self) -> None:
        v: torch.Tensor = torch.tensor([[1.0, -3.0], [0.5, 0.25]], dtype=torch.float64)
        torch.testing.assert_close(
            solvers._max_abs(v), torch.tensor([3.0, 0.5], dtype=torch.float64)
        )


class TestNewtonOutcome:
    def test_fields_present(self) -> None:
        outcome = solvers.NewtonOutcome(
            result=torch.zeros(1, 1),
            best_x=torch.zeros(1, 1),
            best_err=torch.zeros(1),
            status=torch.zeros(1, dtype=torch.long),
            err=torch.zeros(1),
        )
        assert outcome.result.shape == (1, 1)
        assert int(outcome.status[0]) == solvers._RUNNING
