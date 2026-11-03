"""Tests for :mod:`potions.equilibrium`.

The equilibrium solve is exercised on a hand-built one-reaction network
``B = K * A`` together with the mass-conservation row ``A + B = c_tot``.  That
system has a closed-form solution, so the tests can check the analytic Jacobian
against a finite difference, verify the conservation residual at the root, and
confirm the ``num_free == 0`` pass-through.
"""

from __future__ import annotations

import numpy as np
import torch
import pytest

from potions.equilibrium import (
    EquilibriumParameters,
    compute_stoich_null_space,
)

LOG_K: float = float(np.log10(2.0))


def simple_system() -> EquilibriumParameters:
    """One secondary reaction ``B - A = 0`` (so ``B = 2A``) + total ``A``."""
    stoich: np.ndarray = np.array([[-1.0, 1.0]], dtype=np.float64)
    log_eq: np.ndarray = np.array([LOG_K], dtype=np.float64)
    null, x_particular = compute_stoich_null_space(stoich, log_eq)
    return EquilibriumParameters(
        stoich=torch.tensor(stoich, dtype=torch.float64),
        log_eq_consts=torch.tensor(log_eq, dtype=torch.float64),
        total=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        stoich_null_space=torch.tensor(null, dtype=torch.float64),
        x_particular=torch.tensor(x_particular, dtype=torch.float64),
        total_names=("A",),
    )


class TestComputeStoichNullSpace:
    def test_null_space_is_orthogonal_to_stoich(self) -> None:
        stoich: np.ndarray = np.array([[-1.0, 1.0]], dtype=np.float64)
        null, _ = compute_stoich_null_space(stoich, np.array([LOG_K]))
        assert null.shape == (2, 1)
        np.testing.assert_allclose(stoich @ null, np.zeros((1, 1)), atol=1e-12)

    def test_particular_solves_reaction_equations(self) -> None:
        stoich: np.ndarray = np.array([[-1.0, 1.0]], dtype=np.float64)
        log_eq: np.ndarray = np.array([LOG_K])
        _, x_particular = compute_stoich_null_space(stoich, log_eq)
        np.testing.assert_allclose(stoich @ x_particular, log_eq, atol=1e-12)


class TestEquilibriumParameters:
    def test_sizes(self) -> None:
        eq: EquilibriumParameters = simple_system()
        assert eq.num_species == 2
        assert eq.num_free == 1
        assert eq.num_reactions == 1
        assert eq.charge_row is None

    def test_conc_func_enforces_equilibrium_ratio(self) -> None:
        eq: EquilibriumParameters = simple_system()
        x_free: torch.Tensor = torch.tensor([[0.0], [1.0], [-2.0]], dtype=torch.float64)
        conc: torch.Tensor = eq.conc_func(x_free)
        ratio: torch.Tensor = conc[:, 1] / conc[:, 0]
        torch.testing.assert_close(
            ratio, torch.full((3,), 2.0, dtype=torch.float64), atol=1e-10, rtol=1e-10
        )

    def test_residual_is_zero_at_solution(self) -> None:
        eq: EquilibriumParameters = simple_system()
        x_free: torch.Tensor = torch.tensor([[0.3]], dtype=torch.float64)
        conc: torch.Tensor = eq.conc_func(x_free)
        c_tot: torch.Tensor = conc @ eq.total.T
        residual: torch.Tensor = eq.residual(x_free, c_tot)
        torch.testing.assert_close(
            residual, torch.zeros_like(residual), atol=1e-10, rtol=1e-10
        )

    def test_jacobian_matches_finite_difference(self) -> None:
        eq: EquilibriumParameters = simple_system()
        c_tot: torch.Tensor = torch.tensor([[3.0]], dtype=torch.float64)
        x_free: torch.Tensor = torch.tensor([[0.7]], dtype=torch.float64)
        analytic: torch.Tensor = eq.jacobian_residual(x_free)
        eps: float = 1e-6
        f_plus: torch.Tensor = eq.residual(x_free + eps, c_tot)
        f_minus: torch.Tensor = eq.residual(x_free - eps, c_tot)
        fd: torch.Tensor = ((f_plus - f_minus) / (2.0 * eps)).unsqueeze(-1)
        torch.testing.assert_close(analytic, fd, atol=1e-6, rtol=1e-6)

    def test_x_free_solve_conserves_total(self) -> None:
        eq: EquilibriumParameters = simple_system()
        chms: torch.Tensor = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        initial: torch.Tensor = torch.zeros(1, eq.num_free, dtype=torch.float64)
        x_free: torch.Tensor = eq.x_free_solve(chms, initial, raise_on_failure=True)
        conc: torch.Tensor = eq.conc_func(x_free)
        total: torch.Tensor = conc @ eq.total.T
        expected: torch.Tensor = chms @ eq.total.T
        torch.testing.assert_close(total, expected, atol=1e-8, rtol=1e-8)

    def test_x_free_solve_roundtrips_leading_shape(self) -> None:
        eq: EquilibriumParameters = simple_system()
        chms: torch.Tensor = torch.tensor(
            [[1.0, 1.0], [2.0, 0.5]], dtype=torch.float64
        )
        initial: torch.Tensor = torch.zeros(2, eq.num_free, dtype=torch.float64)
        x_free: torch.Tensor = eq.x_free_solve(chms, initial, raise_on_failure=True)
        assert x_free.shape == (2, eq.num_free)

    def test_solve_equilibrium_returns_concentrations(self) -> None:
        eq: EquilibriumParameters = simple_system()
        chms: torch.Tensor = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        out: torch.Tensor = eq.solve_equilibrium(chms, raise_on_failure=True)
        assert out.shape == (1, 2)
        assert float((out[0, 1] / out[0, 0])) == pytest.approx(2.0, rel=1e-6)
        assert float(out.sum()) == pytest.approx(2.0, rel=1e-6)

    def test_zero_free_variables_passes_through(self) -> None:
        # Full-rank stoich -> empty null space -> nothing to solve.
        stoich: np.ndarray = np.eye(2, dtype=np.float64)
        log_eq: np.ndarray = np.zeros(2, dtype=np.float64)
        eq: EquilibriumParameters = EquilibriumParameters(
            stoich=torch.tensor(stoich, dtype=torch.float64),
            log_eq_consts=torch.tensor(log_eq, dtype=torch.float64),
            total=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
            stoich_null_space=torch.tensor(
                compute_stoich_null_space(stoich, log_eq)[0], dtype=torch.float64
            ),
            x_particular=torch.tensor(
                compute_stoich_null_space(stoich, log_eq)[1], dtype=torch.float64
            ),
            total_names=("A",),
        )
        assert eq.num_free == 0
        chms: torch.Tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        assert eq.solve_equilibrium(chms) is chms

    def test_x_free_solve_bad_initial_shape_raises(self) -> None:
        eq: EquilibriumParameters = simple_system()
        chms: torch.Tensor = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        initial: torch.Tensor = torch.zeros(5, eq.num_free, dtype=torch.float64)
        with pytest.raises(ValueError):
            eq.x_free_solve(chms, initial)


class TestChargeRow:
    def test_charge_row_locates_h_plus(self) -> None:
        eq: EquilibriumParameters = simple_system()
        with_h: EquilibriumParameters = EquilibriumParameters(
            stoich=eq.stoich,
            log_eq_consts=eq.log_eq_consts,
            total=torch.zeros(2, 2, dtype=torch.float64),
            stoich_null_space=eq.stoich_null_space,
            x_particular=eq.x_particular,
            total_names=("A", "H+"),
        )
        assert with_h.charge_row == 1
