"""Equilibrium speciation structures (construction + batched speciation solve).

Mirrors the reference ``kinetic_structures.rs`` ``EquilibriumParameters`` type:
the equilibrium stoichiometry matrix ``stoich`` (one row per secondary species,
then one row per exchange reaction), the log10 equilibrium constants
``log_eq_consts``, and the mass/charge conservation matrix ``total`` (one row
per total species = primary + mineral + ``X-``).

Two derived objects are precomputed here exactly as the Rust constructor does
(which literally calls ``scipy``):

- ``stoich_null_space`` = ``scipy.linalg.null_space(stoich)`` — a
  ``(num_species, n_free)`` basis of the null space of the reaction
  stoichiometry matrix;
- ``x_particular``      = ``scipy.linalg.pinv(stoich) @ log_eq_consts``.

Given these, equilibrium concentrations satisfy ``10**(N @ x_free + x_particular)``
for ``x_free`` solving the conservation residual
``c_tot - total @ 10**(N x_free + x_particular) = 0`` (see the plan, section 6.3).
``x_free`` is found with the Rust-compatible batched Newton -> LM solver chain in
``solvers.py`` (Rust ``x_free_solve_rust`` / ``solve_equilibrium``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.linalg
import torch

import math

from .solvers import newton_analytic_then_lm

Tensor: TypeAlias = torch.Tensor

# Plain Python float (not ``numpy``) so the constant stays a plain number in
# the batched torch expressions (no numpy-scalar wrapping).
LNL10: float = math.log(10.0)


def _flatten_batch(t: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    """Reshape ``(..., d)`` tensors to ``(B, d)``, returning the leading shape."""
    leading: tuple[int, ...] = t.shape[:-1]
    return t.reshape(-1, t.shape[-1]), leading


def compute_stoich_null_space(
    stoich: npt.NDArray[np.float64],
    log_eq_consts: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the equilibrium null-space basis and particular solution.

    Mirrors ``math.rs::null_space_scipy`` / ``pinv_scipy`` in the Rust core,
    which call ``scipy.linalg.null_space`` and ``scipy.linalg.pinv``.

    Args:
        stoich:          (num_reactions, num_species) equilibrium stoich matrix.
        log_eq_consts:   (num_reactions,) log10 equilibrium constants.

    Returns:
        ``N``:      (num_species, n_free) null-space basis.
        ``x_particular``: (num_species,) particular solution pinv(stoich) @ logK.
    """
    null: npt.NDArray[np.float64] = scipy.linalg.null_space(stoich)
    x_particular: npt.NDArray[np.float64] = scipy.linalg.pinv(stoich) @ log_eq_consts
    return null, x_particular


@dataclass(frozen=True)
class EquilibriumParameters:
    """Precomputed equilibrium-speciation constants for a reaction network.

    The four matrix/vector tensors are network constants (never gradient
    tracked); they are kept in float64/CPU form and cast by the solver as
    needed.

    Attributes:
        stoich:            (num_reactions, num_species) equilibrium stoich; rows are
                           secondary species then exchange reactions.
        log_eq_consts:     (num_reactions,) log10 equilibrium constants
                           (``sec.eq_consts[1]``, then ``exch.log10_k_eq``).
        total:             (num_total, num_species) mass/charge conservation matrix.
        stoich_null_space: (num_species, n_free) null-space basis (scipy).
        x_particular:      (num_species,) particular solution (scipy pinv).
        total_names:       (num_total,) names of the total species, in row order.
    """

    stoich: Tensor
    log_eq_consts: Tensor
    total: Tensor
    stoich_null_space: Tensor
    x_particular: Tensor
    total_names: tuple[str, ...]

    @property
    def num_free(self) -> int:
        """Number of free variables = columns of the null space (Rust ``num_free``)."""
        return int(self.stoich_null_space.shape[1])

    @property
    def num_species(self) -> int:
        """Number of species = rows of the null space basis."""
        return int(self.stoich_null_space.shape[0])

    @property
    def charge_row(self) -> int | None:
        """Row index of the ``H+`` charge-balance row in ``total`` (None if absent)."""
        for idx, name in enumerate(self.total_names):
            if name == "H+":
                return idx
        return None

    @property
    def num_reactions(self) -> int:
        """Number of equilibrium reactions (rows of ``stoich``)."""
        return int(self.stoich.shape[0])

    def _cast(self, t: Tensor, ref: Tensor) -> Tensor:
        """Move a network constant onto ``ref``'s dtype/device (constants are
        stored float64/CPU; the solver may run float32 on CUDA)."""
        return t.to(dtype=ref.dtype, device=ref.device)

    def conc_func(self, x_free: Tensor) -> Tensor:
        """Evaluate ``10**(N @ x_free + x_particular)`` (Rust ``conc_func_rust``).

        Args:
            x_free: (n_free,) or (B, n_free) free-variable vector(s).

        Returns:
            (num_species,) or (B, num_species) equilibrium concentrations.
        """
        n_mat: Tensor = self._cast(self.stoich_null_space, x_free)
        xp: Tensor = self._cast(self.x_particular, x_free)
        exponent: Tensor = x_free @ n_mat.T + xp
        return exponent.mul(LNL10).exp()

    def residual(self, x_free: Tensor, c_tot: Tensor) -> Tensor:
        """Speciation conservation residual (Rust ``residual_rust``).

        Evaluates ``c_tot - total @ conc_func(x_free)`` for batched free
        variables.

        Args:
            x_free: (B, n_free) free-variable vectors.
            c_tot:  (B, num_total) conservation targets (``H+`` row zeroed).

        Returns:
            (B, num_total) residual.
        """
        total: Tensor = self._cast(self.total, x_free)
        return c_tot - self.conc_func(x_free) @ total.T

    def jacobian_residual(self, x_free: Tensor) -> Tensor:
        """Analytic Jacobian of the speciation residual (Rust ``jacobian_residual_rust``).

        With ``f(x) = c_tot - A g(x)``, ``g(x) = 10^(N x + b)``, the chain rule
        gives ``J = -ln(10) * (A * g[:, None]) @ N``.

        Args:
            x_free: (B, n_free) free-variable vectors.

        Returns:
            (B, num_total, n_free) Jacobian.
        """
        g: Tensor = self.conc_func(x_free)
        prod: Tensor = self._cast(self.total, x_free).unsqueeze(0) * g.unsqueeze(1)
        return prod @ self._cast(self.stoich_null_space, x_free) * (-LNL10)

    def x_free_solve(
        self,
        chms: Tensor,
        initial_x: Tensor,
        *,
        raise_on_failure: bool = False,
        verbose: bool = False,
    ) -> Tensor:
        """Solve the speciation problem for ``x_free`` (Rust ``x_free_solve_rust``).

        Computes the conservation targets ``c_tot = total @ chms`` (zeroing the
        ``H+`` charge-balance row), then solves the residual with Newton (analytic
        Jacobian) falling back to Levenberg-Marquardt; each batch lane follows the
        scalar Rust control flow exactly.

        Args:
            chms:       (num_species,) or (B, num_species) concentrations.
            initial_x:  (n_free,) or (B, n_free) warm start (zeros for a cold start).
            raise_on_failure: raise if a lane fails to converge; otherwise NaN.
            verbose:    print solver progress.

        Returns:
            ``(B, n_free)`` free-variable solution (leading dims of ``chms``
            preserved; a 1-D ``chms`` returns ``(n_free,)``).
        """
        c, leading = _flatten_batch(chms)
        total: Tensor = self._cast(self.total, c)
        batch: int = c.shape[0]
        c_tot: Tensor = c @ total.T
        if self.charge_row is not None:
            c_tot = c_tot.clone()
            c_tot[:, self.charge_row] = 0.0

        x0, x0_leading = _flatten_batch(initial_x)
        if x0_leading != () and x0_leading != leading:
            raise ValueError(
                f"initial_x leading dims {x0_leading} do not match chms leading dims {leading}"
            )
        if x0.shape[0] == 1 and batch > 1:
            x0_batch: Tensor = x0.expand(batch, -1).clone()
        elif x0.shape[0] == batch:
            x0_batch = x0.clone()
        else:
            raise ValueError(
                f"initial_x has {x0.shape[0]} rows but chms has {batch}"
            )

        def f(x: Tensor) -> Tensor:
            return self.residual(x, c_tot)

        def jf(x: Tensor) -> Tensor:
            return self.jacobian_residual(x)
        x_free: Tensor = newton_analytic_then_lm(
            f, jf, x0_batch, raise_on_failure=raise_on_failure, verbose=verbose
        )
        return x_free.reshape(*(leading + (self.num_free,)))

    def solve_equilibrium(
        self,
        chms: Tensor,
        x0: Tensor | None = None,
        *,
        raise_on_failure: bool = False,
        verbose: bool = False,
    ) -> Tensor:
        """Public speciation entry point (Rust ``solve_equilibrium`` / ``solve_equilibrium_rust``).

        With no free variables the concentrations are returned unchanged (there
        is nothing to re-speciate); otherwise the solve is warm-started from
        ``x0`` (or a cold start from zeros) and returns the equilibrium
        concentrations.

        Args:
            chms:       (num_species,) or (B, num_species) input concentrations.
            x0:         optional (n_free,) or (B, n_free) free-variable warm start.
            raise_on_failure: raise if a lane fails to converge; otherwise NaN.
            verbose:    print solver progress.

        Returns:
            Equilibrium concentrations with the leading dims of ``chms`` preserved.
        """
        if self.num_free == 0:
            return chms
        c, leading = _flatten_batch(chms)
        batch: int = c.shape[0]
        device: torch.device = c.device
        if x0 is None:
            x0_batch: Tensor = torch.zeros(
                (batch, self.num_free), dtype=c.dtype, device=device
            )
        else:
            x0_batch = x0
        x_free: Tensor = self.x_free_solve(
            chms, x0_batch, raise_on_failure=raise_on_failure, verbose=verbose
        )
        out: Tensor = self.conc_func(x_free)
        return out.reshape(*(leading + (self.num_species,)))