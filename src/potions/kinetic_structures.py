"""Kinetic rate-law evaluation and per-zone parameter containers.

Mirrors the reference ``kinetic_structures.rs``: the Monod / TST rate laws plus
the mineral ``MineralParameters`` (soil-water + temperature factors) and the
zone ``RtParameters`` / ``RiverParameters`` bundles.  All rate evaluations are
vectorised over a leading batch dimension (each lane is one scalar reference
concentration vector).

**TST matrix convention (important):** the reference Rust code swaps the natural
naming.  The matrix stored in ``timestep_params.stoich`` is built from each TST
reaction's ``dependence`` map, and the matrix stored in ``dep`` is built from the
mineral's own dissolution stoichiometry.  So, in ``tst_rate = D*(1 - Q/K)``:

- ``Q`` exponent matrix = ``stoich`` = TST reaction ``dependence`` values
  (zero where the dependence map lacks the species),
- ``D`` exponent matrix = ``dep``     = mineral stoichiometry values (also zero
  where the dependence map lacks the species).

This is the opposite of the legacy ``potions`` Python wrapper's ``tst_params``,
and the Rust core — which is what the notebooks and golden tests run — is
authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch

Tensor: TypeAlias = torch.Tensor
Scalar: TypeAlias = float | torch.Tensor

#: Number of auxiliary parameters packed per mineral in calibration arrays
#: (Rust ``PARAMETERS_PER_MINERAL``): sw_threshold, sw_exp, q_10, ssa.
PARAMETERS_PER_MINERAL: int = 4


def _coerce(x: Scalar, ref: Tensor) -> Tensor:
    """Materialise a scalar as a broadcastable tensor on ``ref``'s dtype/device."""
    if isinstance(x, (int, float)):
        return torch.tensor(float(x), dtype=ref.dtype, device=ref.device)
    return x.to(dtype=ref.dtype, device=ref.device)


# --------------------------------------------------------------------------- #
# TST activity smoothing (vectorised ports of the Rust helpers)
# --------------------------------------------------------------------------- #

#: Softplus transition scale (mol/L) of :func:`smoothed_conc`.  The transition
#: sits ~10 orders of magnitude below the smallest physical concentration
#: observed (1e-17 mol/L): for ``c > 5e-29`` the branch-1 correction
#: ``delta*exp(-c/delta)`` is below half an ulp of ``c`` (and underflows to
#: exactly zero for ``c > 7.5e-28``), so ``s(c)`` is bit-identical to ``c``
#: (and :func:`smoothed_rel_deriv` to ``1/c``) for every physical state; the
#: smoothing is confined to the non-physical side of the zero plane.
SOFTPLUS_DELTA: float = 1e-30

#: Floor (mol/L) applied in the TST activity expressions before ``log10``.
#: Root finders (Newton, LM) probe trial states with zero or negative
#: concentrations; keeping each log-contribution large but finite instead of
#: NaN is what the solvers' sufficiency checks require.  ``s(c)`` equals this
#: value exactly for ``c < ~= -6.2e-28`` (branch 2, ``delta*exp(z)`` below the
#: floor) and across the branch-3 plateau ``-5e-29 <= c < ~= -3.7e-28``
#: (``1 + exp(z)`` rounds to ``1.0`` so the log is exactly 0); below
#: ``~= -6.2e-28`` this preserves the old extreme-behaviour sanitisation
#: (log10 = -300).
LOG10_CONC_FLOOR: float = 1e-300


def smoothed_conc(c: Tensor) -> Tensor:
    """Smooth stand-in for ``c.clamp_min(LOG10_CONC_FLOOR)`` in TST activity terms.

    ``s(c) = max(d * softplus(c/d), LOG10_CONC_FLOOR)`` with ``d =
    SOFTPLUS_DELTA`` — a vectorised port of the Rust ``smoothed_conc``.

    Governing form (per element, ``z = c/d``)::

        s(c) = c + d*exp(-z)                 if z >  50          (no floor)
             = max(d*exp(z),  FLOOR)         if z < -50
             = max(d*log(1 + exp(z)), FLOOR) if |z| <= 50

    Written as ``c + d*exp(-z)`` (not ``d*(z + softplus(z))``) so the
    ``d*(c/d)`` round-trip cannot perturb ``c``: for every ``c > 5e-29`` the
    correction ``d*exp(-z)`` is below half an ulp of ``c`` (it underflows to
    exactly 0 for ``c > 7.5e-28``) and ``s(c)`` is bit-identical to ``c``.
    The middle branch uses the Rust literal ``log(1 + exp(z))`` — a true
    ``+`` followed by ``log``, *not* ``log1p``: for ``z < ln(2**-53) ~=
    -36.7`` the addition rounds to ``1.0`` and the branch plateaus at exactly
    ``FLOOR`` (log of 1.0 is 0), matching the Rust scalar bit-for-bit,
    plateau included.

    The branches are evaluated *unclamped*: the non-selected sides may
    saturate silently (``exp`` to ``inf`` / 0) exactly as the Rust scalar
    does, and ``torch.where`` drops them.  NaN inputs keep the old
    sanitisation (the floor): Rust's ``f64::max`` returns the non-NaN
    operand, and the only branch a NaN ``z`` can reach (branch 3, since both
    comparisons against NaN are false) is guarded with
    ``where(isnan, FLOOR, ...)``.  ``clamp_min`` there is the true max the
    Rust ``.max(FLOOR)`` performs, not redundant with the NaN guard: on the
    plateau ``x3`` is exactly 0.0, which is below the floor.

    Bit-identical to the old hard clamp for every ``c >= 1e-25`` (indeed for
    every ``c > 5e-29`` by the ulp argument above) and for ``c < ~=
    -6.2e-28``; the change is confined to the non-physical side of the zero
    plane, where the value and its derivative vary smoothly instead of the
    old O(1) kink.  **f64-only**: ``LOG10_CONC_FLOOR`` (1e-300) underflows
    to 0.0 in float32; the batched RT paths (and the Rust reference) are
    f64-only.
    """
    z: Tensor = c / SOFTPLUS_DELTA
    # Branch 1 (z > 50): c + d*exp(-z).  The selected side has -z < -50
    # (exp only underflows); the discarded side saturates to +inf at
    # z < -745 and is dropped by ``where`` — the Rust scalar's own silent
    # saturation.
    b1: Tensor = c + SOFTPLUS_DELTA * torch.exp(-z)
    # Branch 2 (z < -50): max(d*exp(z), FLOOR).  exp(z) < exp(-50) on the
    # selected side (no overflow); the discarded side can saturate to inf
    # (which stays inf through the floor).  NaN only arises from NaN c,
    # which never satisfies z < -50.
    b2: Tensor = (SOFTPLUS_DELTA * torch.exp(z)).clamp_min(LOG10_CONC_FLOOR)
    # Branch 3 (|z| <= 50): max(d*log(1 + exp(z)), FLOOR).  The clamp is a
    # no-op on the selected side (exp(z) <= exp(50) ~= 5e21, no overflow)
    # and keeps the discarded side finite.  NaN x3 (NaN c) maps to FLOOR via
    # the guard (Rust f64::max semantics); ``clamp_min`` is the true max.
    x3: Tensor = SOFTPLUS_DELTA * torch.log(1.0 + torch.exp(z.clamp(max=50.0)))
    b3: Tensor = torch.where(
        x3.isnan(), x3.new_tensor(LOG10_CONC_FLOOR), x3.clamp_min(LOG10_CONC_FLOOR)
    )
    return torch.where(z > 50.0, b1, torch.where(z < -50.0, b2, b3))


def smoothed_rel_deriv(c: Tensor) -> Tensor:
    """Relative derivative ``s'(c)/s(c)`` of :func:`smoothed_conc`.

    Vectorised port of the Rust ``smoothed_rel_deriv``:
    ``sigmoid(c/d) / s(c)``, with ``sigmoid(z) = 1/(1 + exp(-z))``
    branched exactly as the Rust helper (``z > 50 -> 1``; ``z < -50 ->
    exp(z)``; else ``1/(1 + exp(-z))``).  The branches are evaluated
    *unclamped*: no selected branch can overflow (``exp(z) < exp(-50)`` on
    the left, ``exp(-z) <= exp(50)`` in the middle), while the discarded
    sides saturate silently to ``inf`` / 0 exactly as the Rust scalar does
    and are dropped by ``where``.

    NaN inputs map to 0.0, mirroring the Rust helper's up-front
    ``c.is_nan()`` check (the guard also keeps the NaN of the discarded
    sigmoid branches out of the final division).

    This is the quantity the TST Jacobian needs: the TST value is a power
    product ``prod_j s(c_j)**nu_j`` (= ``10**(sum_j nu_j*log10 s(c_j))``
    exactly), whose derivative is ``D * nu * s'/s`` (no ``ln(10)``, which
    cancels) and ``D * mu * s'/s`` on the dependence side.  For
    ``c > 5e-29`` this is bit-identical to the old clamp's ``1/c``
    (sigmoid = 1, s = c); on the non-physical side it stays finite and
    decays smoothly through the zero plane (~0 below ``c ~= -7.5e-28``,
    where ``exp(z)`` underflows — the old zero derivative; ~7.2e29 at
    ``c = 0``) instead of the old 0 / 1/c jump that stalled Newton / LM at
    zero-concentration planes.  **f64-only** (see :func:`smoothed_conc`).
    """
    z: Tensor = c / SOFTPLUS_DELTA
    sig: Tensor = torch.where(
        z > 50.0,
        torch.ones_like(z),
        torch.where(z < -50.0, torch.exp(z), 1.0 / (1.0 + torch.exp(-z))),
    )
    sig: Tensor = torch.where(c.isnan(), torch.zeros_like(z), sig)
    return sig / smoothed_conc(c)


def monod_inhib_rates(
    monod_mat: Tensor, inhib_mat: Tensor, chms: Tensor, out: Tensor | None = None
) -> Tensor:
    """Elementwise Monod ``prod c/(K+c)`` times inhibition ``prod K/(K+c)``.

    Args:
        monod_mat: (num_minerals, num_species) half-saturation constants; ``NaN``
            where the reaction defines no monod term, and all-zero rows for
            minerals without a monod reaction.
        inhib_mat: (num_minerals, num_species) inhibition constants; same
            ``NaN`` / zero conventions as ``monod_mat``.
        chms:      (B, num_species) concentrations.
        out:       optional pre-allocated output tensor.

    Returns:
        (B, num_minerals) per-mineral Monod rate (Rust ``MonodParameters::rate_rust``).
    """
    num_min: int = int(monod_mat.shape[0])
    batch: int = int(chms.shape[0])
    if num_min == 0:
        if out is not None:
            out[:, :] = torch.zeros((batch, 0), dtype=chms.dtype, device=chms.device)
            return out
        return torch.zeros((batch, 0), dtype=chms.dtype, device=chms.device)

    c: Tensor = chms.unsqueeze(1)  # (B, 1, num_species)
    m: Tensor = monod_mat.unsqueeze(0)  # (1, num_minerals, num_species)
    i_mat: Tensor = inhib_mat.unsqueeze(0)

    monod_factor: Tensor = torch.where(
        m.isnan(), torch.ones_like(m), c / (m + c)
    )
    monod_prod: Tensor = monod_factor.prod(dim=-1)  # (B, num_minerals)

    inhib_factor: Tensor = torch.where(
        i_mat.isnan(), torch.ones_like(i_mat), i_mat / (i_mat + c)
    )
    inhib_prod: Tensor = inhib_factor.prod(dim=-1)

    if out is not None:
        # Same elementwise multiply kernel as the non-out path, written
        # directly into the caller's buffer (no intermediate result + copy).
        torch.mul(monod_prod, inhib_prod, out=out)
        return out
    return monod_prod * inhib_prod


def tst_rates(
    tst_stoich: Tensor, tst_dep: Tensor, min_eq_const: Tensor, chms: Tensor
) -> Tensor:
    """Per-mineral TST rate ``D*(1 - Q/K)`` (Rust ``TstParameters::rate_rust``).

    ``Q`` and ``D`` are power products over the *smoothed* species
    concentrations ``s = :func:`smoothed_conc``: ``Q = prod_j
    s(c_j)**nu_j``, ``D = prod_j s(c_j)**mu_j`` (``nu`` / ``mu`` the rows of
    ``tst_stoich`` / ``tst_dep``), evaluated in log space as ``Q =
    10**(stoich @ log10 s(c))``, ``D = 10**(dep @ log10 s(c))`` — the exact
    Rust ``calculate_solubility_product`` / ``calculate_dependence_term``
    form.  Routing ``c`` through ``smoothed_conc`` before the ``log10``
    keeps every log-contribution finite (>= ``-300``) for the zero or
    negative trial concentrations a root finder can probe, instead of the
    ``-inf`` / NaN a raw ``log10(c)`` would give (and instead of the kink
    a hard clamp would introduce into the derivative).

    Args:
        tst_stoich:   (num_minerals, num_species) ``Q`` exponent matrix (built
            from each reaction's ``dependence`` map).
        tst_dep:      (num_minerals, num_species) ``D`` exponent matrix (built
            from each mineral's own stoichiometry).
        min_eq_const: (num_minerals,) equilibrium constants ``K``.
        chms:         (B, num_species) concentrations.

    Returns:
        (B, num_minerals) TST rate.
    """
    # Avoid the shape bug for the empty-mineral network: a (0, n) @ (B, n).T
    # still gives (B, 0) but 10** of an empty tensor must not be taken.
    num_min: int = int(min_eq_const.shape[0])
    batch: int = int(chms.shape[0])

    if num_min == 0:
        empty: Tensor = torch.zeros((batch, 0), dtype=chms.dtype, device=chms.device)
        return empty

    log_conc: Tensor = torch.log10(smoothed_conc(chms))  # (B, num_species)
    e_q: Tensor = log_conc @ tst_stoich.T  # (B, num_minerals)
    e_d: Tensor = log_conc @ tst_dep.T
    q: Tensor = torch.pow(10.0, e_q)
    d: Tensor = torch.pow(10.0, e_d)
    return d * (-q / min_eq_const.unsqueeze(0) + 1.0)


@dataclass(frozen=True)
class MonodParameters:
    """Monod (Michaelis-Menten) constants for every mineral in the network.

    Attributes:
        monod_mat: (num_minerals, num_species) half-saturation constants; entry
            is ``NaN`` where the reaction defines no monod term for that
            species, and an entire row of ``0.0`` for minerals without a monod
            reaction.
        inhib_mat: (num_minerals, num_species) inhibition constants; same NaN /
            zero conventions as ``monod_mat``.
    """

    monod_mat: Tensor
    inhib_mat: Tensor

    @property
    def num_minerals(self) -> int:
        """Number of minerals (rows of the matrices)."""
        return int(self.monod_mat.shape[0])

    @property
    def num_species(self) -> int:
        """Number of species (columns of the matrices)."""
        return int(self.monod_mat.shape[1])

    def rate(self, chms: Tensor) -> Tensor:
        """Monod rate on the batch lanes of ``chms`` (Rust ``rate_rust``).

        Args:
            chms: (B, num_species) concentrations.

        Returns:
            (B, num_minerals) rate factors.
        """
        return monod_inhib_rates(
            self.monod_mat.to(dtype=chms.dtype, device=chms.device),
            self.inhib_mat.to(dtype=chms.dtype, device=chms.device),
            chms,
        )


@dataclass(frozen=True)
class TstParameters:
    """Transition-state-theory constants for every mineral in the network.

    Attributes:
        stoich:        (num_minerals, num_species) matrix used as the ``Q``
                       (solubility-product) exponent.  Built from each TST
                       reaction's ``dependence`` map; zero rows for minerals
                       without a TST reaction.
        dep:           (num_minerals, num_species) matrix used as the ``D``
                       (forward-rate) exponent.  Built from each mineral's own
                       stoichiometry, masked to the species in its dependence
                       map; zero rows for minerals without a TST reaction.
        min_eq_const:  (num_minerals,) equilibrium constant ``K`` = ``10**eq_consts[1]``
                       for TST minerals, ``1.0`` otherwise.
    """

    stoich: Tensor
    dep: Tensor
    min_eq_const: Tensor

    @property
    def num_minerals(self) -> int:
        """Number of minerals (rows of the matrices)."""
        return int(self.stoich.shape[0])

    @property
    def num_species(self) -> int:
        """Number of species (columns of the matrices)."""
        return int(self.stoich.shape[1])

    def calculate_solubility_product(self, chms: Tensor) -> tuple[Tensor, Tensor]:
        """Linear solubility product ``Q`` and its log10 exponent ``e``."""
        return self._exponentiate(self.stoich, chms)

    def calculate_dependence_term(self, chms: Tensor) -> tuple[Tensor, Tensor]:
        """Linear dependence-term ``D`` and its log10 exponent ``e``."""
        return self._exponentiate(self.dep, chms)

    def _exponentiate(self, mat: Tensor, chms: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(10**e, e)`` with ``e = mat @ log10 s(c)`` (batched).

        ``s(c)`` is :func:`smoothed_conc`, so ``10**e`` is the exact power
        product ``prod_j s(c_j)**m_j`` over the row ``m`` of ``mat`` — the
        Rust ``calculate_solubility_product`` / ``calculate_dependence_term``
        form (see :func:`tst_rates`).
        """
        log_conc: Tensor = torch.log10(smoothed_conc(chms))
        e: Tensor = log_conc @ mat.to(dtype=chms.dtype, device=chms.device).T
        return torch.pow(10.0, e), e

    def rates_from_exponents(self, e_dep: Tensor, e_q: Tensor) -> Tensor:
        """TST rate ``10**e_dep * (1 - 10**e_q / K)`` from log-exponents."""
        d: Tensor = torch.pow(10.0, e_dep)
        q: Tensor = torch.pow(10.0, e_q)
        return d * (
            -q / self.min_eq_const.to(dtype=e_dep.dtype, device=e_dep.device) + 1.0
        )

    def rate(self, chms: Tensor) -> Tensor:
        """TST rate on the batch lanes of ``chms`` (Rust ``rate_rust``).

        Args:
            chms: (B, num_species) concentrations.

        Returns:
            (B, num_minerals) TST rates.
        """
        return tst_rates(
            self.stoich.to(dtype=chms.dtype, device=chms.device),
            self.dep.to(dtype=chms.dtype, device=chms.device),
            self.min_eq_const.to(dtype=chms.dtype, device=chms.device),
            chms,
        )


@dataclass(frozen=True)
class MineralAuxParams:
    """Auxiliary kinetic parameters for a *single* mineral (calibration unit).

    Attributes:
        sw_threshold: soil-water-content threshold [-] the activity branch on.
        sw_exp:       exponent applied to the (fractional) soil-water ratio.
        q_10:         temperature-sensitivity factor (rate doubles every 10 K).
        ssa:          specific surface area [cm^2/g].
    """

    sw_threshold: float
    sw_exp: float
    q_10: float
    ssa: float

    def to_array(self) -> Tensor:
        """(PARAMETERS_PER_MINERAL,) packed array (sw_threshold, sw_exp, q_10, ssa)."""
        return torch.tensor(
            [self.sw_threshold, self.sw_exp, self.q_10, self.ssa], dtype=torch.float64
        )

    @staticmethod
    def from_array(arr: Tensor) -> MineralAuxParams:
        """Reconstruct from a packed (PARAMETERS_PER_MINERAL,) array."""
        if arr.shape[0] != PARAMETERS_PER_MINERAL:
            raise ValueError(
                f"Passed incorrect number of parameters to MineralAuxParams::from_array: {arr}"
            )
        return MineralAuxParams(
            sw_threshold=float(arr[0]),
            sw_exp=float(arr[1]),
            q_10=float(arr[2]),
            ssa=float(arr[3]),
        )


@dataclass(frozen=True)
class ZoneDimensions:
    """Hydrologic dimensions of a soil ``RtZone`` (Rust ``ZoneDimensions``).

    Attributes:
        porosity:            volumetric porosity [-] (fraction of depth that is void).
        depth:               zone depth [mm] (the mineral concentration scale).
        passive_water_storage: water immobile in the matrix that never drains [mm].
    """

    porosity: float | Tensor
    depth: float | Tensor
    passive_water_storage: float | Tensor

    @property
    def max_water_volume(self) -> Scalar:
        """Maximum drainable water the zone can hold [mm].

        Returns a ``float`` for scalar dimensions or a ``Tensor`` when any
        dimension carries a batch dimension (batched parameter sets).
        """
        return self.porosity * self.depth - self.passive_water_storage

    def to_array(self) -> Tensor:
        """(3,) packed array (porosity, depth, passive_water_storage)."""
        return torch.tensor(
            [self.porosity, self.depth, self.passive_water_storage], dtype=torch.float64
        )

    @staticmethod
    def from_array(arr: Tensor) -> ZoneDimensions:
        """Reconstruct from a (3,) array."""
        if arr.shape[0] != 3:
            raise ValueError(
                "Incorrect number of arguments passed to `ZoneDimensions::from_array`"
            )
        return ZoneDimensions(
            porosity=float(arr[0]),
            depth=float(arr[1]),
            passive_water_storage=float(arr[2]),
        )


@dataclass(frozen=True)
class MineralParameters:
    """Per-network arrays of auxiliary mineral kinetic parameters.

    Attributes:
        sw_threshold: (num_minerals,) soil-water threshold [-] per mineral.
        sw_exp:       (num_minerals,) soil-water exponent per mineral.
        q_10:         (num_minerals,) temperature-sensitivity factor per mineral.
        ssa:          (num_minerals,) specific surface areas [cm^2/g].
    """

    sw_threshold: Tensor
    sw_exp: Tensor
    q_10: Tensor
    ssa: Tensor

    @property
    def num_minerals(self) -> int:
        """Number of minerals described by these parameters."""
        return int(self.sw_threshold.shape[0])

    def soil_water_factor(self, s_w: Scalar) -> Tensor:
        """Activity factor from soil-water content (Rust ``soil_water_factor_rust``).

        ``( (1-s_w)/(1-threshold) )**sw_exp`` above the threshold, otherwise
        ``( s_w/threshold )**sw_exp``.

        The output has shape ``(..., num_minerals)`` so that it broadcasts
        correctly against per-mineral arrays when ``s_w`` carries a batch
        dimension.
        """
        sw: Tensor = _coerce(s_w, self.sw_threshold)
        # Ensure at least one leading batch axis for uniform broadcasting with
        # per-mineral parameter vectors even when ``s_w`` is a 0-d scalar.
        sw_v: Tensor = sw.unsqueeze(-1)                              # (..., 1)
        numer_hi: Tensor = (1.0 - sw_v) / (1.0 - self.sw_threshold)
        numer_lo: Tensor = sw_v / self.sw_threshold
        return torch.where(
            sw_v >= self.sw_threshold, numer_hi**self.sw_exp, numer_lo**self.sw_exp
        )

    def temperature_factor(self, temp: Scalar) -> Tensor:
        """Activity factor from temperature (Rust ``temperature_factor_rust``).

        Returns ``(..., num_minerals)`` for batched ``temp``.
        """
        t: Tensor = _coerce(temp, self.sw_threshold)
        t_v: Tensor = t.unsqueeze(-1)                                # (..., 1)
        return self.q_10 ** ((t_v - 20.0) / 10.0)

    def factor(self, s_w: Scalar, temp: Scalar) -> Tensor:
        """Combined auxiliary kinetic factor ``sw_factor * temp_factor``.

        Output shape is ``(..., num_minerals)`` matching the leading dims of
        ``s_w`` / ``temp`` (at least one leading dim for the mineral axis).
        """
        return self.soil_water_factor(s_w) * self.temperature_factor(temp)

    def to_array(self) -> Tensor:
        """(PARAMETERS_PER_MINERAL * num_minerals,) packed natural-scale array."""
        parts: list[Tensor] = []
        for i in range(self.num_minerals):
            parts.append(
                torch.tensor(
                    [
                        float(self.sw_threshold[i]),
                        float(self.sw_exp[i]),
                        float(self.q_10[i]),
                        float(self.ssa[i]),
                    ],
                    dtype=torch.float64,
                )
            )
        return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float64)

    @staticmethod
    def from_array(arr: Tensor, natural_scales: bool = True) -> MineralParameters:
        """Unpack a packed array; ``10**`` applied to ssa when ``natural_scales``.

        The reference stores the specific surface area in log10 units in packing
        and re-exponentiates on unpacking (``MineralParameters::from_array``).
        """
        if arr.shape[0] % PARAMETERS_PER_MINERAL != 0:
            raise ValueError(
                "Failed to create new MineralParameters from array because the "
                f"shape is wrong, expected multiple of {PARAMETERS_PER_MINERAL} "
                f"but got {arr.shape[0]}"
            )
        sw_threshold = arr[0::PARAMETERS_PER_MINERAL]
        sw_exp = arr[1::PARAMETERS_PER_MINERAL]
        q_10 = arr[2::PARAMETERS_PER_MINERAL]
        ssa = arr[3::PARAMETERS_PER_MINERAL]
        ssa = (10.0**ssa) if natural_scales else ssa
        return MineralParameters(
            sw_threshold=sw_threshold.clone(),
            sw_exp=sw_exp.clone(),
            q_10=q_10.clone(),
            ssa=ssa.clone(),
        )

    @staticmethod
    def from_batched_array(
        arr: Tensor, natural_scales: bool = True
    ) -> MineralParameters:
        """Build per-lane mineral parameters from a ``[batch_size, 4*num_min]`` array.

        Unlike :meth:`from_array` (a single ``[4*num_min]`` lane, producing
        ``[num_min]`` fields), this treats every row as an *independent*
        parameter lane so that a batch of parameter sets can be stepped in one
        pass.  Each field carries a leading batch-size dimension:

        - ``sw_threshold`` / ``sw_exp`` / ``q_10`` / ``ssa``:
          ``[batch_size, num_min]``.

        The fields are laid out per row in the same interleaved order as the
        scalar form ``(sw_threshold, sw_exp, q_10, ssa)`` per mineral; ``10**``
        is applied to ``ssa`` when ``natural_scales`` is true.

        Args:
            arr:            ``[batch_size, PARAMETERS_PER_MINERAL * num_min]``
                packed array.
            natural_scales: if True, the packed ``ssa`` values are log10 and are
                re-exponentiated on unpacking.

        Returns:
            A :class:`MineralParameters` whose four fields each have shape
            ``(batch_size, num_min)``.
        """
        if arr.ndim != 2 or arr.shape[1] % PARAMETERS_PER_MINERAL != 0:
            raise ValueError(
                "Failed to create batched MineralParameters from array because "
                f"the shape is wrong, expected [batch_size, multiple of "
                f"{PARAMETERS_PER_MINERAL}] but got {tuple(arr.shape)}"
            )
        g0: Tensor = arr[..., 0::PARAMETERS_PER_MINERAL]  # [batch_size, num_min]
        g1: Tensor = arr[..., 1::PARAMETERS_PER_MINERAL]
        g2: Tensor = arr[..., 2::PARAMETERS_PER_MINERAL]
        g3: Tensor = arr[..., 3::PARAMETERS_PER_MINERAL]
        ssa: Tensor = (10.0 ** g3) if natural_scales else g3
        return MineralParameters(
            sw_threshold=g0.clone(),
            sw_exp=g1.clone(),
            q_10=g2.clone(),
            ssa=ssa.clone(),
        )

    @staticmethod
    def from_mineral_parameters(
        minerals: list[MineralAuxParams],
    ) -> MineralParameters:
        """Aggregate a list of per-mineral :class:`MineralAuxParams` objects."""
        sw_thrs: list[float] = []
        sw_exps: list[float] = []
        q_10s: list[float] = []
        ssas: list[float] = []
        for m in minerals:
            sw_thrs.append(m.sw_threshold)
            sw_exps.append(m.sw_exp)
            q_10s.append(m.q_10)
            ssas.append(m.ssa)
        return MineralParameters(
            sw_threshold=torch.tensor(sw_thrs, dtype=torch.float64),
            sw_exp=torch.tensor(sw_exps, dtype=torch.float64),
            q_10=torch.tensor(q_10s, dtype=torch.float64),
            ssa=torch.tensor(ssas, dtype=torch.float64),
        )


@dataclass(frozen=True)
class RtParameters:
    """Per-zone physical/kinetic parameters for a soil ``RtZone``.

    Attributes:
        dimensions:     porosity/depth/passive water storage.
        mineral_params: per-mineral auxiliary kinetic parameters, or ``None``
            when reactions are disabled.
    """

    dimensions: ZoneDimensions
    mineral_params: MineralParameters | None = None

    def to_array(self) -> Tensor:
        """Flattened ``(3 + PARAMETERS_PER_MINERAL*num_min,)`` parameter array."""
        parts: list[Tensor] = [self.dimensions.to_array()]
        if self.mineral_params is not None:
            parts.append(self.mineral_params.to_array())
        return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float64)

    @staticmethod
    def from_batched_array(
        arr: Tensor, natural_scales: bool = True
    ) -> RtParameters:
        """Build per-lane :class:`RtParameters` from a batched array.

        The input has shape ``[batch_size, 3 + 4*num_min]``.  The first three
        columns of every row are the zone dimensions (porosity, depth, passive
        water storage); the trailing ``4*num_min`` columns are the per-mineral
        kinetic blocks in the same interleaved
        ``(sw_threshold, sw_exp, q_10, ssa)`` order as the scalar form.  Every
        field carries a leading batch-size dimension so a batch of parameter
        sets can drive a single ``RtZone.step`` call.

        Args:
            arr:            ``[batch_size, 3 + PARAMETERS_PER_MINERAL*num_min]``
                per-lane array.
            natural_scales: if True, the packed ``ssa`` values are log10 and are
                re-exponentiated on unpacking.

        Returns:
            A :class:`RtParameters` whose :attr:`dimensions` fields are
            ``(batch_size,)`` tensors and whose :attr:`mineral_params` (if
            reactions are enabled) has ``(batch_size, num_min)`` fields.
        """
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(
                "Failed to create batched RtParameters from array because the "
                "shape is wrong, expected [batch_size, 3 + multiple of 4] "
                f"but got {tuple(arr.shape)}"
            )
        dims: ZoneDimensions = ZoneDimensions(
            porosity=arr[:, 0].clone(),
            depth=arr[:, 1].clone(),
            passive_water_storage=arr[:, 2].clone(),
        )
        if arr.shape[1] == 3:
            return RtParameters(dimensions=dims, mineral_params=None)
        minerals: MineralParameters = MineralParameters.from_batched_array(
            arr[:, 3:], natural_scales=natural_scales
        )
        return RtParameters(dimensions=dims, mineral_params=minerals)

    @staticmethod
    def from_array(arr: Tensor, natural_scales: bool = True) -> RtParameters:
        """Reconstruct from a packed array (dimensions first, then minerals).

        Args:
            arr:            ``(3 + 4*num_min,)`` packed array.
            natural_scales: if True, the packed ``ssa`` values are log10 and
                re-exponentiated on unpacking. Defaults to True.

        Returns:
            A new :class:`RtParameters` instance.
        """
        if arr.shape[0] < 3:
            raise ValueError(
                "Failed to create RtParameters from array, the shape is wrong: "
                f"expected at least 3 entries, got {arr.shape[0]}"
            )
        dims: ZoneDimensions = ZoneDimensions.from_array(arr[:3])
        if arr.shape[0] == 3:
            return RtParameters(dimensions=dims, mineral_params=None)
        minerals: MineralParameters = MineralParameters.from_array(arr[3:], natural_scales=natural_scales)
        return RtParameters(dimensions=dims, mineral_params=minerals)


@dataclass(frozen=True)
class RiverDimensions:
    """Dimensions of a ``RiverZone`` (Rust ``RiverDimensions``).

    Attributes:
        bed_depth:              river bed depth [mm] used to scale mineral loads.
        passive_water_storage:  immobile bed water [mm].
    """

    bed_depth: float
    passive_water_storage: float

    @staticmethod
    def num_parameters() -> int:
        """Number of packed parameters (bed_depth, passive_water_storage)."""
        return 2

    def to_array(self) -> Tensor:
        """(2,) packed array (bed_depth, passive_water_storage)."""
        return torch.tensor(
            [self.bed_depth, self.passive_water_storage], dtype=torch.float64
        )

    @staticmethod
    def from_array(arr: Tensor) -> RiverDimensions:
        """Reconstruct from a (2,) array."""
        if arr.shape[0] != RiverDimensions.num_parameters():
            raise ValueError("Incorrect array size passed to RiverDimensions::from_array")
        return RiverDimensions(bed_depth=float(arr[0]), passive_water_storage=float(arr[1]))


@dataclass(frozen=True)
class RiverParameters:
    """Per-zone physical/kinetic parameters for a ``RiverZone``.

    Attributes:
        dimensions:     bed depth + passive water storage.
        mineral_params: per-mineral auxiliary kinetic parameters, or ``None``
            when reactions are disabled.
    """

    dimensions: RiverDimensions
    mineral_params: MineralParameters | None = None

    def to_array(self) -> Tensor:
        """Flattened ``(2 + PARAMETERS_PER_MINERAL*num_min,)`` parameter array."""
        parts: list[Tensor] = [self.dimensions.to_array()]
        if self.mineral_params is not None:
            parts.append(self.mineral_params.to_array())
        return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float64)

    @staticmethod
    def from_array(arr: Tensor, natural_scales: bool = True) -> RiverParameters:
        """Reconstruct from a packed array (dimensions first, then minerals).

        Args:
            arr:            ``(2 + 4*num_min,)`` packed array.
            natural_scales: if True, the packed ``ssa`` values are log10 and
                re-exponentiated on unpacking. Defaults to True.

        Returns:
            A new :class:`RiverParameters` instance.
        """
        dims: RiverDimensions = RiverDimensions.from_array(arr[:2])
        if arr.shape[0] == 2:
            return RiverParameters(dimensions=dims, mineral_params=None)
        minerals: MineralParameters = MineralParameters.from_array(arr[2:], natural_scales=natural_scales)
        return RiverParameters(dimensions=dims, mineral_params=minerals)