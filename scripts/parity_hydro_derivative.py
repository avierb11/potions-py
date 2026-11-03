"""Parity check: analytic hydro residual derivative vs ``torch.autograd.grad``.

Validates that ``HydrologicZone.residual_deriv`` — the analytic flux-by-flux
derivative consumed by the damped Newton driver's ``(r, dr)`` hot path —
matches ``torch.autograd.grad`` of ``HydrologicZone.residual`` **bit-for-bit**,
including at every non-smooth point (kink) of the zone flux laws:

  * snow    melt ``min(s, fmax*(t-tt))`` tie          (``s == fmax*(t-tt)``)
  * surface ET ``min(s/(fc*lp), 1)`` tie             (``s == fc*lp``)
  * surface quickflow ``max(k0*(s-thr), 0)`` kink    (``s == thr``)
  * surface percolation ``q_in*(s/fc)**beta`` at ``s == 0``
            (``beta < 1`` -> infinite one-sided slope, as in autograd)
  * ground  baseflow gate ``where(s >= 1e-12, ...)`` edge (``s == 1e-12``)
  * ground  baseflow at ``s <= 0`` (the ``0**negative -> inf`` pow slope,
            dropped by the scalar-threshold clamp gate on both sides)
  * ground  percolation ``clamp_max(s, perc)`` edge (``s == perc``: hard
            1/0 step for float ``perc``, min/max 0.5 tie for Tensor ``perc``)

The zone cases also run at a **non-dyadic** ``dt = 1.37``: a dyadic ``dt``
scales exactly, so only a non-dyadic value discriminates the dt-*folded*
chains (each flux's chain carries its own ``-dt`` at the autograd position)
from the naive ``dt * d(mass_balance)/ds - 1`` form.

The primitive sweep additionally covers ``clamp_min`` / ``clamp_max`` under
**both** threshold dispatches (a Python ``float`` threshold — the hard-step
kernel with the tie on the flat side — and a 0-dim ``Tensor`` threshold —
the min/max-style subgradient kernel with a 0.5 tie) and the ``pow``
backward slope ``b * x**(b-1)`` against autograd for several exponents
(incl. fractional and 0-dim-tensor exponents) and several
upstream-gradient variants.

The comparison is exact (``a == b`` or both NaN); the 1-ulp allowance the
optimization plan grants for *smooth* points does not apply here — the
analytic formula is written to mirror the forward op sequence so smooth
points agree exactly too, and the kink conventions (half-slopes / clamp
steps / pow-at-zero / ``where``-gate ``+0``) must be exact or the Newton
iterates diverge from the legacy autograd driver.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/parity_hydro_derivative.py
    PYTHONPATH=src .venv/bin/python scripts/parity_hydro_derivative.py --n-random 4096

Exit code 0 iff every lane of every case matches exactly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from potions.common_types import HydroForcing
from potions.hydro import (
    GroundZone,
    GroundZoneB,
    HydrologicZone,
    SnowZone,
    SurfaceZone,
    _d_clamp_max,
    _d_clamp_min,
    _d_max,
    _d_min,
    _d_pow,
)

torch.set_num_threads(1)


def _nextafter(x: float, y: float) -> float:
    """Nearest representable float to ``x`` in the direction of ``y``."""
    return torch.nextafter(torch.tensor(x), torch.tensor(y)).item()


def build_s_grid(kinks: list[float], n_random: int, seed: int) -> torch.Tensor:
    """A sweep grid of trial states ``s`` covering every kink and its
    neighbours, both sides of the non-negativity clamp, and dense random /
    log-uniform coverage of the physical range.

    For each kink value the grid contains the kink itself, its
    predecessor / successor representable floats (ulp neighbours), and
    fixed-distance offsets on both sides, so the half-slope ties, clamp
    steps, and pow-at-zero conventions are all exercised.
    """
    vals: list[float] = []
    specials: list[float] = [
        float("-0.0"),
        0.0,
        float("-1e-9"),
        1e-15,
        1e-12,
        1e-9,
        1e-6,
        0.5,
        1.0,
        2.5,
        10.0,
        100.0,
        500.0,
    ]
    vals.extend(specials)
    for k in kinks:
        if k is None:
            continue
        vals.append(k)
        vals.append(_nextafter(k, -torch.inf))
        vals.append(_nextafter(k, torch.inf))
        for off in (1e-12, 1e-9, 1e-6):
            vals.append(k - off)
            vals.append(k + off)
    g: torch.Generator = torch.Generator().manual_seed(seed)
    uni: torch.Tensor = 500.0 * torch.rand(n_random, generator=g)
    uni2: torch.Tensor = 5.0 * (2.0 * torch.rand(n_random, generator=g) - 1.0)
    logsp: torch.Tensor = 10.0 ** (-15.0 + 17.0 * torch.rand(2048, generator=g))
    out: torch.Tensor = torch.tensor(vals, dtype=torch.float64)
    out = torch.cat(
        [out, uni, uni2, logsp, torch.tensor([float("nan"), float("inf"), float("-inf")])]
    )
    return out


@dataclass(frozen=True)
class Case:
    """One (zone, forcing, dt) configuration to sweep."""

    zone: HydrologicZone
    d: HydroForcing
    dt: float
    s_0: float
    label: str
    kinks: tuple[float, ...]


def build_cases() -> list[Case]:
    """All zone / parameter / forcing combinations, with the kink locations
    each case must exercise (``None`` kinks are omitted)."""
    cases: list[Case] = []

    def add(
        zone: HydrologicZone,
        label: str,
        kinks: tuple[float | None, ...],
        *,
        precip: float = 0.0,
        temp: float = 0.0,
        pet: float = 0.0,
        q_in: float = 0.0,
        dt: float = 1.0,
        s_0: float = 25.0,
    ) -> None:
        cases.append(
            Case(
                zone=zone,
                d=HydroForcing(precip=precip, temp=temp, pet=pet, q_in=q_in),
                dt=dt,
                s_0=s_0,
                label=label,
                kinks=tuple(k for k in kinks if k is not None),
            )
        )

    # ---- SnowZone: melt tie at s == fmax*(t-tt) when t > tt ------------- #
    for tt, fmax, t, tag in [
        (0.0, 1.0, -2.0, "no-melt"),
        (0.0, 1.0, 0.0, "t==tt"),
        (0.0, 1.0, 1e-6, "t~tt"),
        (0.0, 1.0, 1.5, "melt"),
        (0.5, 2.5, 0.5, "t==tt(0.5)"),
        (0.5, 2.5, 3.0, "melt(0.5)"),
    ]:
        melt: float | None = fmax * (t - tt) if t > tt else None
        add(
            SnowZone(tt=tt, fmax=fmax),
            f"snow tt={tt} fmax={fmax} t={t} ({tag})",
            (melt,),
            precip=2.0,
            temp=t,
            pet=1.0,
            q_in=0.5,
        )

    # ---- SurfaceZone: ET tie s == fc*lp; quickflow kink s == thr;
    #      percolation pow-at-zero (beta < 1) ----------------------------- #
    for fc, lp, beta, k0, thr, tag in [
        (100.0, 0.5, 0.4, 0.1, 10.0, "beta<1"),
        (100.0, 0.5, 1.0, 0.1, 10.0, "beta=1"),
        (250.0, 0.2, 2.0, 0.5, 25.0, "beta=2"),
        (100.0, 0.5, 0.4, 0.0, 0.0, "k0=0 (always-tie quickflow)"),
        (10.0, 1.0, 0.05, 0.02, 3.0, "small"),
    ]:
        add(
            SurfaceZone(fc=fc, lp=lp, beta=beta, k0=k0, thr=thr),
            f"surface fc={fc} lp={lp} beta={beta} k0={k0} thr={thr} ({tag})",
            (fc * lp, thr, 0.0),
            pet=2.0,
            q_in=3.0,
        )

    # ---- GroundZone: gate edge s == 1e-12; clamp edge s == perc --------- #
    for k, alpha, perc, tag in [
        (0.5, 0.4, 1.0, "alpha<1"),
        (1e-2, 1.0, 0.3, "alpha=1"),
        (0.2, 2.0, 2.5, "alpha=2"),
    ]:
        add(
            GroundZone(k=k, alpha=alpha, perc=perc),
            f"ground k={k} alpha={alpha} perc={perc} ({tag})",
            (1e-12, perc, 0.0),
            q_in=1.0,
        )

    # ---- GroundZone with a *Tensor* perc: the clamp_max backward then
    #      takes the min/max-style subgradient path (0.5 tie at s == perc,
    #      nan -> 1) instead of the float-threshold hard step ----------- #
    add(
        GroundZone(k=0.5, alpha=0.4, perc=torch.tensor(1.0, dtype=torch.float64)),
        "ground k=0.5 alpha=0.4 perc=1.0 (Tensor perc)",
        (1e-12, 1.0, 0.0),
        q_in=1.0,
    )

    # ---- GroundZoneB: gate edge s == 1e-12 ------------------------------ #
    for k, alpha, tag in [(0.5, 0.4, "alpha<1"), (1e-2, 1.0, "alpha=1"), (0.2, 2.0, "alpha=2")]:
        add(
            GroundZoneB(k=k, alpha=alpha),
            f"groundB k={k} alpha={alpha} ({tag})",
            (1e-12, 0.0),
            q_in=1.0,
        )

    # ---- dt / s_0 variants on one representative zone each ------------- #
    for dt, s_0 in [(0.5, 0.0), (1.0, 0.0), (2.0, 100.0)]:
        add(
            SurfaceZone(fc=100.0, lp=0.5, beta=0.4, k0=0.1, thr=10.0),
            f"surface-dt dt={dt} s0={s_0}",
            (50.0, 10.0, 0.0),
            pet=2.0,
            q_in=3.0,
            dt=dt,
            s_0=s_0,
        )
        add(
            GroundZoneB(k=0.5, alpha=0.4),
            f"groundB-dt dt={dt} s0={s_0}",
            (1e-12, 0.0),
            q_in=1.0,
            dt=dt,
            s_0=s_0,
        )

    # ---- non-dyadic dt = 1.37 (the dt-folding discriminator), all zones - #
    add(
        SnowZone(tt=0.0, fmax=1.0),
        "snow dt=1.37 (non-dyadic)",
        (1.5,),
        precip=2.0,
        temp=1.5,
        pet=1.0,
        q_in=0.5,
        dt=1.37,
    )
    add(
        SurfaceZone(fc=100.0, lp=0.5, beta=0.4, k0=0.1, thr=10.0),
        "surface dt=1.37 (non-dyadic)",
        (50.0, 10.0, 0.0),
        pet=2.0,
        q_in=3.0,
        dt=1.37,
    )
    add(
        GroundZone(k=0.5, alpha=0.4, perc=1.0),
        "ground dt=1.37 alpha<1 (non-dyadic)",
        (1e-12, 1.0, 0.0),
        q_in=1.0,
        dt=1.37,
    )
    add(
        GroundZoneB(k=0.5, alpha=0.4),
        "groundB dt=1.37 alpha<1 (non-dyadic)",
        (1e-12, 0.0),
        q_in=1.0,
        dt=1.37,
    )

    return cases


def check_residual_deriv(case: Case, n_random: int, seed: int) -> int:
    """Sweep one case: analytic ``residual_deriv`` vs ``autograd.grad`` of
    ``residual``, lane-by-lane exact comparison. Returns the mismatch count."""
    s: torch.Tensor = build_s_grid(list(case.kinks), n_random, seed)
    s_leaf: torch.Tensor = s.clone().detach().requires_grad_(True)

    r_ad: torch.Tensor = case.zone.residual(s_leaf, case.s_0, case.d, case.dt)
    g_ad: torch.Tensor = torch.autograd.grad(
        r_ad, s_leaf, grad_outputs=torch.ones_like(r_ad), create_graph=False
    )[0]
    g_an: torch.Tensor = case.zone.residual_deriv(
        s.detach(), case.s_0, case.d, case.dt
    )

    match: torch.Tensor = (g_ad == g_an) | (g_ad.isnan() & g_an.isnan())
    bad: torch.Tensor = ~match
    n_bad: int = int(bad.sum().item())
    if n_bad:
        idx: torch.Tensor = bad.nonzero(as_tuple=True)[0][:10]
        print(f"  MISMATCH ({n_bad} of {s.numel()} lanes) in: {case.label}")
        for i in idx.tolist():
            print(
                f"    s={s[i].item():+.6e}  autograd={g_ad[i].item():+.10e}  "
                f"analytic={g_an[i].item():+.10e}"
            )
    return n_bad


def check_primitives(n_random: int, seed: int) -> int:
    """Unit-level parity of the ``_d_*`` subgradient primitives against
    autograd, over dense (x, c) pairs including ties and tensor ``c``."""
    g: torch.Generator = torch.Generator().manual_seed(seed)
    x: torch.Tensor = torch.cat(
        [
            torch.tensor(
                [-2.0, -1e-9, float("-0.0"), 0.0, 1e-9, 0.5, 1.0, 1.5, 3.0],
                dtype=torch.float64,
            ),
            torch.rand(n_random, generator=g) * 4.0 - 2.0,
        ]
    )
    cvals: list[float] = [-1.0, -0.25, 0.0, 0.5, 1.0, 1.000001, 2.0]
    total_bad: int = 0
    # minimum/maximum run their tensor-tensor form (as the zone flux laws
    # do); clamp runs *both* threshold dispatches, since its backward
    # depends on the threshold's type (float = hard step, Tensor =
    # min/max-style subgradient).
    ops: list[tuple[str, str]] = [
        ("minimum", "0d-tensor"),
        ("maximum", "0d-tensor"),
        ("clamp_max", "scalar"),
        ("clamp_max", "0d-tensor"),
        ("clamp_min", "scalar"),
        ("clamp_min", "0d-tensor"),
    ]
    for op, ckind in ops:
        for c in cvals:
            c_t: torch.Tensor = torch.tensor(c, dtype=torch.float64)
            if ckind == "scalar":
                c_val: float | torch.Tensor = c
            else:
                c_val = c_t
            xt: torch.Tensor = x.clone().detach().requires_grad_(True)
            if op == "minimum":
                y: torch.Tensor = torch.minimum(xt, c_val)
                g_an: torch.Tensor = _d_min(x, c_val)
            elif op == "maximum":
                y = torch.maximum(xt, c_val)
                g_an = _d_max(x, c_val)
            elif op == "clamp_max":
                y = xt.clamp_max(c_val)
                g_an = _d_clamp_max(x, c_val)
            else:
                y = xt.clamp_min(c_val)
                g_an = _d_clamp_min(x, c_val)
            g_ad: torch.Tensor = torch.autograd.grad(
                y, xt, grad_outputs=torch.ones_like(y), create_graph=False
            )[0]
            match: torch.Tensor = (g_ad == g_an) | (g_ad.isnan() & g_an.isnan())
            n_bad: int = int((~match).sum().item())
            total_bad += n_bad
            if n_bad:
                idx: torch.Tensor = (~match).nonzero(as_tuple=True)[0][:8]
                print(f"  MISMATCH ({n_bad}) primitive={op} ({ckind} c) c={c}")
                for i in idx.tolist():
                    print(
                        f"    x={x[i].item():+.6e}  autograd={g_ad[i].item():+.10e}  "
                        f"analytic={g_an[i].item():+.10e}"
                    )
    return total_bad


def check_pow(n_random: int, seed: int) -> int:
    """Parity of the ``pow`` backward slope ``_d_pow(x, b) = b*x**(b-1)``
    against autograd: ``autograd.grad(x**b, x, grad_outputs=g) == g * _d_pow``
    (ATen's ``g * (b * x**(b-1))`` order), over an exponent grid (incl.
    fractional and non-dyadic ``b`` and a 0-dim *tensor* ``b``) and an
    upstream-gradient grid (1-dim varying / constant / zero, and a 0-dim
    constant routed through a mul node, as the zone chains do)."""
    g: torch.Generator = torch.Generator().manual_seed(seed)
    xs: list[float] = [
        float("-inf"), float("-0.0"), 0.0, 1e-300, 0.25, 0.5, 1.0, 1.37,
        3.7, 1e300, float("inf"), float("nan"),
    ]
    x: torch.Tensor = torch.cat(
        [
            torch.tensor(xs, dtype=torch.float64),
            torch.rand(n_random, generator=g) * 4.0 - 2.0,
        ]
    )
    n: int = x.numel()
    total_bad: int = 0
    for b in (0.25, 0.4, 0.5, 0.75, 1.0, 1.37, 1.5, 2.0, 3.0):
        g_vary: torch.Tensor = 1.0 + 0.001 * torch.arange(n, dtype=torch.float64)
        g_neg: torch.Tensor = -1.37 * torch.ones(n)
        g_zero: torch.Tensor = torch.zeros(n)
        g_0d: torch.Tensor = torch.tensor(-3.151)
        # 0-dim upstream gradient: route through a 0-dim mul node (a
        # gradient identity) so autograd's 0-dim path is exercised.
        for gname, gout in [
            ("vary-1d", g_vary),
            ("neg-1d", g_neg),
            ("zero-1d", g_zero),
        ]:
            xt: torch.Tensor = x.clone().detach().requires_grad_(True)
            ref: torch.Tensor = torch.autograd.grad(
                xt ** b, xt, grad_outputs=gout, create_graph=False
            )[0]
            got: torch.Tensor = gout * _d_pow(x, b)
            match: torch.Tensor = (ref == got) | (ref.isnan() & got.isnan())
            n_bad: int = int((~match).sum().item())
            total_bad += n_bad
            if n_bad:
                idx: torch.Tensor = (~match).nonzero(as_tuple=True)[0][:8]
                print(f"  MISMATCH ({n_bad}) pow b={b} g={gname}")
                for i in idx.tolist():
                    print(
                        f"    x={x[i].item():+.6e}  autograd={ref[i].item():+.10e}  "
                        f"analytic={got[i].item():+.10e}"
                    )
        for bname, b_t in [("float", b), ("0d-tensor", torch.tensor(b))]:
            xt = x.clone().detach().requires_grad_(True)
            y0d: torch.Tensor = (xt ** b_t) * g_0d
            ref0d: torch.Tensor = torch.autograd.grad(
                y0d, xt, grad_outputs=torch.ones_like(y0d), create_graph=False
            )[0]
            got0d: torch.Tensor = g_0d * _d_pow(x, b_t)
            match = (ref0d == got0d) | (ref0d.isnan() & got0d.isnan())
            n_bad = int((~match).sum().item())
            total_bad += n_bad
            if n_bad:
                idx = (~match).nonzero(as_tuple=True)[0][:8]
                print(f"  MISMATCH ({n_bad}) pow b={bname} b={b} g=const-0d")
                for i in idx.tolist():
                    print(
                        f"    x={x[i].item():+.6e}  autograd={ref0d[i].item():+.10e}  "
                        f"analytic={got0d[i].item():+.10e}"
                    )
    return total_bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-random", type=int, default=2048,
                        help="random lanes per sweep grid (default 2048)")
    args = parser.parse_args()

    print("=" * 78)
    print("Hydro analytic-derivative parity (analytic vs torch.autograd.grad)")
    print("=" * 78)

    print("\n[1] _d_min / _d_max / _d_clamp_max / _d_clamp_min primitives")
    prim_bad: int = check_primitives(args.n_random, seed=7)
    print(f"  primitive mismatches: {prim_bad}")

    print("\n[2] _d_pow backward slope (exponent grid x upstream-gradient grid)")
    pow_bad: int = check_pow(args.n_random, seed=11)
    print(f"  pow mismatches: {pow_bad}")

    print("\n[3] residual_deriv per zone (dense kink grid + random sweeps)")
    cases: list[Case] = build_cases()
    total_bad: int = 0
    for i, case in enumerate(cases):
        n_bad: int = check_residual_deriv(case, args.n_random, seed=1000 + i)
        total_bad += n_bad
        print(f"  {'OK  ' if n_bad == 0 else 'FAIL'} {case.label}")

    print("\n" + "=" * 78)
    total: int = total_bad + prim_bad + pow_bad
    print(
        f"RESULT: {total} mismatched lane(s) "
        f"({prim_bad} primitive + {pow_bad} pow + {total_bad} zone-case)"
        f" across {len(cases)} case(s)"
    )
    if total == 0:
        print("PASS: analytic derivative matches autograd bit-for-bit everywhere,")
        print("      including all kinks (ties, clamp steps, pow-at-zero, NaNs).")
    print("=" * 78)
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
