# potions-py — Project Plan

## Goal
Re-implement the `potions` hydrology / reactive-transport ODE model (Rust core + Python wrappers in
`/home/andrew/Documents/Research/Projects/potions`) as a standalone, **pure Python + PyTorch** package:
no Rust anywhere, Jacobians computed by **autograd** (no finite differences), supporting **batched
parameter sets** and **GPU** acceleration.

## Environment / constraints (user-confirmed)
- Package name: `potions`; location: `/home/andrew/Documents/Research/Projects/potions-py`.
- Isolated venv (uv, Python 3.12 — system python is 3.12.9; old compiled `potions.core` is
  cpython-3.13-only, so it cannot be imported here and must not shadow; validation uses ported
  hand-computed tests, and optionally a one-time disposable 3.13 env for Rust baseline fixtures).
- Batch semantics: forcing (RtForcing, HydroForcing, dt) shared across the batch dim; only
  parameter sets (model params) vary.
- Precision: `float64` on CPU (default, for parity); opt-in `float32` on GPU.
- Style (AGENTS.md): functional where possible; type hints on everything (builtin generics);
  comprehensive docstrings; equation/implementation notes written **as** work is done;
  no `_internal(...)` methods.

## Public API (parity with old `src/potions`)
- `HydroModel` — run batched hydrology; `HydroForcing`, `HydroModelParameters` (pydantic).
- `RtModel` — run batched reactive transport; `RtForcing`, `RtModelParameters`.
- `ReactionNetwork` (+ `default_network(s)`s e.g. carbon network, soil_network).
- `database` module: `Database` (pydantic, loadable from `default_database.json`).
- Zoned components: `SnowZone`, `SurfaceZone`, `GroundZone`, `GroundZoneB`, `RtZone`, `RiverZone`.
- Solvers: `find_root`, `bisect`, `find_root_multi`, `levenberg_marquardt` (torch, autograd).
- Kinetics: `MonodParameters`, `TstParameters`, `EquilibriumParameters`.

## Architecture (all math in torch, batched over `[B, ...]`)
- `math.py` — root finders (secant + bisection fallback), multi-dim damped Newton with autograd
  Jacobian (`torch.autograd.functional.jacobian` or per-col `grad`), least-squares fallback solve,
  Levenberg-Marquardt. Replaces `rust/src/math.rs`; `approx_fprime` (finite differences) is dropped.
- `common_types.py` — `HydroForcing`, `HydroStep`, `RtForcing`, `RtStep`, `MiscData`,
  `LapseRateParameters`, `ZERO_CONC = 1e-20`.
- `database.py` — pydantic `Database` mirroring `default_database.json` schema (species molar mass,
  charge, size param, stoichiometry, eq consts, molar volumes, kinetics constants).
- `reaction_network.py` — species ordering [primary aq, secondary, mineral, exchange];
  stoich matrices, total/conservation matrix incl. charge-balance row; parameter matrices.
- `kinetic_structures.py` —
  - `EquilibriumParameters`: speciation. log10 c = Z@x + pinv(S)@logK with Z = null(S);
    residual(x) = c_tot_observed − T@c(x); solve by Newton (autograd J). Replaces manual linear null-space
    speciation solve — the main autograd win.
  - `MonodParameters`: rate = k · ∏ C_j/(m_ij + C_j) · ∏ i_ij/(i_ij + C_j) (per mineral);
    NaN entries = unused.
  - `TstParameters`: Q = 10^(Σ s_ij log10 C_j), D = 10^(Σ d_ij log10 C_j),
    rate = D(1 − Q/K_eq). NaN hazard: log10(0)·0 = NaN in IEEE — mirror baseline init convention.
  - `MineralParameters`: aux rate = soil_water_factor · q10^((T−20)/10) with piecewise water-content
    factor; mineral rate = 86400 · rate_const · ssa · molar_mass · conc · aux.
- `hydro.rs → hydro.py` — `HydrologicZone.implicit_euler_step`: residual (s0 − s) + dt·mass_balance(s, d);
  `find_root` then `bisect(f, 1000.0)` fallback; `s_new = max(root, 0)`.
  Zones: `SnowZone` (accumulation/melting), `SurfaceZone(fc, lp, beta, k0, thr)`
  (vert = q_in(s/fc)^beta), `GroundZone` (perc = min(s, ...)), `GroundZoneB`.
- `rt_zone.py` / `river_zone.py` — implicit Euler over full state vector:
  residual = c0 − c + dt·R(c); transport term for aqueous = (q_out/v)·(c0 − c) when |v| ≥ 1e-6 else 0;
  mineral moles = c · depth (aqueous moles = c · v); v = 0 → conc = 1e-20 (ZERO_CONC), state frozen;
  optional post-step `solve_equilibrium`; outputs: lat/vert flux mass = c_after·q_int·frac,
  conc = mass/q, mineral rates.
  NOTE: `RiverZone` slices minerals `[num_spec − num_min:]` vs `RtZone` `[num_aq:num_aq+num_min]` —
  port as-is (matches baseline), document quirk.
- `model.py` — batched orchestration: stack RtModelParameters over B, run zone steps, collect
  RtSteps; forces per-column time evolution (coupling river/RT zones via flows).

## Module build order
1. Scaffold: pyproject (`potions`, src layout, deps torch/numpy/pandas/scipy/pydantic/pytest),
   uv venv, package skeleton, `PROGRESS.md`.
2. `common_types.py`
3. `math.py` (autograd Newton et al.)
4. `database.py`
5. `reaction_network.py`
6. `kinetic_structures.py` (equilibrium speciation first — core of the autograd win)
7. `rt_zone.py`, `river_zone.py`
8. `hydro.py`
9. `model.py` orchestration
10. Tests: port `test_zones.py` (SnowZone 1.0; SurfaceZone state ≈ 49, vert 0.5, vap 1),
    `test_equilibrium.py` (A⇌B logK=log10 2 total 1 → [1/3, 2/3]; carbonate),
    `test_kinetics.py` (Monod 2/(1+2)=2/3; inhibition).

## Validation strategy
- Ported hand-computed unit tests (no baseline required).
- Optional: disposable Python 3.13 env → run old Rust potions → save golden fixtures
  (states/fluxes for example networks) → compare in-pipeline here.

## Batched design
- All zone residual/rate functions take tensors of shape `[B, n]`; forcing `[B, ...]` (broadcasting
  allowed); parameters `[B, ...]`. GPU: move everything to `cuda`, float32 option.
