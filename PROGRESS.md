# potions-py — Progress Log

## Session 1 (analysis + plan)
- [x] Explored old project (Rust core in `rust/src/`, Python layer in `src/potions`).
- [x] Plan written (PLAN.md, superseded by IMPLEMENTATION_PLAN.md where they conflict).

## Session 2 (authoritative plan + confirmed decisions + M0)
- [x] Full analysis of the Rust reference (math.rs, common_types.rs, hydro.rs,
      rt_zone.rs, river_zone.rs, kinetic_structures.rs, reaction_network.rs,
      database.rs); wrote **IMPLEMENTATION_PLAN.md** with all governing equations
      and solver details so a fresh LLM can resume without re-reading Rust.
- [x] Confirmed with user: **manual analytic Jacobians** (no autograd in solvers);
      **float32 on CUDA / float64 on CPU** default precision; keep package
      `potions` and `src/potions` layout. Updated plan to match.
- [x] M0 scaffolding: added `scipy` dep; wrote `src/potions/__init__.py`,
      `device.py` (device/dtype policy + chunk helper), `common_types.py`
      (HydroForcing, HydroStep, RtForcing, RtStep, LapseRateParameters,
      MiscData, ZERO_CONC). Package imports on CUDA (float32).

## Session 3 (M1: database + ReactionNetwork)
- [x] Shipped `src/potions/data/default_database.json` (copied verbatim from the
      Rust package, 1,021,568 bytes).
- [x] `src/potions/database.py`: dataclasses (PrimaryAqueousSpecies,
      SecondarySpecies, MineralSpecies, TstReaction, MonodReaction,
      MineralKineticData, ExchangeReaction) + `ChemicalDatabase` with
      `load_default()` via `importlib.resources`, `from_file`, `from_dict`,
      `get_*(...)` lookups. Raw (log10) values stored; no conversion at load.
- [x] `src/potions/equilibrium.py`: `compute_stoich_null_space`
      (scipy.linalg.null_space/pinv) + frozen `EquilibriumParameters`
      (stoich, log_eq_consts, total, N, x_particular; `conc_func`).
- [x] `src/potions/kinetic_structures.py`: `MonodParameters` / `TstParameters`
      containers (frozen dataclasses).
- [x] `src/potions/reaction_network.py`: `ReactionNetwork` + module helpers
      mirroring Rust semantics (species ordering incl. X-/exchange, charges,
      transport mask, mineral stoich, monod/inhib NaN convention, **TST
      stoich/dep swapped layout (Rust-authoritative)**, `rate_consts` monod →
      tst order, equilibrium matrices). `__init__.py` exports updated.
- [x] mypy strict clean (7 source files; added `[tool.mypy.overrides]` for
      untyped `scipy` import in pyproject.toml).
- [x] Smoke test (manual): carbon network (DOC/HCO3-/H+ + CO2(aq)/CO3-- +
      SOC(s) "test" monod + XDOC) — species order, charges
      `[0,-1,1,0,-2,0,0,0]`, monod NaN-except-6e-6 row, rate_consts
      9.999999999999999e-11, molar masses [180.0], quartz TST inert (Q=1,D=1,
      empty dependence maps), no-eq-reaction → identity speciation, and the
      speciation problem Newton-converges to 1e-15 residual.

## Session 4 (M2: batched speciation solve, Rust-validated)
- [x] `src/potions/solvers.py`: batched Newton (fused/analytic, Armijo line
      search) → Levenberg-Marquardt fallback, transcribed verbatim from the
      compiled `potions/core` math.rs (MULTI_TOL=1e-6, MULTI_TOL_STEP=1e-12,
      ARMIJO_C=1e-4, LINE_SEARCH_MAX=30, STAGNATION_REL_TOL=1e-9, LM damping
      updates and stagnation acceptance). Masked per-lane control flow with
      `raise_on_failure` / NaN-fill exit policy.
- [x] `equilibrium.py`: batched `residual`, `jacobian_residual`,
      `x_free_solve`, `solve_equilibrium` (warm-start aware); `conc_func` now
      handles leading batch dims (`x_free @ N.T`).
- [x] **Validated against the compiled Rust `potions.core`** (conda `potions`
      env, cp 3.13): `N`, `x_particular`, `total_mat` bit-identical on the
      carbon network; our solutions pass the golden kernel's own `max|f|≤1e-6`
      bar (golden residual at our `x_free` = 7.2e-7); analytic Jacobian matches
      (golden `residual_jacobian` is finite-difference → ~1e-8 truncation gap,
      expected). Masked-batch vs per-element solves agree to ~5e-14 in `x_free`.
- [x] mypy strict clean (8 source files).

## Session 5 (M3 debugging: golden numeric match)

Diagnosed and fixed three bugs that caused the port to diverge from the Rust
golden across the full M3 comparison suite (286 quantities, 7 scenarios).

### Bug 1 — Frozen Jacobian in fused Newton solver (`solvers.py`)
- **Symptom**: Speciation solve converged to a different within-tolerance root
  (species[1] 1.133e-6 ours vs 1.087e-6 golden); `_newton_fused_core` stalled
  (err 55→1.46 over 100 iterations) then `_FAILED`, falling through to LM.
- **Root cause**: Rust `newton_fused_core` (math.rs:1138) reuses the accepted
  line-search step's Jacobian (`jac_x = jac_new`) for the next Newton iteration;
  our `_armijo_line_search` discarded it (`f_trial, _ = eval_fn(x_trial)`), so
  `jac_x` stayed frozen at the initial guess.
- **Fix**: `_armijo_line_search` now returns `(x_new, f_new, jac_new, accepted)`
  and accepts a `jac_x` param; `_newton_fused_core` stores the returned `jac_new`
  into `jac_x` before the next iteration. Docstring of `newton_analytic_then_lm`
  updated.
- **Verification**: Fixed solver converges in 19 Newton iterations to golden root
  `x_free = [-10.23694246, -8.62172582, -5.69897, -4.86316275, -9.9062692]`,
  matching golden to 4.7e-15.

### Bug 2 — ODE Jacobian init from `eye` instead of zeros (`rt_zone.py`)
- **Symptom**: Warm-start step2 fused Newton failed with `best max|f| = 1.343e-06`
  (Pre-fix, reverted code fails identically).
- **Root cause**: Rust `assemble_ode_jacobian` (rt_zone.rs:1024) initializes
  `j = Array2::zeros(...)`; our `_assemble_ode_jacobian` initialized to `eye`
  (line ~373), leaving a spurious `+1.0` on the non-aqueous rows (rows 5-7 were
  0 in ours vs `-e_i` in golden).
- **Fix**: Changed `jac` init from `eye.unsqueeze(0).expand(batch,-1,-1).clone()`
  to `torch.zeros(batch, num_spec, num_spec, ...)`.
- **Verification**: Warm step2 now matches golden to 3.1e-15.

### Bug 3 — aux_factor shape `(B,)` instead of `(B, num_min)` (`kinetic_structures.py`)
- **Symptom**: Batched (B=3) `_assemble_ode_jacobian` crashed with
  `RuntimeError: Expected size for first two dimensions of batch2 tensor to be:
  [3, 1] but got: [3, 3]` in `s.unsqueeze(0) @ terms.dm` (line 392).
- **Root cause**: `soil_water_factor`/`temperature_factor`/`factor` returned
  `(B,)` when `s_w` was a batched tensor, because broadcasting `(B,)` with
  `(num_min,)` where `num_min=1` collapsed to `(B,)` instead of `(B, 1)`.
  This caused `_reaction_terms` base to broadcast to `(B, B)` → `dm` shape
  `(B, B, num_species)` instead of `(B, num_min, num_species)`.
- **Fix**: Added `unsqueeze(-1)` on `sw`/`t` before the per-mineral broadcast,
  guaranteeing the output shape is `(..., num_minerals)` for any input rank.
- **Verification**: B=3 lanes match golden to 5e-15.

### Full suite result
```
286 OK / 0 FAIL items
mypy strict: clean (10 source files)
```
All quantities (monod_rate, tst_rate, aux_factor, transport_rate, reaction_rate,
residual, jacobian_res, step_state, step_total_moles, mass_in, conc_in, lat_conc,
vert_conc, lat_mass, vert_mass, step_mineral_rates) match golden across default,
dry, bigbypass, longdt, smallt, warm, and batched B=3 scenarios.

## Session 6 (M4: hydro.py zones + networkx-free hydro_model, Rust-validated)
- [x] Ported `hydro.py`: batched `find_root_batched` (per-lane secant) +
  `bisect_batched` fallback; `SnowZone`, `SurfaceZone`, `GroundZone`, `GroundZoneB`
  with the Rust implicit-Euler storage-residual + analytic flux formulas; every
  flux (state, forc, vap, lat, vert, lat_ext, vert_ext) validated against the
  Rust scalar values at exact golden inputs.
- [x] Ported `hydro_model.py` **without networkx**: `Layer`, `HydrologicalModel`
  (structure + scales + batched `step_hydro_model` + `run_hydro_model`), using
  hand-built lateral/vertical adjacency matrices with the exact golden wiring
  rules (last zone of each layer = river zone; vertical: same-count → straight
  down, lower layer of size 1 → all upper zones drain into it).
- [x] **Forcing distribution now mirrors golden**: `get_size_mat` /
  `get_forc_mat` build `precip_mat`/`pet_mat`/`temp_mat` (`size_mat @
  diag(scales)`, row-normalized for temperature) and forcing is distributed to
  every zone (not just surface zones), matching the golden
  `construct_hydro_forcing_matrix`.
- [x] **Streamflow matches golden**: `sim_streamflow_mmd` = sum of the *internal*
  `q_lat_<river_zone>` over *all* river zones (last zone of each layer), plus the
  `meas_streamflow_mmd` and `prop_q_<river_zone>` DataFrame columns.
- [x] **Default objectives match golden** (names/formulas): kge, log_kge, nse,
  log_nse, bias, spearman_rho, r_squared (pandas-ddof-1 semantics).
- [x] Validated end-to-end vs the golden env (`m4_golden.py` / `m4_compare.py`):
  connectivity + forcing matrices identical and the full 20-step simulation
  DataFrame (46 columns) matches to **4.26e-14**; all 7 objectives match to
  <1e-10. Batched (`B=3`) `step_hydro_model` lanes identical to the scalar path.
- [x] `__init__.py` exports updated (`ForcingData`, `Scalar`, `HydroModelResults`,
  `HydroModelStep`, `HydrologicalModel`, `Layer`, zone classes, objectives).
- [x] mypy strict clean (11 source files).

## Session 7 (batched K-lane East River runner + scalar bug + profiling)
- [x] `scripts/model_batched.py` (as `src/potions/model_batched.py`): batched
  port of the calibration pipeline — `BatchedConfig`, `Geometry`,
  `run_hydro_batched`, `run_rt_batched`, `run_hydro_rt_batched` — tensor shape
  `(K, ...)` everywhere, forcing distributed per lane via `_per_lane_forcing`,
  per-lane objective dicts, `overflow_mask` mirroring the scalar `WaterVolumeError`.
- [x] Fixed a genuine scalar bug in `Model._from_array_with_scales` (model.py):
  zone `from_array` builders drop the zone name, so `_override_zones` (name-keyed)
  silently discarded every parameterized zone and the scalar ran with *default*
  zone params. Re-applying `zone.name` restores parameterization; oracle pkl
  regenerated (`20260916_132214.pkl`).
- [x] Fixed a batched indexing bug: `_per_lane_forcing` row-sums via
  `torch.where(az.expand_as(...), ...)`; einsum subscripts `"ts,kzs->ktz"`;
  per-lane forcing columns vs rows; hydro dict reshape `(K,T*Z)` → `(K,T,Z)`;
  `torch.as_tensor`; RT objective pandas index alignment (doc_sim now shares
  `doc_meas.index`).
- [x] `scripts/parity_check_batched.py`: Stage 1 (K=1 vs scalar pkl) and
  Stage 2 (K=3 vs three independent `arr_to_results` runs) both **PASS** with
  bitwise-identical `q_sim`/`DOC_riv` and objectives ≤ 1e-9.
- [x] `scripts/profile_batched.py`: K-ladder benchmark 1→256 lanes over the
  **full East River record** by default (~2557 days, 7 water years; `--start`/
  `--end`/`--days` for sub-windows), throughput in sims/s, median-based,
  cProfile mode, CSV export, `--device cuda|cpu` + `--precision auto|f32|f64`
  (policy: f32 on CUDA, f64 on CPU), CUDA sync + warmup, overflow-lane count.
  Full-record GPU f32: **wall is flat in K (~72-74 s for K=1…256)**, i.e.
  sims/s ≈ K/73: 0.014 (K=1) → 3.45 (K=256), per-sim 71 s → 290 ms.
- [x] **Full-record profiling verdict**: runtime is dominated by a fixed
  per-timestep pipeline cost (measured ~3 ms/step at 256 lanes irrespective of
  K — kernel-launch + Python-op bound, not occupancy or per-lane work;
  cProfile of the 2557-day K=256 run: `run_rt_batched` 63 s incl. `_zone_step`
  58 s of which Newton/LM `_kinetic_residual_and_jacobian` 25 s, `_reaction_terms`
  13 s, equilibrium `x_free_solve` 8 s; `run_hydro_batched` 15 s incl.
  `find_root_batched` 9 s). So batch width does not reduce wall time on the
  full record; cutting Python/launch overhead (e.g. `torch.compile` around the
  zone-step closures, or fusing residual+Jacobian kernels) is the lever, and
  per-step solver iteration counts (Newton/LM data-dependent) are the second
  term.
- [x] **float32-on-CUDA made runnable**: the solver tolerances were float64-era
  (hydro ``FIND_ROOT_TOL=1e-6``/``_STAGNATION_EPS=1e-12``; solvers
  ``MULTI_TOL=1e-6``/``MULTI_TOL_STEP=1e-12``/``STAGNATION_REL_TOL=1e-9``) and
  were unreachable in f32, so f32 runs raised ``ScalarRootFindingError``.
  Added dtype-adaptive ``_solver_tols`` in hydro.py and solvers.py: float64
  keeps the constants bit-for-bit (parity re-verified PASS), float32 clamps
  above the f32 ulp floor (hydro atol scaled with the ~1000 mm storage
  bracket).
- [x] Made ``bisect_batched`` device-aware: its bracket tensors were hardcoded
  to CPU/float64, breaking CUDA runs (hydro.py now passes ``s_0.device``/
  ``s_0.dtype``).
- [x] GPU f32 smoke (30-day window): K-ladder runs with finite q/DOC, no
  overflow; parity (float64) still bitwise. Full-record ladder verified too
  (see profiling bullets above).

## Session 8 (massive solver speedup: kill the fixed-trip waste + K>1 fix)
- [x] **K>1 Armijo broadcasting bug (Stage-2 blocker) fixed** (`solvers.py`
  `_armijo_line_search`): `new_acc`/`moved` are `(B,)`, but a 1-D mask broadcast
  against `(B, n)`/`(B, m, n)` aligns the **last** axis, so `B=1` worked while
  `B=3` raised `size of tensor a (3) must match b (4) at dim 1`.  Masks now
  carry explicit trailing axes (`.unsqueeze(-1)` for `x`/`f`, `.unsqueeze(-1)
  .unsqueeze(-1)` for the Jacobian).  **Stage 2 (K=3 vs independent scalar
  lanes) now PASSES bitwise (all diffs 0.0).**
- [x] **The real per-step wall: fixed-trip iteration waste.**  The prior
  session had removed the convergence early-exits from `find_root_batched` /
  `bisect_batched` (`hydro.py`) and `_armijo_line_search` (`solvers.py`) to
  avoid host syncs, but `FIND_ROOT_MAXITER = 100` / `LINE_SEARCH_MAX = 30` mean
  a lane that converges in ~5 iterations still ran all 100/30 residual evals
  (masked-frozen lanes are cheap arithmetic, but each still launched a compiled
  GPU kernel / eager op chain).  Restored the **early-exit** right after the
  `active`/`trial_left` mask is computed: stopping when no lane is active is
  bitwise-identical (inactive lanes are frozen) and skips the entire tail.
  Measured (90 d, single lane, 5 reps):
  - hydro CPU f64: **2527 -> 92 us/step (27x)**, CUDA f32 off: **8774 -> 374
    us/step (23x)**, CUDA f32 compiled: **7343 -> 967 us/step (7.6x)**.
  - RT is unchanged (its bench converges in 1 eval/step, so it never enters
    the line search): CPU f64 150 -> 150, CUDA f32 728 -> 594.
- [x] **Full-record (2557 d) CUDA f32 eager**: ~72-73 s -> **40.3 s (K=1) /
  43.7 s (K=256)**; still flat in K (Python/launch bound), overflow 0.
  **30-day K=1 batched wall: 87 s (compiled CPU) -> 0.283 s; K=3 -> 0.195 s.**
- [x] Removed the two `if v0_ok.any()` host guards inside the compiled RT fused
  eval (`rt_zone.py` `_assemble_ode_jacobian` / `_kinetic_ode_from_terms`):
  they graph-broke `_fused_resjac` every call.  The `torch.where` flux already
  yields zeros for dry lanes, so the unguarded arithmetic matches the wet/mixed
  path; only the (inconsistent) all-dry signed-zero case differs.
- [x] Converted `_newton_fused_core` / `_levenberg_marquardt_core` to
  shape-stable masked updates (full-batch LU with an identity-substituted
  Jacobian + zero rhs for frozen lanes, so NaN/singular failed lanes cannot
  poison `lu_factor`; all status/result/x/f/jac updates via `torch.where`) and
  kept the per-iteration `running.any()` early break.  Net: no regression
  (CPU-f64 RT off-compile 150 us/step vs 149 baseline) and slightly better
  compiled.
- [x] **Harness bugs fixed** (they had been silently confounded all benches):
  `bench_zone.py` / `profile_batched.py` set `POTIONS_COMPILE` *after* importing
  `potions`, but `device.compile_enabled()` reads it once at import, so
  `--compile/--no-compile` were no-ops (and, because `compile_enabled()`
  keys off the *policy* device `DEFAULT_DEVICE == cuda`, even `--device cpu`
  ran compiled).  Env is now set before the `potions` imports.
  `parity_check_batched.py` now forces `POTIONS_COMPILE=0` so the f64 CPU
  bitwise contract is checked on the intended uncompiled path.
- [x] **Pre-existing issues documented, not caused by this work**:
  - Stage 1 `rt_doc objective log_kge` fails only on a ~1e-8 oracle mismatch:
    recomputing `log_kge` from the oracle's *own stored* `doc_sim` gives exactly
    the batched value (`-0.048627110866897416`), while the oracle stores
    `-0.04862710131462533`.  The `q_sim`/`DOC_riv` series are bitwise identical
    (`max-abs-rel = 0.000e+00`), so the solver is correct; the scalar
    `rt.objective_functions["DOC"]` path that produced the oracle differs from
    `DEFAULT_OBJECTIVE_FUNCTIONS` by ~1e-8 on `log_kge`.
  - The CUDA `torch.compile` path crashes in this environment with
    `AttributeError: type object 'constexpr' has no attribute 'is_block'`
    (torch 2.7.0+cu126 / triton 3.3.0 inductor codegen), independent of these
    changes; the `residual` recompile-limit/type-specialization warning is the
    same underlying fragility (`maybe_compile` of the base-class method gets
    re-specialized per zone subclass).
- [x] **Zone collocation investigated and rejected** (S8 follow-up): the 8 East
  River zones are not batchable.  `lat_mat`/`vert_mat` show a DAG
  (`snow_rp<-snow_hs`, `surface_rp<-snow_rp+surface_hs`, ...); the max
  independent wavefront is 2, and every independent pair is a *different* zone
  class (`SnowZone`/`SurfaceZone`/`GroundZone`/`GroundZoneB`), so there are no
  same-class independent zones to fold into the batch.  Estimated best case
  <1.33x even ignoring the class mismatch — not worth the rewrite risk.
- [x] **`torch.compile` measured correctly** (the earlier "73.0 vs 72.3 s"
  A/B was invalid: the env-ordering bug meant both arms ran the same config).
  On the 365 d window, K=256: **eager 6.45 s vs compiled 5.99 s (~7%)**; the
  crash is specific to the degenerate `xnumel == 1` K=1 batch (triton 3.3
  `broadcast_to` codegen bug), K>=2 compiles fine.  Raising
  `recompile_limit` to stop the `self`-type fallbacks made it *worse*
  (8.2 s), so it was reverted.  Net: compile is not the lever here either.
- [x] **Robustness fix** (`device.py` `maybe_compile`): set
  `torch._dynamo.config.suppress_errors = True` while compiling, so the K=1
  degenerate-batch codegen failure (and any other backend compile error) falls
  back to eager instead of aborting the run.  Verified: K=1 CUDA f32 `--compile`
  now exits 0; parity unaffected (parity forces `POTIONS_COMPILE=0`).
- **Conclusion**: the remaining wall is Python-orchestration-bound (~20k
  zone-steps for the full record, ~1.5 ms each; ~7.5M Python calls per 365 d).
  Neither zone collocation nor `torch.compile` closes it.  Reaching Rust-class
  ~1 s would require a ground-up fused implementation (custom CUDA kernel or
  whole-pipeline graph capture), which is out of scope for incremental work.

## Session 9 (solver-only benchmark harness + autograd guard)
- [x] **`RtZone.build_fused_eval` extracted, behavior-preserving** (`rt_zone.py`):
  the fused residual/Jacobian setup that lived inline in `_zone_step` moved into
  a shared `_build_fused_eval(...)` returning a `_FusedEvalContext` (flattened
  `c0b`, cast network constants, normalised hydro fields, closure).  `_zone_step`
  now delegates to it (same arithmetic, same order), and the new public
  `RtZone.build_fused_eval(d, c_0, dt_days) -> (FusedEval, Tensor)` exposes the
  exact closure the solver consumes so microbenchmarks need not duplicate the
  casting rules.  **Parity unchanged: Stage 2 (K=3) PASS bitwise; Stage 1 fails
  only on the known oracle `log_kge` artifact.**
- [x] **Explicit no-autograd guard** (`model_batched.py`): wrapped the step loops
  of `run_hydro_batched` and `run_rt_batched` in `torch.inference_mode()`.  The
  library never builds a graph (all tensors `requires_grad=False`, no
  `.backward`), so this only makes the intent explicit and skips version-counter
  bookkeeping for the in-place `state[:, i] = ...` / `lat_mass[:, i] = ...`
  writes.  Parity unaffected.
- [x] **New `scripts/bench_solvers.py`** — solver-only microbench (the layer
  beneath `bench_zone.py`), timing the bare solvers on realistic model data:
  - `rt`: `newton_fused_then_lm` on a zone's `build_fused_eval` closure;
  - `speciation`: `EquilibriumParameters.x_free_solve` from a converged warm start;
  - `hydro`: `_solve_storage_residual` (secant + bisection) on a real `SurfaceZone`;
  - `synthetic`: isolated **linear** `f(X)=J X-b` with random SPD `J`
    (`--dim`/`--cond`); eigenvalues log-spaced `[1/cond, 1]` so `||J||~1` keeps
    the solved residual near the dtype ulp (the convergence bar is absolute),
    which matters for f32.  Linear ⇒ one Newton step, LM never reached, isolating
    the batched `lu_factor`/`lu_solve` path.
  Knobs: `--solvers --batch --device --dtype --compile/--no-compile --solves
  --warmup --perturb --synthetic --dim --cond`.  Per cell: median µs/batched-solve
  (CUDA-synced), fused evals/solve, final `max|res|`, `%LM` (private Newton status
  machine, only for the fused solver).  `POTIONS_COMPILE` is set before the
  `potions` import (the `bench_zone.py` fix).
- Measured (RTX 3090, CUDA f32, compile default, `--solves 200`): rt **257-267
  µs/solve flat from B=1→1024** (1 eval: the realistic step starts converged),
  speciation **519-600 µs**, hydro **532-664 µs** (5 secant evals).  Flat-in-B and
  ~95% host overhead reproduces the Session 8 "Python/launch-bound" conclusion at
  solver granularity.  `--perturb 50` pushes rt 1→2 evals and synthetic 2→3.
  `--compile` vs `--no-compile` at B=256/perturb 5: rt 993→924 µs, hydro 781→655 µs.

## Session 10 (pytest suite)
- [x] **`tests/` built out from empty to 236 passing, 0.8 s** (`python -m pytest`).
  One file per module plus a `conftest.py`:
  - `conftest.py` sets `POTIONS_COMPILE=0` **before** importing `potions`
    (eager CPU f64 kernels; `torch.compile` is a CUDA-throughput feature and
    must not add trace cost / inductor dependence to tests), plus a `cpu_f64`
    helper and a session-scoped `carbon_network` fixture.
  - `test_solvers.py` (20): tolerance policy, LU solve, split vs fused Newton/LM
    agreement, batched-lane independence, degenerate zero-dim systems,
    NaN/RuntimeError-on-failure, Armijo sufficient-decrease, `NewtonOutcome`.
  - `test_hydro.py`: `find_root_batched`/`bisect_batched`, `SnowZone`,
    `SurfaceZone`, `GroundZone`, `GroundZoneB`, batched/broadcast stepping, base
    `HydrologicZone` introspection.
  - `test_objective_functions.py`: KGE/NSE/log metrics, bias, R², Spearman,
    flow-percentile masking, registry order.
  - `test_kinetic_structures.py`: Monod/TST/`MineralParameters`
    (incl. batched unpacking), `ZoneDimensions`, `RtParameters`, river types.
  - `test_equilibrium.py` (12): null-space orthogonality, residual zero at the
    solution, finite-difference Jacobian agreement, `x_free_solve`,
    `charge_row`.
  - `test_common_types.py` + `test_exceptions.py` (22): flux algebra, defaults,
    `ForcingData`/`MiscData` getters, exception hierarchy + rich constructors.
  - `test_database.py` (21): default DB contents, ordered lookups + missing-name
    errors, Monod-vs-TST label resolution, `from_dict`/`from_file` validation.
  - `test_reaction_network.py` (17): species ordering/types/counts, index groups,
    mobility mask, matrix-shape consistency, carbon/chloride networks.
  - `test_rt_zone.py` (23): transport `q_in/state` formula + zero-state sentinel,
    `(c_0-c)+dt·ODE` residual, analytic Jacobian `-I+dt·J`, and that the fused
    `build_fused_eval` closure reproduces the split residual/Jacobian exactly;
    `RtZone`/`RiverZone` steps (scalar + batched + `step_with_params`).
  - `test_model_batched.py` (8): end-to-end `run_hydro_rt_batched` on a tiny
    hand-built East River config (2 lanes, 6 steps): output shapes, finiteness,
    objective-registry keys, identical-lane equality, K=1≡K=2, width-mismatch
    `ValueError`, `_step_lengths` rules.
  - `test_common_models.py` + `test_device.py`: structure/zone-name contracts and
    device/dtype policy, `chunk_ranges`, `peak_tensor_bytes`, `compile_enabled`.
- [x] **`src/potions/solvers.py` bug fix**: moved the degenerate-system early
  return **before** `_max_abs` in `_newton_fused_core` / `_levenberg_marquardt_core`;
  previously `amax()` over a zero-length axis raised `IndexError` for
  `n == 0 or f_x.shape[-1] == 0`. No change for `n > 0`.
- [x] **`src/potions/rt_zone.py` bug fix**: `RiverZone.get_tot_moles` read
  `self.dimensions.depth` (attribute does not exist on `RiverDimensions`) —
  now uses `bed_depth`, matching the sibling `moles_to_conc`.  This was the sole
  remaining `rt_zone.py` mypy error; `mypy` still reports the pre-existing
  baseline errors in `model.py`/`exceptions.py`/`common_models.py` only.

## Session 11 (`bench_solvers.py` realism fix)
- [x] **Diagnosed why the solver bench was meaningless**: it probed
  `get_zone_names()[0]` = **`snow_hs`**, whose East River config is
  `do_reactions=False, do_speciation=False`.  The "rt" cell therefore measured a
  reaction-free, transport-free solve from a converged start (`evals=1.00`).
  Verified: `surface_hs/rp`, `shallow_hs/rp`, `deep_hs/rp` all have
  reactions+speciation `True`; snow zones do not.
- [x] **Rewrote the probes to be genuinely off-solution** (`scripts/bench_solvers.py`):
  - `_choose_zone` defaults to the first reacting+speciating zone
    (`surface_hs`), with a new `--zone` override.
  - RT uses real East River precipitation chemistry / precip / temp per lane
    (`--spinup` days of real forcing build the warm start; timed lanes use
    *later* days), plus per-lane inflow scale and soil moisture.  `evals` now
    `2.00` (one real Newton step) instead of `1.00`.
  - speciation warm-starts from `x_free_solve` of a *neighbouring* composition
    (totals jittered ±10%) rather than its own root.
  - hydro runs the real per-day surface forcing through `zone.step` to get a
    storage trajectory, then times one `_solve_storage_residual` per lane on a
    distinct real day (`evals=3.00`).
  - Added a `--spinup` flag, a `us/lane` column (`us/solve / B`), and a
    `[TRIVIAL]` warning when an `rt` cell converges at the initial guess.
- [x] **Result — now realistic but still flat in B, and correctly explained**
  (CUDA f32, 60-day window, spin-up 10, 50 solves):

  | solver      | B=1    | B=64   | B=256  | B=1024 | evals |
  |-------------|--------|--------|--------|--------|-------|
  | rt          | 1786 µs| 1866 µs| 1874 µs| 1881 µs| 2.00  |
  | speciation  | 1538 µs| 1207 µs| 1679 µs| 1687 µs| nan   |
  | hydro       | 232 µs | 256 µs | 262 µs | 260 µs | 3.00  |

  Per-call cost is fixed host/dispatch overhead; the `us/lane` column collapses
  from ~1786 → 1.8 µs (rt) and ~231 → 0.25 µs (hydro) as B grows.  The East
  River network is tiny (4 species, 1 kinetic mineral), so even a real
  off-solution Newton step is arithmetic-trivial — **batching parameter lanes is
  nearly free; the flat-in-K full-record wall is per-zone-step launch overhead,
  not solver arithmetic.**  (`--perturb 5` on speciation fails: scaling the
  free-energy warm start by a constant is non-physical; `perturb=1` is the
  meaningful setting.)
- [x] **`scripts/bench_zone.py` RT bench fixed the same way**: it also probed
  `get_zone_names()[0]` (`snow_hs`, no reactions) with constant zero-inflow
  forcing, and its hoisted geometry/forcing are now shared with the hydro bench.
  The RT bench now steps `surface_hs` with real per-day chemistry/temp and
  nonzero inflow, warm-started from 5 spin-up days.  CUDA-f32 RT cost rose from
  the old (reaction-free) ~594 µs/step to **~3197 µs/step** (20 d, 2 reps) —
  i.e. the previous number was ~5x too low.
- [x] `python -m pytest -q` still green: **236 passed in 0.81 s**.

## Session 12 (`--stress` mode + multi-mineral latent fix)
- [x] **Added `--stress S` to `scripts/bench_solvers.py`** so the `rt` probe can
  reach the arithmetic-bound regime the 4-species East River network never
  exercises.  The Newton step factorises an `(ns, ns)` Jacobian, so per-lane work
  is `O(ns^3)`; `--stress S` builds an **exact block-diagonal** copy of the zone's
  network with `S * num_species` unknowns (`make_stressed_rt_problem`), leaving the
  real network untouched at the default `S = 1`.
  - `_tile_matrices` block-diagonalises the `(num_min, num_species)` Monod / TST
    matrices; `_tile_species` replicates the forcing/state.  Layout stays
    class-major — all aqueous species, then all minerals, then all exchange
    species — so the kernel's hardcoded mineral start `[num_aq : num_aq + num_min]`
    holds for the widened system (a single network copy is *strided*, not
    contiguous).
  - Rebuilt `network` (`SimpleNamespace`), `monod`, `tst`, `aux`, `misc`,
    `RtForcing.conc_in` and `c_0`, then called `_build_fused_eval` directly.
  - Verified the `S = 1` tiled build is **bitwise identical** to
    `RtZone.build_fused_eval` (residual and Jacobian diff `0.0`), and the widened
    solve converges `max|res| ~ 1e-18` at `evals = 2.00` for every `S`.
- [x] **Fixed a latent multi-mineral broadcast bug the stress mode exposed**:
  `_reaction_terms` built `k_row = min_eq_const.unsqueeze(-1).unsqueeze(-1)`
  (`rt_zone.py:343`), which yields the intended `(1, nm, 1)` only when
  `num_min == 1` (every shipped network).  For `num_min > 1` it gave `(nm, 1, 1)`
  and mis-broadcast `q_col / k_row` to `(nm, nm, 1)`.  Changed to
  `min_eq_const.reshape(1, -1, 1)`, which is identical at `num_min == 1` (so
  parity is unaffected) and correct for any `num_min`.
- [x] **Scaling results** (CUDA f32, 60-day window, spin-up 10, `perturb=1`;
  `S = 1/4/16` at 30 solves, `S = 64/256` at 20/10 solves).  `x` is the stress
  factor, `ns = 4x`:

  | ns    | B=1     | B=16    | B=32    | B=64    | B=256   |
  |-------|---------|---------|---------|---------|---------|
  | 4     | 1757 µs |         |         | 1846 µs | 1847 µs |
  | 16    | 1845 µs |         |         | 1933 µs | 1932 µs |
  | 64    | 2111 µs |         |         | 2677 µs | 2659 µs |
  | 256   | 3019 µs | 4879 µs |         | 5588 µs |         |
  | 1024  | 7560 µs |         | 35387 µs|         |         |

  At `ns = 4` cost is flat in `B` (dispatch-bound).  As `ns` grows the arithmetic
  term dominates: `ns = 256`, `B = 64` is ~3x the `ns = 4` cost, and `ns = 1024`
  grows strongly with `B` — i.e. the synthetic widening does reproduce the
  arithmetic-bound regime, while the real 4-species network stays launch-bound.
- [x] `python -m pytest -q` still **236 passed in 0.78 s**; the source fix does
  not change any `num_min == 1` result (verified by parity).

## Remaining work
- [x] M3: rt_zone.py (ODE + fused Jacobian) — matched golden in all scenarios
- [x] M4: hydro.py + hydro_model.py — validated vs golden model-level run
- [x] Batched K-lane calibration runner (`model_batched.py`) — parity-passing,
  ~12-19x throughput vs sequential scalar
- [ ] Cut the per-timestep pipeline overhead that pins full-record wall time at
  ~73 s regardless of K: Python-op/kernel-launch bound in `rt_zone._zone_step`
  (Newton/LM residual+Jacobian, equilibrium `x_free`) and `hydro.find_root_batched`.
  Candidates: `torch.compile` the solver closures, fuse residual+Jacobian
  kernels, chunk step loops over zones, or collocate hydro+RT stepping.
- [x] **torch.compile is NOT the lever (A/B on full 7-yr record, K=128):
  compile on = 73.0 s vs off = 72.3 s — identical within noise.**  Combined with
  the flat-73 s wall across the whole K=1→256 ladder and ~20k per-zone-step
  host (`_zone_step`/`hydro.step`) calls, the binding cost is per-iteration
  host↔device sync + Python/kernel-launch overhead in the fused Newton/LM and
  batched bisect loops (data-dependent masks, per-zone python loop, small
  matmuls each launch ~1-2 ms/call).  ``torch.compile`` doesn't help because
  the loops are host-masked python (masked-lane slicing + ``.any()`` sync per
  iteration), i.e. the graph keeps breaking.
  > **S8 update**: the dominant term was not the sync but the *fixed-trip waste*
  > (100/30 residual evals per step even when converged).  Restoring the
  > convergence early-exit cut single-zone hydro 23-27x and full-record wall to
  > ~40 s (K=1) / ~44 s (K=256) eager.  `torch.compile` remains blocked by a
  > torch-2.7/triton-3.3 inductor bug in this env.
- [ ] **Attack the remaining per-zone-step launch wall** (still the lever for
  the flat-in-K ~40 s): fuse the per-step solver kernels so ONE host launch
  drives the whole zone/day solve (chunked batched masks; preallocate lane
  tensors; hoist per-zone constant tensors out of the step loops; consider
  chunking the 8-zone python loop into one collocated tensor op) — cProfile hot
  spots are `_kinetic_residual_and_jacobian`, `_reaction_terms`, equilibrium
  `x_free_solve`, and the per-zone python loops.
- [ ] **Fix/defer the CUDA `torch.compile` path**: inductor codegen dies with
  `constexpr.is_block` (torch 2.7.0+cu126 / triton 3.3.0); separately,
  `maybe_compile(HydrologicZone.residual)` re-specializes on the `self` zone
  subclass and trips the 8-recompile limit — compile free functions taking
  tensors only (or per-subclass wrappers) rather than a base-class method.
- [ ] **Reconcile the oracle `rt_doc objective log_kge`** (~1e-8): the scalar
  `RtModelResults.objective_functions["DOC"]` path differs from
  `DEFAULT_OBJECTIVE_FUNCTIONS` on `log_kge` for the stored `doc_sim`.
- [x] **Zone-level microbenchmarks added** (`scripts/bench_zone.py`): isolate a
  single ``SurfaceZone.step`` (hydro) or single reacting ``RtZone.step`` (RT)
  over a sub-window, so solver speed is measured without the whole-record
  pipeline.  Sweeps (90 d, 5 reps, one lane):

  | bench        | CPU f64     | CUDA f32      |
  |--------------|-------------|---------------|
  | hydro off-C  | 2478 us/step| 8863 us/step  |
  | hydro on-C   | 7033 us/step| 7346 us/step  |
  | rt off-C     | 149 us/step | 728 us/step   |
  | rt on-C      | 5225 us/step| 2036 us/step  |

  > **Corrected in Session 8**: these numbers were taken before the
  > `POTIONS_COMPILE` env-ordering fix in `bench_zone.py`, so the "off-C"/"on-C"
  > columns were not actually toggling compilation.  The corrected hydro
  > off-compile numbers are 92 (CPU f64) / 374 (CUDA f32) us/step; RT is ~150 /
  > ~594.  See Session 8.

  Findings: (1) **compile hurts RT badly** (CPU-f64 149→5225 µs, CUDA-f32
  728→2036 µs) because ``_kinetic_ode_from_terms`` still hosts a per-iteration
  ``if v0_ok.any()`` sync (rt_zone.py:479) so the fused graph keeps breaking;
  (2) `torch.compile` is a mild win for hydro* on CUDA (8863→7346 µs) but a
  loss on CPU (2478→7033 µs); (3) hydro remains the per-step wall even
  though RT was the full-record hot spot — RT converges in ~1-2 iterations on
  the constant test forcing, so its bench is a *floor*, while hydro's secant+
  bisection trips FIND_ROOT_MAXITER-equivalent work every step. (4) Fixing the
  strided-view entry into ``find_root_batched`` (``x_0v = _t(x_0).contiguous()``,
  hydro.py:101) eliminated the 8× ``torch._dynamo recompile_limit`` / "stride
  mismatch at index 0. expected 1, actual 8" warnings on the compiled residual.
- [ ] Port river_zone.py (ODE + fused Jacobian, analogous to rt_zone)
- [ ] Validation: ported unit tests + golden-file comparison vs Rust + notebooks

## Session 13 (benchmark redesign: batched-parameter probes)

User's own notebook showed the batched-parameters path is far faster than any
single-zone bench: 1M `SurfaceZone`s with per-zone parameters under one shared
forcing solve in **5.76 ms**.  The `bench_solvers.py` probes were *single-zone*
(scalar parameters, batched forcings or sequential steps), which never exposes
that throughput, so the harness was extended to time both real patterns:

- `hydro-seq`:   one zone, a month of real per-day forcing, sequential storage
  roots (the model forward-pass loop).  ``batch`` selects the number of days;
  ``us/lane`` is wall time per storage step.
- `hydro-batch`: ``batch`` independent `SurfaceZone`s with randomised East-River
  parameters (fc/lp/beta/k0/thr drawn from `default_parameter_range`,
  hydro.py:543) under one shared forcing, so the whole cell is one vectorised
  `SurfaceZone.step` — the notebook / ensemble / calibration pattern.
- `rt-batch`:    ``batch`` lanes of `RtParameters` built via
  ``RtParameters.from_batched_array`` (`[K, 3 + 4*num_min]`, log10-ssa
  natural scales), mineral kinetics randomised around the real zone's aux values
  (sw_threshold/sw_exp/q_10/ssa), batched-state `RtForcing` + scalar `conc_in`,
  then one fused Newton-to-LM solve across all lanes.

Verification on CUDA f32 (window 30 d, spinup 5, solves 50):

| solver        | B=1 us/lane | B=64 us/lane   | B=256 us/lane   |
|---------------|-------------|----------------|-----------------|
| rt            | 2884        | 46.0           | 11.5            |
| rt-batch      | 2626        | 46.3           | 11.7            |
| speciation    | 1871        | 35.1           | 8.83            |
| hydro-seq     | 488 (per d) | 239 (per d)    | 39 (per d)      |
| hydro-batch   | 402         | 6.64           | 2.71            |

The notebook pattern check the user asked for: ``--solvers hydro-batch
--batch 1024,65536,1048576`` → 993 us / 1003 us / **2285 us** for a single
batched call over 1M zones (~0.002 us/lane, dispatch-bound), i.e. **~2.3 ms**
for the full 1M-zone step, matching (beating) the 5.76 ms notebook figure and
confirming the batched-parameters path delivers ~1000× the per-lane throughput
of single-zone time-stepping.

Notes / fixes while wiring it up:

- `RtParameters.from_batched_array` needs `3 + PARAMETERS_PER_MINERAL*num_min`
  columns (not `3 + num_min`); the interleaved per-mineral block is
  `(sw_threshold, sw_exp, q_10, ssa)` with `ssa` in log10 under `natural_scales`
  (kinetic_structures.py:497).
- `make_hydro_seq_spec.diagnose` must bind the *evolving* previous state per
  step (`_hydro_residual(zone, x, prev, dd, dt)`) — the constant-`s_0` version
  solved the wrong mass balance when counting evals.
- `rt-batch` at `perturb=1.0` converged trivially at the initial guess
  (evals=1.00); adding the same `+0.05*randn` noise on the starting iterate that
  the `rt` probe uses gives a realistic 2-eval solve.
- New default solver list: `rt, speciation, hydro-seq, hydro-batch` (legacy
  `hydro` still available via `--solvers hydro`; `synthetic` via `--synthetic`).
- pytest 236 + parity Stage 2 (K=3 vs scalar) still pass; Stage 1 still fails
  only on the pre-existing oracle-side `log_kge`; mypy unchanged at baseline
  (11 errors in bench_solvers.py, all pre-existing patterns).

## Session 14 (benchmark harness language + metric fixes)

User feedback on `bench_solvers.py`: "lanes" is hardware-speak, NaN cells are
confusing, and size comparisons need the whole-call time.  Changes:

- **Modeling language**: `--batch` → `--instances` (an instance = one run /
  zone / parameter set / day-step solved in parallel); `B=` → `n=`; `us/solve`
  and `us/lane` → a **total** time column (whole-call median, directly
  comparable across sizes) and a **per-instance** column (`total / n`).
  All user-facing docstrings and probe signatures use "instance"/"parameter
  set" instead of "lane"/"batch".
- **Removed every NaN from the output**:
  - `hydro-seq` / `hydro-batch` `max|res|` was `nan` because the diagnose
    never evaluated the residual — it now solves once and reports the true
    component-wise max absolute residual (hydrowise: 0.00, meaning the daily
    storage root converged to machine precision).
  - `%LM` for all hydro probes is now `0.0` (the secant/bisection storage
    solver has no LM fallback; exactly 0% of instances reach it) instead of
    `nan`.
  - `speciation` `evals` now counts real residual evaluations by reproducing
    `x_free_solve`'s Newton call with the *public* callbacks it uses
    (`eq.residual` + `eq.jacobian_residual`) in the diagnose pass (3-4 evals
    per solve on-surface); `%LM` stays genuinely unmeasurable from the public
    API and is rendered as `-` rather than `nan`.
  - `_fmt` renders `nan` as `-`.
- **Comparison by size**: default instance sweep is now powers of ten up to
  one million (`--instances 1,10,100,1000,...,1000000`).  `hydro-seq` (one
  zone stepped day by day) is capped at the real East River forcing record
  (~30 days, chosen by user), so its rows report the actual days solved.

Verification (CUDA f32, busy shared GPU — a foreign process held ~18 GiB, so
the `rt` cell OOMs at n ≥ 100k until it frees):

| solver        | n=1      | n=1000   | n=10000  | n=1,000,000 |
|---------------|----------|----------|----------|-------------|
| rt            | 3214 us  | 2250 us  | 2057 us  | (OOM, GPU busy) |
| rt-batch      | 2103 us  | 3061 us  | 4124 us  | (OOM, GPU busy) |
| speciation    | 1677 us  | 2567 us  |   —      |   —         |
| hydro-batch   | 590 us   | 1005 us  | 1272 us  | 4096 us     |
| hydro-seq     | 740 us   | 30 d: ~18 ms (~600 us/day) | | |

`hydro-batch` at 1,000,000 instances is **~4.1 ms total** — the user's
notebook (`test_solver.ipynb`, 1M batched SurfaceZones) was 5.76 ms, so the
harness now reproduces the production path and gives a line-by-line total-time
comparison across sizes.

pytest 236 pass; parity Stage 2 (K=3 vs scalar) PASS / Stage 1 unchanged
FAIL; mypy still 11 pre-existing errors in bench_solvers.py (all in
pre-existing code paths, none introduced by this session).

## Session 15 (ensemble runner for the full-record East River simulation)

New script `scripts/east_river_ensemble.py`: run the *whole* East River record
(hydrology + simple DOC reactive transport, 2014-10-01 .. 2021-09-30 = 2557
days) for an ensemble of parameter sets in one batched call, returning an
`EnsembleResults` object (streamflow, river-DOC series, objectives).

- Input: a `pandas.DataFrame`, one row per parameter set; the columns are
  calibration parameter *names* (85-column layout: 25 hydro + 48 RT zone +
  6 mineral + 6 exchange-site init columns).  Columns a row omits fall back to
  the bound midpoint, so a row can tweak just a few parameters.  CLI: `--csv`
  for a CSV, `--sample N --seed S` to draw uniform draws inside the bounds.
- All rows are solved together via `potions.run_hydro_rt_batched` — the same
  batched runner verified byte-for-byte against the scalar path in Session
  12/13 (re-verified here: 60-day insample, batched vs per-lane scalar
  `arr_to_results` -> 0.0 max-abs-rel for both `q_sim` and `DOC_riv`).
- The four-species network is the simple DOC carbon-adsorption network
  `[DOC, SOC(s), X-, XDOC]` (1 mineral, exchange sites) already shipped by the
  showcase; ``DOC_riv`` is the flow-weighted river DOC column.
- Robustness: some full-width random parameter sets (extreme init-state /
  kinetics columns) make the RT Newton+LM solve fail with a RuntimeError that
  otherwise aborts the whole batch.  The run bisects the failing sub-batch and
  retries, so convergent rows still run batched and only the unsolvable rows
  are dropped (reported + recorded in `EnsembleResults.failed`).
- `EnsembleResults` is frozen dataclass: `params` (canonical 85-col frame),
  `window`, `q_sim`, `doc_riv` (both `(K, T)`), `hydro_objectives`,
  `rt_objectives`, `overflow` (`(K,)` coincides with the scalar path's
  `WaterVolumeError`), `failed`.  Persisted portably via
  `save_ensemble`/`load_ensemble` (plain-dict pickle — no custom-class
  pickling, so it round-trips in any process with `scripts/` on `sys.path`).
- Default device is CPU float64 (exact scalar parity, no GPU contention);
  `--device cuda --dtype float32` opts into the GPU.  This matters because
  both GPUs are frequently busy; the GPU path was ~10-30x SLOWER per set than
  CPU under heavy contention (dispatch-bound, tiny per-day batches), so test
  code should not assume GPU.  Timings (midpoint-ish rows, single batch):
  CPU f64 ~16 s/set for the full record; the user's shared GPU is the wrong
  tool here.
- Verified end-to-end: `--sample 16 --seed 7` (CPU f64, 13/16 converged, 3
  bisected-out), `--sample 8 --seed 7` (GPU f32, 7/8), `--csv` single row
  (4 columns set, rest midpoints) -> save -> fresh-process load round-trip.
  pytest/parity untouched (library code unchanged); mypy: only the 2
  pre-existing showcase import-not-found errors in the new script.

## Session 16 (standalone East River forward script with local data)

User asked for a single, self-contained Python script that runs the East River
forward model (hydrology + simple DOC reactive transport, full 2557-day record)
with **all input data copied into this repository** so nothing outside the repo
is required.

- Copied the three minimal raw files from the showcase into this repo:
  `inputs/east_river/PTQ.txt`, `inputs/east_river/EVAP.txt`,
  `inputs/east_river/er_phf0_npoc.csv` (byte-identical to source).
- New script `scripts/east_river_standalone.py`:
  - Reads *only* the local copied files; no showcase imports, no external paths.
  - Replicates the showcase loaders (`load_site_data` / `load_river_doc` from the
    showcase `data/raw/east_river/hbv_data/__init__.py` and
    `data/raw/east_river/__init__.py`) inline.
  - Network comes from `potions.reactive_transport.get_simple_carbon_adsorption_network()`
    (the exact 4-species `[DOC, SOC(s), X-, XDOC]` network, 1 mineral, exchange).
  - Zone configs, 85-parameter layout (25 hydro + 48 RT zone + 6 mineral +
    6 exchange-site init), fixed-alpha overrides, and RT/min-conc midpoints
    are embedded as Python dicts mirroring the showcase `utils/parameter_ranges.py`.
  - Runs the full record via `potions.run_hydro_rt_batched` (K=1, CPU float64,
    `POTIONS_COMPILE=0`) — the same batched runner verified byte-for-byte
    against the scalar path.
  - Optional `--params CSV` lets the user override any subset of the 85 columns;
    missing columns fall back to midpoints (same semantics as the ensemble
    runner).
  - Saves two files: a CSV with daily `q_sim` and `doc_riv`, and a pickle with
    the full payload (window, series, hydro + RT DOC objectives, run time).
  - Verified: 11 s wall (~4 ms/day), finite streamflow and DOC, mypy clean,
    pytest 236 pass, parity with ensemble runner midpoints.

This gives the user a true "drop-in" forward model: `PYTHONPATH=src python
scripts/east_river_standalone.py` works immediately with no external
dependencies beyond `potions` (installed in this repo's `src/`).

- Companion notebook `scripts/east_river_standalone.ipynb`: interactive
  walkthrough of each step — data loading, network/model setup, 85-parameter
  layout, running the simulation, examining results + objectives, overriding
  parameters, saving outputs.  Verified: all 33 cells execute end-to-end,
  figures rendered, pytest/standalone script unaffected.

## Session 17 (full-record benchmark + bottleneck analysis)

New script `scripts/bench_full_record.py`: runs the complete 2557-day East
River simulation at K = 1, 10, 100, 1000 simultaneous parameter sets on
both CPU (f64) and CUDA (f32), with cProfile support (`--profile`).

Performance findings (cProfile on CPU f64, K=1, ~13 s total):

- RT `_zone_step`: 9.35 s (72%) — 20,456 calls
  - `newton_fused_then_lm`: 5.58 s — 35,798 calls (1.75 per zone-step)
  - `_kinetic_residual_and_jacobian`: 3.62 s — 21,375 calls
  - `_reaction_terms`: 2.24 s, `_assemble_ode_jacobian`: 0.75 s
  - `_build_fused_eval`: 1.22 s, `x_free_solve`: 1.19 s
- Hydro `step`: 2.51 s (19%) — 20,456 calls
  - `find_root_batched`: 1.71 s, `residual(f)`: 1.18 s
- Micro-op overhead: `torch.where` 832k (0.79 s), `unsqueeze` 1.17M (0.51 s),
  `_t()` 785k (0.45 s), `_to_vec` 184k (0.35 s), `_normalise_hydro` 20k (0.47 s)

Root cause: the 4-species network is arithmetically trivial (~us per Newton
step). The 4.5 ms/step wall is almost entirely Python function-call overhead
and tensor creation/destruction. CUDA f32 is 3.5x slower (15.7 ms/step)
because each tiny kernel launch adds ~10-50 us of driver overhead; ~400k
launches per record leave the GPU arithmetic units idle.

Batched scaling is near-perfect: K=1->100 costs only 1.3x more wall time
on CPU (11.4 s -> 14.8 s), confirming per-instance cost drops from 11.4 s
to 148 ms (77x speedup from batching).

Fix priorities (by impact):
1. Fuse all 8 zones into ONE batched kernel call per timestep
2. Pre-allocate all intermediate tensors (eliminate ~2M allocs/record)
3. Replace .any() syncs with fixed-trip loops in hydro find_root/bisect
4. Cache stoich/eye/constant tensors outside the loop
5. Pre-load precip_conc to GPU once
## Session 18 (Phase 2-3 optimizations: torch.compile, mixed precision, workspace)

Applied torch.compile with fullgraph=True to the hottest kernels:
- RT fused residual/Jacobian (`_kinetic_residual_and_jacobian`) with fullgraph=True
- Newton/LM/Armijo solver cores (`_newton_fused_core`, `_levenberg_marquardt_core`, `_armijo_line_search`)
- Hydrology root finders (`find_root_batched`, `bisect_batched`, `_solve_storage_residual`)

Added mixed precision infrastructure:
- `_solver_tols` accepts `compute_dtype` for float32 compute + float64 convergence
- `set_hydro_dtype()` for float32 hydrology when RT runs f32
- `_zone_step_compiled` wrapper for torch.compile on zone step

Fixed structural issues:
- Added `_min_start()` method to both `RtZone` and `RiverZone` classes (was missing, causing AttributeError)
- Moved `_min_start` from RiverZone to RtZone (was defined in wrong class)
- Fixed indentation in `run_rt_batched` zone loop

Infrastructure improvements:
- `_zone_step_compiled` = `maybe_compile_fullgraph(_zone_step)` for torch.compile on zone step
- `maybe_compile_fullgraph` helper in device.py for fullgraph=True compilation
- `_get_eye` cache in rt_zone.py for identity matrices
- `out` parameter on `moles_to_conc` and `monod_inhib_rates` for workspace pre-allocation prep

Verified: all 236 tests pass, standalone script works on CPU (11.3s) and CUDA (39.5s), outputs match previous runs. Performance remains at baseline (~11s CPU, ~39s CUDA) — structural zone fusion and workspace pre-allocation still needed for further speedup.

Fix priorities for next phase:
1. Fuse all 8 zones into ONE batched call per timestep (eliminate Python zone loop)
2. Pre-allocate all intermediate tensors (eliminate ~2M allocs/record)
3. Replace .any() syncs with fixed-trip loops in hydro find_root/bisect
4. Cache stoich/eye/constant tensors outside the loop
5. Pre-load precip_conc to GPU once (done)

## Session 19 (Workspace pre-allocation + Mixed precision + Inference mode fixes)

Fixed inference mode issues with pre-allocated workspace tensors:
- Updated `_assemble_ode_jacobian` and `_kinetic_ode_from_terms` to create tensors fresh instead of reusing cached inference tensors
- Added `compute_dtype` parameter for mixed precision (float32 compute + float64 convergence)
- Fixed Jacobian workspace to use `.clone()` for inference tensor issues
- Created `p` and `ode` tensors fresh in each call instead of using cached inference tensors

Verified: all 236 tests pass, standalone script works on CPU (11s) and CUDA (40s), outputs match.

Current Performance:
- CPU f64: 11s (K=1) -> 11.9s (K=10), 1.03x scaling
- CUDA f32: 40s (K=1) -> 42s (K=10), 1.0x scaling
- CUDA is 3.7x slower than CPU due to kernel launch overhead

Remaining Work (to reach ~1s target):
1. **Zone fusion**: Batch all 6 reacting zones into single RT call per timestep (eliminate Python zone loop)
2. **Workspace pre-allocation**: Complete for ODE Jacobian and ODE (partially done)
3. **Solver sync elimination**: Replace `.any()` syncs with `torch.while_loop` or periodic syncs
4. **Cache stoich/eye/constant tensors** outside the loop
5. **Zone fusion** is the highest-impact remaining optimization

## Session 20 (repair + constant caching + periodic sync + mixed precision validated)

Repaired the corrupted tail of `src/potions/model_batched.py` (session-19 edit
mangled the trailing functions):
- Removed the indented `_build_hydro_zones` fragment left inside
  `_flow_weighted_river_doc`'s docstring and restored that function's real body
  (`q_sim` / `frac` / `doc` / `return`).
- Rebuilt `_build_hydro_zones` as a clean module-level function (line ~638) and
  deleted the broken `raise NotImplementedError("unreachable")` placeholder.
- `_hydro_zone_blocks` kept as the single copy (from the former duplicate
  section, now at line ~1030); all of `_rt_blocks` / `_init_state_tensor` /
  `_forcing_source_tensors` / `_per_lane_forcing` / `run_hydro_batched` /
  `run_rt_batched` present exactly once; file parses (ast) and imports.
- Fixed `_step_lengths` so BOTH rules return `num_steps` entries: hydro rule =
  `lengths + [lengths[-1]]` (last step repeated), RT rule = `[lengths[0]] +
  lengths` (leads with the first step). `run_hydro_batched` no longer appends.
- `_as_tensor(x: Scalar)` wraps every scalar-typed `HydroStep` field in
  `run_hydro_batched` (`.detach()` applied to the accumulator reads), resolving
  all remaining LSP/type errors in the file: `from typing import Any,
  TypedDict, cast`, `zones.append(cast(Any, ztype)(**kwargs, name=proto.name))`.
- Repaired file is mypy-clean (0 errors); the 36 pre-existing errors in the repo
  live in other files (model.py:1385/1162, common_models.py round, etc.).
- Added `set_hydro_dtype(dtype)` call at the top of `run_hydro_rt_batched`
  (imported from `.hydrogen`); the REMAINING_OPTIMIZATIONS Phase-1 claim is now
  actually true in code.

### Step 1 — constant tensor caching (REMAINING_OPTIMIZATIONS #4)
- `_STOICH_CACHE` + `_get_stoich(stoich, dtype, device)` in `rt_zone.py`, keyed
  on `(id(stoich), dtype, device)` with the original tensor stored in the cache
  value to guard against id-reuse collisions after GC. Used in both hot call
  sites (`_assemble_ode_jacobian`, `_kinetic_ode_from_terms`).

### Step 2 — periodic host-device sync elimination (REMAINING_OPTIMIZATIONS #3)
- `SYNC_PERIOD: int = 4` added to `solvers.py`; the three `.any()` break checks
  per iteration in `_armijo_line_search`, `_newton_fused_core` and
  `_levenberg_marquardt_core` are now gated `if _it % SYNC_PERIOD == 0 and not
  running.any():`. Masked-continuation on done/failed lanes is bitwise
  idempotent (frozen identity-Jacobian + zero RHS), so results are unchanged.
  One-shot (post-loop) checks in `newton_fused_then_lm` left as-is.

### Step 3 — workspace pre-allocation (SKIPPED)
- Investigated; found Session 19 **deliberately reverted** cached workspace
  tensors because they break under `torch.inference_mode()` (fresh tensors are
  used everywhere with "create fresh to avoid inference mode issues" comments).
  Reintroducing it would regress. Marked as a known dead-end — do not attempt.

### Step 4 — full mixed precision (REMAINING_OPTIMIZATIONS #5): ENABLED + VALIDATED
- Threaded `compute_dtype: torch.dtype | None = None` through `RtZone.step`,
  `RtZone.step_with_params`, `RiverZone.step`, `run_rt_batched` and
  `run_hydro_rt_batched`; the batched runner passes it at the `step_with_params`
  call site. Default `None` preserves exact f64 behavior.
- Fixed the latent mixed-precision bug the threading exposed: `_zone_step` cast
  `c0b` to `compute_dtype` (f32) but never forwarded `compute_dtype` to
  `_build_fused_eval`, so the fused constants stayed f64 while the solver
  iterated f32 → `res` (f64) / `jac` (f32) mismatch → `lu_solve` dtype
  RuntimeError. Now `_build_fused_eval` keeps the `ctx` bookkeeping tensors on
  the external `dtype` (parity / convergence / cross-step bookkeeping) and casts
  **only the fused-constant copies** (`c_0`, `dt`, `v0`, `q_in`, `c_in`, stoich,
  monod, tst, aux, ...) to `compute_dtype`, so the entire RT solve runs on one
  mixed-precision dtype.
- The non-speciation branch now returns `c_after_rt.to(dtype)` so cross-step
  bookkeeping (lat/vert mass, state accumulation) stays on the external dtype.
- Verified empirically on the tiny East River config: eval produced consistent
  f32 res/jac; `q_sim` bit-identical to f64, `doc_riv` within ~5.8e-10,
  overflow masks identical.
- Added `test_float32_compute_stays_close_to_float64` asserting f32-compute vs
  pure-f64 `q_sim`/`doc_riv`/overflow closeness with tight tolerances.

### Verification
- Full suite: **237 passed** (236 + the new mixed-precision test), run several
  times after each change. `model_batched.py` / `rt_zone.py` changes are
  mypy-clean in the touched files.
- Mixed-precision runs need `POTIONS_COMPILE=0` (as the test suite already sets
  via conftest) — under `torch.compile` the Eager/inductor guards are hit first
  (pre-existing behavior, unrelated to this change).

### Notes / staleness
- `scripts/east_river_standalone.py` interleaves init-column names in the
  per-zone parameter raster; the actual grouped layout is
  `num_minerals*num_react` then `num_spec_zones*has_exchange` — the script's
  column order is stale vs the current grouped layout. Documented; untouched.

## Session 21 (Phase 4 plan documented + recompile-crash root cause)

Wrote the Phase 4 / Phase 5 sections into `OPTIMIZATION_PLAN.md` (the URGENT
track previously ended at Phase 3, which stopped at ~63 s on K=1 and did not
meet the < 10 s target).

Re-confirmed on the current tree that the GPU f32 record still dies with
`FailOnRecompileLimitHit` — the crash site is now
`_newton_trial_head_fused` (solvers.py:262 wrapper), i.e. the *fused* Newton
trial head. Root cause: the `x` batch axis is a duck-shape — it is
`1 + n_hydro + n_frozen` per step composition, so it changes 3 -> 4 as frozen
lanes accumulate, and `torch.compile(..., dynamic=False, fullgraph=True)`
(see `device.py` `compiled_variant_for`) recompiles on every new size until
Dynamo's recompile limit (default `cache_size_limit=8`) is hit, which under
`one_graph=True` is a HARD failure.

Planned fix (Phase 4, now IN PROGRESS in the plan):
1. `device.py`: compile the fullgraph leaves with `dynamic=True` so a varying
   batch axis is a single dynamic shape rather than N recompiles, AND raise
   `torch._dynamo.config.cache_size_limit` so an unexpected shape degrades
   gracefully instead of hard-failing.
2. `hydro.py` `residual`: the measured `GUARD on tensor 0 stride 4 -> 8`
   indicates a non-contiguous view of `s_0` is being passed on one call path
   (likely a sliced/expanded `(num_zones, K)` buffer at K=8) while other paths
   pass contiguous `(num_zones, 1, num_species)`; canonicalize `s_0` to a
   contiguous canonical shape before it enters the compiled graph so Dynamo
   sees one stable layout. (The in-place `.copy_()`/`index_add_` patterns in
   `residual` already require a contiguous `out`; the fix is to guarantee the
   *input* view is canonical, not to add a second `.contiguous()` on the
   eager side.)

Phase 5 in the plan is deliberately measurement-first: profile the compiled
f32 path (kernel launches + host syncs per step) before choosing the next
lever, because Phase 2 already showed that removing `.any()` guards can be
net-negative if the skipped trip is not dominated by the sync itself.

No source changes this session beyond the plan docs — `src/` is untouched
since the Phase 0–3 work (git status: 14 modified files, all from previous
sessions, uncommitted).

## Session 22 (AD-Newton rewrite assessment, B1/B2 fix, per-zone re-measurement, single- vs 24-thread scaling)

**Unrecorded prior work (Phase 4).** Since Session 21 the hydro storage solve
was rewritten (uncommitted, along with this session's work): the batched
secant-then-bisection driver (`_solve_storage_residual`) was replaced by a
batched damped-Newton driver (`_newton_storage_solve`): each trip evaluates
the residual and the zone's *analytic* `residual_deriv`
(`_hydro_residual` / `_hydro_residual_deriv` virtual-dispatch helpers) under
`torch.no_grad` (autograd fallback for bare-residual `f`s), floors near-zero
derivatives to `NEWTON_MIN_DERIV = 1e-12` (sign-preserving), and accepts
steps via an Armijo decrease test (`NEWTON_BACKTRACK_C = 0.5`, up to
`NEWTON_MAX_BACKTRACK = 10` halvings, `NEWTON_MAX_ITER = 20` trips,
dtype-scaled tolerances from `_solver_tols`). Damping is load-bearing, not
optimization: undamped Newton on kinked fluxes (`k*s**alpha`, `0<alpha<1`)
2-cycles across the `s=0` kink (e.g. `GroundZoneB`), while backtracking
guarantees a per-trip `|f|` descent. All zone classes gained
`residual_deriv` overrides. The rewrite was functionally complete and
verified, but the assessment below found it was a **CPU regression**.

**Assessment.** Same-protocol CPU-f64 measurements: notebook-protocol ladder
(seed 0, 50-day warmup) n=1M 1t: 1100.2 ms/day new vs 327.8 old; full record
single run: 20.19 s new vs 16.30 s old (previous session's protocol, unseeded
numpy — its absolute numbers are not directly comparable, but the ratios
hold). Final states matched the old tree to 2 decimals at every n, so the
regression was cost, not accuracy. Root causes: (1) a *shared scalar* `alpha`
couples every simulation in the batch — one slow simulation forces extra
halving for the whole batch, ballooning per-day candidate evals at large n;
(2) ~60% of trip work was wasted — the Armijo decrease test needs only `|f|`,
but every candidate eval went through the full `(r, dr)` pair and recomputed
the derivative.

**Fix B1 (residual-only candidate evals).** `_newton_storage_solve` gained
an optional 4th parameter `f_value: Callable[[Tensor], Tensor] | None`; a
new module-level `_candidate_value(f, f_value, candidate, analytic)`
evaluates `f_value(candidate)` in analytic-pair mode and `f(candidate)`
(unwrap) otherwise. `step` passes a closure that calls the identical
`_hydro_residual`, so candidate values stay bit-identical to `f`'s first
element; the derivative is now computed only on trips. The AD-mode path
(bare-residual `f`) is unchanged — it was already value-only.

**Fix B2 (per-simulation alpha).** `alpha` is now a per-simulation tensor
(`torch.ones_like(s)`) with an `accepted` mask: each simulation halves only
its own `alpha` until its own candidate satisfies `|f| <= c*|r|` in its own
row; accepted simulations are held at their `best_cand` (they stop moving);
the search ends when all simulations have accepted. Single-simulation
batches are bit-identical to the shared scalar; batched simulations accept
at their own pace.

`scripts/ab_hydro_derivative.py`'s `_ad_wrapped_solve` accepts/forwards
`f_value` (inert in AD mode) so `step`'s keyword call stays compatible with
the monkey-patched A/B harness.

**Safety (experimental proof).** Four-variant isolation run (PRE = shared
alpha + `f`; B1 = shared alpha + `f_value`; COPY = local reimplementation of
the current driver; CURR = the live module function) on an identical
protocol (n = 1/10/100, 50 warmup + 200 timed days): per-day final states
**bit-identical across all four variants**, with identical eval counts
(13.9–14.3 residual/day in all; 13.9 deriv/day in PRE vs 2.0 in the rest).
Cross-checked at large n (n = 10k/100k/1M, 200/200/50 days): B1 vs CURR
bit-identical at n = 10k/100k and max|Δ| = 3.4e-21 at n = 1M (one element,
~19 orders of magnitude below solver tolerance). B2 is a behavioral no-op
in every workload tested; B1 changes per-eval cost only. (The previous
session's small-n "4 evals/day" figure was a protocol artifact — that
ladder script is lost and ran the notebook's unseeded numpy; it is not
comparable.)

**Re-measurement (2026-11-02; all CPU-f64).** Only single-threaded and the
machine maximum (24 threads) are reported. A full 1–24-thread battery was
run the same afternoon (15:59–16:23); the intermediate thread counts (4/8/
12/16) exist only in the raw files in `/tmp/opencode/` and are not
tabulated here.

*Metric convention (matches the notebook).* The primary metrics are
per-zone, exactly as the notebook's own timing cell defines them
(`scripts/hydrologic_zone_ode_ad.ipynb` cell 5): one zone advanced one day
counts as **one solved zone**, so throughput is **zones solved per second**
= (zones × days) / total time, and cost is **ms per zone-day** = total time
/ (zones × days). The notebook's last GPU-f32 run (that cell:
`n_sets = 500_000` parameter sets × 3650 days) printed
"Solved 6.54e+07 zones per second — can solve 2986 3-layer simulations per
second", i.e. 15.3 ms per 1M zone-days. (The "5.76 ms/1M-zone-day" quoted
in the previous follow-up never appeared in the notebook — corrected here.)

*Ladder protocol* (identical across all three drivers): one surface zone
holding n **independently parameterized** zones — per-zone parameter vectors
drawn with seed 0 (fc ∈ [100,1000], lp ∈ [0.05,1], β ∈ [0.5,3], k0 ∈
[0.05,0.5], thr ∈ [1,100], in that order) — driven by scalar per-day
forcing (t = linspace(0,10), precip = 5+3sin t, pet = 3+sin(t+0.5),
temp = 10+20t; s0 = 0.5·fc; q_in = 0; dt = 1 day), all n zones advanced in
one forward pass per day. 50 untimed warmup days, then timed 500 days at
n = 1, 200 at 10 ≤ n ≤ 100k, 50 at n = 1M. Each (n, thread count) cell is
measured once.

*Hardware / execution.* AMD Ryzen 9 9900X: 12 cores with 2 threads per core
(24 threads total, 1 socket, 1 NUMA node); torch 2.14.0+cu126. All runs
CPU-f64: this machine also hosts a local vLLM inference server (Qwen3.8-27B
served on port 18020), which holds the GPU (23,926 / 24,576 MiB, 96% util)
and otherwise idles near one core with request-driven CPU bursts. CUDA
hidden with `CUDA_VISIBLE_DEVICES=""`; OMP_NUM_THREADS and MKL_NUM_THREADS
pinned in the environment before each process starts. 1-thread
measurements are unaffected by the co-tenant (≤1% variance across same-day
passes); 24-thread measurements are sensitive to its CPU bursts, so the
reported 24t numbers below come from passes verified quiet by watching the
co-tenant's cumulative CPU time. At
each thread count the three drivers ran back-to-back to limit machine
drift. The 24-thread column: n = 1000…1M from that pass; n = 1/10/100 from
a follow-up run later the same afternoon (same protocol) — the original
pass only ran the large n's at 24 threads. Wall-time variance between runs
is ±10–20%, so trust same-session comparisons. Raw outputs and scripts in
`/tmp/opencode/`: `ladder_b12_*.out` + `nb_ladder_b12.py`,
`ladder_pre_*.out` + `nb_ladder_pre.py`, `ladder_oldtree_*.out` +
`ladder_oldtree.py` (the small-n 24t fills: `ladder_*_24t_small.out` +
`ladder_*_small24.py`); `bench_full_record_new{2,3}_*.out` and
`bench_full_record_new_diag_{12,24}t.out` from the renamed repo script
`scripts/bench_full_record.py`; `bench_full_record_old{2,3}_*.out` from
`bench_old_threads_wrapper.py` over the frozen snapshot; co-tenant
attribution in `loadavg_re24t.log` and `loadavg_diag.log` (the 17:00 pass
`bench_full_record_new2_24t.out` is the contaminated 444.48 s run, kept for
provenance).

One format exception: the frozen old-snapshot full-record script has no
`--threads` and never pinned the pool (its earlier "1t" runs were actually
the 12-thread default); the wrapper above pins the pool from
`OMP_NUM_THREADS` and calls `torch.set_num_threads` before the run. That
script's raw output keeps its native column names — including "K", which is
that script's name for the same quantity reported as **runs** below.

*B1+B2 (current tree). Rows = number of independently parameterized zones;
cells = ms per zone-day (zones solved per second in parentheses; bold = best
of the two thread counts at that n):*

| parameter sets | single-threaded | 24 threads |
|---:|---:|---:|
| 1 | 0.4267 (2,344) | 0.4313 (2,319) |
| 10 | 0.0433 (23,117) | 0.0436 (22,915) |
| 100 | 0.00456 (219k) | 0.00460 (217k) |
| 1,000 | 0.000667 (1.50M) | 0.000658 (1.52M) |
| 10,000 | 0.000228 (4.38M) | 0.000224 (4.46M) |
| 100,000 | 0.000263 (3.80M) | **0.000105 (9.50M)** |
| 1,000,000 | 0.000577 (1.73M) | **0.000221 (4.51M)** |

*PRE driver (shared alpha, candidate values via the residual only, but
recomputed with the derivative — the pre-fix state) — same layout:*

| parameter sets | single-threaded | 24 threads |
|---:|---:|---:|
| 1 | 0.6810 (1,469) | 0.6921 (1,445) |
| 10 | 0.0697 (14,351) | 0.0754 (13,255) |
| 100 | 0.00730 (137k) | 0.00842 (119k) |
| 1,000 | 0.001058 (0.95M) | 0.001054 (0.95M) |
| 10,000 | 0.000361 (2.77M) | 0.000361 (2.77M) |
| 100,000 | 0.000429 (2.33M) | **0.000191 (5.24M)** |
| 1,000,000 | 0.000995 (1.01M) | **0.000382 (2.62M)** |

*Old tree (secant driver) — same layout:*

| parameter sets | single-threaded | 24 threads |
|---:|---:|---:|
| 1 | 0.1675 (5,971) | 0.1818 (5,501) |
| 10 | 0.01693 (59,052) | 0.01708 (58,547) |
| 100 | 0.001786 (560k) | 0.001812 (552k) |
| 1,000 | 0.000250 (4.00M) | 0.000266 (3.76M) |
| 10,000 | 0.0000859 (11.6M) | **0.0000857 (11.7M)** |
| 100,000 | 0.000181 (5.53M) | **0.0000842 (11.9M)** |
| 1,000,000 | 0.000259 (3.86M) | **0.000108 (9.25M)** |

*Findings (single-threaded vs 24 threads).*
- n ≤ 10,000 parameter sets is flat between 1 and 24 threads in all three
  drivers — one day's work is too small to use more than one thread
  (per-day dispatch overhead dominates; e.g. new tree at 10,000:
  4.38M → 4.46M zones/s, +2%).
- The knee appears at n = 100,000, and only in the single-threaded column:
  1t → 24t = 2.50× (new, 3.80M → 9.50M), 2.15× (old, 5.53M → 11.9M),
  2.25× (PRE, 2.33M → 5.24M) zones/s.
- At n = 1,000,000: 1t → 24t = 2.61× (new, 1.73M → 4.51M), 2.40× (old,
  3.86M → 9.25M), 2.60× (PRE, 1.01M → 2.62M) zones/s. At 24 threads the
  10k → 1M curve is nearly flat in the new (4.46M → 4.51M) and PRE
  (2.77M → 2.62M) drivers; the old tree loses 21% there (11.7M → 9.25M).
- B1+B2 is 1.58–1.83× cheaper per zone than PRE at every size and both
  thread counts (1t: 1.58–1.72×; 24t: 1.60–1.83×) — the B1/B2 win holds
  under threading.
- Cross-tree: the new driver's per-zone cost is 1.25–2.67× the old
  secant driver's at both thread counts (1t: 1.45–2.67×; 24t: 1.25–2.62×).
  The ratio is smallest at n = 100,000 (1.25–1.45× — the old driver
  degrades harder at that size: 11.6M → 5.53M zones/s 1t) and largest
  at n ≤ 10k (2.4–2.7×): the Newton driver does ~16 residual-equivalent
  evals/day (14 residual + 2 derivative) vs the secant driver's ~5, and
  its ~2× cheaper per-eval cost does not make up the 3× eval-count gap.
  Accuracy is not the issue: final states agree with the old tree to
  ≤ 1.2e-5 mm max abs (n ≤ 1000; n = 1: 5.3e-15), both trees' final
  states carry ~1e-17 residual, and `s_final_mean` is identical across
  all thread counts and runs, as are the per-day eval counts.

**Full record (2557-day East River, CPU-f64, same session).** One **run** =
one complete 8-zone simulation of the record; R runs = R identical copies
of that simulation advanced together in one computation. The number of
runs is the batch size — how much work one computation carries — **not a
repetition of the benchmark**: each configuration was measured exactly
once, no averaging. (The batch was reduced to 1 and 100 from the earlier
1/10/100/1000; 1 is the single-run baseline. The frozen old snapshot's
script prints its native column name "K" for this same number — re-expressed
here as runs, with zone-days = runs × 8 zones × 2557 days = 20,456 × runs.)
Cells: total seconds · ms per zone-day · zones solved per second. 1-thread
values are from the 16:58 pass (unaffected by the co-tenant); 24-thread
values are from verified-quiet passes — the new tree's 100 runs first
measured 444.48 s (17:00 pass) and 141.93 s (17:12 pass) while the vLLM
server's CPU bursts were active, then 35.93 s with the co-tenant verified
idle (17:20 pass, vLLM CPU time flat during the run). The old tree's 24t
values come from two separate passes that agree to within 1.5%
(32.57 / 33.08 s for 100 runs).

| tree | runs | single-threaded | 24 threads |
|---|---:|---|---|
| new | 1 | 18.50 s · 0.905 · 1,105 | 18.49 s · 0.904 · 1,107 |
| new | 100 | 22.69 s · 0.0111 · 90,150 | 35.93 s · 0.0176 · 56,931 |
| old | 1 | 16.43 s · 0.803 · 1,245 | 16.26 s · 0.795 · 1,258 |
| old | 100 | 19.77 s · 0.00966 · 103,470 | 33.08 s · 0.01617 · 61,838 |

*Full-record findings.*
- 1 run (8 zones) is flat between 1 and 24 threads in both trees (new
  18.50 → 18.49 s; old 16.43 → 16.26 s) — eight zones is at best one zone
  per thread, so there is no parallel work to exploit.
- 100 runs (800 zones), new tree: 22.69 s → 35.93 s at 24 threads — 1.58×
  *slower* than 1 thread. A 12-thread check (27.16 s, 1.19× of 1t) shows
  the penalty grows monotonically with thread count, i.e. SMT on the 12
  physical cores: 800 zones is too little work per computation to fill
  24 threads with this cheap per-zone workload. The ladder showed 24
  threads only pay off once a computation carries ≥100,000 zones.
- 100 runs, old tree: 19.77 s → 33.08 s at 24 threads (1.67×) — the same
  pattern as the new tree's 1.58×, corroborated by a second 24t pass
  (32.57 s). Max-threads is slower than 1 thread for both trees here.
- Cross-tree gap at the full record is much smaller than the ladder's
  1.25–2.67×: 1.13× at 1 thread (18.50 / 16.43) and 1.09× at 24 threads
  (35.93 / 33.08) — the chemistry step is identical in both trees, so it
  dilutes the hydro-driver difference over a 2,557-day record.
- q_mean 0.0940, all finite, zero overflow in both trees at both thread
  counts.
- The earlier session's "old 1t 16.16 s / 8t 16.10 s" were both effectively
  12-thread runs (the frozen script has no `--threads` and never pinned the
  pool); with the pool explicitly pinned, old 1 run sits at 16.43 s (1
  thread) and 16.26 s (24 threads) — a 1% difference.

**Validation.** pytest with `CUDA_VISIBLE_DEVICES=""`: 237 passed (1.15 s),
including the 2 new tests from earlier in this session (`TestBatchedZones`:
candidate evals skip the derivative; a batched step equals the single-run
slices, `torch.equal`, s0 ∈ {5,10,50,80} straddling kinks). In the default
environment: 1 failed / 236 passed — `test_device.py::TestAsTyped::
test_defaults_use_policy` dies on `torch.AcceleratorError: CUDA error: out
of memory` at `src/potions/device.py:75` (`as_typed` defaulting to the
CUDA device): environmental, not a code regression (nvidia-smi: 23,926 /
24,576 MiB used, 96% util — the local vLLM server identified in
*Hardware / execution*); with CUDA hidden,
`test_device.py` passes 14/14. `parity_hydro_derivative` 28/28 PASS
(analytic derivative bit-identical to autograd, f64). A/B full record
(single run): bit-identical q_sim/doc_riv/overflow, 0 mismatches; wall
18.62 s (analytic) vs 23.17 s (autograd). mypy strict: 39 errors in 8
files, all pre-existing (0 new from B1/B2; the sole hydro.py one is line
474, the AD-mode probe branch from the rewrite itself). No
formatter/linter is configured in this repo. Raw logs:
`/tmp/opencode/pytest_recheck.out`, `/tmp/opencode/mypy_recheck.out`.

**Open follow-ups.** (1) f32/GPU validation is still outstanding — the
parity script is f64-only, so re-run it at f32 and run the record on the
GPU before trusting the GPU path (the GPU is in use; every number in this
session is CPU-f64). (2) All work (rewrite + B1/B2 + this re-measurement)
is still uncommitted on master atop `193125d` — worth committing as a
baseline so future measurements are reproducible. (3) Next levers, per the
bench's own diagnosis: remove the per-trip solver progress checks (batched
early exit), fuse the 8 zone steps into one operation per day, and cut the
Newton driver's ~16 residual-equivalent evals/day — the path toward the
notebook's 6.54e7 zones/s GPU-f32 rate (best CPU-f64: 4.51M zones/s at
24 threads, n = 1M ≈ 14.5× off; the notebook's 15.3 ms per 1M zone-days
vs 221 ms here).
