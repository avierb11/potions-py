# POTIONS-Py: PyTorch-Accelerated Reactive Transport Model

Implementation plan for a pure-Python, GPU-accelerated rewrite of the reactive
transport model from the `potions` project (Rust/PyO3 core). The new package
lives in this directory (`potions-py`) and is designed so that an LLM or human
can resume implementation from this document without re-deriving the model.

---

## 1. Goals and constraints (the "why")

- **Speed and batch throughput**: run **~1,000 parameter sets per batch** (an
  evolutionary-algorithm generation), and up to **~1,000,000 parameter sets**
  (in chunks, e.g. for Sobol-style screening / rejection sampling). The batch
  dimension maps directly onto GPU tensor lanes.
- **Automatic differentiation is NOT required through the time loop.** Gradients
  of the objective w.r.t. parameters would blow up over ~1000 steps. Instead,
  the value proposition is **vectorized batched execution on GPU** (float32).
  The model takes a `(batch, ...)` parameter matrix and a fixed forcing/setup,
  and returns a `(batch, ...)` output tensor. Anything resembling an optimizer
  (evolutionary search, random/quasi-random sampling, rejection sampling,
  population MCMC) consumes batches of outputs; it does not consume gradients.
- **Single-parameter-set correctness must match the `potions` Rust output.**
  The notebooks in `/home/andrew/Documents/Research/Projects/potions/notebooks`
  plus `test_data` in `/home/andrew/Documents/Research/Projects/potions-rs/test_data`
  are the integration/validation suite.
- **Numerics (user-confirmed policy)**: **float32** on CUDA, **float64** on CPU
  by default; both with a configurable dtype override knob. The multi-dimensional
  root solvers (Newton + Levenberg-Marquardt) and equilibrium speciation may run
  at float64 internally when the batch requires it.
- **Jacobians (user-confirmed)**: **manual analytic Jacobians** for kinetics,
  transport, and equilibrium speciation — derived from the Rust source and
  documented in this file — *not* autograd. Fastest on batched GPU and matches
  the Rust solver exactly.
- **12 GB VRAM target**; must scale gracefully to more/less memory by chunking
  the batch dimension.
- Follow AGENTS.md conventions: functional style where possible, full type
  hints (modern style, no `typing.List`), docstrings, a `NOTES` file describing
  governing equations.

## 2. Reference source layout (read these before coding)

All values/equations below were transcribed from:

- Rust core: `/home/andrew/Documents/Research/Projects/potions/rust/src/`
  - `math.rs` — root finders (scalar + multi-dimensional Newton/LM)
  - `common_types.rs` — `HydroForcing`, `HydroStep`, `RtForcing`, `RtStep`,
    `MiscData`, `ZERO_CONC`
  - `hydro.rs` — `HydrologicZone` + `SnowZone`, `SurfaceZone`, `GroundZone`,
    `GroundZoneB`
  - `reactive_transport/{mod.rs, rt_zone.rs, river_zone.rs,
    kinetic_structures.rs, reaction_network.rs, database.rs}`
  - `reactive_transport/default_database.json` — embedded chemical DB
- Python wrappers: `/home/andrew/Documents/Research/Projects/potions/src/potions/`
  - `hydro_model.py`, `model.py`, `reactive_transport_model.py`,
    `reactive_transport/*`, `common_types.py`, `utils.py`, `calibration.py`
- The Rust crate is built via maturin (see `pyproject.toml` in `potions`); the
  PyO3 module name is `potions.core`. **We are NOT re-using it** — we reimplement
  in torch.

## 3. Model overview

The model is a **multi-zone water-balance + reactive transport simulator**,
stepped with implicit (backward) Euler.

1. **Hydrology run**: for each zone `z` and time step `t`, solve scalar
   root-finding for the next storage state `s`; record a `HydroStep` containing
   the state and all water fluxes.
2. **Reactive transport run**: for each time step, **Gauss-Seidel** over zones
   (in fixed zone order). Each zone advances a vector of species concentrations
   `c` via an implicit solve of a mass-balance ODE (transport + optional kinetic
   reactions) using Newton with analytic Jacobian, then optionally re-speciates
   equilibrium chemistry. Water/solute fluxes from already-solved zones feed the
   "incoming concentration" of the next zone downstream.

### 3.1 Species ordering (critical)

Species are ordered (see `reaction_network.rs::species_order`):

```
[ primary_aqueous..., secondary..., mineral..., (X-, exchange...) ]
```

- `num_aq    = num_primary_aqueous + num_secondary`
- `num_min   = num_minerals`
- `num_spec  = num_aq + num_min + (has_exchange ? 1 + num_exchange : 0)`
- `has_exchange` is true iff `exchange_species.len() > 0` (the `X-` site is
  always index `num_aq + num_min` when exchange exists).
- **RtZone**: mineral concentrations are at indices `num_aq .. num_aq+num_min`.
- **RiverZone**: mineral concentrations are at the END: `num_spec-num_min ..`.
  That is, `min_start = num_species - num_minerals` for RiverZone and
  `min_start = num_aqueous` for RtZone.

### 3.2 Data types and constants

- `ZERO_CONC = 1e-20` — sentinel "zero" concentration (used to keep logs finite).
- Water fluxes/state in mm; concentrations in mol/L; depth in mm
  (1 mm over 1 m^2 = 1 L). Kinetic rates convert via the `86_400.0` factor*.

## 4. Hydrology

### 4.1 Zone flux formulas

Each zone implements these scalar functions of `s` (current state) and
`d: HydroForcing(precip, temp, pet, q_in)`.

**Generic implicit Euler**: solve `f(s) = (s_0 - s) + dt * mass_balance(s, d) = 0`
for `s_new`, then clamp with `.max(0.0)`. `mass_balance = q_in + forc_flux -
vap_flux - lat_flux - vert_flux`.

- **SnowZone** (`tt`, `fmax`):
  - `forc_flux = precip` if `temp <= tt` else `0`
  - `vert_flux = fmax*(temp-tt)` if `temp > tt` else `0`, capped at `min(s, melt)`
  - `vert_flux_ext = vert_flux + precip` if `temp > tt` else `vert_flux`
  - `mass_balance = q_in + forc_flux - vert_flux`
  - params: `[tt, fmax]`; defaults `(0.0, 1.0)`
- **SurfaceZone** (`fc`, `lp`, `beta`, `k0`, `thr`):
  - `vap_flux = pet * min(s / (fc*lp), 1)`
  - `lat_flux = max(k0 * (s - thr), 0)`
  - `vert_flux = q_in * (s/fc)**beta`
  - `lat_flux_ext = lat_flux`; `vert_flux_ext = vert_flux`
  - params: `[fc, lp, beta, k0, thr]`; defaults `(100, 0.5, 1.0, 0.1, 10)`,
    init state `25.0`
- **GroundZone** (`k`, `alpha`, `perc`):
  - `lat_flux = k * max(s,0)**alpha` if `s >= 1e-12` else `0`
  - `vert_flux = min(s, perc)`
  - `lat_flux_ext = lat_flux`; `vert_flux_ext = vert_flux`
  - `mass_balance = q_in - vert_flux - lat_flux`
  - params: `[k, alpha, perc]`; defaults `(1e-2, 1.0, 1.0)`, init state `10.0`.
  - Note: `k` is passed in **log10 scale** when `natural_scales=True`
    (`from_array` does `10**vals[0]`).
- **GroundZoneB** (`k`, `alpha`): same as GroundZone with `vert_flux = 0`,
  `mass_balance = q_in - lat_flux`. `k` likewise log10 in natural scales.

Scalar solvers (see `math.rs`):
- `find_root_secant`: initial `x_1 = x_0 + 0.1`; tolerance `1e-6`; max 100 iters;
  stop if `|fx_0 - fx_1| < 1e-12`; error on NaN / iteration budget.
- `bisect`: bracket `[0, 1000]`; if sign not opposite at endpoints → NaN error;
  tolerance `1e-6`; max 100 iters.
- `step` tries secant, falls back to bisect.

### 4.2 HydrologicalModel / forced matrix wiring

The Python wrapper (`hydro_model.py`) builds:
- `lat_mat`, `vert_mat`: `(n_zones, n_zones)` binary coupling matrices; a 1 at
  `[i,j]` means zone `j`'s lateral/vertical outflow feeds zone `i`. In words the
  model graph is a DAG: snow → surface → ground → ground_bottom, with surface
  draining to river laterally.
- `surface_zone_ids`: indices whose incoming solute concentration is
  precipitation chemistry (not upstream flux).
- `forc` matrices distributed per zone (precip/temp/pet/Qin).

For the torch re-write we just need the same `lat_mat`/`vert_mat`/
`surface_zone_ids`/scales, produced by a pure-Python graph builder that takes a
list of zone specs. Hydrology stepping is **not** the main expense; implement it
as a torch `(batch, n_zones)` implicit-Euler solve with the **vectorized secant
solver** below, or — since each zone's residual is monotone in `s` — a couple of
deterministic fixed-point / bisection iterations per batch.

**HydroStep** fields (all mm): `state, forc_flux, lat_flux, vert_flux,
vap_flux, q_in, lat_flux_ext, vert_flux_ext`. Derivers used downstream:
- `q_internal() = lat_flux + vert_flux`
- `q_external() = lat_flux_ext + vert_flux_ext - q_internal()`
- `total_water_out() = lat_flux + vert_flux + vap_flux`

## 5. Chemical database and ReactionNetwork

### 5.1 Database JSON schema

(`reactive_transport/default_database.json`, `database.rs`):

```json
{
  "primary_species":   {"DOC": {"name":..., "molar_mass":..., "charge":..., "dh_size_param":...}},
  "secondary_species": {"HCO3-": {"name":..., "molar_mass":..., "charge":...,
                                    "dh_size_param":..., "eq_consts":[...],
                                    "stoichiometry": {"H+":1,"CO3--":..., ...}}},
  "mineral_species":   {"SOC(s)": {"name":..., "molar_mass":..., "molar_volume":...,
                                    "eq_consts":[...], "stoichiometry": {...}}},
  "exchange_reactions":{"XDOC": {"name":..., "log10_k_eq":..., "charge":...,
                                  "stoichiometry": {"X-":-1.0, "DOC":1.0}}},
  "tst_reactions":     {"mineral_name": {"label": {"mineral_name":..., "label":...,
                                                    "rate_constant":<log10 k 25C>,
                                                    "dependence": {"species":coeff}}}},
  "monod_reactions":   {"mineral_name": {"label": {...,
                                                    "monod_terms": {"species":Khalf},
                                                    "inhib_terms": {"species":Kinh}}}}
}
```

### 5.2 ReactionNetwork construction

`ReactionNetwork(primary_aqueous, mineral, secondary, mineral_kinetics,
exchange_species)`. Everything boils down to matrices we precompute **once as
torch tensors** (constants, not gradient-tracked):

- `species_order`, `species_names`, `num_species`, `num_aqueous_species`,
  `num_minerals`, `num_exchange_species`, `has_exchange`, `total_species`
  (primary + mineral + `X-`).
- **`mineral_stoichiometry`**: `S` (num_spec, num_min); column i = mineral i's
  stoichiometry over all species (`0` for species absent from the map).
- **charge vector**: (num_spec,) — primary/secondary charges, 0 for minerals and
  `X-`, exchange charges after.
- **`monod_mat` / `inhib_mat`**: (num_min, num_spec); entry is `NaN` where the
  reaction has no monod/inhib term for that species; minerals with no
  monod reaction get a row of `0.0`. (Rust uses `f64::NAN` for missing and `0.0`
  for inactive minerals — keep this exact.)
- **TST data**: `stoich` (num_min, num_spec), `dep` (num_min, num_spec), and
  `min_eq_const` = `10**m.eq_consts[1]` for TST minerals, else `1.0`. Rows for
  non-TST minerals are all zeros.
  - **IMPORTANT — stoch/dep swap vs the legacy Python wrapper.** In the Rust
    kernel, `stoich` holds the TST reaction's **dependence** map (exponent of
    Q/K) and `dep` holds the mineral's own stoichiometry restricted to the same
    dependence keys (exponent of D). The legacy Python wrapper
    (`reactive_transport/reaction_network.py::tst_params`) puts the **mineral
    stoichiometry** in `stoich` and the **dependence** in `dep` — the opposite.
    Reproduce Rust exactly (validated: our TST build pushes the reaction
    `dependence` values into `stoich` and the masked mineral stoich into `dep`).
  - Consequence for the **default database**: every TST `dependence` map is
    empty (`{}`), so both matrices are all-zeros → `Q_i = 1` and `D_i = 1` for
    every TST mineral; the TST machinery is effectively **inert** with the
    shipped data (`tst_rate_i = 1·(1 − 1/K_i)`). Do not "fix" this to use the
    mineral stoichiometry.
- **`rate_consts`**: `10**log10(k)` (num_min,), searched first in `monod_reactions`
  then `tst_reactions` (Rust order — verified), else `KeyError("Cannot find
  mineral {name}")`. Note `10.0**-10.0` in float64 is `9.999999999999999e-11`,
  not `1e-10` — expected rust-equivalent (`powf`) behavior.
- **`mineral_molar_mass`**: (num_min,).
- **Equilibrium (speciation) matrices** — see §6.3.

## 6. Reactive transport core

### 6.1 Concentration/moles conversions

Let `state` = water storage `v0` (mm), `depth` = zone depth (RtZone) or
`bed_depth` (RiverZone).

```
tot_moles[i] = chms[i] * v0      if i < num_aq      (mmol over unit area)
tot_moles[i] = chms[i] * depth   otherwise           (mineral/exchange "per volume" scale)
```

```
moles_to_conc(tot_moles, d):
  conc[i] = (v0 >= 1e-6 ? tot_moles[i]/v0 : ZERO_CONC)   if i < num_aq
  conc[i] = tot_moles[i] / depth                          else
```

### 6.2 Kinetic rate models (per RtZone/RiverZone, given `chms`)

**Monod rate** (num_min,):
```
monod_i = Π_j  chms[j]/(K_ij + chms[j])      for finite K_ij
inhib_i = Π_j  K_ij/(K_ij + chms[j])          for finite K_ij
monod_rate_i = monod_i * inhib_i
```

**TST rate** (num_min,):
```
log_conc = log10(chms)
e_q   = stoich @ log_conc        Q_i = 10^e_q
e_dep = dep     @ log_conc       D_i = 10^e_dep
tst_rate_i = D_i * (1 - Q_i / K_i)
```
(Safer torch form: compute exponents first, then `10**exponent`.)

**Aux + mineral production rate** (per zone with `do_reactions`):
```
sw_factor_i   = ((1 - s_w)/(1 - thr_i))**sw_exp_i   if s_w >= thr_i
             else (s_w/thr_i)**sw_exp_i
temp_factor_i = (q10_i)**((temp - 20)/10)
aux_factor_i  = sw_factor_i * temp_factor_i

min_conc_i = chms[min_start + i]                       # mineral concentration
mineral_conc_rate_i = 86_400 * rate_const_i * ssa_i * molar_mass_i
                     * min_conc_i * aux_factor_i * (monod_rate_i + tst_rate_i)

# -> to species rates:  S @ mineral_conc_rate   then zero rows i >= num_aq
```

### 6.3 Equilibrium speciation

**Setup** (in `ReactionNetwork.equilibrium_parameters`):
- Build `stoich_eq`: (n_reactions, num_spec). One row per **secondary species**
  (its stoichiometry vector over all species) followed by one row per
  **exchange reaction** (its stoichiometry vector). `log_eq_consts` are
  `sec.eq_consts[1]` and `exch.log10_k_eq`.
- Build `total_mat`: (num_total, num_spec), one row per **total species**
  (primary + mineral + `X-`). For a non-`H+` total species `T`:
  `total_mat[row] = 1 at species T`,
  + `sec.stoichiometry[T]` for each secondary whose map contains `T`,
  + `exch.stoichiometry[T]` for each exchange whose map contains `T`.
  For the `H+` total species the row is the **charge vector**. (Rust does **not**
  apply `.abs()` to total rows — the legacy Python wrapper does; do not.)
- `N = null_space(stoich_eq)` via **scipy.linalg.null_space** (num_spec, n_free);
  **the Rust code literally calls scipy**, so we can too. Replicate exactly (do
  NOT use a different null-space convention without re-validating).
- `x_particular = pinv(stoich_eq) @ log_eq_consts` (scipy.linalg.pinv).
- `n_free` = number of columns of `N` and always equals `num_total`. **Corrected
  claim:** `n_free == 0` happens *only* for a truly empty network (no total
  species). scipy `null_space` of a `(0, num_spec)` matrix returns an
  `(num_spec, num_spec)` identity-like span and `pinv` of it gives the
  transpose, so for empty chemistry the solve is a harmless identity-like no-op
  returning the input concentrations — Rust always runs the solver when
  `num_free > 0`; we mirror that and do **not** special-case-skip.

**Given concentrations `chms`, solve for equilibrium `c_eq`:**
```
c_tot = total_mat @ chms          # (num_total,)
if any total species is "H+": c_tot[charge_row] = 0.0

conc_func(x_free)     = 10**(N @ x_free + x_particular)       # (num_spec,)
residual(x_free)      = c_tot - total_mat @ conc_func(x_free) # (num_total,)
jacobian(x_free)      = -ln10 * (total_mat * conc[None,:]) @ N # (num_total, n_free)

# Solve residual == 0 for x_free via Newton (analytic Jacobian) then LM.
c_eq = conc_func(x_free*)
```

Implemented (M2) in `solvers.py` (Newton → LM chain, Rust-verbatim: masked,
per-lane control flow) + `equilibrium.py` (`residual`, `jacobian_residual`,
`x_free_solve`, `solve_equilibrium`, all batched over a leading `(B, ...)` dim).
Failed lanes: by default return NaN; `raise_on_failure=True` raises. torch note:
`linalg.lu_solve` needs `B` as `(…, n, 1)` (a plain `(…, n)` RHS is rejected),
and batched-masked results are pointwise identical to per-element solves up to
matmul-kernel rounding (~1e-15). Validated vs the compiled `potions.core`:
`N`/`x_particular`/`total` bit-identical, solution residual within the golden
kernel's own `≤ 1e-6` bar, analytic Jacobian matching the kernel (its exposed
`residual_jacobian` is a finite-difference approximation, so expect ~1e-8
truncation-level disagreement there).

**Warm starting**: each zone stores the previous step's `x_free`; initialize
Newton from it (lengths must match; else zeros). Cold start = zeros.

### 6.4 Mass-balance ODE and its analytic Jacobian

State = concentrations `c` (num_spec,). Implicit Euler residual:

```
residual(c) = c_0 - c + dt_days * ODE(c)
jacobian(c) = -I + dt_days * dODE/dc
```

**ODE(c)** (per zone):
```
if |v0| >= 1e-6:  ODE_transport[j] = (q_in/v0) * (c_in[j] - c[j])  for j < num_aq
else:             ODE_transport[j] = ZERO_CONC  (Rust returns full-vector ZERO_CONC; but in fused impl zero)
if do_reactions:
   chms = moles_to_conc(c)
   MCR_i = base_i * chms[min_start+i] * (monod_i + D_i(1 - Q_i/K_i))
   react = S @ MCR ; zero react[i>=num_aq]
   ODE += react
```
Note: `min_start = num_aq` for RtZone, `num_spec - num_min` for RiverZone.
**Quirk (verified in Rust `reaction_terms_impl`):** the Jacobian builds the
per-mineral concentration term from `chms[num_aq + i]` (hardcoded index), not
`chms[min_start + i]` — these coincide for RtZone (no exchange shifts the
mineral block) but differ for RiverZone with exchange species, where `min_start
= num_spec - num_min ≠ num_aq`. Replicate the hardcode as-is (it also means
`dm` is computed for `i < num_min` regardless of exchange).

**dODE/dc** (`residual_jacobian_impl` / `assemble_ode_jacobian`):
```
J      = zeros(num_spec, num_spec)
if |v0| >= 1e-6:  J[r,r] = -q_in/v0         for r < num_aq   # transport diagonal
if do_reactions:
   P_i  = 1/v0 if i < num_aq (0 if v0 too small) else 1/depth
   for each mineral i, species j (aqueous rows only):
      A_i  = 86_400 * rate_const_i * ssa_i * molar_mass_i * aux_factor_i
      Mn_i = monod_rate_i
      T_i  = D_i*(1 - Q_i/K_i)
      # Monod derivative
      g = 0
      if Kmonod_ij finite and c_j>0:  g += Kmonod/(c_j*(Kmonod+c_j))
      if Kinhib_ij finite and c_j>0:  g -= Kinhib/(c_j*(Kinhib+c_j))
      dMn = Mn_i * g
      # TST derivative
      if c_j > 0:
         dT = D_i*dep_ij/c_j*(1 - Q_i/K_i) - D_i*Q_i*stoich_ij/(c_j*K_i)
      else: dT = 0
      delta = (j == min_start + i)
      dm[i,j] = A_i*( delta*(Mn_i+T_i) + chms[min_start+i]*(dMn + dT) )
   # J_ode = J_transport + S @ dm @ diag(P), only rows r<num_aq
```

### 6.5 Solver: Newton with analytic Jacobian → Levenberg-Marquardt fallback

Constants (`math.rs`): `MULTI_TOL=1e-6`, `MULTI_MAXITER=100`,
`MULTI_TOL_STEP=1e-12`, `ARMIJO_C=1e-4`, `LINE_SEARCH_MAX=30`,
`STAGNATION_REL_TOL=1e-9`.

**Newton core** (for `f(c)` residual + `J(c)` analytic Jacobian, fused):
```
x = x_0; fx = f(x); err = max|fx|; best = (x, err)
for i in 0..100:
   if err <= 1e-6:  return x
   if any NaN in x / fx / J: error
   step = solve(J, fx)                     # PartialPivLu; fallback QR lstsq
   alpha = 1.0
   for _ in 0..30:
       x_new = x - alpha*step
       (f_new, J_new) = fused(x_new)
       if |f_new|^2 <= (1 - 1e-4*alpha) * |fx|^2:  accept; break
       alpha *= 0.5
   if not accepted: stall -> break
   x = x_new; fx = f_new; J = J_new; err = max|fx|
   track best (x, err)
end
if best_err <= 1e-6: return best_x else raise / fallback LM
```

**LM fallback** (resume from Newton's best iterate; uses analytic Jacobian):
```
x = best_x; err = 0.5*|fx|^2; err_max = max|fx|; lambda = 1e-6
for i in 0..100:
   if err_max <= 1e-6: return x
   J = jac(x);  A = J^T J ; A_diag += lambda * diag(A)
   b = J^T fx;  step = solve(A_damped, b)
   step_converged = max|step| <= 1e-12 * max(max|x|,1)
   rho = (0.5|fx|^2 - 0.5|f(x-step)|^2) / (step·b - 0.5 step·A step)   # guard denom ~0
   if rho > 0:  accept; lambda = max(1e-16, lambda * max(1/3, 1-(2*rho-1)^3))
   else:        reject; lambda = min(1e16, 2*lambda)
   if step_converged and best_err <= max(1e-6, 1e-9*initial_err_max): return best
end
```

**Batched implementation notes**: the per-zone speciation/kinetic solves are the
hot path. Batch over (param-set × zones) as a single leading axis. Newton
iterations terminate per-element (masked), so inner loops process the full batch
each iteration but only elements whose `err > tol` and not-NaN are "active".
Use `torch.linalg.solve` on the batched Jacobian `(B, n, n)`; guard singular
cases with a per-batch fallback to least-squares (`torch.linalg.lstsq`).
The Armijo/search loop over `alpha` is `while` + `torch.where` selection.

**M2 status**: the above is implemented in `solvers.py` — masked Newton core
(`_newton_fused_core`) + masked LM (`_levenberg_marquardt_core`), exposed as
`newton_fused_then_lm` / `newton_analytic_then_lm`, with per-lane status
(`done` / `failed`) and `raise_on_failure` / NaN-fill exit policy. Note the
reference's `solve_with_fallback_f64` only falls back to QR least-squares when
the matrix is non-contiguous (never in practice), so our LU-only path is
faithful; a singular pivot yields Inf/NaN steps that the line search then
rejects, exactly like faer's `PartialPivLu`.

### 6.6 RtZone step (single zone, single time step)

Mirrors `RtZone::step_internal` exactly:

```
tot_moles_init = get_tot_moles(c_0, d)
tot_mass_in    = conc_in * q_in
q_int = q_internal() = lat_flux + vert_flux
q_ext  = q_external() = lat_flux_ext + vert_flux_ext - q_internal()
tot_moles_ext  = q_ext * conc_in

# 1) kinetic/transport implicit solve
c_after_rt = newton_lm( fused_residual_jacobian, x0 = c_0 )

# 2) optional equilibrium speciation (warm-started)
c_after_eq = speciate(c_after_rt)  if do_speciation else c_after_rt

tot_moles_after = get_tot_moles(c_after_eq, d)
set_minerals_zero(x)[i>=num_aq] = 0
tot_moles_out_internal = set_minerals_zero(c_after_eq * q_int)
tot_moles_out          = set_minerals_zero(tot_moles_ext + tot_moles_out_internal)

total_q_out = q_int + q_ext
frac_lat = lat_flux_ext / total_q_out
frac_vert = vert_flux_ext / total_q_out
if total_q_out > 1e-6:
    lat_mass = tot_moles_out * frac_lat
    vert_mass = tot_moles_out * frac_vert
    lat_conc  = where(|q_lat|<=1e-6, ZERO_CONC, lat_mass/q_lat)   # aqueous only
    vert_conc = where(|q_vert|<=1e-6, ZERO_CONC, vert_mass/q_vert) # aqueous only
else:
    all = ZERO_CONC

mineral_rates = reaction_rate(c_after_eq, d, minerals_only=True) if do_reactions else zeros(num_min)
```
**Quirk (verified in Rust `reaction_real::reaction_rate` and `step_internal`):**
`mineral_rates` is computed by calling `reaction_rate_rust(&c_after_eq, d, true)`
on the concentration array already converted from moles — `reaction_rate_rust`
applies `moles_to_conc` **again**, so aqueous species in the reported
`mineral_rates` are double-divided (minerals only divided by `depth`, second
division is by `state`). This is separate from the in-ODE rate; replicate
faithfully (guard any future refactor with a golden test on `mineral_rates`).

Output `RtStep` fields: `state, total_moles, conc_in, mass_in, lat_conc,
vert_conc, lat_mass, vert_mass, mineral_rates`.

### 6.7 `run_zone_steps` — Gauss-Seidel coupling (RT only)

Per time step, zones are solved in index order `0..n_zones`; **this ordering
matters** (surface zones first, because their incoming conc is precipitation
and doesn't depend on other zones):

```
lat_mass_accum = zeros(n_zones, num_spec);  vert_mass_accum = zeros(n_zones, num_spec)
for i in 0..n_zones:
    d = forcings[i]
    if i not in surface_zone_ids:
        mass_in = lat_mat[i] @ lat_mass_accum + vert_mat[i] @ vert_mass_accum   # row-vector accumulation over already-solved k
        tot_water_in = d.hydro_step.q_in + d.hydro_step.forc_flux
        d.conc_in = mass_in / tot_water_in if tot_water_in > 1e-6 else zeros
    step_i = RtZone.step(states[i], d, dt)
    lat_mass_accum[i] = step_i.lat_mass
    vert_mass_accum[i] = step_i.vert_mass
    states[i] = step_i.state
```

Because `lat_mat`/`vert_mat` are binary and the DAG is small (3–8 zones), this
is a tiny matvec; keep it as a torch op over the batch axis.

### 6.8 RiverZone

Same machinery with two differences:
- `min_start = num_species - num_minerals` (minerals at end of vector).
- Transport uses `q_internal()` instead of `q_in`; `get_tot_moles` uses
  `bed_depth` instead of `depth`.
- Not coupled via `lat_mat`/`vert_mat` in `run_zone_steps`; the Python wrapper
  (see `reactive_transport_model.py`) computes the river's incoming conc by
  **weighted mixing** of zone lateral/downstream fluxes:
  `q_lat_{zone} / q_sim` × zone concentration, summed.

## 7. Full-model orchestration (Python side)

Replicate the API of `ReactiveTransportModel` (subclass of `HydrologicalModel`):

- `Model` (`model.py`) + `ReactiveTransportModel` (`reactive_transport_model.py`)
  expose:
  - `run_hydro_model(forc, meas_streamflow)` → `HydroModelResults`
  - `run_rt_model(hydro_res, precip_conc, initial_state, ...)` → `RtModelResults`
  - `to_array()` (`[hydro_params..., rt_params...]`) and `from_array(...)` for
    calibration — this is the parameter vector an optimizer feeds us.
  - `_run_model` / batched variants: the torch version replaces the
    `multiprocessing.Pool` batch runner with batched tensors.
- `initial_state` handling: build the species concentration vector per zone from
  `ChemicalState(primary=..., secondary=..., mineral=..., exchange=...)` with
  defaults: primary/secondary default `1e-3`; minerals: required if
  `do_reactions and contains_minerals` else `1e-20`; `X-`: required if
  `do_speciation and has_exchange` else `1e-20`; exchange species default `1e-6`.
- Dart outputs: for each (species, zone): `{name}_{zone}` `{name}_{zone}_mass_in`
  `_lat_mass` `_lat_conc` `_vert_mass` `_vert_conc` `_total_moles`, and for each
  mineral `{min}_rate_{zone}`. Plus `{species}_riv` (flow-weighted river conc).
- Objective functions over measured river conc (`DEFAULT_OBJECTIVE_FUNCTIONS`).

## 8. Package layout for `potions-py`

**Confirmed: keep the existing `potions` package name and `src/potions` layout**
(pyproject.toml, `.venv`, and `src/potions/py.typed` are already present).

```
potions-py/
├── pyproject.toml                 # torch, numpy, pandas, scipy, pydantic deps (add scipy)
├── AGENTS.md                      # project coding rules (already present)
├── NOTES.md                       # governing equations + implementation notes (append as you work)
├── IMPLEMENTATION_PLAN.md         # this file (authoritative plan)
├── PLAN.md                        # earlier-session plan (keep for history; superseded where conflicting)
├── PROGRESS.md                    # progress log (update after each session)
├── tests/
│   ├── test_kinetics.py           # vectorized vs scalar reference formulas
│   ├── test_equilibrium.py        # speciation vs scipy/rust reference
│   ├── test_solvers.py            # Newton/LM convergence on published cases
│   ├── test_hydro.py              # zone ODE + implicit euler batched vs scalar
│   ├── test_transport.py          # RtZone/RiverZone single-zone step
│   ├── test_model.py              # full run; mass balance checks
│   └── test_vs_rust.py            # compare against potions.core outputs (golden files)
└── src/potions/
    ├── __init__.py
    ├── py.typed                   # (exists)
    ├── common_types.py            # dataclasses: HydroForcing, HydroStep, RtForcing, RtStep, ...
    ├── device.py                  # get_device(), dtype policy (f32 GPU / f64 CPU), chunk helper
    ├── database.py                # load default_database.json -> dataclasses
    ├── reaction_network.py        # ReactionNetwork -> torch tensor matrices
    ├── hydro.py                   # zone flux functions + (→batched) root solvers
    ├── hydro_model.py             # HydrologicalModel: lat/vert mats, time loop
    ├── kinetic_structures.py      # monod/tst/aux rates (vectorized)
    ├── equilibrium.py             # speciation solver (Newton+LM, batched)
    ├── solvers.py                 # Newton + LM root finders (batched, analytic J)
    ├── rt_zone.py                 # RtZone + RiverZone step logic
    ├── model.py                   # ReactiveTransportModel orchestration (torch)
    └── objective.py               # objective functions over batched outputs
```

## 9. Batching strategy, memory, and precision

### 9.1 Shapes

- Parameter tensor: `(B, n_params)` float32 on GPU. Built from
  `to_array()` template with a `(B, n_params)` sample matrix (diagnostic maps:
  `_from_arrays` inverse).
- Hydrology tensors: `(B, n_zones)` states/fluxes.
- RT tensors: `(B, n_zones, num_spec)` states; forcing stacked as
  `(B, n_zones, 8+num_spec)` cached flux/concentration arrays.
- Outputs: `(B, n_steps, n_zones, num_spec)` (conc) plus fluxes
  `(B, n_steps, n_zones, num_spec)`... — but see 9.3 memory note; usually we
  only retain a few bandits.

### 9.2 Memory budget (float32, 12 GB)

Per time step, the biggest live working set is the Newton/Jacobian solves:
- `c`: `B × n_zones × num_spec` floats.
- J per active zone: `B × n_zones × num_spec × num_spec` floats (dense).

Example: `B=1000, n_zones=8, num_spec=6` → J = 1000×8×36×4 B ≈ **1.15 MB** →
trivial. Even `B=1e6, n_zones=8, num_spec=10` → 320 MB for J; with solvers'
temporaries (LU factor ~4×, plus fx, step, x-fit buffers) plan ≤ ~1.5 GB.
**Conclusion: the RT solve can comfortably batch 1000s of cases; for 1,000,000
cases, chunk the batch axis into ~`ceil(12GB / peak_per_element)` chunks and
stream results to host/disk.** Balance memory: don't keep `(B,n_steps,...)`
history for all B on the GPU — accumulate only the diagnostics needed by the
objective (e.g. per-zone river conc, water balances), or copy chunks back.

### 9.3 Precision

- **Confirmed policy: float32 on CUDA, float64 on CPU** (module-level
  `DEFAULT_DEVICE` / `DEFAULT_DTYPE` in `device.py`, changeable; warn if a
  default-dtype differs from device policy).
- Wrap solver-critical internals so a dtype override stays possible (functions
  take an explicit `dtype` argument defaulting to `DEFAULT_DTYPE`).
- The equilibrium speciation (powers of ten over ~15+ orders of magnitude) may
  need float64 even when the rest is float32; make the speciation module use a
  `dtype` argument and test both.
- Conversion caveat: `10**x`, `.clamp_min(1e-20)`, division by `v0` produce
  inf/NaN easily in float32 — use safe helpers (`torch.where` + masked
  operations) exactly as the Rust code branches.

### 9.4 vectorized scalar-solver (hydrology)

Every zone's storage residual is monotone; a **batched secant** with
element-wise masking mirroring `find_root_rust` (start `x1 = x0+0.1`,
`for i < 100`, update `x = x1 - fx1*(x1-x0)/(fx1-fx0)`, track `|fx|`)
with a fallback **batched bisection** on `[0, 1000]` for the elements that
failed, exactly reproducing the Rust step semantics, is ~30 lines.

## 10. Validation plan (critical — do not skip)

1. **Unit-level equality**: implement scalar (non-batch) reference functions in
   pure NumPy matching each Rust function 1:1 (formulas above), then assert
   torch batched results with `B=1` equal the NumPy reference to ~1e-4 (float32)
   or ~1e-8 (float64).
2. **Golden files from the existing model**: run `potions.core` (Rust, via the
   existing env at `potions`) on the test data
   `potions-rs/test_data/Reactive Transport/hbv_data.pickle` and the
   `Run Reactive Transport Model.ipynb` workflow; serialise per-step states to
   `.npz`. Compare our torch output step-by-step for a few representative zone
   configs (with/without reactions, with/without speciation, with/without
   river). This is the strongest integration test.
3. **Notebooks** that must reproduce (used as acceptance tests):
   - `notebooks/Run Reactive Transport Model.ipynb` (CO2/carbonate C network,
     XDOC adsorption, SOC(s) kinetics, 4 zones + river)
   - `notebooks/Run Hydrological Model.ipynb`
   - `notebooks/Equilibrium.ipynb`
   - `notebooks/example-reactive-transport.ipynb`
   - `notebooks/Markov-Chain Monte Carlo.ipynb` (objective-function plumbing)
   Port these into `tests/` as scripts that assert the reference outputs.
4. **Mass balance**: cumulative water and solute checks per batch element
   (transport closure: `state_{t+1} ≈ state_t + q_in - q_vap - q_lat - q_vert`
   within solver tolerance).
5. **Gradcheck** (optional, cheap): since we don't need reverse-mode through the
   time loop, gradient checks are not required; but `torch.autograd.gradcheck`
   on a **single-step** forward (kinetics/speciation as pure functions) is a good
   cheap correctness probe for the analytic Jacobians (finite-difference match).
6. **The Rust `test_*` files** (`src/potions/tests/*.py`) contain
   reference-validated cases (`test_equilibrium.py`, `test_kinetics.py`,
   `test_solver_convergence.py`, `test_math.py`, `test_zones.py`) — port them.

## 11. Implementation order

Suggested milestone order, each independently testable:

1. **M0 – Scaffolding**: `pyproject.toml` (add scipy), package skeleton,
   `device.py`, `common_types.py`, load `default_database.json`. Test: import +
   DB loads.
2. **M1 – Network matrices**: `network.py` produces all tensors
   (`mineral_stoich`, `monod`, `inhib`, TST mats, eq mats, `N`, `x_particular`,
   `total_mat`). Test: print shapes / verify against `potions.core` (+
   scipy-equivalent).
3. **M2 – Equilibrium** (DONE): `solvers.py` (batched Newton → LM chain) +
   `equilibrium.py` (batched `x_free_solve` / `solve_equilibrium`, warm-start
   aware). Test: `test_equilibrium.py`, `Equilibrium.ipynb`, the Rust
   `x_free_solve` failure cases (network without equilibrium → skip; mis-sized
   system → raise). Validated against compiled `potions.core` on the carbon
   network (see §6.3).
4. **M3 – Kinetics + RT solve**: `kinetics.py`, `rtmath.py`
   (Newton+LM batched), `rtzone.py` (single-zone step, no coupling).
   Test: `test_kinetics.py`, `test_solver_convergence.py`, golden single-zone
   steps from `potions.core`.
5. **M4 – Hydrology**: `hydro.py`, `hydromodel.py` (batched time loop).
   Test: `test_hydro.py`, `Run Hydrological Model.ipynb`.
6. **M5 – Full model**: `model.py` orchestration, Gauss-Seidel coupling,
   river mixing, outputs. Test: `Run Reactive Transport Model.ipynb`,
   `test_model.py`, mass balances, and per-step comparison vs Rust golden files.
7. **M6 – Batched calibration path**: `(B, n_params) → (B, objective...)`
   function; chunked driver; host-side accumulation. Test: Sobol-style sweep on
   `hbv_data.pickle`, verify B=1000 runs in one forward pass and times
   beat the multiprocessing baseline; then scale to 1,000,000 in chunks.
8. **M7 – Optional**: torch.compile the hot inner kernels
   (`@torch.compile` on fresh tensors, no graph capture across time steps since
   shapes change); fuse kernel if needed.

## 12. Pitfalls and exactness notes (from the Rust source)

- **Newton vs LM layout bug**: the old code silently misordered the Jacobian
  (row-major misinterpreted as col-major), stalling the residual at `~1e-8`.
  In torch, never `.t()` accidentally; use `torch.linalg.solve` with consistent
  (batch, m, n) contiguity; verify single-zone residuals hit `~1e-13` in float64.
- **Empty null space** (`n_free == 0`): skip speciation entirely — return input.
- **Empty system** (`x.len() == 0 || f(x).len() == 0`): return x0 (vacuous).
- **ZERO_CONC (1e-20)** appears in: no-water transport, mineral rows of
  `tot_moles_ext`/`lat_conc`/`vert_conc` guards, and set-minerals-to-zero.
- **Inactive minerals**: a mineral contributes nothing when `rate_const`,
  `molar_mass`, `chms[min]`, `Mn`, `D`, `Q`, `K` are non-finite or `|K| <= 0`,
  or when `aux`/`ssa` is missing/non-finite.
- **`q_external` can be negative** (bypass water) — keep the exact formula;
  `tot_moles_ext = q_ext * conc_in`.
- **Exchange species**: `X-` always occupies index `num_aq + num_min` when
  exchange exists; `num_exchange_species = 1 + len(exchange)`.
- **Rate constant / eq-const scales**: database stores **log10** values; convert
  once at load time.
- **The `86_400` factor** makes implicit-Euler `dt_days` consistent with
  daily units: `residual = c_0 - c + dt_days * ODE`, with ODE including
  `86400 * ...` in the kinetic rate. Keep identical.
- **Speciation resets charge balance**: `c_tot[charge_row] = 0` **before or
  during solving** (Rust sets it after `total_mat @ chms` and before the solve).

## 13. Open decisions to revisit

- Whether to keep a small scipy dependency (only used at construction time for
  `null_space`/`pinv` — yes, cheap and matches Rust exactly) vs a torch SVD
  reimplementation (only if GPU-only construction becomes necessary).
  → **Keep scipy (decided);** add to `${pyproject}` dependencies.
- Whether `river_zone` coupling is a required milestone for M5; the notebooks
  exercise it, so include it in M5. → **Yes (decided).**
- Chunk size tuning for `B=1e6` (start with `B=4096`/GPU and benchmark).
- **Resolved (user-confirmed, Session 2):** Jacobians = manual analytic
  everywhere; precision = float32 GPU / float64 CPU; package name/layout = keep
  `potions` / `src/potions`.

## 14. Definition of "done" per milestone

- All tests ported/run, no skips.
- Batched (`B≥1000`) forward pass runs on GPU in one graph; results match
  single-path NumPy reference to float32 tolerance.
- The `Run Reactive Transport Model.ipynb` carbon-network workflow produces
  equivalent `DOC_*`/`XDOC_*`/streamflow time series vs the Rust run.
- Timing: `B=1000 × ~1800 steps × 4-zone + river` budget target
  < the current multiprocessing-Python baseline (measure in M6).