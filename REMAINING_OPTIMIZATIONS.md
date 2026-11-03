# Remaining Optimizations for Potions-Py

> **Terminology note:** in historical entries below, `K` denotes the batch
> size — the number of identical parameter sets (lanes) advanced together in
> one batched call. Current code and scripts use the explicit name
> `batch_size`.

## Current State (After Sessions 1-20)

### ✅ Completed Optimizations

| Phase | Optimization | Files Modified |
|-------|--------------|----------------|
| **Phase 1** | Eye matrix cache (`_get_eye`) | `rt_zone.py` |
| **Phase 1** | Pre-convert `precip_conc` to GPU tensor | `model_batched.py` |
| **Phase 1** | Pre-alloc `state_out` (no `.clone()`) | `model_batched.py` |
| **Phase 1** | `_t()` dtype parameterization | `hydro.py`, `model_batched.py` |
| **Phase 1** | `set_hydro_dtype()` call | `model_batched.py` |
| **Phase 1** | `FusedEvaluator` class | `rt_zone.py` |
| **Phase 1** | `out` param on `moles_to_conc` & `monod_inhib_rates` | `rt_zone.py`, `kinetic_structures.py` |
| **Phase 2** | `@maybe_compile_fullgraph` on RT fused kernel | `rt_zone.py` |
| **Phase 2** | `@maybe_compile_fullgraph` on Newton/LM/Armijo | `solvers.py` |
| **Phase 2** | `@maybe_compile_fullgraph` on hydro root finders | `hydro.py` |
| **Phase 2** | `compute_dtype` for mixed precision in solvers | `solvers.py` |
| **Phase 2** | `compute_dtype` for mixed precision in RT | `rt_zone.py` |
| **Phase 2** | `_min_start()` method on `RtZone`/`RiverZone` | `rt_zone.py` |
| **Phase 2** | `_zone_step_compiled` wrapper | `model_batched.py` |
| **Phase 2** | Workspace pre-allocation for kinetic terms | `rt_zone.py` (local tensors in `_reaction_terms`) |
| **Phase 2** | Jacobian workspace cache (with `.clone()` fix) | `rt_zone.py` |
| **Phase 2** | ODE workspace pre-allocation | `rt_zone.py` |
| **Phase 3** | Mixed precision infrastructure (`compute_dtype` param) | `solvers.py`, `rt_zone.py` |
| **Phase 3** | `set_hydro_dtype()` call in `run_hydro_rt_batched` | `model_batched.py` |
| **Phase 3** | Float32 RT + float64 convergence checks | Enabled + validated |
| **Session 20** | Stoich constant caching (`_STOICH_CACHE`/`_get_stoich`) | `rt_zone.py` |
| **Session 20** | Periodic sync (`SYNC_PERIOD=4`) in Newton/LM/Armijo | `solvers.py` |
| **Session 20** | `compute_dtype` threaded through step APIs & batched runner | `rt_zone.py`, `model_batched.py` |
| **Session 20** | Fused-eval constants cast to `compute_dtype` (fixes res/jac dtype mismatch) | `rt_zone.py` |
| **Session 20** | `model_batched.py` tail repair + `_step_lengths` fix + `_as_tensor` wraps | `model_batched.py` |

---

## 🎯 Remaining Optimizations (Model-Agnostic)

These optimizations apply to any reactive-transport model using the Potions framework.

---

### 1. Zone Fusion (Highest Impact)

**Problem**: Python-level Gauss-Seidel loop over zones per timestep (8 zones × 2557 steps = 20,456 zone-steps/record). Each zone step involves Python dispatch, solver calls, and tensor allocations.

**General Solution**: Batch zones within each layer that have no inter-dependencies.

- **Layer 0** (snow): 2 zones, non-reacting → sequential (no fusion possible, non-reacting)
- **Layer 1** (surface): 2 zones, reacting → **batchable**
- **Layer 2** (shallow): 2 zones, reacting → **batchable**
- **Layer 3** (deep): 2 zones, reacting → **batchable**

**General Implementation Pattern**:
```python
def run_rt_layer_batched(
    layer_zone_indices: list[int],
    rt_zones: list[RtZone],
    hydro: dict[str, Tensor],
    # ... other args
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Process all zones in a layer as a single batched RT step.
    Zones within a layer have no inter-dependencies (only upstream layer coupling).
    """
    # Stack zone inputs (c_0, d, aux, etc.) along zone dimension
    # Call single batched RT step for all zones in layer
    # Unstack outputs
    pass
```

**Expected Speedup**: 2-3× reduction in Python dispatch overhead (6/8 zones batched).

---

### 2. Complete Workspace Pre-allocation (Partially Done)

**Current State**: Pre-allocation for `_reaction_terms` done. Missing for:
- `_assemble_ode_jacobian` (allocates `jac`, `p`, `eye_batch`, `react` per call)
- `_kinetic_ode_from_terms` (allocates `ode`, `react`, `mcr` per call)

**General Solution**:
```python
# Add to _get_*_workspace functions
def _get_jacobian_workspace(batch, num_spec, dtype, device) -> dict:
    return {
        "jac": torch.zeros(batch, num_spec, num_spec, dtype=dtype, device=device).clone(),
        "p": torch.zeros(batch, num_spec, dtype=dtype, device=device),
        "eye_batch": torch.eye(num_spec, dtype=dtype, device=device).unsqueeze(0).expand(batch, num_spec, num_spec).clone(),
        "zero_rhs": torch.zeros(batch, num_spec, dtype=dtype, device=device),
        "react": torch.zeros(batch, num_spec, num_spec, device=device).clone(),
        "s": torch.empty(num_spec, num_min, dtype=dtype, device=device),
        "p": torch.empty(batch, num_spec, dtype=dtype, device=device),
    }

def _get_ode_workspace(batch, num_spec, num_min, dtype, device) -> dict:
    return {
        "ode": torch.zeros(batch, num_spec, dtype=dtype, device=device),
        "react": torch.zeros(batch, num_spec, dtype=dtype, device=device),
        "mcr": torch.zeros(batch, num_min, dtype=dtype, device=device),
    }
```

**Expected Speedup**: Eliminates ~2M tensor allocations/record.

---

## 3. Solver Host-Device Sync Elimination (High Impact)

**Problem**: ~1.3M host-device syncs/record from `.any()` checks in Newton/LM/Armijo loops.

**Current State**: Fixed-trip loops restored (early exits removed for compile compatibility). Re-introduce early exits without sync.

**General Solutions** (in order of preference):

1. **`torch.while_loop` with tensor condition** (preferred):
   ```python
   def newton_cond(state):
       status = state[6]
       running = status == _RUNNING
       return running.any()
   
   def newton_body(state):
       # ... all operations using torch.where for masking
       return new_state
   
   final_state = torch.while_loop(newton_cond, newton_body, init_state)
   ```
   **Status**: `torch.while_loop` in PyTorch 2.7 has meta-consistency issues with tuple state. Needs investigation.

3. **Periodic Sync** (fallback):
   ```python
   # Sync every N iterations instead of every iteration
   for iter in range(MULTI_MAXITER):
       if iter % 4 == 0:
           if not running.any(): break
   ```
   Reduces syncs by 4× with minimal convergence impact.

**Expected Speedup**: 2-3× on GPU (eliminates sync latency).

---

## 4. Constant Tensor Caching (Low Effort)

**Problem**: Re-allocating/re-casting constant tensors in hot path.

**Caches Already Exist**:
- `_get_eye()` for identity matrices ✅

**Missing Caches**:
```python
# In _assemble_ode_jacobian
s = _get_stoich_cached(stoich, dtype, device)  # stoichiometry matrix

# In _kinetic_ode_from_terms
s = _get_stoich_cached(stoich, dtype, device)

# In _kinetic_residual_and_jacobian
eye = _get_eye(num_spec, dtype, device)  # Already cached ✅

# In _kinetic_ode_from_terms
s = _get_stoich_cached(stoich, dtype, device)
```

**General Implementation**:
```python
_STOICH_CACHE: dict[tuple, Tensor] = {}

def _get_stoich_cached(stoich: Tensor, dtype: torch.dtype, device: torch.device) -> Tensor:
    key = (id(stoich), dtype, device)
    if key not in _STOICH_CACHE:
        _STOICH_CACHE[key] = stoich.to(dtype=dtype, device=device)
    return _STOICH_CACHE[key]
```

---

## 5. Full Mixed Precision (DONE)

**Status**: Complete — `compute_dtype` is threaded through `RtZone.step`,
`RtZone.step_with_params`, `RiverZone.step`, `run_rt_batched` and
`run_hydro_rt_batched`. `_build_fused_eval` casts only its fused-constant copies
to `compute_dtype` while keeping the `ctx` bookkeeping tensors on the external
dtype, so the whole RT solve runs on one dtype and cross-step bookkeeping stays
on the convergence dtype.

```python
# In _zone_step:
compute_dtype = compute_dtype or dtype  # e.g., float32

c_after_rt = newton_fused_then_lm(
    fused, c0b.to(compute_dtype), 
    raise_on_failure=True, 
    verbose=verbose,
    compute_dtype=compute_dtype  # NEW
)

# Speciation stays in dtype (float64) for stability
c_after_eq = eq.conc_func(x_free.to(dtype))  # convergence in float64
```

**Validated**: `test_float32_compute_stays_close_to_float64` compares
float32-compute vs pure float64 on the batched runner — q_sim bit-identical,
doc_riv within ~5.8e-10, overflow masks identical.

---

## 7. `torch.while_loop` Migration (Future)

**Goal**: Replace manual for-loops with `torch.while_loop` for full graph capture.

**Requirements**:
- State as tuple of tensors with consistent shapes
- Condition function returns scalar boolean tensor
- Body function returns same tuple structure
- All tensors must have consistent shapes across iterations

**Current Blocker**: `torch.while_loop` meta-consistency issues in PyTorch 2.7 with tuple state.

**Workaround**: Flatten all loop state into single tensor with fixed shape, or use periodic sync fallback.

---

## Summary: Priority Order

| Priority | Optimization | Expected Speedup | Effort | Status |
|----------|--------------|------------------|--------|--------|
| 1 | Zone Fusion (layers 1-3) | 2-3× | High | **Skipped** (Gauss-Seidel parity) |
| 2 | Solver Sync Elimination | 2-3× on GPU | Medium-High | **DONE** (periodic sync §3) |
| 3 | Complete Workspace Pre-allocation | 1.5-2× | Medium | **Skipped** (inference-mode regression) |
| 4 | Constant Tensor Caching | 1.2-1.5× | Low | **DONE** (§4) |
| 5 | Full Mixed Precision | 1.5-2× | Medium | **DONE** (§5) |
| 6 | `torch.while_loop` | 1.5-2× | High | Blocked (PyTorch 2.7) |

**Total Potential**: ~10-15× combined (11s → ~1s CPU, 40s → ~3s CUDA)

> Session 20 note: zone fusion was skipped by explicit user decision (Session 8
> measured <1.33×; within-layer fusion breaks Gauss-Seidel zone coupling).
> Complete workspace pre-allocation was skipped because Session 19 deliberately
> reverted cached workspace tensors (they break under `torch.inference_mode()`);
> current code creates fresh tensors and should stay that way. Remaining
> actionable item: `torch.while_loop` migration once meta-consistency allows.

---

## Current Performance Baseline (After Sessions 1-19)

| Config | Time | Per-instance (K=10) |
|--------|------|---------------------|
| CPU f64 K=1 | 11s | 11s |
| CPU f64 K=10 | 11.9s | 1.2s (77x speedup) |
| CUDA f32 K=1 | 40s | 40s |
| CUDA f32 K=10 | 42s | 4.2s (90x speedup) |

**Batched scaling is excellent** (10× params = 1.05× wall time). Remaining gains must come from per-instance speedup.

**Target**: 1s full record (40× speedup needed on CUDA).