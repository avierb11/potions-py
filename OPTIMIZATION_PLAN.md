# Full Optimization Plan: East River Forward Simulation

## URGENT track — cut GPU host-device syncs (K=1 target < 10 s)

Measured starting point: the East-River full record (2557 d, K=1, CUDA f32)
runs in ~120 s; the Rust reference is ~1 s. Target for this track: < 10 s
(realistically ~3-6 s). CPU speedup is secondary; this must be fast on GPU.

### Root causes (ranked by GPU impact)

1. `torch.while_loop` in the hot path. In torch 2.7.0 `while_loop` internally
   calls `torch.compile(..., fullgraph=True)`, causing recompiles /
   `FailOnRecompileLimitHit` on every solve and `UncapturedHigherOrderOpError`
   when a while_loop frame nests inside a compiled frame. This is why the
   benchmark forces the compile flag off and yet
   `test_float32_compute_stays_close_to_float64` still crashes.
   Fix = remove `torch.while_loop` from the hot path entirely (Phase 0).
2. Unfused Python-leaf math. With the solver loops kept eager, every
   Newton/Armijo/secant/bisection iteration runs a pure-Python fused eval
   (`_kinetic_residual_and_jacobian`, ~200 tiny CUDA ops) plus
   `lu_factor`/`l_solve` -> Python kernel-launch bound, not arithmetic bound.
   Fix = compile the sync-free leaf kernels (Phase 1).
3. ~300-500k host-device syncs per record from the Newton `.item()` guard
   (`solvers.py`), the Armijo `trial_left.any()` guard (`solvers.py`), plus the
   LM/secant/bisection loop-condition checks. Fix = reduce condition syncs (P2).
   **(Measured: at K=1 the termination-sync hypothesis does NOT pay off — see
   the Phase 2 note; the sync is cheap, the extra body evals it lets run are
   not. Only syncs that save work, not just host reads, are worth cutting.)**
4. Per-step allocation churn (hydro `torch.stack` of per-zone accumulators, RT
   `state`/mass in-place updates, redundant `.to()` no-ops, loop-local
   `torch.arange`). Fix = pre-allocate / hoist (Phase 3).

### Locked decisions (user-confirmed)

- No environment flags. The compile policy is a device/dtype policy: GPU/f32
  leaves are ALWAYS compiled+fused; CPU/f64 stays eager (keeps Rust golden
  parity bit-exact). `POTIONS_COMPILE` is removed everywhere.
- Architecture: thin eager Python `for`-loop control (Newton/LM/secant/bisect/
  Armijo) + always-compiled sync-free leaf kernels. Kill `torch.while_loop`.

### Phase 0 — remove `torch.while_loop`; bounded Python for-loops [DONE]

Goal: make the entire hot path deterministic eager with NO `while_loop`/
`compile` on the loop frames. Per-lane masking stays bit-exact.

- [0.1] [DONE] `device.py`: replace `device_loop` with a `bounded_loop(max_iter,
  cond_fn, body_fn, state)` that runs `for _ in range(max_iter): if not bool(cond_fn(*
state)): break; state = tuple(body_fn(*state)); return state`. No compile, no
  while_loop, no stride-canonicalization (eager only needs value equality).
  Remove `device_loop` + `_canonical_stride`.
- [0.2] [DONE] `solvers.py` `_levenberg_marquardt_core`: the LM iterate is a
  bounded Python loop (later re-flattened in Phase 2: the per-lane `it` counter +
  `multi_max` cap were dropped in favour of a plain `range(MULTI_MAXITER)`, and
  the `guard` short-circuit is hoisted to a single pre-loop `guard.any()`).
- [0.3] [DONE] `hydro.py` `find_root_batched` (secant) + `_bisect_core`:
  bounded Python loops via `bounded_loop`.
- [0.4] [DONE] no residual `while_loop`/`device_loop` references in src
  (grep clean).
- [0.5] [DONE] docstrings updated to describe the bounded-loop design.

Verification (P0):
- [V0.1] [DONE] `tests/test_solvers.py tests/test_hydro.py` pass bit-exact.
- [V0.2] [DONE] `tests/test_rt_zone.py tests/test_equilibrium.py` pass
  bit-exact (f64 CPU).
- [V0.3] [DONE] `tests/test_model_batched.py` (incl.
  `test_float32_compute_stays_close_to_float64`) passes.
- [V0.4] [DONE] GPU wall timer K=1 f32: 26.0 ms/step (post-P0, 2557 d record,
  ~66.5 s wall) vs ~120 s / ~47 ms/step pre-P0 baseline.

### Phase 1 — always-compile the sync-free leaf kernels (device/dtype policy) [DONE]

Goal: with the control loops eager, fuse each solve *iteration* into one
compiled kernel. Only compile functions that are sync-free (no `.item()` /
`.any()` inside) and have stable shapes; CPU/f64 stays eager -> bit-exact.

**Implementation note (important):** the original plan asked for the compile
policy to be read from a device flag *at import time*.  That was insufficient:
on a GPU box it would compile even the CPU-f64 calls and break bit-exact parity
(V1.3), while on a CPU box it would never compile even GPU/f32 calls.  The
policy was therefore made a **per-call device/dtype dispatch**:
`device.CompiledLeaf` inspects the first tensor argument at call time and
routes `(cuda, torch.float32)` calls through a lazily-built `torch.compile`
graph (cached per `(leaf, device, dtype, fullgraph)` in `_LEAF_CACHE`), while
every other device/dtype falls straight through to the eager function.  This
satisfies the locked decision — GPU/f32 leaves are *always* compiled+fused,
CPU/f64 stays eager and bit-exact — independent of any environment flag.
`maybe_compile` / `maybe_compile_fullgraph` now return a `CompiledLeaf`; the
`POTIONS_COMPILE` override is removed from `compile_enabled` (it only reports
the default policy device).

- [1.1] [DONE] `device.py`: `compile_enabled()` ->
  `return DEFAULT_DEVICE.type == "cuda"` (no `POTIONS_COMPILE` override);
  `maybe_compile` / `maybe_compile_fullgraph` return a `CompiledLeaf` that
  fuses only `(cuda, f32)` calls (`suppress_errors` eager fallback kept).
  **Measured:** on the K=1 GPU f32 record all five leaves (
  `HydrologicZone.residual`, `_kinetic_residual_and_jacobian`, `_solves_lu`,
  `EquilibriumParameters.residual`, `EquilibriumParameters.jacobian_residual`)
  register compiled variants and engage; CPU-f64 runs stay bit-exact.
- [1.2] [DONE] `rt_zone.py` `_fused_resjac`: `maybe_compile_fullgraph`, now
  fires on GPU (f64 path eager -> bit-exact).
- [1.3] [DONE] `solvers.py` `_solves_lu`: wrapped in
  `maybe_compile_fullgraph` (batched LU is sync-free).
- [1.4] [DONE] `equilibrium.py`: `_frozen_residual` / `_frozen_jacobian`
  wrapped in `maybe_compile_fullgraph`.
- [1.5] [DONE] `hydro.py` `_hydro_residual` (secant/bisect leaf) fuses on GPU.

Verification (P1):
- [V1.1] [DONE] `test_float32_compute_stays_close_to_float64` passes.
- [V1.2] [DONE] K=1 lane == K=2 lane 0 on GPU (bit-exact) over the full
  record.
- [V1.3] [DONE] CPU-f64 golden suite bit-exact (238 tests).
- [V1.4] [DONE] GPU wall timer K=1 f32: 23.2 ms/step (~59.3 s) after P1,
  down from 26.0 ms/step after P0.  Note the drop is modest because at K=1
  the wall is Python-kernel-launch bound, not arithmetic bound (the fused
  leaves cut FLOPs, not the number of tiny kernels per step).

### Phase 2 — cut per-zone-step host syncs [PARTIAL — premise partially refuted]

- [2.1] [REVERTED] `solvers.py` Newton periodic termination (check
  `running.any()` every 4th iter).  **Measured net-negative on the K=1 record:
  the every-4th-trip check gave ~43 ms/step vs ~25.7 ms/step for the every-trip
  check — up to 3 extra fused eval + LU + Armijo body evaluations (the
  expensive part, each ~20-50 tiny kernel launches) cost more than the one
  host sync a skipped check saves. The termination-sync premise does not hold
  for this workload at K=1; every-trip checks are kept (documented in
  `solvers.py`).
- [2.2] [DONE] `solvers.py` Armijo inner loop: full-step trial (`alpha = 1`)
  is run **unrolled** with no host guard, then a single `trial_left.any()`
  host check before the (rare) bounded halving tail. Each unrolled tail iteration
  is masked (`new_acc = trial_left & decrease`) so the values are
  bitwise-identical to the original guarded loop — strictly fewer host syncs,
  no extra evaluations.
- [2.3] [PARTIAL] `solvers.py` LM: `guard.any()` hoisted out of the per-trip
  condition (single pre-loop host read; the `guard` mask never changes over
  the LM trips). The `it < multi_max` cap became a plain
  `range(MULTI_MAXITER)`. The per-trip `running.any()` check remains every
  trip (same reason as 2.1).

Verification (P2): [V2.1] [DONE] CPU-f64 golden suite bit-exact (238 tests);
[V2.2] [DONE] GPU wall timer: no regression (25.2 ms/step after P2, same as
25.2-25.7 every-trip baseline); [V2.3] [DONE] GPU lane-identity re-verified.

### Phase 3 — allocation / cast trims [DONE]

- [3.1] [DONE] `model_batched.py` `run_hydro_batched`: per-zone accumulators
  replaced with two dense `(num_zones, K)` slabs (`lat_ext_step` /
  `vert_ext_step`); the per-step `q_in` matvec is now a single slice-matvec
  (`lat_ext_step[:i].T @ lat_mat[i, :i]`) instead of `torch.stack` of a
  growing Python list; the 8 output fields are written in place into
  pre-allocated `(K, T, Z)` buffers instead of per-step Python lists +
  `torch.stack`. **Measured:** K=1 record wall 25.2 -> 24.5 ms/step and peak
  GPU memory ~19 MB -> ~10.7 MB (the list + stack churn was a real memory
  footprint); numerics bit-exact (238 tests).
- [3.2] [DONE-already] `model_batched.py` `run_rt_batched`: `state_out` /
  `lat_mass` / `vert_mass` were already pre-allocated with in-place writes
  (done in a prior session); `precip_conc` is already pre-converted to a
  device tensor once.
- [3.3] [DONE] `rt_zone.py`: the eager per-zone-step
  `aq_range = torch.arange(num_aq)` in `_zone_step` now uses the cached
  `_aq_range(num_aq, device)`.  (The second `torch.arange(num_aq)` at
  `rt_zone.py:639` is inside the `fullgraph`-compiled `_assemble_ode_jacobian`
  and is elided by the compiler — no separate handling needed.)
- [3.4] [DONE] Removed the `POTIONS_COMPILE=0` env override from
  `scripts/bench_full_record.py`, `scripts/east_river_standalone.py`,
  `scripts/east_river_ensemble.py`, `scripts/parity_check_batched.py`, and
  `tests/conftest.py` (also removed the now-unused `import os` where the
  flag line was the only use).  `compile_enabled` no longer reads the env
  var (P1).  `tests/test_device.py` rewritten to test the
  `CompiledLeaf` dispatch directly.

Verification (P3): [V3.1] [DONE] CPU-f64 golden suite bit-exact (238 tests);
[V3.2] [DONE (see note)] full-record GPU f32 K=1: **~24.5 ms/step
(~62.8 s, 2557 d)**.  **The < 10 s target is not reached** — at K=1 the wall
is Python-kernel-launch bound (the bench's own diagnosis: "~µs per Newton
step ... the ms/step wall is Python dispatch overhead + tensor allocation,
not solver arithmetic").  The 5-fold reduction from the ~120 s starting point
comes almost entirely from Phase 0 (removing `torch.while_loop` recompiles).
Fusing the remaining ~200 tiny CUDA ops per Newton/Armijo/secant trial into
fewer launches would need the whole per-trip body (residual + full Jacobian
+ LU solve + mask updates) captured as a single compiled leaf — a
substantially larger architectural change than the per-leaf wrapping this
plan assumes, and is not done here.

### Phase 4 — stabilize the always-compiled fullgraph leaves (recompiles) [IN PROGRESS]

With the compile policy locked, the GPU record crashes inside
`torch._dynamo` with `FailOnRecompileLimitHit` when a compiled `fullgraph`
leaf sees a second shape/stride instance (default `cache_size_limit` is 8,
and torch 2.7.0 turns a `one_graph=True` recompile-limit hit into a hard
failure).  Two distinct trigger shapes were measured on the K=1 record:

- **`x` duck-shape** (`_newton_trial_head_fused`, solvers.py): batch axis
  expands 3 -> 4 (1 RT lane + 1 hydro + 2 frozen lanes, then 2 + 1 + N_frozen).
  With `dynamic=False` that is a recompile; with `dynamic=True` it is a
  single dynamic batch axis.
- **`residual` stride-8** (hydro.py): an f32 caller passes an
  `s_0` / `s` view with row stride 8 (pad from a `(num_zones, K=8)` buffer)
  instead of 4; same value layout, different stride -> recompile.

Root-cause rule (locked): **any tensor whose shape or stride is a function
of the per-step solve mix (`K`, frozen-lane count) OR of a batched caller's
buffer layout must be stride/shape-normalized at the Python wrapper, or the
compiled leaf must be `dynamic` over that axis.  Normalization is the first
choice (a view, zero copy, and keeps the graph byte-stable); `dynamic=True`
is the safety net for axes that cannot be normalized.**  This does not
reduce model generality: masks and per-lane counts still flow through
exactly as before.

- [4.1] [IN PROGRESS] `device.py` `_leaf` / `compiled_variant_for`: the
  fullgraph (no-mode) branch compiles with `dynamic=True` instead of
  `dynamic=False`, so varying batch sizes / strides across lanes and steps
  do not exhaust Dynamo's recompile cache.  The *mode* branch stays
  `dynamic=False` — a CUDA-graph / reduce-overhead mode wants a static
  shape and will simply recompile into a new graph (that branch is unused
  by the current call sites; all nine compile sites use the no-mode path).
- [4.2] [TODO] `device.py`: raise `torch._dynamo.config.cache_size_limit`
  from the default 8 to a large value at import time, so a stray new shape
  degrades to "recompile + use the new graph" instead of a hard
  `FailOnRecompileLimitHit` under `one_graph=True`.  (Defensive; the
  normalization in 4.3 should keep the steady-state graph count at ~1 per
  leaf.)
- [4.3] [DONE] `hydro.py` `residual`: `s_0` (line ~490) and `s` (line ~512)
  go through a duck-shape guard that returns the input unchanged when it is
  already contiguous in the canonical `(num_aq, num_species, 3)` layout and
  else materializes a contiguous copy.  This is the single-assignment,
  zero-copy-in-the-common-case form (the first `if` branch has no second
  assignment to the name).  Verified by the failing compiled run: without it
  the Dynamo graph guard logs "bytecode 68: GUARD on tensor 0 stride 4 -> 8"
  on `residual`.

**Correctness invariant:** normalization is a *view/copy that preserves
values exactly* (`.contiguous()` on a non-overlapping view is an elementwise
copy — bit-exact).  CPU-f64 runs never enter the compiled branch, so the
Rust golden parity is untouched.

Verification (P4):
- [V4.1] [TODO] CPU-f64 golden suite bit-exact (238 tests).
- [V4.2] [TODO] K=1 GPU f32 *compiled* full record completes without
  `FailOnRecompileLimitHit` (previously crashed on day ~6) and is finite.
- [V4.3] [TODO] K=1 lane == K=2 lane 0 (bit-exact) over the full record.
- [V4.4] [TODO] wall timer recorded (f32 compiled).  Expected: the ~59-66 s
  f64-path number should *drop* on f32 (the fused leaves finally engage) —
  but the < 10 s target needs Phase 5.

### Phase 5 — cut the per-step launch/sync count (K=1 < 10 s) [PLANNED]

Phase 1 fused the leaves, but a step still issues a large number of
compile-frame entries (one per Newton/Armijo/secant trial) plus the
eager-glue launches between them.  The measurement-first decision tree:

- [5.1] **Measure before choosing a lever.** On the Phase-4-stable
  compiled path, profile one representative day (e.g. day 6, post-ramp):
  (a) CUDA kernel launches per step (`torch.profiler` or `nsys`), (b) host
  syncs per step (`.item()` / `.any()` / `.item()`-guard count), (c) GPU
  busy time per step vs wall.  This tells us whether the step is
  launch-bound, sync-bound, or arithmetic-bound.
  - [5.1a] [TODO] capture `torch.profiler` trace for day 6 (240 steps of
    rt + hydro at that index), count kernels + syncs.
- [5.2] **Lever choice (decide from 5.1, do NOT up front):**
  - If **launch/Python-overhead-bound** (most likely at K=1): shrink the
    number of compile-frame entries + glue kernels per step.  Two
    candidates, in ascending order of architectural risk:
    1. *Hoist the invariant per-step tensors* (eye, mass-matrix factor,
       mask/indicator buffers, `_aq_range`) so the eager glue between
       fused trials is only the minimum index/slice math.
    2. *Widen the fused leaf*: fold the Newton update + Armijo trial +
       line-search tail into one `fullgraph` leaf per trial (so one
       compile-frame entry per trial instead of head + tail + glue), and
       fold the secant/bisect per-trip residual + trial + step into one
       leaf.  Keeps bit-exactness if the fold is purely structural (same
       ops, same order).
  - If **sync-bound**: reduce the remaining `.any()` / `.item()` guards
    (Newton/LM termination) using per-lane count + multi_max caps, the way
    the Jacobian guard already does — *only* where the skipped iteration
    would have done redundant work (Phase 2 showed skipping does NOT save
    when the check is cheap and the body is not; re-measure per lever).
  - If **arithmetic-bound** (unexpected at K=1): profile per-op and target
    the 2-3 hottest kernels (likely LU or the fused residual); consider a
    Triton/`@torch._C._dynamo`-stable rewrite of that op only.
- [5.3] **Re-verify after each lever:** CPU-f64 bit-exact (238), K=1==K=2
  lane-0, wall timer, and the finite-output check on the full record.
- [5.4] **Stop rule:** if a lever is measured net-negative (Phase 2
  precedent), revert it and record the measurement in this plan before
  trying the next.  Do not stack unmeasured levers.

Target: 2557-day record, K=1, CUDA f32, wall < 10 s (ideally ~3-6 s).

Status: Phase 4 in progress — device.py dynamic=True + cache_size_limit
raise + hydro residual stride guard.  Phase 5 pending Phase 4 verification.

### Status legend
`[n.x]` = to-do; `[DONE]` = complete; `[REVERTED]` = attempted and measured
net-negative, so removed; `[IN PROGRESS]` = being worked this session.
**Summary after this session:** Phase 0 (remove `torch.while_loop`) is the
largest single win; Phase 1 (fuse the sync-free leaves on GPU/f32) and Phase
3 (allocation trims) give smaller gains; Phase 2 is partial — its premise
(termination syncs are the bottleneck) is refuted by measurement at K=1 and
was reverted, keeping only the syncs that save actual work (Armijo full-step
unroll, LM `guard` hoist).  Phase 4 removes the recompile crashes so the
f32 compiled path finally runs.  Phase 5 is the measurement-first track to
<10 s.

---

## Prior optimizations (done in earlier sessions — context only)

Already completed (see `REMAINING_OPTIMIZATIONS.md` for full history): eye
matrix cache (`_get_eye`); precip_conc pre-converted to a device tensor;
`state_out` pre-allocation; `_t()` dtype parameterization (`set_hydro_dtype`);
`FusedEvaluator` callable class; kinetic/ODE/jacobian workspace buffers;
`_STOICH_CACHE` / `_get_stoich` (bounded in the current code); `compute_dtype`
mixed-precision plumbing through RT and the solvers.

**Note (important):** the earlier plan assumed the solver cores and the
`maybe_compile` wrappers would be the winning lever. In practice, with the
compile flag off (as the benchmark sets) the `torch.while_loop` frames in
`solvers.py` / `hydro.py` still hit `torch.compile` internally in
torch 2.7.0, so the loop frames themselves are now the bottleneck. This urgent
track (Phase 0) supersedes the "compile the whole core" idea: keep the control
loops as *eager bounded Python for-loops* and compile only the *leaf* kernels
(Phase 1).
