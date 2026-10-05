# Warp-Level Programming

A **warp** is CUDA's fundamental scheduling unit: 32 threads that execute in
lockstep on the same SM. Because all 32 threads share an instruction
pointer, they can exchange data directly through **warp shuffle**
instructions — register-to-register transfers that cost roughly one cycle
and require no shared memory, no barriers, and no synchronization.

cuda-oxide exposes the full warp intrinsic set through `cuda_device::warp`.
This chapter covers shuffle, vote, and the patterns they unlock: warp
reductions, broadcasts, scans, and ballot-based filtering.

:::{seealso}
[CUDA Programming Guide — Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-programming-guide/#warp-shuffle-functions)
for PTX encoding details and the full set of width variants.
:::

---

## Lane and warp identity

Every thread in a block has a **lane ID** (0–31) within its warp:

```rust
use cuda_device::warp;

let lane = warp::lane_id();     // 0..31, hardware register %laneid
let warp = warp::warp_id();     // threadIdx.x / 32
```

`warp_id()` is derived from `threadIdx.x / 32`. For multi-dimensional
blocks, this only accounts for the x dimension — which is usually fine,
since most kernels that care about lane identity use 1D blocks.

---

## Shuffle: register-to-register data exchange

The four shuffle variants let a thread read another thread's register
without going through memory:

| Function                    | What it does                | Source lane       |
| :-------------------------- | :---------------------------| :-----------------|
| `shuffle(val, src)`         | Read from a specific lane   | `src`             |
| `shuffle_xor(val, mask)`    | Read from `lane_id ^ mask`  | `lane_id ^ mask`  |
| `shuffle_down(val, delta)`  | Read from `lane_id + delta` | `lane_id + delta` |
| `shuffle_up(val, delta)`    | Read from `lane_id - delta` | `lane_id - delta` |

Each variant exists for both `u32` and `f32`:

```rust
let partner_val = warp::shuffle_xor_f32(my_val, 1);
let broadcast   = warp::shuffle_f32(my_val, 0);     // lane 0's value to all
let neighbor    = warp::shuffle_down_f32(my_val, 1); // next lane's value
```

All shuffles are **warp-synchronous** — they implicitly synchronize the
warp. No `sync_threads()` is needed, and in fact calling `sync_threads()`
inside a shuffle pattern would be both unnecessary and wasteful.

---

## Warp reduction

The most common shuffle pattern is a **butterfly reduction**: in
⌈log₂(32)⌉ = 5 steps, every lane accumulates the sum (or min, max, etc.)
of all 32 values. No shared memory, no barriers, five shuffle steps.

```{figure} images/warp-shuffle-reduction.svg
:align: center
:width: 100%

Butterfly reduction using `shuffle_xor`. At each step, lanes exchange
values with their XOR partner and add. After 5 steps (masks 16, 8, 4, 2,
1), lane 0 holds the sum of all 32 values.
```

```rust
use cuda_device::warp;

fn warp_reduce_sum(mut val: f32) -> f32 {
    val += warp::shuffle_xor_f32(val, 16);
    val += warp::shuffle_xor_f32(val, 8);
    val += warp::shuffle_xor_f32(val, 4);
    val += warp::shuffle_xor_f32(val, 2);
    val += warp::shuffle_xor_f32(val, 1);
    val
}
```

After the reduction, **all 32 lanes** hold the sum (because XOR is
symmetric — both partners accumulate). If you only need the result in
lane 0, you can use `shuffle_down` instead:

```rust
fn warp_reduce_sum_lane0(mut val: f32) -> f32 {
    val += warp::shuffle_down_f32(val, 16);
    val += warp::shuffle_down_f32(val, 8);
    val += warp::shuffle_down_f32(val, 4);
    val += warp::shuffle_down_f32(val, 2);
    val += warp::shuffle_down_f32(val, 1);
    val
}
```

With `shuffle_down`, only lane 0 holds the correct result — the others
hold partial sums. This is fine when only lane 0 writes the output.

:::{tip}
Need a block-wide reduction? Reduce within each warp using shuffles, write
the 32 per-warp results to shared memory, `sync_threads()`, then reduce
the warp-level results with one final warp. This hybrid approach is faster
than a pure shared memory tree because it eliminates 5 levels of barriers.
:::

### Integers on Ampere: one instruction instead of ten

Every reduction above -- the hand-written butterflies, and the generic
`cooperative_groups::warp_reduce` -- costs five shuffles plus five combines.
For **integer** reductions on `sm_80` and newer there is a single hardware
instruction, `redux.sync`, exposed one function per operation:

| function | reduces |
|:---------|:--------|
| `warp::redux_sync_add(mask, v)` | wrapping sum, `u32` |
| `warp::redux_sync_min_u32` / `_min_i32` | minimum |
| `warp::redux_sync_max_u32` / `_max_i32` | maximum |
| `warp::redux_sync_and` / `_or` / `_xor` | bitwise |

```rust
use cuda_device::warp;

const FULL_MASK: u32 = 0xffff_ffff;

// The whole reduction. One instruction, every lane holds the result.
let total = warp::redux_sync_add(FULL_MASK, my_value);
```

Measured on an A10G (`sm_86`), 1M threads each performing 64 full-warp `u32`
sum reductions: the butterfly form takes 159.4 µs per launch, the `redux.sync`
form 33.6 µs -- **4.74x**, with bit-identical results. Wrapping `u32` addition
is associative modulo 2³² and min/max/and/or/xor are exact, so reduction order
cannot change the answer.

That figure is the primitive measured in isolation: the probe kernel does
almost nothing but reduce. A kernel that reduces once at the end of a
memory-bound pass will see very little. Reach for this where reductions sit in
an inner loop.

Three constraints:

- **Integers only on Ampere.** There is an `f32` form, but it needs Blackwell;
  see the next subsection. `f64` has no `redux.sync` form at all, so `f64`
  reductions keep the butterfly everywhere.
- **A full, converged warp.** All 32 lanes must be live and must reach the same
  call with the same mask -- the same contract the shuffle reductions already
  require. Sub-warp tiles and short tail warps keep the shuffle path (see
  `reduce_sum_f32_partial` and friends for the tail case).
- **It will not assemble below `sm_80`.** Gate the call the way the `redux_sum`
  example does, on `ctx.compute_capability()`, or restrict the build to Ampere
  and newer.

`warp_reduce` does not select this form for you today; making it do so
automatically needs a way for device code to know its target architecture,
which is the open question in
[#811](https://github.com/NVlabs/cuda-oxide/issues/811).

### Floats on Blackwell: `redux.sync` for `f32`

`redux.sync` gained an `f32` min/max form, exposed as eight functions. Each
takes `(mask: u32, value: f32)` and returns `f32`, like the integer ones:

| function | reduces |
|:---------|:--------|
| `warp::redux_sync_min_f32` / `warp::redux_sync_max_f32` | minimum / maximum, NaN inputs ignored |
| `warp::redux_sync_min_abs_f32` / `warp::redux_sync_max_abs_f32` | same, over the absolute values (`.abs`) |
| `warp::redux_sync_min_nan_f32` / `warp::redux_sync_max_nan_f32` | NaN-propagating (`.NaN`) |
| `warp::redux_sync_min_abs_nan_f32` / `warp::redux_sync_max_abs_nan_f32` | both modifiers |

So unlike the integer family this is not one function per operation: `.abs`
and `.NaN` are independent instruction modifiers, and each combination is its
own entry point.

The `.NaN` forms matter for reductions that must not lose a NaN. The plain and
`.abs` forms *ignore* NaN inputs; the `.NaN` forms propagate NaN when any lane
contributes one. Pick deliberately. The two differ only when a lane holds a
NaN, and that is when the choice matters.

Say one lane holds `NaN` and the smallest ordinary value in the warp is `-1.0`:

- `redux_sync_min_f32(mask, v)` returns `-1.0` (the NaN is ignored)
- `redux_sync_min_nan_f32(mask, v)` returns `NaN` (the NaN propagates)

**This is Blackwell-only, and narrower than "sm_100+".** The f32 forms need
PTX 8.6 and exist only on `sm_100a`, `sm_100f`, `sm_103a`, and `sm_103f`. They
are not available on `sm_110a` or `sm_120a`, so consumer Blackwell is out.
Build for one of those four targets or compilation fails.

The `redux_f32` example is the working reference. It builds for `sm_100a`, so
CI compiles it and checks the emitted PTX rather than running it. Its `main`
skips unless the device reports `sm_100` and nothing else.

---

## Broadcast

Broadcasting lane 0's value to all lanes is a single shuffle:

```rust
let leader_val = warp::shuffle_f32(my_val, 0);
```

Any lane can be the source. This replaces the shared-memory pattern of
"lane 0 writes to shared, sync, all lanes read" — one instruction instead
of three operations.

---

## Inclusive prefix sum (scan)

An **inclusive scan** computes a running total: lane `i` holds the sum of
values from lanes 0 through `i`. The pattern uses `shuffle_up`:

```rust
fn warp_inclusive_scan(mut val: f32) -> f32 {
    let mut offset = 1u32;
    while offset < 32 {
        let n = warp::shuffle_up_f32(val, offset);
        if warp::lane_id() >= offset {
            val += n;
        }
        offset *= 2;
    }
    val
}
```

After 5 steps, each lane holds the prefix sum up to and including its own
value. This is the building block for stream compaction, histogram building,
and parallel scan algorithms.

---

## Vote: warp-wide predicates

Vote operations let the warp collectively evaluate a boolean condition:

| Function              | Returns                                                              |
| :-------------------- | :------------------------------------------------------------------- |
| `warp::all(pred)`     | `true` if *every* active lane's `pred` is `true`                     |
| `warp::any(pred)`     | `true` if *any* active lane's `pred` is `true`                       |
| `warp::ballot(pred)`  | A `u32` bitmask — bit `i` is set if lane `i`'s `pred` is `true`      |
| `warp::popc(pred)`    | Population count: how many active lanes have `pred == true`          |

### Filtering with ballot

A common pattern is to compact an array, keeping only elements that pass
a predicate. `ballot` + `popc` gives you the count and the per-lane write
offset:

```rust
use cuda_device::{DisjointSlice, kernel, thread, warp};

#[kernel]
pub fn compact_positive(
    input: &[f32],
    mut output: DisjointSlice<f32>,
    mut count: DisjointSlice<u32>,
) {
    let idx = thread::index_1d();
    let val = input[idx.get()];
    let is_positive = val > 0.0;

    let mask = warp::ballot(is_positive);
    let lane = warp::lane_id();

    // Count bits below this lane to get the write position
    let offset = (mask & ((1u32 << lane) - 1)).count_ones();

    if is_positive {
        unsafe {
            *output.get_unchecked_mut(offset as usize) = val;
        }
    }

    // Lane 0 records total count for this warp
    if lane == 0 {
        unsafe {
            *count.get_unchecked_mut(warp::warp_id() as usize) = mask.count_ones();
        }
    }
}
```

The `ballot` mask encodes the entire warp's predicate result in one
register. No communication, no shared memory — the hardware computes it in
a single cycle.

---

## When to use warp primitives vs. shared memory

| Task                        | Warp shuffle                 | Shared memory                  |
| :-------------------------- | :--------------------------- | :----------------------------- |
| Reduce 32 values            | 5 shuffles, ~5 cycles        | 5 loads + 5 syncs, ~50 cycles  |
| Reduce 256 values           | Shuffle + 1 sync + shuffle   | Tree reduction, ~10 syncs      |
| Stencil (neighbor access)   | `shuffle_up`/`shuffle_down`  | Good for 2D stencils           |
| Data visible to other warps | Not possible                 | Required                       |
| Random access patterns      | Not supported                | Index freely                   |
| Persists across warp        | Not applicable               | Persists for block lifetime    |

The rule of thumb: if the data fits in a single warp (32 elements) and
the access pattern is regular, shuffle is faster. If you need cross-warp
communication, larger data sets, or random access, shared memory is the
tool.

---

## A complete example: warp-level dot product

Putting shuffles and votes together, here is a kernel that computes the
dot product of two vectors using warp reduction:

```rust
use cuda_device::{DisjointSlice, kernel, thread, warp};

#[kernel]
pub fn warp_dot_product(
    a: &[f32],
    b: &[f32],
    n: u32,
    mut result: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();

    // Each thread computes one element of the pointwise product
    let product = if idx.get() < n as usize {
        a[idx.get()] * b[idx.get()]
    } else {
        0.0f32
    };

    // Warp-level reduction
    let mut sum = product;
    sum += warp::shuffle_xor_f32(sum, 16);
    sum += warp::shuffle_xor_f32(sum, 8);
    sum += warp::shuffle_xor_f32(sum, 4);
    sum += warp::shuffle_xor_f32(sum, 2);
    sum += warp::shuffle_xor_f32(sum, 1);

    // Lane 0 of each warp writes its partial sum
    if warp::lane_id() == 0 {
        unsafe {
            *result.get_unchecked_mut(warp::warp_id() as usize) = sum;
        }
    }
}
```

For a full dot product, launch a second pass that reduces the per-warp
results — either with another warp kernel or with atomics. The first
pass eliminates the vast majority of the work using only register shuffles.

:::{seealso}
- [Shared Memory and Synchronization](shared-memory-and-synchronization.md) —
  the block-wide counterpart for larger-than-warp operations
- [Tensor Memory Accelerator](tensor-memory-accelerator.md) — hardware that
  accelerates the global→shared data movement that feeds these patterns
:::
