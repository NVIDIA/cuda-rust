# coop_groups_demo

## Cooperative Groups — End-to-End Smoke Test

21 end-to-end checks for `cuda_device::cooperative_groups`, exercising every
supported group type, every warp collective, and the full
`(reduce/scan, warp/block, op, type)` matrix.

The established checks have real-hardware coverage. The synchronized-shuffle
source-boundary assertions added during the generated-intrinsics migration
currently have compile evidence only. No GPU was available for that batch, so
those new assertions remain unexecuted.

The crate doubles as a hand-debug aid: the generated `coop_groups_demo.ptx`
is the canonical reference for "what does this typed handle actually
lower to" across the whole module.

## Build and Run

```bash
cargo oxide run coop_groups_demo

# or, picked up by the workspace smoketest runner:
just smoketest -o '^coop_groups_demo$'
```

A single console line at the end:

```text
=== All cooperative-groups checks PASSED ===
```

is the success marker the smoketest runner looks for. Any individual
check that fails prints the offending tile/lane/op and exits non-zero.

## What's Covered

The 21 checks split into three layers:

### Layer 1 — raw warp/grid intrinsics (4 checks)

| Kernel              | What it verifies                                               |
|:--------------------|:---------------------------------------------------------------|
| `active_mask`       | every lane in a full warp sees `0xFFFFFFFF`                    |
| `match_any_sync`    | bucket detection: `value = lane / 4` ⇒ `0xF << (group * 4)`    |
| `match_all_sync`    | constant input ⇒ full warp mask                                |
| `grid_sync`         | every block sees every other block's pre-barrier marker write  |

`grid_sync` (and `typed_grid_sync` in Layer 2) run as cooperative
launches through the typed path: both kernels carry
`#[cooperative_launch]` inside a `#[cuda_module]` module, so their
generated launch methods submit via `cuLaunchKernelEx` with
`CU_LAUNCH_ATTRIBUTE_COOPERATIVE`. The same module also holds
`test_cluster_coop_grid_sync`, which combines `#[cluster_launch(2, 1, 1)]`
with `#[cooperative_launch]` and a `#[launch_contract]`. That kernel is
prepared and launched through the safe `PreparedLaunch` path. Pre-Hopper
devices never get this far: `main` prints its `thread block clusters
require sm_90+` skip line and exits before launching anything.

### Layer 2 — typed cooperative-groups handles (5 checks)

| Kernel                     | What it verifies                                                        |
|:---------------------------|:------------------------------------------------------------------------|
| `typed_warp32_ballot`      | full-warp ballot plus `match_any` / `match_all` fast-path semantics      |
| `typed_warp16_ballot`      | sub-warp ballot is **tile-relative** (16-bit mask, not 32-bit)          |
| `typed_warp16_shfl`        | tile-relative shuffle/match masks plus sparse coalesced ballot/match packing and shuffle sources |
| `typed_grid_sync`          | `this_grid().sync()` matches the raw `grid::sync()` semantics           |
| `typed_grid_rank`          | `this_grid().thread_rank()` is the identity permutation `0..total`      |

### Layer 3 — reductions and scans (12 checks)

Every cell in the matrix below is a separate launched kernel; each
runs all listed ops in the same kernel call and the host verifies
every (op, lane/thread) pair.

| Primitive       | `u32` ops                              | `i32` ops          | `f32` ops          |
|:----------------|:---------------------------------------|:-------------------|:-------------------|
| `warp_reduce`   | Sum, Min, Max, BitAnd, BitOr, BitXor   | Sum, Min, Max      | Sum, Min, Max      |
| `warp_scan`     | Sum, Min, Max, BitAnd, BitOr, BitXor   | Sum, Min, Max      | Sum, Min, Max      |
| `block_reduce`  | Sum, Min, Max, BitAnd, BitOr, BitXor   | Sum, Min, Max      | Sum, Min, Max      |
| `block_scan`    | Sum, Min, Max, BitAnd, BitOr, BitXor   | Sum, Min, Max      | Sum, Min, Max      |

All scans are **inclusive**: thread `i` receives the reduction of
values from threads `0..=i`. Block-scoped variants take a
`*mut SharedArray<T, NUM_WARPS>` for warp-totals scratch. `NUM_WARPS`
is a capacity: it must be at least `ceil(block_threads / 32)`, and extra
slots are allowed. Partial final warps are supported. The demo uses
`&raw mut SMEM` so callers don't need an `unsafe` block.

## Layout Conventions

Block tests use `block_dim = 96` (3 warps), 4 blocks. The choice is
deliberate:

- `BitAnd` input `!(1 << lane)` — AND across 3 of each bit clears all bits.
- `BitOr`  input `1 << lane`    — OR  across 3 of each bit sets all bits.
- `BitXor` input `1 << lane`    — XOR across 3 (odd count) sets all
  bits. A 32-warp block would have an even count and degenerate to
  identity, hiding bugs in the across-warps phase.

Output buffer layout for reduce/scan kernels:

| Kernel kind          | Cells per row | Row index                      |
|:---------------------|:--------------|:-------------------------------|
| `warp_reduce_<T>`    | 6 (u32) / 3   | warp index (one row per warp)  |
| `warp_scan_<T>`      | 6 (u32) / 3   | global thread id               |
| `block_reduce_<T>`   | 6 (u32) / 3   | block index (one row per block)|
| `block_scan_<T>`     | 6 (u32) / 3   | global thread id               |

`u32` rows carry six cells (`Sum, Min, Max, BitAnd, BitOr, BitXor`);
`i32`/`f32` rows carry three (`Sum, Min, Max`).

## What's Intentionally Out of Scope

- **`Cluster<...>` checks** — covered by the dedicated `cluster` example.
- **DSMEM via `dsmem_read_u32`** — also in the `cluster` example.
- **Sub-warp `WarpTile<N>` for `N` other than 32 and 16** — the
  `tiled_partition` machinery is generic over `N ∈ {1, 2, 4, 8, 16, 32}`
  with a `const { assert!(...) }` validating the choice; the structural
  correctness of the lower powers is covered by the `WarpTile<16>`
  checks in Layer 2 plus the warp-reduce/scan tests in Layer 3 (which
  exercise the same mask machinery at `N = 32`).
- **General `CoalescedThreads` algorithms** — `typed_warp16_shfl` covers sparse
  even-lane and irregular groups, including group-relative ballot/match packing
  and invalid-source shuffle self fallback. Broader divergent layouts remain
  exercised by `hashmap_v2` and `hashmap_v3`'s typed warp-find paths.

## Hardware Requirements

- **Minimum GPU**: Hopper (sm_90) or newer. `main` skips cleanly below that
  (`if major < 9`) because the demo launches thread block clusters. The
  `match.any.sync`/`match.all.sync` primitives it also exercises are sm_70+,
  but the cluster launch is what sets the floor.
- **Cooperative launch**: any GPU that reports
  `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH) == 1`
  (essentially all post-Pascal GPUs).
- **CUDA Driver**: 11.0+.

## Inspecting the Generated PTX

```bash
# After running once, the .ptx lives next to the binary:
less cuda-oxide/crates/rustc-codegen-cuda/examples/coop_groups_demo/coop_groups_demo.ptx
```

Useful greps:

| What to look for                        | rg pattern                              |
|:----------------------------------------|:----------------------------------------|
| Per-kernel entry points                 | `^\.visible \.entry test_`              |
| Surviving typed wrapper functions       | `^\.visible \.func.*WarpTile`           |
| Butterfly shuffles inside warp_reduce   | `shfl\.sync\.bfly`                      |
| Up-shuffles inside warp_scan            | `shfl\.sync\.up`                        |
| Block barriers in block_reduce/scan     | `bar\.sync`                             |
| Cooperative-launch entry markers        | `\.entry test_(typed_)?grid_sync`       |

The reduction/scan kernels lower into their underlying `warp_reduce`/
`warp_scan` wrappers as standalone `.visible .func` definitions
(rustc's MIR `Inline` cost threshold currently keeps them outlined for
the larger ops). This is the same trade-off the typed warp paths in
`hashmap_v2` (`find_kernel_warp_typed`) and `hashmap_v3` (the
`tile_32` / `tile_16` find kernels) make; functional correctness is
unaffected.

The typed i32/f32 shuffle leaves are generated as `i0050`-`i0057`; the
cooperative-group API remains handwritten. For sub-warp and sparse groups, the
wrapper substitutes the calling lane when XOR, down, or up would select a lane
outside the member mask. An invalid coalesced `shfl` source rank also selects
the caller. This avoids an undefined PTX read while preserving the existing
API and the example's success output.

## Adding a New Check

The recipe is identical for every test in the file:

1. Write a `#[kernel]` function that performs the operation under test
   and writes its result into a `DisjointSlice<T>`.
2. In `main`, allocate a `DeviceBuffer<T>` of the right size, launch
   via `unsafe { cuda_launch! { ... } }` (the macro cannot check the
   argument list, so each site carries a SAFETY comment), copy back
   with `to_host_vec`, and compare each cell against a host-computed
   expected value.
3. Print one summary line ending in `yes` or `NO`. On `NO`, also
   print the first few mismatches and `std::process::exit(1)` so the
   smoketest harness flags it.

Following this pattern keeps the harness uniform — every check is
self-contained and the success/fail signal is unambiguous.

Block reductions and scans take an unsafe raw scratch pointer. Every block thread must use the same live shared allocation; no other operation may access it, and a block barrier is required before reuse.
