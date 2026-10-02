# wgmma_mma_bf16

Compile-only integration example for BF16 Hopper WGMMA lowering.

The example covers the public `cuda_device::wgmma` API from Rust source through
MIR import, WGMMA region selection, LLVM lowering, and PTX generation.

## What this tests

The crate contains five compile-only kernels.

### m64n64 full drain

```text
wgmma_fence
mma m64n64 acc0
commit_group
wait_group<0>
```

For the canonical `[[f32; 8]; 4]` accumulator, the compiler selects the
value-threaded path and exposes the 32 accumulator values to LLVM only outside
the complete asynchronous WGMMA lifetime.

### Repeated reborrows

```text
wgmma_fence
mma &mut acc
mma &mut acc
commit_group
wait_group<0>
```

Each call creates a distinct Rust `&mut` SSA value. The compiler follows both
reborrows back to the same accumulator storage when checking the WGMMA region.

### Counted-loop reborrow

```text
wgmma_fence
repeat 4 times: mma &mut acc; advance descriptors
commit_group
wait_group<0>
```

The reborrow is created inside the loop. The compiler follows it back to the
pre-loop accumulator storage before hoisting the value-threaded adapter.

### m64n64 partial wait

```text
wgmma_fence

mma acc0
commit_group

mma acc1
commit_group

wait_group<1>
wait_group<0>
```

The two independent accumulator objects form two accumulator slots. The
compiler can therefore keep two committed groups in flight, lower the static
`wait_group<1>`, and still require a final `wait_group<0>` before either
accumulator is observed.

The example deliberately stops after two groups. Round-robin slot reuse and
the associated legality checks are covered by `mir-lower` integration tests.

### m64n128 full drain

```text
wgmma_fence
mma m64n128 wide_acc
commit_group
wait_group<0>
```

The canonical `[[f32; 8]; 8]` accumulator carries 64 `f32` values per thread.
The compiler keeps all 64 values tied through one convergent inline-PTX scope
and exposes them only after the final full drain. Counted loops and partial
waits remain intentionally unsupported for this shape.

## Usage

```bash
cargo oxide build wgmma_mma_bf16 --arch sm_90a
```

The command must complete successfully and generate PTX containing both BF16
`m64n64k16` and `m64n128k16` WGMMA instructions, a counted-loop carrier, and
the m64n64 partial wait.

To run the repository smoketest:

```bash
cuda-oxide/scripts/smoketest.sh -x -v '^wgmma_mma_bf16$'
```

## Expected smoketest marker

```text
SUCCESS: BF16 WGMMA value-threaded, reborrow, counted-loop, and partial-wait lowering compiled.
```

## Important

This is a compile-only example.

All kernels use zero-valued WGMMA descriptors so compilation and PTX generation
can be tested without allocating Hopper shared-memory tiles. The kernels must
not be launched with those descriptors.

Functional execution requires an `sm_90a` Hopper GPU, valid shared-memory
descriptors, and warpgroup-uniform participation.

## Supported lowering shapes

The current BF16 `m64n64k16.f32.bf16.bf16` lowering recognizes three
conservative shapes:

- linear full-drain regions ending in `wait_group<0>`;
- a canonical counted K-loop with affine descriptor recurrences;
- straight-line static partial-wait pipelines using `N + 1` independent
  accumulator slots for `wait_group<N>`.

BF16 `m64n128k16.f32.bf16.bf16` supports canonical linear full-drain regions
only, using a 64-value `[[f32; 8]; 8]` accumulator carrier.

Every accepted asynchronous lifetime ends in `wait_group<0>` before
accumulator values escape.

Unsupported m64n64 BF16 pointer shapes retain the deferred pointer-form fallback
where the full-drain sequence can still be proven safe. m64n128 has no pointer
fallback. Dynamic waits, unsupported control flow, malformed pipeline schedules,
m64n128 non-linear shapes, and unsupported data-type/shape combinations fail
closed.
