# Dynamic shared memory passed to a device helper

This example checks the raw-pointer pattern:

```rust
let smem = DynamicSharedArray::<u8, 1024>::get_raw();
helper(smem, offset);

unsafe fn helper(smem: *mut u8, offset: usize) -> u64 {
    let full = smem.add(1024).cast::<Barrier>();
    // Initialize, arrive at, and wait on the shared-memory barrier.
    // Read and update a separate shared-memory data slot.
}
```

The launch supplies 2048 dynamic shared bytes. Four independent CTAs each use
one thread. Nine launches cover three data offsets and three input values,
including wrapping arithmetic. Every CTA checks both a volatile shared-memory
data round trip and a barrier transition from pending to complete. Barrier
polling is bounded so a broken barrier reports failure instead of hanging.

The operation body is a macro shared by three variants, so only the function
boundary changes:

| Features | Placement |
| --- | --- |
| None | Directly in the kernel; control case |
| `local-helper` | Helper in the kernel's crate |
| `cross-crate-helper` | Helper in the `helper-lib` dependency crate |

Helpers use `#[inline(never)]` by default. Add `force-inline` to request
`#[inline(always)]`, or `heuristic-inline` to omit an inline attribute. These
source attributes alone are not evidence that a call survives or disappears
in generated GPU code.

## Reproduce

Run from the repository root with the repository's Rust and CUDA toolchains.
The barrier probe requires SM90 or newer; the commands below target an RTX
5090 (`sm_120a`).

Build and run the direct control first. `set -e` prevents execution of an old
binary if this build fails.

```bash
set -e
cargo oxide build shared_memory_helper --arch sm_120a
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  crates/rustc-codegen-cuda/examples/shared_memory_helper/target/release/shared_memory_helper
/usr/local/cuda/bin/compute-sanitizer --tool synccheck --error-exitcode 1 \
  crates/rustc-codegen-cuda/examples/shared_memory_helper/target/release/shared_memory_helper
```

Build each helper variant separately; these currently fail as described below.
Do not run a binary left over from the successful control build as evidence
that a failed helper build worked.

```bash
cargo oxide build shared_memory_helper \
  --arch sm_120a --features local-helper
cargo oxide build shared_memory_helper \
  --arch sm_120a --features cross-crate-helper
```

Repeat with `--features local-helper,force-inline`,
`cross-crate-helper,force-inline`, `local-helper,heuristic-inline`, or
`cross-crate-helper,heuristic-inline` to compare inlining policies.
Set `CUDA_OXIDE_DUMP_MIR=1` to inspect the imported helper definition and call.

## Observed results

Tested on 2026-09-17 against `main` at
`7ce30ec798c490e6ed772dfcbad193b3733fe55a`, with this example added, on an
RTX 5090, NVIDIA driver 580.173.02, and CUDA tools 13.3.

| Variant | Result |
| --- | --- |
| Direct control | Builds; GPU checks pass |
| Local helper, all three inline policies | MIR verification rejects the call |
| Dependency helper, all three inline policies | Same rejection |

Both Compute Sanitizer runs of the direct control reported:

```text
PASS direct: 9 launches, 36 CTA data/barrier checks
========= ERROR SUMMARY: 0 errors
```

All six helper builds failed with:

```text
MirCallOp argument 0 type does not match callee signature
```

The imported call passes `mir.ptr<ui8, mutable:true, addrspace:3, kind:RawMut>`
(shared memory), while the helper parameter is
`mir.ptr<ui8, mutable:true, addrspace:0, kind:RawMut>` (generic memory). The
helper body is present in the imported module, including for the dependency
crate. Ordinary Rust calls do not receive the address-space adaptation that
the importer applies to foreign calls. Strict call verification rejects the
mismatch before backend code generation. Requesting `inline(always)` did not
remove the call before this verification in the tested builds.

This reproduces a compile-time rejection of this raw-byte-pointer boundary.
It does not demonstrate silent miscompilation or prove runtime correctness
of a surviving helper call, because no helper variant reached GPU execution.
The dependency test imports available MIR across crates; it does not test an
opaque, separately linked device binary. The runtime control checks one-thread
barrier behavior, not synchronization between multiple threads or TMA traffic.

The diagnostic currently exposes an internal MIR mismatch. This reproducer
does not change the diagnostic or implement the missing call conversion.
Pointers whose pointee is `Barrier` or `SharedArray` already receive a shared
address space, so this result should not be generalized to every helper that
accepts shared memory.
