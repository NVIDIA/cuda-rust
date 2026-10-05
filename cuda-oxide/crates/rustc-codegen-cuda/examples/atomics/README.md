# atomics

## Sound GPU Atomics with Explicit Scope and Ordering

Comprehensive test suite for cuda-oxide's atomic operations. Demonstrates
all atomic types, all RMW operations, memory orderings, and the LLVM
fence-splitting workaround -- running end-to-end on real hardware.

## Prerequisites

### LLVM 22+

Atomic operations require **LLVM 22 or newer** for correct syncscope
generation. Without it, scopes (`.gpu`, `.cta`, `.sys`) will be missing
from generated PTX.

The pinned Rust toolchain (`nightly-2026-08-28`) ships LLVM 23 with NVPTX
enabled via the `llvm-tools` component, so the default onboarding path
already satisfies this requirement:

```bash
rustup component add llvm-tools  # already listed in rust-toolchain.toml
```

The pipeline auto-picks `<sysroot>/lib/rustlib/<host>/bin/llc` first.

If you prefer a system LLVM install (Ubuntu / Debian):

```bash
wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && sudo ./llvm.sh 22
llc-22 --version  # Should show 22.x
```

Resolution order: `$CUDA_OXIDE_LLC` → rustup `llc` → `llc-23` → `llc-22` →
`llc-21` → `llc` on `PATH`. To pin a specific binary:

```bash
export CUDA_OXIDE_LLC=/path/to/llc
```

### Hardware

- **GPU**: sm_80+ (Ampere or newer). Tested on A100.
- **CUDA Driver**: 11.0+

## Build and Run

```bash
cargo oxide run atomics

# Or with explicit llc path:
CUDA_OXIDE_LLC=/path/to/llc-22 cargo oxide run atomics

# Verbose mode (shows which llc was used):
CUDA_OXIDE_VERBOSE=1 cargo oxide run atomics
```

## Test Suite (22 runtime tests)

### Phase 1: Core operations (AtomicU32/I32)

| #  | Test                             | What it verifies                                                           |
|----|----------------------------------|----------------------------------------------------------------------------|
| 1  | `atomic_fetch_add_test`          | AtomicU32 fetch_add (Relaxed) -- basic counter                             |
| 2  | `atomic_load_store_test`         | AtomicU32 load/store (Acquire/Release)                                     |
| 3  | `atomic_cas_test`                | AtomicU32 compare_exchange (AcqRel) -- single winner CAS race              |
| 4  | `atomic_fetch_add_acqrel_test`   | Fence-splitting workaround: `fence.acq_rel` + `atom.add` + `fence.acq_rel` |
| 5  | `atomic_fetch_add_seqcst_test`   | SeqCst pattern: `fence.sc` + `atom.add` + `fence.sc`                       |
| 6  | `atomic_i32_test`                | AtomicI32 fetch_add + CAS with negative values (-42)                       |
| 7  | `atomic_multiblock_test`         | Device-scope atomics across 4 CTAs (256 threads total)                     |

### Phase 2: Extended types and RMW operations

| #  | Test                             | What it verifies                                                           |
|----|----------------------------------|----------------------------------------------------------------------------|
| 8  | `atomic_u64_fetch_add_test`      | AtomicU64 -- 64-bit unsigned atomics                                       |
| 9  | `atomic_i64_test`                | AtomicI64 fetch_add + CAS with i64 (-100)                                  |
| 10 | `atomic_fetch_sub_test`          | fetch_sub -- LLVM lowers to `atom.add ..., -1`                             |
| 11 | `atomic_bitwise_test`            | fetch_and, fetch_or, fetch_xor (`.b32` PTX types)                          |
| 12 | `atomic_swap_test`               | swap (`atom.exch`) with sentinel 0xDEADBEEF                                |
| 13 | `atomic_minmax_test`             | AtomicI32 fetch_min/fetch_max (signed `.s32`, range -128..+127)            |
| 14 | `atomic_f32_fetch_add_test`      | AtomicF32 fetch_add via `atomicrmw fadd`; see the lowering note below |

### Floating-point add lowering

With the pinned `nightly-2026-08-28` toolchain's LLVM 23 NVPTX defaults,
`atomic_f32_fetch_add_test` uses a compare-and-swap (CAS) loop for global memory.
Generic-address-space `f32` adds also use a loop; shared `f32` and `f64` adds
still use native instructions on supported targets. For example:

```text
global f32: add.f32 + atom.relaxed.gpu.global.cas.b32 + retry
shared f32: atom.relaxed.gpu.shared.add.f32
```

This is a property of that LLVM snapshot and its options, not every LLVM 23
build or the separate libNVVM backend. Native global `atom.add.f32` flushes
subnormal inputs and results to zero; the loop preserves them under the default
floating-point mode. Switching instructions therefore changes numerical
behavior, not just performance. See [#1234](https://github.com/NVlabs/cuda-oxide/issues/1234)
for the investigation.

### Phase 3: Remaining types, scopes, and coverage

| #  | Test                             | What it verifies                                                              |
|----|----------------------------------|-------------------------------------------------------------------------------|
| 15 | `atomic_f64_fetch_add_test`      | AtomicF64 -- hardware `atom.add.f64` (64-bit float)                           |
| 16 | `atomic_f32_swap_test`           | AtomicF32 swap (`atom.exch.b32`) with float value 3.14                        |
| 17 | `atomic_unsigned_minmax_test`    | AtomicU32 fetch_min/fetch_max (UMin/UMax, unsigned `.u32`)                    |
| 18 | `atomic_block_scope_test`        | BlockAtomicU32 fetch_add (`.cta` scope, Relaxed)                              |
| 19 | `atomic_block_scope_acqrel_test` | BlockAtomicU32 fetch_add (`.cta` scope, AcqRel -- proves `fence.acq_rel.cta`) |

### Phase 4: Standard library atomics (`core::sync::atomic`)

| #  | Test                                  | What it verifies                                                        |
|----|---------------------------------------|-------------------------------------------------------------------------|
| 20 | `core_atomic_fetch_add_test`          | `core::sync::atomic::AtomicU32` fetch_add (system scope, Relaxed)       |
| 21 | `core_atomic_ptr_test`                | `core::sync::atomic::AtomicPtr<u16>` load/store/swap/compare_exchange   |
| 22 | `core_atomic_local_test`              | Private pointer/integer atomics, helper calls and wrapping arithmetic   |
| 23 | `core_atomic_ptr_shared_value_test`   | AtomicPtr round-trip of a shared-memory pointer value                   |
| 24 | `core_atomic_ptr_shared_storage_test` | AtomicPtr storage backed by shared memory                               |

The pointer test gives each thread a distinct mutable storage slot and checks
both successful and failed compare-exchange, then reads through the loaded
pointer. The local test calls the same pointer helper with per-thread storage.
The shared-value test round-trips a pointer originating in shared memory
through AtomicPtr load/store/swap/compare_exchange and dereferences the final
pointer. The shared-storage test places the AtomicPtr backing storage in shared
memory and exercises the same pointer operations from device code.

Pointer types survive compiler lowering; legacy NVVM uses scoped PTX for
pointer exchange and compare-exchange. Private atomic storage uses ordinary
accesses because no other thread can observe it.

Contended counters use raw writable device pointers and the existing
`from_ptr` atomic constructors. The host keeps their allocations alive and
excludes non-atomic access until synchronization. Block barriers are reached
by every thread, including the thread that performs the update.

The separate `core_atomic_ordering_probe` kernel provides compile-only
standard-library atomic ordering coverage.

## Expected Output

```text
=== Unified Atomics Test ===

--- Test 1: atomic_fetch_add_test ---
  Counter final value: 256 (expected 256)
  All 256 fetch_add return values are unique

--- Test 2: atomic_load_store_test ---
  All 256 threads read 42 after atomic store

--- Test 3: atomic_cas_test ---
  Exactly 1 winner (tid NN)
  255 threads lost the CAS race

--- Test 4: atomic_fetch_add_acqrel_test ---
  Counter = 256 with AcqRel ordering, all 256 values unique

--- Test 5: atomic_fetch_add_seqcst_test ---
  Counter = 256 with SeqCst ordering, all 256 values unique

--- Test 6: atomic_i32_test ---
  i32 counter = 256 (expected 256)
  i32 CAS: 0 -> -42 (expected -42)
  Thread 0 CAS result = 1 (1 = success)

--- Test 7: atomic_multiblock_test ---
  Counter = 256 across 4 blocks x 64 threads, all 256 values unique

--- Test 8: atomic_u64_fetch_add_test ---
  u64 counter = 256, all 256 values unique

--- Test 9: atomic_i64_test ---
  i64 counter = 256 (expected 256)
  i64 CAS: 0 -> -100 (expected -100)
  Thread 0 CAS result = 1 (1 = success)

--- Test 10: atomic_fetch_sub_test ---
  fetch_sub: 256 -> 0, all 256 old values unique

--- Test 11: atomic_bitwise_test ---
  fetch_or:  0xFFFFFFFF (expected 0xFFFFFFFF)
  fetch_and: 0x0000FFFF (expected 0x0000FFFF)
  fetch_xor: 0x00000000 (expected 0x00000000)

--- Test 12: atomic_swap_test ---
  swap: old=0x00000000 (expected 0), target=0xDEADBEEF

--- Test 13: atomic_minmax_test ---
  fetch_min: -128 (expected -128)
  fetch_max: 127 (expected +127)

--- Test 14: atomic_f32_fetch_add_test ---
  f32 counter = 256 (expected 256)

--- Test 15: atomic_f64_fetch_add_test ---
  f64 counter = 256 (expected 256)

--- Test 16: atomic_f32_swap_test ---
  target = 3.14 (expected ~3.14), thread 0 ok

--- Test 17: atomic_unsigned_minmax_test ---
  min = 0 (expected 0), max = 255 (expected 255)

--- Test 18: atomic_block_scope_test ---
  counter = 256 (expected 256), all old values unique

--- Test 19: atomic_block_scope_acqrel_test ---
  counter = 256 (expected 256), all old values unique

--- Test 20: core_atomic_fetch_add_test (core::sync::atomic) ---
  counter = 256 (expected 256), all old values unique

--- Test 21: core_atomic_ptr_test (core::sync::atomic::AtomicPtr) ---
  all 256 threads passed AtomicPtr load/store/swap/CAS

--- Test 22: core_atomic_local_test ---
  all 256 threads passed private pointer and integer atomic operations

--- Test 23: core_atomic_ptr_shared_value_test ---
  all 256 threads passed shared-pointer AtomicPtr load/store/swap/CAS/deref

--- Test 24: core_atomic_ptr_shared_storage_test ---
  all 256 threads passed shared-storage AtomicPtr load/store/swap/CAS

=== SUCCESS: All 24 runtime atomic tests passed! ===
```

## Available Atomic Types

All types are defined in `cuda_device::atomic`:

| Scope            | Integer types                                                              | Float types                          |
|------------------|----------------------------------------------------------------------------|--------------------------------------|
| Device (`.gpu`)  | `AtomicU32`, `AtomicI32`, `AtomicU64`, `AtomicI64`                         | `AtomicF32`, `AtomicF64`             |
| Block  (`.cta`)  | `BlockAtomicU32`, `BlockAtomicI32`, `BlockAtomicU64`, `BlockAtomicI64`     | `BlockAtomicF32`, `BlockAtomicF64`   |
| System (`.sys`)  | `SystemAtomicU32`, `SystemAtomicI32`, `SystemAtomicU64`, `SystemAtomicI64` | `SystemAtomicF32`, `SystemAtomicF64` |

### Supported operations

| Operation          | Integer                                              | Float                        |
|--------------------|------------------------------------------------------|------------------------------|
| `load`             | Yes                                                  | Yes                          |
| `store`            | Yes                                                  | Yes                          |
| `fetch_add`        | Yes                                                  | Yes (native or a CAS loop; see the lowering note above) |
| `fetch_sub`        | Yes                                                  | --                           |
| `fetch_and`        | Yes                                                  | --                           |
| `fetch_or`         | Yes                                                  | --                           |
| `fetch_xor`        | Yes                                                  | --                           |
| `fetch_min`        | Yes (signed: `.s32`/`.s64`, unsigned: `.u32`/`.u64`) | --                           |
| `fetch_max`        | Yes (signed/unsigned, same as `fetch_min`)           | --                           |
| `swap`             | Yes                                                  | Yes (`atom.exch.b32/b64`)    |
| `compare_exchange` | Yes                                                  | -- (PTX has no float CAS)    |

## LLVM 21/22 Limitations (and Workarounds)

### `atomicrmw` ordering is silently dropped (fixed in LLVM 23)

The LLVM 21/22 NVPTX backend ignores memory orderings on `atomicrmw`
instructions. All `atomicrmw add/sub/and/or/xor/exch/min/max` produce bare
`atom.<op>` without ordering or scope qualifiers.

**Workaround (fence splitting):** We emit explicit fences around the
operation:

```text
AcqRel example:
  fence.acq_rel.gpu      <- release semantics
  atom.add.u32 ...       <- monotonic (ordering dropped anyway)
  fence.acq_rel.gpu      <- acquire semantics
```

**Fix:** PR [#176015](https://github.com/llvm/llvm-project/pull/176015)
landed in **LLVM 23**, which the pinned toolchain now ships: verified
against the pinned llc-23, `atomicrmw ... syncscope("block") acq_rel`
lowers to `atom.acq_rel.cta.global.add.u32` directly. Fence splitting is
still emitted so the LLVM 21/22 floor keeps producing correctly ordered
PTX; it can be removed when the supported-llc floor moves to 23.

### Syncscopes work on everything except `atomicrmw` (fixed in LLVM 23)

| Instruction    | llc-21/22 scope support          | llc-23 |
|----------------|----------------------------------|--------|
| `load atomic`  | Yes (`.gpu`, `.cta`, `.sys`)     | Yes    |
| `store atomic` | Yes                              | Yes    |
| `cmpxchg`      | Yes                              | Yes    |
| `fence`        | Yes                              | Yes    |
| `atomicrmw`    | No (same bug as orderings)       | Yes    |

The fence-splitting workaround also fixes scope on 21/22: the fences carry
the correct syncscope even though the `atomicrmw` itself doesn't.

**Note:** For `Relaxed` ordering (no fences emitted) on llc-21/22, the bare
`atomicrmw` instruction loses its scope entirely. For example,
`BlockAtomicU32::fetch_add` with `Relaxed` emits `atom.add.u32` (defaulting
to `.gpu`) instead of `atom.cta.add.u32`. This is functionally correct
(`.gpu` is a superset of `.cta`) but not optimal. The pinned llc-23 keeps
the scope even on `Relaxed` (`atom.relaxed.cta.global.add.u32`). Non-Relaxed
orderings correctly emit scoped fences (e.g., `fence.acq_rel.cta`), as
verified by test 19.

### `fetch_sub` lowers to `atom.add ..., -N`

LLVM has no native `atomicrmw sub` for NVPTX. It converts subtraction to
addition of the negated value. This is semantically correct.

## Compilation Pipeline

```text
cuda_device::atomic::AtomicU32::fetch_add(...)     ← Rust stub (never executed)
  -- OR --
core::sync::atomic::AtomicU32::fetch_add(...)    ← Compiles to std::intrinsics::atomic_xadd
        │
        ▼
mir-importer  (intrinsics/atomic.rs)              ← Intercepts call by name,
        │                                            emits NvvmAtomicRmwOp
        ▼
dialect-nvvm  (ops/atomic.rs)                     ← NvvmAtomicRmwOp { kind: Add,
        │                                            ordering: AcqRel, scope: Device }
        ▼
mir-lower     (convert/intrinsics/atomic.rs)      ← Fence splitting + LlvmAtomicRmwOp
        │
        ▼
llvm-export   (export/)                           ← Textual LLVM IR:
        │                                            fence syncscope("device") release
        │                                            atomicrmw add ptr %p, i32 %v ...
        │                                            fence syncscope("device") acquire
        ▼
llc                                               ← PTX:
                                                     fence.acq_rel.gpu
                                                     atom.add.u32 ...
                                                     fence.acq_rel.gpu
```

## Potential Errors

| Error                                    | Cause                 | Solution                                           |
|------------------------------------------|-----------------------|----------------------------------------------------|
| Scope missing from PTX                   | Using llc-21 or older | Set `CUDA_OXIDE_LLC=/path/to/llc-22` (or newer)    |
| `PTX generation failed: llc not found`   | No LLVM installed     | `sudo apt install llvm-22` or set `CUDA_OXIDE_LLC` |
| `CUDA_ERROR_NO_DEVICE`                   | No GPU available      | Ensure NVIDIA driver is installed                  |
| `Failed to load PTX module`              | PTX file missing      | Run via `cargo oxide run atomics`                  |
