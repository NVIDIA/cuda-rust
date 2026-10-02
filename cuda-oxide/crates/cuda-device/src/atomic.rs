/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU atomic operations with explicit scope and memory ordering.
//!
//! This module provides atomic types for CUDA device code. Unlike
//! `core::sync::atomic`, these types have explicit **scope** (which threads
//! see the atomic) baked into the type, and all memory orderings map directly
//! to PTX ordering qualifiers.
//!
//! # Scope
//!
//! | Prefix           | PTX scope | Which threads observe it            |
//! |------------------|-----------|-------------------------------------|
//! | `DeviceAtomic*`  | `.gpu`    | All threads on the GPU (device)     |
//! | `BlockAtomic*`   | `.cta`    | Same thread block only              |
//! | `SystemAtomic*`  | `.sys`    | GPU **and** CPU (unified memory)    |
//!
//! # Ordering
//!
//! All five standard orderings are supported:
//!
//! | Variant   | PTX qualifier | Cost     |
//! |-----------|---------------|----------|
//! | `Relaxed` | `.relaxed`    | Cheapest |
//! | `Acquire` | `.acquire`    | Low      |
//! | `Release` | `.release`    | Low      |
//! | `AcqRel`  | `.acq_rel`    | Medium   |
//! | `SeqCst`  | `fence.sc` + op | Highest  |
//!
//! # Types
//!
//! | Type                         | Operations                                 |
//! |------------------------------|--------------------------------------------|
//! | Integer (U32, I32, U64, I64) | load, store, all RMW ops, compare_exchange |
//! | Float (F16, F32, F64)        | load, store, fetch_add, fetch_sub, swap    |
//!
//! Float atomics do **not** support compare_exchange (PTX has no `atom.cas`
//! for float types) or bitwise operations.
//!
//! # Example
//!
//! ```rust,ignore
//! use cuda_device::atomic::{DeviceAtomicU32, AtomicOrdering};
//!
//! #[kernel]
//! fn histogram(data: &[u32], bins: &mut [DeviceAtomicU32]) {
//!     let idx = cuda_device::thread::index_1d();
//!     let val = data[idx];
//!     bins[val as usize].fetch_add(1, AtomicOrdering::Relaxed);
//! }
//! ```
//!
//! # Performance: a scoped atomic load cannot be served from L1
//!
//! `load` and `store` here are *coherent* at their scope. An L1 cache is not
//! coherent across SMs, so a device- or system-scoped atomic access bypasses L1
//! by construction, on every call, however weak its ordering. Relaxed does not
//! make it cheap; relaxed only removes ordering, not coherence.
//!
//! That matters when the access is on a hot path and the algorithm does not
//! actually need coherence. A common shape is a hash table where a thread
//! publishes a key and then an index, and readers must not observe the first
//! without the second. Reading the index with `load` on every lookup pays an
//! uncached access every time, when the index is almost always already
//! published. Reading it plainly first and falling back to `load` only while
//! genuinely waiting keeps the common path in L1.
//!
//! Measured on such a probe loop: doing it the first way took the L1 sector hit
//! rate from 54% to 29%, pushed 1.7x the sectors through to L2, and made the
//! kernel 24% slower, with the entire difference showing up as
//! `long_scoreboard` warp stalls. No instruction count changes; only where the
//! data is allowed to live.
//!
//! Use these types where you need cross-thread visibility. Do not reach for
//! them by reflex just because a location is shared.
//!
//! # Naming and overlap with `core::sync::atomic`
//!
//! Device-scope types are named **`DeviceAtomic*`** (e.g. `DeviceAtomicU32`) so they
//! do not collide with `core::sync::atomic::AtomicU32`. Block and system scopes
//! keep the `BlockAtomic*` and `SystemAtomic*` prefixes.
//!
//! The ordering enum is **`AtomicOrdering`** (not `Ordering`) so you can use
//! `Ordering` from the standard library without a name clash. For example:
//! `use core::sync::atomic::{AtomicU32, Ordering};` for host/cpu code, and
//! `use cuda_device::atomic::{DeviceAtomicU32, AtomicOrdering};` for device code.
//!
//! # Host–device ABI
//!
//! The ABI is the same for atomics on host and device: `core::sync::atomic::AtomicT`
//! and `DeviceAtomicT` / `BlockAtomicT` / `SystemAtomicT` have the same layout
//! (a single `T`). The same allocation can be used from host and device in
//! unified memory without conversion.
//!
//! # Implementation
//!
//! Each method is a stub that the cuda-oxide compiler recognizes by name and
//! replaces with the correct LLVM atomic instruction. The method bodies are
//! never executed -- they exist only so the type checker and borrow checker
//! work normally during development.

use core::cell::UnsafeCell;

// =============================================================================
// Memory Ordering
// =============================================================================

/// Memory ordering for atomic operations.
///
/// Mirrors `core::sync::atomic::Ordering` with the same semantics.
/// The compiler reads the discriminant value to select the PTX ordering.
/// Named `AtomicOrdering` so you can use `core::sync::atomic::Ordering` in
/// the same crate without a name clash.
///
/// Discriminant values (must stay in sync with mir-importer):
/// - `Relaxed` = 0
/// - `Acquire` = 1
/// - `Release` = 2
/// - `AcqRel`  = 3
/// - `SeqCst`  = 4
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AtomicOrdering {
    /// No ordering guarantees. Cheapest. Use when you only need atomicity
    /// (e.g., a counter where you don't care about ordering of other writes).
    Relaxed = 0,
    /// If I see a released value, I also see everything the releaser wrote
    /// before it. Valid for: `load`, `fetch_*`, `compare_exchange`.
    Acquire = 1,
    /// All my prior writes become visible to whoever acquires this value.
    /// Valid for: `store`, `fetch_*`, `compare_exchange`.
    Release = 2,
    /// Both acquire and release in one operation.
    /// Valid for: `fetch_*`, `compare_exchange`.
    AcqRel = 3,
    /// All the above, plus a single global total order. Most expensive.
    /// Valid for: all operations.
    SeqCst = 4,
}

// =============================================================================
// Macros for generating atomic types
// =============================================================================

/// Defines an integer atomic type with all supported operations.
///
/// Generated methods:
/// - `new(val)` — constructor
/// - `from_ptr(ptr)` — non-owning view over existing `*mut T` memory
/// - `load`, `store` — atomic load/store
/// - `fetch_add`, `fetch_sub` — arithmetic RMW
/// - `fetch_and`, `fetch_or`, `fetch_xor` — bitwise RMW
/// - `fetch_min`, `fetch_max` — comparison RMW
/// - `swap` — atomic exchange
/// - `compare_exchange_raw` (private) — raw CAS intrinsic returning T
/// - `compare_exchange` — CAS returning `Result<T, T>`
///
/// All stub methods are `#[inline(never)]` with `unreachable!()` bodies.
/// The cuda-oxide compiler intercepts calls by method name and replaces
/// them with the appropriate LLVM atomic instructions.
macro_rules! define_integer_atomic {
    (
        $(#[$outer:meta])*
        pub struct $Name:ident ($ty:ty);
    ) => {
        $(#[$outer])*
        #[repr(transparent)]
        pub struct $Name {
            inner: UnsafeCell<$ty>,
        }

        // SAFETY: Atomic types are specifically designed for shared access.
        // All operations go through atomic instructions emitted by the compiler.
        unsafe impl Sync for $Name {}

        impl $Name {
            /// Create a new atomic with the given initial value.
            pub const fn new(val: $ty) -> Self {
                $Name { inner: UnsafeCell::new(val) }
            }

            /// Reinterpret a raw pointer as a reference to this atomic.
            ///
            /// This is the non-owning view pattern: hand it a pointer to an
            /// existing plain `T` and use atomic semantics over that location.
            /// Mirrors `core::sync::atomic::AtomicU32::from_ptr` in shape and
            /// is the closest cuda-oxide equivalent of C++'s
            /// `cuda::atomic_ref<T, Scope>` constructor.
            ///
            /// # Safety
            ///
            /// * `ptr` must be aligned to `align_of::<Self>()` (which equals
            ///   `align_of::<T>()` for this `#[repr(transparent)]` type).
            /// * The memory at `ptr` must be valid for reads and writes for
            ///   the entire lifetime `'a`.
            /// * No non-atomic accesses to `*ptr` may occur during `'a`.
            ///   All accesses through the returned reference are atomic; mixing
            ///   atomic and non-atomic access to the same location is UB.
            /// * For block-scoped types, only threads in the same thread block
            ///   may access this memory.
            /// * For system-scoped types, the memory must be reachable by both
            ///   the GPU and the CPU (unified memory).
            #[inline(always)]
            pub const unsafe fn from_ptr<'a>(ptr: *mut $ty) -> &'a Self {
                // SAFETY: `Self` is `#[repr(transparent)]` over `UnsafeCell<$ty>`,
                // which is layout-compatible with `$ty`. The caller upholds all
                // alignment, validity, and aliasing invariants.
                unsafe { &*(ptr as *const Self) }
            }

            // ── Load / Store ───────────────────────────────────────────

            /// Atomically load the value.
            ///
            /// `order` must be `Relaxed`, `Acquire`, or `SeqCst`.
            #[inline(never)]
            pub fn load(&self, order: AtomicOrdering) -> $ty {
                let _ = order;
                unreachable!(concat!(stringify!($Name), "::load called outside CUDA kernel context"))
            }

            /// Atomically store a value.
            ///
            /// `order` must be `Relaxed`, `Release`, or `SeqCst`.
            #[inline(never)]
            pub fn store(&self, val: $ty, order: AtomicOrdering) {
                let _ = (val, order);
                unreachable!(concat!(stringify!($Name), "::store called outside CUDA kernel context"))
            }

            // ── Arithmetic RMW ─────────────────────────────────────────

            /// Atomically add `val` and return the **previous** value.
            #[inline(never)]
            pub fn fetch_add(&self, val: $ty, order: AtomicOrdering) -> $ty {
                let _ = (val, order);
                unreachable!(concat!(stringify!($Name), "::fetch_add called outside CUDA kernel context"))
            }

            /// Atomically subtract `val` and return the **previous** value.
            #[inline(never)]
            pub fn fetch_sub(&self, val: $ty, order: AtomicOrdering) -> $ty {
                let _ = (val, order);
                unreachable!(concat!(stringify!($Name), "::fetch_sub called outside CUDA kernel context"))
            }

            // ── Bitwise RMW ────────────────────────────────────────────

            /// Atomically bitwise-AND with `val` and return the **previous** value.
            #[inline(never)]
            pub fn fetch_and(&self, val: $ty, order: AtomicOrdering) -> $ty {
                let _ = (val, order);
                unreachable!(concat!(stringify!($Name), "::fetch_and called outside CUDA kernel context"))
            }

            /// Atomically bitwise-OR with `val` and return the **previous** value.
            #[inline(never)]
            pub fn fetch_or(&self, val: $ty, order: AtomicOrdering) -> $ty {
                let _ = (val, order);
                unreachable!(concat!(stringify!($Name), "::fetch_or called outside CUDA kernel context"))
            }

            /// Atomically bitwise-XOR with `val` and return the **previous** value.
            #[inline(never)]
            pub fn fetch_xor(&self, val: $ty, order: AtomicOrdering) -> $ty {
                let _ = (val, order);
                unreachable!(concat!(stringify!($Name), "::fetch_xor called outside CUDA kernel context"))
            }

            // ── Comparison RMW ─────────────────────────────────────────

            /// Atomically compute the minimum and return the **previous** value.
            #[inline(never)]
            pub fn fetch_min(&self, val: $ty, order: AtomicOrdering) -> $ty {
                let _ = (val, order);
                unreachable!(concat!(stringify!($Name), "::fetch_min called outside CUDA kernel context"))
            }

            /// Atomically compute the maximum and return the **previous** value.
            #[inline(never)]
            pub fn fetch_max(&self, val: $ty, order: AtomicOrdering) -> $ty {
                let _ = (val, order);
                unreachable!(concat!(stringify!($Name), "::fetch_max called outside CUDA kernel context"))
            }

            // ── Exchange ───────────────────────────────────────────────

            /// Atomically swap with `val` and return the **previous** value.
            #[inline(never)]
            pub fn swap(&self, val: $ty, order: AtomicOrdering) -> $ty {
                let _ = (val, order);
                unreachable!(concat!(stringify!($Name), "::swap called outside CUDA kernel context"))
            }

            // ── Compare-and-exchange ───────────────────────────────────

            /// Low-level compare-and-swap. Returns the **previous** value.
            ///
            /// This is the raw intrinsic intercepted by the compiler.
            /// Use [`compare_exchange`](Self::compare_exchange) for the
            /// `Result`-returning public API.
            #[inline(never)]
            fn compare_exchange_raw(
                &self,
                current: $ty,
                new: $ty,
                success: AtomicOrdering,
                failure: AtomicOrdering,
            ) -> $ty {
                let _ = (current, new, success, failure);
                unreachable!(concat!(
                    stringify!($Name),
                    "::compare_exchange_raw called outside CUDA kernel context"
                ))
            }

            /// Compare-and-swap. If `*self == current`, set `*self = new`
            /// and return `Ok(current)`. Otherwise return `Err(actual_value)`.
            ///
            /// `success` is the ordering for the RMW if the comparison succeeds.
            /// `failure` is the ordering for the load if the comparison fails.
            #[inline(always)]
            pub fn compare_exchange(
                &self,
                current: $ty,
                new: $ty,
                success: AtomicOrdering,
                failure: AtomicOrdering,
            ) -> Result<$ty, $ty> {
                let old = self.compare_exchange_raw(current, new, success, failure);
                if old == current {
                    Ok(old)
                } else {
                    Err(old)
                }
            }
        }
    };
}

/// Defines a float atomic type with the subset of operations that
/// hardware supports for floating-point values.
///
/// Generated methods:
/// - `new(val)` — constructor
/// - `from_ptr(ptr)` — non-owning view over existing `*mut T` memory
/// - `load`, `store` — atomic load/store
/// - `fetch_add`, `fetch_sub` — arithmetic RMW
/// - `swap` — atomic exchange (`atomicrmw xchg`)
///
/// **Not supported** (PTX hardware limitation):
/// - `compare_exchange` — no `atom.cas` for float types
/// - `fetch_and/or/xor` — bitwise ops meaningless on floats
/// - `fetch_min/max` — not in initial implementation
macro_rules! define_float_atomic {
    (
        $(#[$outer:meta])*
        pub struct $Name:ident ($ty:ty);
    ) => {
        $(#[$outer])*
        #[repr(transparent)]
        pub struct $Name {
            inner: UnsafeCell<$ty>,
        }

        // SAFETY: Atomic types are specifically designed for shared access.
        // All operations go through atomic instructions emitted by the compiler.
        unsafe impl Sync for $Name {}

        impl $Name {
            /// Create a new atomic with the given initial value.
            pub const fn new(val: $ty) -> Self {
                $Name { inner: UnsafeCell::new(val) }
            }

            /// Reinterpret a raw pointer as a reference to this atomic.
            ///
            /// See the matching method on the integer atomic types for the
            /// full safety contract; the rules are identical for floats.
            ///
            /// # Safety
            ///
            /// * `ptr` must be aligned to `align_of::<Self>()`.
            /// * The memory at `ptr` must be valid for reads and writes for
            ///   the entire lifetime `'a`.
            /// * No non-atomic accesses to `*ptr` may occur during `'a`.
            /// * Block-scoped: only same-block threads may access.
            ///   System-scoped: memory must be reachable by GPU and CPU.
            #[inline(always)]
            pub const unsafe fn from_ptr<'a>(ptr: *mut $ty) -> &'a Self {
                // SAFETY: `Self` is `#[repr(transparent)]` over `UnsafeCell<$ty>`,
                // which is layout-compatible with `$ty`. The caller upholds all
                // alignment, validity, and aliasing invariants.
                unsafe { &*(ptr as *const Self) }
            }

            // ── Load / Store ───────────────────────────────────────────

            /// Atomically load the value.
            ///
            /// `order` must be `Relaxed`, `Acquire`, or `SeqCst`.
            #[inline(never)]
            pub fn load(&self, order: AtomicOrdering) -> $ty {
                let _ = order;
                unreachable!(concat!(stringify!($Name), "::load called outside CUDA kernel context"))
            }

            /// Atomically store a value.
            ///
            /// `order` must be `Relaxed`, `Release`, or `SeqCst`.
            #[inline(never)]
            pub fn store(&self, val: $ty, order: AtomicOrdering) {
                let _ = (val, order);
                unreachable!(concat!(stringify!($Name), "::store called outside CUDA kernel context"))
            }

            // ── Arithmetic RMW ─────────────────────────────────────────

            /// Atomically add `val` and return the **previous** value.
            ///
            /// Uses LLVM `atomicrmw fadd`.
            #[inline(never)]
            pub fn fetch_add(&self, val: $ty, order: AtomicOrdering) -> $ty {
                let _ = (val, order);
                unreachable!(concat!(stringify!($Name), "::fetch_add called outside CUDA kernel context"))
            }

            /// Atomically subtract `val` and return the **previous** value.
            ///
            /// Lowered as `atomicrmw fadd` of the negated value, so the
            /// backend can keep using native PTX add atomics.
            #[inline(never)]
            pub fn fetch_sub(&self, val: $ty, order: AtomicOrdering) -> $ty {
                let _ = (val, order);
                unreachable!(concat!(stringify!($Name), "::fetch_sub called outside CUDA kernel context"))
            }

            // ── Exchange ───────────────────────────────────────────────

            /// Atomically swap with `val` and return the **previous** value.
            #[inline(never)]
            pub fn swap(&self, val: $ty, order: AtomicOrdering) -> $ty {
                let _ = (val, order);
                unreachable!(concat!(stringify!($Name), "::swap called outside CUDA kernel context"))
            }
        }
    };
}

// =============================================================================
// Device scope (default) — `.gpu`
//
// All threads on the GPU observe the atomic. This is the right choice for
// the vast majority of GPU atomics operating on global memory.
// =============================================================================

define_integer_atomic! {
    /// 32-bit unsigned atomic, **device scope** (`.gpu`).
    pub struct DeviceAtomicU32(u32);
}

define_integer_atomic! {
    /// 32-bit signed atomic, **device scope** (`.gpu`).
    pub struct DeviceAtomicI32(i32);
}

define_integer_atomic! {
    /// 64-bit unsigned atomic, **device scope** (`.gpu`).
    pub struct DeviceAtomicU64(u64);
}

define_integer_atomic! {
    /// 64-bit signed atomic, **device scope** (`.gpu`).
    pub struct DeviceAtomicI64(i64);
}

define_float_atomic! {
    /// 16-bit float atomic, **device scope** (`.gpu`).
    ///
    /// `fetch_add`/`fetch_sub` lower to hardware `atom.add.noftz.f16` (sm_70+).
    /// On pre-sm_70 targets llc expands them to a compare-and-swap loop instead.
    pub struct DeviceAtomicF16(f16);
}

define_float_atomic! {
    /// 32-bit float atomic, **device scope** (`.gpu`).
    ///
    /// `swap` lowers to hardware `atom.exch.b32`; this API does not expose
    /// `compare_exchange` for floats. `fetch_add` uses LLVM `atomicrmw fadd`.
    /// With the pinned `nightly-2026-08-28` toolchain's LLVM 23 NVPTX defaults,
    /// global and generic `f32` adds use a compare-and-swap loop; shared `f32`
    /// and `f64` adds still use native instructions on supported targets.
    ///
    /// Native global `atom.add.f32` flushes subnormal inputs and results to
    /// zero. The loop preserves subnormals under the default floating-point
    /// mode, so instruction choice also affects numerical behavior. This is
    /// specific to that LLVM snapshot and its options, not a guarantee for
    /// every LLVM 23 build or the separate libNVVM backend. See
    /// <https://github.com/NVlabs/cuda-oxide/issues/1234>.
    pub struct DeviceAtomicF32(f32);
}

define_float_atomic! {
    /// 64-bit float atomic, **device scope** (`.gpu`).
    ///
    /// Supports `fetch_add` via hardware `atom.add.f64` and `swap` via
    /// `atom.exch.b64`. Requires sm_60+ (we target sm_80+).
    pub struct DeviceAtomicF64(f64);
}

// =============================================================================
// Block scope — `.cta`
//
// Only threads within the **same thread block** observe the atomic.
// Cheaper than device scope, but the caller must ensure that no other
// block accesses this memory location. Misuse is undefined behavior.
// =============================================================================

define_integer_atomic! {
    /// 32-bit unsigned atomic, **block scope** (`.cta`).
    pub struct BlockAtomicU32(u32);
}

define_integer_atomic! {
    /// 32-bit signed atomic, **block scope** (`.cta`).
    pub struct BlockAtomicI32(i32);
}

define_integer_atomic! {
    /// 64-bit unsigned atomic, **block scope** (`.cta`).
    pub struct BlockAtomicU64(u64);
}

define_integer_atomic! {
    /// 64-bit signed atomic, **block scope** (`.cta`).
    pub struct BlockAtomicI64(i64);
}

define_float_atomic! {
    /// 16-bit float atomic, **block scope** (`.cta`).
    ///
    /// `fetch_add`/`fetch_sub` lower to `atom.add.noftz.f16` with `.cta` scope.
    pub struct BlockAtomicF16(f16);
}

define_float_atomic! {
    /// 32-bit float atomic, **block scope** (`.cta`).
    pub struct BlockAtomicF32(f32);
}

define_float_atomic! {
    /// 64-bit float atomic, **block scope** (`.cta`).
    pub struct BlockAtomicF64(f64);
}

// =============================================================================
// System scope — `.sys`
//
// All threads on the GPU **and** the CPU observe the atomic. Most expensive
// scope. Use for CPU-GPU shared data via HMM / unified memory.
// =============================================================================

define_integer_atomic! {
    /// 32-bit unsigned atomic, **system scope** (`.sys`).
    pub struct SystemAtomicU32(u32);
}

define_integer_atomic! {
    /// 32-bit signed atomic, **system scope** (`.sys`).
    pub struct SystemAtomicI32(i32);
}

define_integer_atomic! {
    /// 64-bit unsigned atomic, **system scope** (`.sys`).
    pub struct SystemAtomicU64(u64);
}

define_integer_atomic! {
    /// 64-bit signed atomic, **system scope** (`.sys`).
    pub struct SystemAtomicI64(i64);
}

define_float_atomic! {
    /// 16-bit float atomic, **system scope** (`.sys`).
    pub struct SystemAtomicF16(f16);
}

define_float_atomic! {
    /// 32-bit float atomic, **system scope** (`.sys`).
    pub struct SystemAtomicF32(f32);
}

define_float_atomic! {
    /// 64-bit float atomic, **system scope** (`.sys`).
    pub struct SystemAtomicF64(f64);
}

// =============================================================================
// Packed Atomic Add (f16x2, bf16x2)
//
// These are standalone functions (not atomic-type methods) because the
// hardware operates on raw `u32` words carrying two packed 16-bit lanes.
// They bypass the scoped-type system and use `atom.global` directly.
// =============================================================================

include!("generated/atomic.rs");
