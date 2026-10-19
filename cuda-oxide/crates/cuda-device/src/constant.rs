/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Constant-memory support for CUDA kernels.
//!
//! [`ConstantMemory<T>`] is a wrapper for module-scope statics that live in CUDA
//! constant memory (PTX `.const`, address space 4). The host populates the
//! storage via `cuMemcpyHtoD`; device code reads it as if it were an
//! ordinary `static T`.
//!
//! There are two accessors, and for anything bigger than a scalar the choice
//! matters: [`ConstantMemory::get`] produces `T` by value, while
//! [`ConstantMemory::get_ref`] borrows it in place. A value has no address, so
//! indexing a `get()` copy at runtime spills the whole `T` to the thread's local
//! depot first — one store per element in every thread. Use `get_ref` for
//! tables, and see its docs for which address space suits which access pattern.
//!
//! # Usage
//!
//! Declare the static inside a `#[cuda_module]` and tag it with `#[constant]`:
//!
//! ```ignore
//! use cuda_device::{constant, cuda_module, kernel, thread, ConstantMemory, DisjointSlice};
//!
//! #[cuda_module]
//! mod kernels {
//!     use super::*;
//!
//!     #[constant]
//!     static COEFFS: ConstantMemory<[f32; 4]> = ConstantMemory::UNINIT;
//!
//!     #[kernel]
//!     pub fn apply(mut out: DisjointSlice<f32>) {
//!         let c = COEFFS.get_ref();    // safe borrow; no load yet
//!         let i = thread::index_1d().get();
//!         if let Some(e) = out.get_mut(thread::index_1d()) {
//!             *e = c[0] + c[1] * (i as f32);
//!         }
//!     }
//! }
//! ```
//!
//! Host code populates the constant with the macro-generated `set_<name>`
//! methods on the loaded module:
//!
//! ```ignore
//! module.set_coeffs(&stream, &[10.0, 20.0, 30.0, 40.0])?;
//! ```
//!
//! # Initialization limitation
//!
//! CUDA Oxide currently emits [`ConstantMemory::UNINIT`] as an all-zero
//! placeholder. It does not lower arbitrary non-zero Rust static initializers
//! into PTX constant-memory data.
//!
//! Populate the constant before any kernel reads it. Use `set_<name>` before
//! the kernel launch on the same stream, or use `set_<name>_blocking`. The
//! placeholder bytes are zero, and [`ConstantMemoryValue`] restricts `T` to
//! types for which the all-zero bit pattern is valid.
//!
//! # Why a wrapper type instead of a bare `static`
//!
//! A plain `static COEFFS: [f32; 4] = [1.0; 4];` would be constant-folded by
//! rustc — every read in device code is replaced with the literal initializer
//! values, and the host's `cuMemcpyHtoD` update becomes invisible. Wrapping
//! the storage in [`UnsafeCell`] prevents the fold by signalling interior
//! mutability, restoring the read-from-memory semantics that CUDA constant
//! memory requires.
//!
//! # Soundness: `Sync`
//!
//! Unlike [`SharedArray`](crate::SharedArray) (which is `!Sync` because
//! shared memory is per-block and requires barriers), `ConstantMemory<T>` is
//! `Sync`. CUDA constant memory has a single, host-controlled value visible
//! identically to every thread on the device, with no in-kernel writes; a
//! `&ConstantMemory<T>` from any thread is sound to read concurrently.

use core::cell::UnsafeCell;

/// Marker trait for values that may be stored in [`ConstantMemory`].
///
/// A constant-memory value is copied byte-for-byte from the host and may be
/// created as an all-zero placeholder by [`ConstantMemory::UNINIT`] before the
/// host populates it.
///
/// # Safety
///
/// Implementors must be safe to duplicate with a byte-for-byte copy, and the
/// all-zero bit pattern must be a valid value of `Self`. Do not implement this
/// trait for references, `NonZero*`, or any type containing a niche that makes
/// the all-zero bit pattern invalid. Custom structs should use an explicit
/// layout such as `#[repr(C)]` before opting in.
pub unsafe trait ConstantMemoryValue: Copy {}

macro_rules! impl_constant_memory_value {
    ($($ty:ty),+ $(,)?) => {
        $(
            unsafe impl ConstantMemoryValue for $ty {}
        )+
    };
}

impl_constant_memory_value!(
    (),
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    f16,
    f32,
    f64
);

unsafe impl<T: ConstantMemoryValue, const N: usize> ConstantMemoryValue for [T; N] {}
unsafe impl<T: ?Sized> ConstantMemoryValue for *const T {}
unsafe impl<T: ?Sized> ConstantMemoryValue for *mut T {}

macro_rules! impl_constant_memory_value_tuple {
    ($($name:ident),+ $(,)?) => {
        unsafe impl<$($name: ConstantMemoryValue),+> ConstantMemoryValue for ($($name,)+) {}
    };
}

impl_constant_memory_value_tuple!(A);
impl_constant_memory_value_tuple!(A, B);
impl_constant_memory_value_tuple!(A, B, C);
impl_constant_memory_value_tuple!(A, B, C, D);
impl_constant_memory_value_tuple!(A, B, C, D, E);
impl_constant_memory_value_tuple!(A, B, C, D, E, F);
impl_constant_memory_value_tuple!(A, B, C, D, E, F, G);
impl_constant_memory_value_tuple!(A, B, C, D, E, F, G, H);

/// A `static`-friendly wrapper that places `T` in CUDA constant memory
/// (`addrspace(4)`).
///
/// See the [module docs](self) for the full usage pattern.
#[repr(transparent)]
pub struct ConstantMemory<T: ConstantMemoryValue>(UnsafeCell<T>);

// SAFETY: ConstantMemory<T> is a host-populated, device-readonly cell. The host
// performs writes via `cuMemcpyHtoD` (synchronized with the calling thread);
// device code only reads. No concurrent writer exists on either side, so
// shared `&ConstantMemory<T>` across threads is sound.
unsafe impl<T: ConstantMemoryValue + Send> Sync for ConstantMemory<T> {}

impl<T: ConstantMemoryValue> ConstantMemory<T> {
    /// All-zero placeholder for a `#[constant]` static.
    ///
    /// CUDA Oxide does not currently lower arbitrary non-zero Rust static
    /// initializers into PTX constant-memory data. Populate the value before
    /// any kernel reads it by calling the macro-generated `set_<name>` before
    /// the kernel launch on the same stream, or by using
    /// `set_<name>_blocking`.
    ///
    /// This follows the placeholder convention used by
    /// [`SharedArray::UNINIT`](crate::SharedArray) and
    /// [`Barrier::UNINIT`](crate::barrier::Barrier).
    ///
    /// `UNINIT` means that the constant has not received its application value;
    /// its underlying bytes are zero. The [`ConstantMemoryValue`] bound rules
    /// out types for which this all-zero placeholder would violate Rust's
    /// validity invariants. Custom types may opt in with an
    /// `unsafe impl ConstantMemoryValue` only when their layout and zero value
    /// satisfy that contract.
    #[allow(clippy::declare_interior_mutable_const)]
    pub const UNINIT: Self = ConstantMemory(UnsafeCell::new(unsafe {
        core::mem::MaybeUninit::<T>::zeroed().assume_init()
    }));

    /// Read the current value.
    ///
    /// Returns a by-value copy of the storage. Safe because constant memory
    /// is read-only from the device — there is no possibility of observing
    /// a torn write from another thread.
    ///
    /// The `UnsafeCell` interior prevents the compiler from hoisting reads
    /// across `set_<name>` boundaries, which means a `.get()` inside a hot
    /// loop will re-read on every iteration.
    ///
    /// For anything larger than a scalar, prefer [`get_ref`](Self::get_ref):
    /// `get` must produce the whole `T` as a value, and a value indexed at
    /// runtime has to be spilled to the thread's local depot first.
    #[inline(always)]
    pub fn get(&self) -> T {
        // SAFETY: read-only from device, and `T: Copy` means we never alias
        // a mutable reference. The host updates this storage only between
        // kernel launches via `cuMemcpyHtoD`, which is synchronized
        // out-of-band relative to device execution.
        unsafe { *self.0.get() }
    }

    /// Borrow the storage in place, reading nothing until something is read.
    ///
    /// This is the accessor to reach for whenever `T` is a table. [`get`](Self::get)
    /// returns `T` *by value*, which is right for a scalar but leaves no way to
    /// index a `ConstantMemory<[f32; N]>` at runtime without materializing the
    /// whole array first: the copy has no address, so it is spilled to the
    /// thread's local depot, one `st.local` per element in *every thread*, and
    /// the lookup then reads thread-private memory.
    ///
    /// Borrowing instead keeps the address in constant space, so a runtime index
    /// is one `ld.const`:
    ///
    /// ```ignore
    /// #[constant]
    /// static TABLE: ConstantMemory<[f32; 256]> = ConstantMemory::UNINIT;
    ///
    /// let t = TABLE.get_ref();          // no load yet
    /// acc += t[i & 255];                // one ld.const
    /// ```
    ///
    /// # Which address space to prefer
    ///
    /// Constant memory is served by a broadcast-oriented cache, so its cost
    /// depends on how much the lanes of a warp agree. `.const` is not simply
    /// "faster memory", and the qualifier is the wrong thing to choose by:
    ///
    /// - **warp-uniform index** — every lane wants the same entry, which is one
    ///   broadcast. This is what constant memory is for, and it is the one case
    ///   that beats global memory.
    /// - **divergent index** — lanes want different entries, and the constant
    ///   cache serves distinct addresses in sequence. A table read this way
    ///   belongs in ordinary global memory, where a warp's accesses coalesce and
    ///   a small table stays resident in L1. A plain `const TABLE: [f32; N]`
    ///   already lowers to exactly that, so it needs no host upload and no
    ///   `#[constant]` at all.
    ///
    /// Measured on an A10G (sm_86), 256-entry `f32` table, 64 dependent lookups
    /// per thread over 8388608 threads, all variants bit-identical:
    ///
    /// | index        | `get()` | `get_ref()` | `const [f32; N]` (global) |
    /// |--------------|--------:|------------:|--------------------------:|
    /// | divergent    | 16048us |      7616us |                 **322us** |
    /// | warp-uniform | 16065us |  **258us**  |                     322us |
    ///
    /// So `get_ref` is 2.1x the throughput of `get` on a divergent index and 62x
    /// on a warp-uniform one — but on a divergent index constant memory is still
    /// 23.7x behind a plain array constant, because that is the pattern its cache
    /// cannot serve in one go.
    ///
    /// # Reads are not folded away
    ///
    /// The returned reference borrows through the [`UnsafeCell`], so the storage
    /// stays mutable as far as the compiler is concerned and a `set_<name>`
    /// between launches remains visible — the same property that makes the
    /// wrapper necessary in the first place. Repeated reads of one entry within a
    /// single launch may still be merged, which is sound: the host writes only
    /// between launches.
    #[inline(always)]
    pub fn get_ref(&self) -> &T {
        // SAFETY: device code only ever reads this storage, so no `&mut` to it
        // exists to alias. The host updates it via `cuMemcpyHtoD` between kernel
        // launches, synchronized out-of-band relative to device execution, which
        // is the same contract `get` relies on; holding a shared reference for
        // the duration of a launch observes no write that `get` would not.
        unsafe { &*self.0.get() }
    }
}
