// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Internal, versioned boundary for native device library calls.
//!
//! This module is an unsafe compiler integration point, not a collective API.

/// Read a Cargo-provided registration and track its native build dependencies.
///
/// Pass the environment variable name as a string literal inside the registration
/// getter. Materialized builds query the provider for file, directory and
/// environment dependencies before rustc emits dep-info. Host checks and rustdoc
/// do not query CUDA inputs. Cargo's timestamp freshness rules still apply;
/// replacements with unchanged or older timestamps may not be detected.
///
/// This snippet requires a build script to set `MY_LIBRARY_REGISTRATION`; the
/// `native_library` example compiles and runs the same registration pattern.
///
/// ```ignore
/// fn registration() -> &'static str {
///     cuda_device::library::registration!("MY_LIBRARY_REGISTRATION")
/// }
/// ```
pub use cuda_macros::library_registration as registration;

/// Invoke a library recipe identified by a constant registration getter.
///
/// Call this marker directly inside a wrapper; the marker itself is not a
/// device function pointer or callable-trait value.
///
/// `Registration` must be a function item returning one constant UTF-8 JSON
/// string. The compiler reads its MIR without calling it. `Args` must be a
/// tuple of the exact scalar and raw-pointer types the provider declares.
/// `T` supplies the scalar type, and the const parameters supply the exact
/// one-dimensional block shape `[BLOCK_THREADS, 1, 1]` and per-thread extent.
///
/// # Safety
///
/// The caller must meet the provider's pointer, lifetime, initialization,
/// participation, and launch-geometry requirements. Library scratch views in
/// one kernel alias one dynamic shared base; the caller must synchronize before
/// reusing it. This version does not permit mixing native dynamic
/// shared views with library calls. It returns no allocation or ownership token.
#[inline(never)]
pub unsafe fn cuda_oxide_library_v1_246e25db_call<
    T: Copy,
    Registration: Fn() -> &'static str + Copy,
    Args: Copy,
    const BLOCK_THREADS: usize,
    const ITEMS_PER_THREAD: usize,
>(
    _registration: Registration,
    _args: Args,
) {
    panic!("library::call_v1 called outside device compilation");
}

pub use cuda_oxide_library_v1_246e25db_call as call_v1;

/// Invoke a full-block library operation with compiler-checked participation.
///
/// The exact block shape comes from the root kernel's launch contract.
/// `uniform` is `(memory_pointer, scalar_controls...)`; `payload` contains
/// lane-local const or mutable payload pointers matching the provider ABI.
/// The compiler checks all call contexts before reconstructing the provider's
/// ordinary scalar/pointer ABI.
///
/// # Safety
///
/// The caller supplies the correct provider recipe, live aligned memory and
/// payload pointers, sufficient capacity, and the provider's initialization and
/// aliasing requirements. Group checks do not prove bounds or exclusive memory
/// ownership. Scratch aliases the kernel's dynamic shared base; the caller must
/// synchronize before conflicting scratch or output reuse. Completion never
/// follows from ending a Rust borrow or running a destructor.
#[inline(never)]
pub unsafe fn cuda_oxide_library_v1_246e25db_checked_block_call<
    T: Copy,
    Registration: Fn() -> &'static str + Copy,
    UniformArgs: Copy,
    PayloadArgs: Copy,
    const ITEMS_PER_THREAD: usize,
>(
    _registration: Registration,
    _uniform: UniformArgs,
    _payload: PayloadArgs,
) {
    // A backend which does not recognize this version must fail to link rather
    // than silently compiling a panic/trap as the collective implementation.
    unsafe extern "C" {
        fn cuda_oxide_checked_block_call_requires_supported_backend();
    }
    unsafe { cuda_oxide_checked_block_call_requires_supported_backend() }
}

pub use cuda_oxide_library_v1_246e25db_checked_block_call as checked_block_call_v1;

/// Invoke a scratch-free full-block operation with a typed memory witness.
///
/// The compiler accepts `&[T]` for reads, a reference to
/// `DisjointSlice<T, CollectiveTiles<C>>` for reads, and a mutable reference to
/// that carrier for writes. It tracks the original kernel allocation through
/// helpers and requires a full-block barrier before conflicting reuse.
/// The witness is not passed to the native function. `uniform` and `payload`
/// have the same roles as [`checked_block_call_v1`].
///
/// # Safety
///
/// The native pointer and accessed range must describe `memory`. The wrapper
/// must validate capacity, canonical group ownership, arithmetic, and bounds
/// before forming that pointer. Native memory effects must stay within that
/// range and the lane-local payload; `WRITE = false` forbids writes to `memory`.
/// The recipe, payload initialization and native ABI must be correct. Compiler
/// provenance and participation checks do not verify these obligations of the
/// unsafe wrapper implementation.
#[inline(never)]
pub unsafe fn cuda_oxide_library_v1_246e25db_checked_block_resource_call<
    T: Copy,
    Registration: Fn() -> &'static str + Copy,
    Memory,
    UniformArgs: Copy,
    PayloadArgs: Copy,
    const ITEMS: usize,
    const WRITE: bool,
>(
    _registration: Registration,
    _memory: Memory,
    _uniform: UniformArgs,
    _payload: PayloadArgs,
) {
    unsafe extern "C" {
        fn cuda_oxide_checked_block_resource_call_requires_supported_backend();
    }
    unsafe { cuda_oxide_checked_block_resource_call_requires_supported_backend() }
}

pub use cuda_oxide_library_v1_246e25db_checked_block_resource_call as checked_block_resource_call_v1;

/// Invoke a Warp operation with a typed memory witness and implicit scratch.
///
/// WIDTH is a power of two in 1..=32. The root's full exact block shape must contain only
/// complete physical warps. Controls and participation agree within WIDTH;
/// sibling groups may differ. A containing canonical WarpTile or block barrier
/// completes conflicting accesses within that scope. The compiler allocates
/// disjoint scratch replicas for canonical groups of each width and inserts a
/// matching barrier before dirty scratch reuse. First use and scratch-free
/// operations supply no completion, and no trailing barrier is implied.
///
/// # Safety
///
/// The pointer, witness, native recipe, payload and bounds must satisfy
/// [`checked_block_resource_call_v1`]. The range must additionally stay inside
/// this group's fixed, canonical lane ownership interval. The compiler checks
/// origin and participation, not these pointer/bounds obligations. No explicit
/// scratch or unsupported group mapping may be composed into this operation.
#[inline(never)]
pub unsafe fn cuda_oxide_library_v1_246e25db_checked_warp_call<
    T: Copy,
    Registration: Fn() -> &'static str + Copy,
    Memory,
    UniformArgs: Copy,
    PayloadArgs: Copy,
    const WIDTH: usize,
    const ITEMS: usize,
    const WRITE: bool,
>(
    _registration: Registration,
    _memory: Memory,
    _uniform: UniformArgs,
    _payload: PayloadArgs,
) {
    unsafe extern "C" {
        fn cuda_oxide_checked_warp_call_requires_supported_backend();
    }
    unsafe { cuda_oxide_checked_warp_call_requires_supported_backend() }
}

pub use cuda_oxide_library_v1_246e25db_checked_warp_call as checked_warp_call_v1;

/// Canonical x-fastest group index within the block.
///
/// WIDTH must be a power of two in 1..=32. This pure query does not assert
/// participation and does not require a native provider or a launch contract.
#[inline(never)]
pub fn cuda_oxide_library_v1_246e25db_warp_group_index<const WIDTH: usize>() -> u32 {
    assert!(WIDTH > 0 && WIDTH <= 32 && WIDTH.is_power_of_two());
    let rank = crate::thread::threadIdx_x()
        + crate::thread::blockDim_x()
            * (crate::thread::threadIdx_y()
                + crate::thread::blockDim_y() * crate::thread::threadIdx_z());
    rank / WIDTH as u32
}

pub use cuda_oxide_library_v1_246e25db_warp_group_index as warp_group_index_v1;

/// Synchronize the canonical aligned tile containing the executing lane.
///
/// This genuine marker has the same device fallback as WarpTile::sync. It
/// requires no provider or native resource plan.
///
/// # Safety
/// WIDTH must be a power of two in 1..=32. Every lane in the canonical tile
/// must execute this synchronization in converged control flow. Checked roots
/// enforce this participation and credit only completion within that tile.
/// Outside checked roots, even the safe `WarpTile::sync` interface relies on
/// the caller to satisfy this collective participation requirement.
#[inline(never)]
pub unsafe fn cuda_oxide_library_v1_246e25db_checked_warp_sync<const WIDTH: u32>() {
    assert!(WIDTH > 0 && WIDTH <= 32 && WIDTH.is_power_of_two());
    let mask = if WIDTH == 32 {
        u32::MAX
    } else {
        ((1u32 << WIDTH) - 1) << ((crate::warp::lane_id() / WIDTH) * WIDTH)
    };
    crate::warp::sync_mask(mask);
}

pub use cuda_oxide_library_v1_246e25db_checked_warp_sync as checked_warp_sync_v1;

/// Internal full-block scratch policy. Its private token has no address identity.
/// Allocation and synchronization are finalized per rooted static use site.
/// This descriptor is neither a Rust borrow of shared memory nor completion.
pub struct StorageDescriptor<
    const CAPACITY: usize = 0,
    const ALIGN: usize = 0,
    const AUTO_SYNC: bool = false,
    const EXCLUSIVE: bool = false,
> {
    _token: u8,
}

/// A compiler-planned typed shared view. No Rust reference is created.
/// Releasing the view does not synchronize or initialize its contents.
#[repr(transparent)]
pub struct StorageReservation<T, const N: usize> {
    pointer: *mut T,
}

impl<T, const N: usize> StorageReservation<T, N> {
    /// Return the raw shared pointer. Dereferencing it requires the caller to
    /// prove initialization, bounds, aliasing and phase synchronization.
    #[inline(always)]
    pub fn as_mut_ptr(&self) -> *mut T {
        self.pointer
    }
}

/// Create a compiler-owned descriptor origin at this rooted static call site.
/// Zero capacity/alignment mean inferred; explicit alignment is a power of two.
///
/// # Safety
/// All block lanes must execute this marker with the same policies and origin.
/// Only a supported compiler may interpret the returned descriptor.
#[inline(never)]
pub unsafe fn cuda_oxide_library_v1_246e25db_storage_new<
    const CAPACITY: usize,
    const ALIGN: usize,
    const AUTO_SYNC: bool,
    const EXCLUSIVE: bool,
>() -> StorageDescriptor<CAPACITY, ALIGN, AUTO_SYNC, EXCLUSIVE> {
    unsafe extern "C" {
        fn cuda_oxide_storage_requires_supported_backend();
    }
    unsafe { cuda_oxide_storage_requires_supported_backend() }
    StorageDescriptor { _token: 0 }
}
pub use cuda_oxide_library_v1_246e25db_storage_new as storage_new_v1;

/// Consume a descriptor and select its policy before any use.
///
/// # Safety
/// `Old` and `New` must be genuine storage descriptors. The compiler verifies
/// their policies and preserves the original static identity.
#[inline(never)]
pub unsafe fn cuda_oxide_library_v1_246e25db_storage_reconfigure<Old, New>(_old: Old) -> New {
    unsafe extern "C" {
        fn cuda_oxide_storage_reconfigure_requires_supported_backend() -> !;
    }
    unsafe { cuda_oxide_storage_reconfigure_requires_supported_backend() }
}
pub use cuda_oxide_library_v1_246e25db_storage_reconfigure as storage_reconfigure_v1;

/// Invoke a full-block native operation with planned storage and a memory witness.
/// `storage` is `()` (managed reuse) or a mutable borrow of a genuine descriptor.
///
/// # Safety
/// The obligations of [`checked_block_resource_call_v1`] apply. Provider scratch
/// effects must fit its declared layout. Explicit manual storage requires a
/// full-block barrier before reuse; automatic storage adds a trailing barrier.
#[inline(never)]
pub unsafe fn cuda_oxide_library_v1_246e25db_checked_block_storage_call<
    T: Copy,
    Registration: Fn() -> &'static str + Copy,
    Memory,
    Storage,
    UniformArgs: Copy,
    PayloadArgs: Copy,
    const ITEMS_PER_THREAD: usize,
    const WRITE: bool,
>(
    _registration: Registration,
    _memory: Memory,
    _storage: Storage,
    _uniform: UniformArgs,
    _payload: PayloadArgs,
) {
    unsafe extern "C" {
        fn cuda_oxide_storage_requires_supported_backend();
    }
    unsafe { cuda_oxide_storage_requires_supported_backend() }
}
pub use cuda_oxide_library_v1_246e25db_checked_block_storage_call as checked_block_storage_call_v1;

/// Reserve a typed view at a static use site; all lanes receive a raw address.
///
/// # Safety
/// Descriptor and participation must be block-uniform. `T` must be a supported
/// numeric scalar, `N` positive, and alignment zero or a power of two. The caller
/// initializes and accesses shared elements without conflicting raw accesses.
/// Automatic trailing synchronization is incompatible with reservations.
#[inline(never)]
pub unsafe fn cuda_oxide_library_v1_246e25db_storage_reserve<
    T: Copy,
    Descriptor,
    const N: usize,
    const ALIGN: usize,
>(
    _descriptor: &mut Descriptor,
) -> StorageReservation<T, N> {
    unsafe extern "C" {
        fn cuda_oxide_storage_requires_supported_backend();
    }
    unsafe { cuda_oxide_storage_requires_supported_backend() }
    StorageReservation {
        pointer: core::ptr::null_mut(),
    }
}
pub use cuda_oxide_library_v1_246e25db_storage_reserve as storage_reserve_v1;

/// End a reservation view without completing pending shared or output accesses.
///
/// # Safety
/// No subsequent raw access may use this view. Synchronize the full block after
/// release and before any overlapping reuse. Dropping/forgetting is not release.
#[inline(never)]
pub unsafe fn cuda_oxide_library_v1_246e25db_storage_release<T: Copy, const N: usize>(
    _reservation: StorageReservation<T, N>,
) {
    unsafe extern "C" {
        fn cuda_oxide_storage_requires_supported_backend();
    }
    unsafe { cuda_oxide_storage_requires_supported_backend() }
}
pub use cuda_oxide_library_v1_246e25db_storage_release as storage_release_v1;

/// Invoke a synchronous native operation with a compiler-rooted scalar callback.
///
/// The thunk must be a function item with signature
/// `unsafe extern "C" fn(*const Environment, i32, i32) -> i32`.
/// `Environment` is a function item or an immutable closure capturing fixed-width
/// numeric scalars by value. The first version supports `T = i32` only.
/// The compiler retains the exact thunk and supplies its symbol to the provider;
/// neither witness is passed as a runtime function pointer.
///
/// # Safety
///
/// All requirements of [`call_v1`] apply. Every participant must use the same
/// callable specialization and capture values. The provider may synchronously
/// pass the opaque environment pointer back to the thunk, but must not inspect,
/// mutate, retain, free, or publish it. The environment remains live throughout
/// the native call. The callback must not perform collectives, synchronization,
/// shared-memory access, foreign calls, or indirect calls. This raw boundary
/// does not prove capture-value uniformity or pointer ownership.
#[inline(never)]
pub unsafe fn cuda_oxide_library_v1_246e25db_call_with_callback<
    T: Copy,
    Registration: Fn() -> &'static str + Copy,
    Args: Copy,
    Environment: Copy,
    Thunk: Copy,
    const BLOCK_THREADS: usize,
    const ITEMS_PER_THREAD: usize,
>(
    _registration: Registration,
    _args: Args,
    _thunk: Thunk,
    _environment: &Environment,
) {
    unsafe extern "C" {
        fn cuda_oxide_callback_call_requires_supported_backend();
    }
    unsafe { cuda_oxide_callback_call_requires_supported_backend() }
}

pub use cuda_oxide_library_v1_246e25db_call_with_callback as call_with_callback_v1;
