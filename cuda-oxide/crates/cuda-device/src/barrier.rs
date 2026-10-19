/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Async barrier primitives for Hopper+ architectures.
//!
//! Hardware barriers (`mbarrier`) enable efficient synchronization for async
//! operations like TMA copies. Unlike `sync_threads()`, barriers can track
//! transaction completion asynchronously.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │  TMA (Hardware DMA)                                     │
//! │       │                                                 │
//! │       │ cp.async.bulk.tensor...                         │
//! │       ▼                                                 │
//! │  ┌─────────────┐    mbarrier.arrive    ┌─────────────┐  │
//! │  │   Shared    │◄─────────────────────►│   Barrier   │  │
//! │  │   Memory    │                       │  (64-bit)   │  │
//! │  └─────────────┘    mbarrier.wait      └─────────────┘  │
//! │       ▲                                      ▲          │
//! │       │                                      │          │
//! │  Threads read data              Threads check completion│
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage Pattern
//!
//! ```rust,ignore
//! use cuda_device::{kernel, thread, SharedArray};
//! use cuda_device::barrier::{Barrier, mbarrier_init, mbarrier_arrive, mbarrier_wait};
//!
//! #[kernel]
//! pub fn async_copy_kernel(...) {
//!     // Barrier in shared memory
//!     static mut BAR: Barrier = Barrier::UNINIT;
//!
//!     // Thread 0 initializes (expected arrivals = block size)
//!     if thread::threadIdx_x() == 0 {
//!         unsafe { mbarrier_init(&mut BAR, 128); }
//!     }
//!     thread::sync_threads();
//!
//!     // ... TMA copy would arrive at barrier when done ...
//!
//!     // All threads arrive and wait
//!     let token = unsafe { mbarrier_arrive(&BAR) };
//!     unsafe { mbarrier_wait(&BAR, token); }
//!
//!     // Barrier phase complete - safe to read data
//! }
//! ```
//!
//! # Hardware Support
//!
//! - **sm_80+ (Ampere)**: Basic mbarrier support
//! - **sm_90+ (Hopper)**: Full TMA integration with transaction tracking
//! - **sm_120 (Blackwell)**: Enhanced barrier operations

// =============================================================================
// Barrier Type
// =============================================================================

/// Hardware barrier for async synchronization.
///
/// This is a 64-bit value stored in shared memory that tracks:
/// - Expected arrival count
/// - Current arrival count
/// - Phase bit (for reuse across iterations)
///
/// # Memory Layout (conceptual)
///
/// ```text
/// [63:48] Phase + Hardware State
/// [47:32] Expected Arrival Count
/// [31:0]  Current Arrival Count
/// ```
///
/// # Safety
///
/// - Must be declared as `static mut` in shared memory
/// - Must be initialized before use with `mbarrier_init`
/// - All threads that will arrive must be accounted for in expected count
#[repr(C, align(8))]
#[derive(Copy, Clone)]
pub struct Barrier {
    /// Internal 64-bit state managed by hardware
    _state: u64,
}

impl Barrier {
    /// Uninitialized barrier constant for `static mut` declarations.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// static mut BAR: Barrier = Barrier::UNINIT;
    /// ```
    pub const UNINIT: Self = Self { _state: 0 };
}

include!("generated/mbarrier_basic.rs");
include!("generated/mbarrier_extended.rs");
include!("generated/counted_barrier.rs");

// =============================================================================
// Barrier Wait Operations
// =============================================================================

/// Wait for barrier phase to complete (blocking).
///
/// Blocks until all expected arrivals have occurred for the given phase.
/// This is implemented as a loop over `mbarrier_test_wait`.
///
/// # Parameters
///
/// - `bar`: Pointer to barrier in shared memory
/// - `token`: Phase token from `mbarrier_arrive`
///
/// # Safety
///
/// - `bar` must be initialized
/// - `token` must be from a matching `mbarrier_arrive` call
/// - Calling thread must have arrived at the barrier
///
/// # Example
///
/// ```rust,ignore
/// let token = unsafe { mbarrier_arrive(&BAR) };
/// unsafe { mbarrier_wait(&BAR, token); }
/// // Barrier complete - safe to access synchronized data
/// ```
#[inline(always)]
pub unsafe fn mbarrier_wait(bar: *const Barrier, token: u64) {
    // test_wait keeps this helper available on Ampere.
    while !unsafe { mbarrier_test_wait(bar, token) } {
        // spin
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Arrive and immediately wait (combined operation).
///
/// Convenience function for the common pattern of arriving and then
/// waiting on the same barrier. Equivalent to:
///
/// ```rust,ignore
/// let token = mbarrier_arrive(bar);
/// mbarrier_wait(bar, token);
/// ```
///
/// # Safety
///
/// - `bar` must be initialized
/// - All arriving threads must call this for the barrier to complete
#[inline(always)]
pub unsafe fn mbarrier_arrive_and_wait(bar: *const Barrier) {
    let token = unsafe { mbarrier_arrive(bar) };
    unsafe { mbarrier_wait(bar, token) };
}

// =============================================================================
// Typestate-Based Managed Barrier API
// =============================================================================
//
// This section provides a safer, typestate-based API for barrier management.
// It prevents common mistakes like using a barrier before initialization or
// double-invalidation through compile-time type checking.

use core::marker::PhantomData;

// =============================================================================
// State Markers
// =============================================================================

/// State: Barrier has been claimed but not initialized.
pub struct Uninit;

/// State: Barrier is initialized and ready for arrive/wait operations.
pub struct Ready;

/// State: Barrier has been invalidated and cannot be used.
pub struct Invalidated;

// =============================================================================
// Kind Markers (users can define their own)
// =============================================================================

/// Marker type for TMA-related barriers.
pub struct TmaBarrier;

/// Marker type for MMA/tcgen05 compute barriers.
pub struct MmaBarrier;

/// Marker type for general-purpose barriers.
pub struct GeneralBarrier;

// =============================================================================
// Barrier Token (Newtype)
// =============================================================================

/// Token returned from `arrive()`, must be passed to `wait()`.
///
/// This newtype prevents accidentally passing raw `u64` values
/// where a barrier token is expected.
///
/// # Example
///
/// ```rust,ignore
/// let token = barrier.arrive();
/// barrier.wait(token);  // Type-safe!
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct BarrierToken(u64);

impl BarrierToken {
    /// Get the raw token value (escape hatch for advanced patterns).
    #[inline(always)]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Create from a raw value (for interop with low-level APIs).
    #[inline(always)]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
}

// =============================================================================
// ManagedBarrier
// =============================================================================

/// A barrier with typestate lifecycle management.
///
/// This type uses Rust's type system to enforce correct barrier usage:
/// - Cannot `arrive()` on an uninitialized barrier
/// - Cannot double-initialize
/// - Cannot use after invalidation
/// - `inval()` consumes the barrier, preventing reuse
///
/// # Type Parameters
///
/// - `State`: Current lifecycle state (`Uninit`, `Ready`, `Invalidated`)
/// - `Kind`: Marker type distinguishing different barriers (`TmaBarrier`, `MmaBarrier`, etc.)
/// - `ID`: Const generic index for multiple barriers of the same kind (default 0)
///
/// # Thread Requirements
///
/// | Operation       | Thread Requirement        |
/// |-----------------|---------------------------|
/// | `from_static()` | Single thread (thread 0)  |
/// | `init()`        | Single thread (thread 0)  |
/// | `arrive()`      | All participating threads |
/// | `wait()`        | All participating threads |
/// | `inval()`       | Single thread (thread 0)  |
///
/// # Example
///
/// ```rust,ignore
/// // Thread 0: Create and initialize
/// let bar = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut BAR);
/// let bar = unsafe { bar.init(128) };  // Now Ready
/// fence_proxy_async_shared_cta();
/// sync_threads();
///
/// // All threads: arrive and wait
/// let token = bar.arrive();
/// bar.wait(token);
///
/// // Thread 0: Cleanup
/// sync_threads();
/// unsafe { bar.inval(); }
/// ```
pub struct ManagedBarrier<State, Kind, const ID: usize = 0> {
    ptr: *const Barrier,
    _state: PhantomData<State>,
    _kind: PhantomData<Kind>,
}

// Safety: Barrier pointer is only accessed through synchronized operations
unsafe impl<S, K, const ID: usize> Send for ManagedBarrier<S, K, ID> {}

// =============================================================================
// Uninit State Implementation
// =============================================================================

impl<Kind, const ID: usize> ManagedBarrier<Uninit, Kind, ID> {
    /// Create an Uninit barrier from an explicit static declaration.
    ///
    /// Wrap a `static mut Barrier` in the typestate wrapper. Each barrier
    /// must be declared as a separate `static mut` variable, following the
    /// same pattern as `SharedArray`.
    ///
    /// # Thread Requirements
    ///
    /// **Single-thread**: Only call from the initialization thread (thread 0).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// static mut BAR0: Barrier = Barrier::UNINIT;
    /// static mut BAR1: Barrier = Barrier::UNINIT;
    ///
    /// if tid == 0 {
    ///     let bar0 = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut BAR0);
    ///     let bar1 = ManagedBarrier::<Uninit, GeneralBarrier>::from_static(&raw mut BAR1);
    ///
    ///     let bar0 = unsafe { bar0.init(32) };
    ///     let bar1 = unsafe { bar1.init(32) };
    /// }
    /// ```
    pub fn from_static(ptr: *mut Barrier) -> Self {
        ManagedBarrier {
            ptr,
            _state: PhantomData,
            _kind: PhantomData,
        }
    }

    /// Initialize the barrier with an expected arrival count.
    ///
    /// **All threads in the block should call this.** Only thread 0 performs
    /// the actual initialization; all threads synchronize and receive a `Ready` handle.
    ///
    /// This is a convenience wrapper for `init_by(count, 0)`.
    ///
    /// # Block-Scoped Barriers
    ///
    /// mbarrier operates at **block scope** - each block has its own barrier in
    /// shared memory. Exactly ONE thread per block must call `mbarrier_init`.
    ///
    /// # Safety
    ///
    /// - Must be called before any arrive/wait operations
    /// - All participating threads in the block must call this together
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// static mut BAR: Barrier = Barrier::UNINIT;
    ///
    /// // ALL threads call init - only thread 0 actually initializes
    /// let bar = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut BAR);
    /// let bar = unsafe { bar.init(128) };  // All threads get Ready handle
    /// ```
    #[inline(always)]
    pub unsafe fn init(self, count: u32) -> ManagedBarrier<Ready, Kind, ID> {
        unsafe { self.init_by(count, 0) }
    }

    /// Initialize the barrier with a specific thread performing initialization.
    ///
    /// **All threads in the block should call this.** Only the thread with
    /// `threadIdx.x == init_thread` performs the actual initialization;
    /// all threads synchronize and receive a `Ready` handle.
    ///
    /// # Block-Scoped Barriers
    ///
    /// mbarrier operates at **block scope** - each block has its own barrier in
    /// shared memory. Exactly ONE thread per block must call `mbarrier_init`.
    /// Any thread can be the initializer, not just thread 0.
    ///
    /// # Parameters
    ///
    /// - `count`: Expected number of arrivals before barrier completes
    /// - `init_thread`: Thread ID (threadIdx.x) that performs initialization
    ///
    /// # Safety
    ///
    /// - Must be called before any arrive/wait operations
    /// - All participating threads in the block must call this together
    /// - `init_thread` must be a valid thread ID within the block
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// static mut BAR: Barrier = Barrier::UNINIT;
    ///
    /// // Use thread 31 (last thread in first warp) as initializer
    /// let bar = ManagedBarrier::<Uninit, TmaBarrier>::from_static(&raw mut BAR);
    /// let bar = unsafe { bar.init_by(128, 31) };
    /// ```
    #[inline(always)]
    pub unsafe fn init_by(self, count: u32, init_thread: u32) -> ManagedBarrier<Ready, Kind, ID> {
        if crate::thread::threadIdx_x() == init_thread {
            unsafe {
                mbarrier_init(self.ptr as *mut Barrier, count);
                fence_proxy_async_shared_cta();
            }
        }
        // All threads synchronize - ensures init is visible to all
        crate::thread::sync_threads();

        ManagedBarrier {
            ptr: self.ptr,
            _state: PhantomData,
            _kind: PhantomData,
        }
    }
}

// =============================================================================
// Ready State Implementation
// =============================================================================

impl<Kind, const ID: usize> ManagedBarrier<Ready, Kind, ID> {
    /// Get the raw pointer to the underlying barrier.
    ///
    /// Useful for interop with low-level APIs.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const Barrier {
        self.ptr
    }

    /// Arrive at the barrier.
    ///
    /// Returns a token that must be passed to `wait()` or `try_wait()`.
    ///
    /// # Thread Requirements
    ///
    /// All participating threads must call this.
    #[inline(always)]
    pub fn arrive(&self) -> BarrierToken {
        unsafe { BarrierToken(mbarrier_arrive(self.ptr)) }
    }

    /// Arrive at the barrier expecting TMA transaction bytes.
    ///
    /// Use when this barrier tracks TMA copy completion. The barrier won't
    /// complete until both all arrivals occur AND all expected bytes transfer.
    ///
    /// # Thread Requirements
    ///
    /// **Single-thread**: The thread that issued the TMA copy.
    #[inline(always)]
    pub fn arrive_expect_tx(&self, bytes: u32) -> BarrierToken {
        unsafe { BarrierToken(mbarrier_arrive_expect_tx(self.ptr, 1, bytes)) }
    }

    /// Wait for barrier completion (blocking).
    ///
    /// Blocks until all expected arrivals have occurred.
    ///
    /// # Thread Requirements
    ///
    /// All participating threads should wait.
    #[inline(always)]
    pub fn wait(&self, token: BarrierToken) {
        unsafe { mbarrier_wait(self.ptr, token.0) }
    }

    /// Try to wait for barrier completion (non-blocking).
    ///
    /// Returns `true` if the barrier phase is complete, `false` otherwise.
    /// Preferred over busy-looping on `test_wait` due to better scheduling hints.
    #[inline(always)]
    pub fn try_wait(&self, token: BarrierToken) -> bool {
        unsafe { mbarrier_try_wait(self.ptr, token.0) }
    }

    /// Test if barrier phase is complete (non-blocking).
    #[inline(always)]
    pub fn test_wait(&self, token: BarrierToken) -> bool {
        unsafe { mbarrier_test_wait(self.ptr, token.0) }
    }

    /// Try wait using parity (for tcgen05.commit patterns).
    ///
    /// Use when the producer arrives via operations that don't return tokens
    /// (like `tcgen05_commit`).
    #[inline(always)]
    pub fn try_wait_parity(&self, parity: u32) -> bool {
        unsafe { mbarrier_try_wait_parity(self.ptr, parity) }
    }

    /// Invalidate the barrier.
    ///
    /// **All threads in the block should call this.** Only thread 0 performs
    /// the actual invalidation; all threads synchronize before returning.
    ///
    /// This is a convenience wrapper for `inval_by(0)`.
    ///
    /// Consumes the `Ready` barrier and returns an `Invalidated` barrier.
    /// The underlying memory can be reused after this.
    ///
    /// # Safety
    ///
    /// - All threads must have completed their wait operations before calling
    /// - All participating threads in the block must call this together
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // ALL threads call inval - only thread 0 actually invalidates
    /// let _dead = unsafe { bar.inval() };  // Consumes Ready, returns Invalidated
    /// ```
    #[inline(always)]
    pub unsafe fn inval(self) -> ManagedBarrier<Invalidated, Kind, ID> {
        unsafe { self.inval_by(0) }
    }

    /// Invalidate the barrier with a specific thread performing invalidation.
    ///
    /// **All threads in the block should call this.** Only the thread with
    /// `threadIdx.x == inval_thread` performs the actual invalidation;
    /// all threads synchronize before returning.
    ///
    /// Consumes the `Ready` barrier and returns an `Invalidated` barrier.
    ///
    /// # Parameters
    ///
    /// - `inval_thread`: Thread ID (threadIdx.x) that performs invalidation
    ///
    /// # Safety
    ///
    /// - All threads must have completed their wait operations before calling
    /// - All participating threads in the block must call this together
    /// - `inval_thread` must be a valid thread ID within the block
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Use thread 31 as the invalidator
    /// let _dead = unsafe { bar.inval_by(31) };
    /// ```
    #[inline(always)]
    pub unsafe fn inval_by(self, inval_thread: u32) -> ManagedBarrier<Invalidated, Kind, ID> {
        // Ensure all threads are done with the barrier before invalidating
        crate::thread::sync_threads();

        if crate::thread::threadIdx_x() == inval_thread {
            unsafe { mbarrier_inval(self.ptr as *mut Barrier) };
        }

        // All threads synchronize - ensures inval is complete
        crate::thread::sync_threads();

        ManagedBarrier {
            ptr: self.ptr,
            _state: PhantomData,
            _kind: PhantomData,
        }
    }
}

// =============================================================================
// Type Aliases for Convenience
// =============================================================================

/// TMA barrier handle (single instance, ID=0)
pub type TmaBarrierHandle<S> = ManagedBarrier<S, TmaBarrier, 0>;

/// MMA barrier handle (single instance, ID=0)
pub type MmaBarrierHandle<S> = ManagedBarrier<S, MmaBarrier, 0>;

/// Double-buffered TMA barrier #0
pub type TmaBarrier0<S> = ManagedBarrier<S, TmaBarrier, 0>;

/// Double-buffered TMA barrier #1
pub type TmaBarrier1<S> = ManagedBarrier<S, TmaBarrier, 1>;
