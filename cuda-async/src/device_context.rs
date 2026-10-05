/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU device state and scheduling policy management.
//!
//! ## Architecture
//!
//! - **Global (process-wide)**: one [`Device`] per device id, shared across all threads
//!   via [`OnceLock`].
//!
//! - **Per-thread**: Scheduling policy and deallocator stream remain thread-local, since
//!   different threads may want different stream assignments.
//!
//! The compiled-kernel cache and its single-flight dedup live in `cutile::tile_kernel`,
//! since they key on a type from `cutile-compiler`.

use crate::error::{device_assert, device_error, DeviceError};
use crate::scheduling_policies::{SchedulingPolicy, StreamPoolRoundRobin};
use cuda_core::{Device, Function, MemPool, Module, Stream};
use std::cell::Cell;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

/// The GPU device used when no explicit device is specified. Device 0 is the first GPU.
pub const DEFAULT_DEVICE_ID: usize = 0;

/// The number of GPU devices initialized by default.
pub const DEFAULT_NUM_DEVICES: usize = 1;

/// The number of CUDA streams in the default round-robin pool.
///
/// With a pool of 4 streams, consecutive operations cycle through streams 0 → 1 → 2 → 3 → 0 → …,
/// allowing up to 4 independent operations to overlap on the GPU. Increasing this value adds more
/// potential concurrency at the cost of additional stream resources; decreasing it (down to 1)
/// makes behavior equivalent to [`SingleStream`](crate::scheduling_policies::SingleStream).
pub const DEFAULT_ROUND_ROBIN_STREAM_POOL_SIZE: usize = 4;

pub trait FunctionKey: Hash {
    /// Short human-readable digest of the key, for log lines and dump filenames.
    ///
    /// **Not an identity.** It is a 64-bit `DefaultHasher` output, so distinct
    /// keys collide at a rate that matters once a process caches enough kernels,
    /// and `DefaultHasher`'s algorithm is unspecified across Rust releases.
    /// Caches key on the whole `Hash + Eq` value; nothing may key on this string
    /// or persist it.
    fn display_hash(&self) -> String {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        let hash_value: u64 = hasher.finish();
        format!("{:x}", hash_value)
    }
}

#[derive(Debug, Clone)]
pub enum ValidParamType {
    Scalar(ScalarParamType),
    Pointer(PointerParamType),
    Tensor(TensorParamType),
}

#[derive(Debug, Clone)]
pub struct ScalarParamType {
    pub element_type: String,
}

#[derive(Debug, Clone)]
pub struct PointerParamType {
    pub mutable: bool,
    pub element_type: String,
}

// TODO (hme): This is note entirely tile-agnostic with this param type.
#[derive(Debug, Clone)]
pub struct TensorParamType {
    pub element_type: String,
    pub shape: Vec<i32>,
}

#[derive(Debug, Clone)]
pub struct Validator {
    pub params: Vec<ValidParamType>,
    /// Compiler-emitted checks to run at launch, before `cuLaunchKernel`. Each
    /// is a canonical [`crate::predicate::Predicate`] the compiler hoisted out
    /// of the device kernel; the host evaluates it against the launched tensors'
    /// extents. Empty unless a kernel hoists a launch-known safety check.
    pub launch_checks: Vec<crate::predicate::LaunchCheck>,
}

// ── Global Device (process-wide, per-device singleton) ─────────────────────

/// Global per-device handles. Shared across all threads so that
/// `Module`/`Function` loaded against a device can be used from any thread.
static DEVICES: OnceLock<Mutex<HashMap<usize, Arc<Device>>>> = OnceLock::new();

fn devices() -> &'static Mutex<HashMap<usize, Arc<Device>>> {
    DEVICES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Get or create the global [`Device`] for a device ordinal.
///
/// The first call for a given `device_id` creates the device handle; subsequent
/// calls return the same `Arc<Device>`.
fn get_or_init_device(device_id: usize) -> Result<Arc<Device>, DeviceError> {
    let mut devices = devices()
        .lock()
        .map_err(|_| device_error(device_id, "device map lock poisoned"))?;
    if let Some(device) = devices.get(&device_id) {
        return Ok(Arc::clone(device));
    }
    let device = Device::new(device_id)?;
    devices.insert(device_id, Arc::clone(&device));
    Ok(device)
}

// ── Compiled kernels ────────────────────────────────────────────────────────

/// A compiled kernel: module, function handle, and parameter validator.
///
/// The cache that holds these lives in `cutile::tile_kernel`, keyed on
/// `TileFunctionKey` directly. It cannot live here: that key names types from
/// `cutile-compiler`, which sits downstream of this crate.
#[derive(Debug)]
pub struct CompiledKernel {
    pub module: Arc<Module>,
    pub function: Arc<Function>,
    pub validator: Arc<Validator>,
}

// ── Per-thread device state ──────────────────────────────────────────────────

/// Per-thread, per-device state: scheduling policy, deallocator stream, and
/// optional memory pool.
///
/// The CUDA context is global (see above). This struct only holds thread-local
/// state.
pub struct AsyncDeviceContext {
    #[expect(dead_code, reason = "will be used when multi-device is implemented")]
    device_id: usize,
    deallocator_stream: Arc<Stream>,
    policy: Arc<dyn SchedulingPolicy>,
    pool: Option<Arc<MemPool>>,
}

pub struct AsyncDeviceContexts {
    default_device: Cell<usize>,
    devices: Cell<Option<HashMap<usize, AsyncDeviceContext>>>,
}

// Manage a statically accessible device context, and their associated streams.
thread_local!(static DEVICE_CONTEXTS: AsyncDeviceContexts = const {
    AsyncDeviceContexts {
        default_device: Cell::new(DEFAULT_DEVICE_ID),
        devices: Cell::new(None),
    }
});

/// Returns the current thread's default GPU device ID.
///
/// This is the device used by `.sync()`, `.await`, and other operations that do not
/// specify a device explicitly. Defaults to [`DEFAULT_DEVICE_ID`] (0).
pub fn get_default_device() -> usize {
    DEVICE_CONTEXTS.with(|ctx| ctx.default_device.get())
}

/// Initialize the device context map for the current thread.
///
/// Call this **before** any GPU work if you need to change the default device or
/// pre-allocate contexts for multiple devices. Individual device contexts are still
/// lazily created on first access (with the default round-robin policy) if not
/// explicitly added via [`init_device`].
///
/// # Errors
///
/// Returns [`DeviceError::Context`] if contexts have already been initialized on
/// this thread. The existing contexts are left untouched.
pub fn init_device_contexts(
    default_device_id: usize,
    num_devices: usize,
) -> Result<(), DeviceError> {
    DEVICE_CONTEXTS.with(|ctx| {
        // `Cell<Option<_>>` has no non-consuming `is_some`, so the map has to
        // be taken out to inspect it. Put a live map straight back: the old
        // `replace(None).is_none()` check discarded every existing context
        // (streams, pools, policies) while reporting "already initialized".
        if let Some(existing) = ctx.devices.take() {
            ctx.devices.set(Some(existing));
            return Err(device_error(
                default_device_id,
                "Context already initialized.",
            ));
        }
        ctx.default_device.set(default_device_id);
        ctx.devices.set(Some(HashMap::with_capacity(num_devices)));
        Ok(())
    })
}

pub fn init_device_contexts_default() -> Result<(), DeviceError> {
    let default_device = get_default_device();
    // TODO (hme): Detect number of devices.
    init_device_contexts(default_device, DEFAULT_NUM_DEVICES)
}

/// Create a new [`AsyncDeviceContext`] with a custom scheduling policy.
///
/// This is the low-level constructor. Most users should use [`init_device`] or let the
/// runtime auto-initialize with the default policy.
pub fn new_device_context(
    device_id: usize,
    policy: Arc<dyn SchedulingPolicy>,
) -> Result<AsyncDeviceContext, DeviceError> {
    let device = get_or_init_device(device_id)?;
    let deallocator_stream = device.new_stream()?;
    Ok(AsyncDeviceContext {
        device_id,
        deallocator_stream,
        policy,
        pool: None,
    })
}

/// Add a device with a specific scheduling policy to the context map.
///
/// # Example: Using 8 streams instead of the default 4
///
/// ```rust,ignore
/// use cuda_async::device_context::*;
/// use cuda_async::scheduling_policies::*;
///
/// // Before any GPU work:
/// init_device_contexts(0, 1).unwrap();
/// // Then add device 0 with a custom stream pool size:
/// let policy = unsafe { StreamPoolRoundRobin::new(0, 8) };
/// // (use with_global_device_context_mut or init_device internally)
/// ```
pub fn init_device(
    hashmap: &mut HashMap<usize, AsyncDeviceContext>,
    device_id: usize,
    policy: Arc<dyn SchedulingPolicy>,
) -> Result<(), DeviceError> {
    let device_context = new_device_context(device_id, policy)?;
    let pred = hashmap.insert(device_id, device_context).is_none();
    device_assert(device_id, pred, "Device is already initialized.")
}

pub fn init_with_default_policy(
    hashmap: &mut HashMap<usize, AsyncDeviceContext>,
    device_id: usize,
) -> Result<(), DeviceError> {
    let device = get_or_init_device(device_id)?;
    let policy = StreamPoolRoundRobin::new(&device, DEFAULT_ROUND_ROBIN_STREAM_POOL_SIZE)?;
    let deallocator_stream = device.new_stream()?;
    let device_context = AsyncDeviceContext {
        device_id,
        deallocator_stream,
        policy: Arc::new(policy),
        pool: None,
    };
    let pred = hashmap.insert(device_id, device_context).is_none();
    device_assert(device_id, pred, "Device is already initialized.")
}

/// Holds the thread's context map while a closure runs against it, and puts it
/// back into the thread-local `Cell` on **every** exit path, including `?`
/// early returns and panics. Without this, a failed lookup (an invalid device
/// ordinal, a driver error while creating the default policy) dropped the map
/// on the floor — and with it every live stream, pool, and policy of the
/// thread, which the next call then silently re-created from scratch.
struct MapLease<'c> {
    cell: &'c Cell<Option<HashMap<usize, AsyncDeviceContext>>>,
    map: Option<HashMap<usize, AsyncDeviceContext>>,
}

impl Drop for MapLease<'_> {
    fn drop(&mut self) {
        self.cell.set(self.map.take());
    }
}

/// Runs `f` against the current thread's context map, lazily initializing the
/// map itself if this thread has none yet. The map is leased for the duration
/// of `f` and restored afterwards (see [`MapLease`]).
fn with_context_map<R>(
    device_id: usize,
    f: impl FnOnce(&mut HashMap<usize, AsyncDeviceContext>) -> Result<R, DeviceError>,
) -> Result<R, DeviceError> {
    DEVICE_CONTEXTS.with(|ctx| {
        let map = match ctx.devices.take() {
            Some(map) => map,
            None => {
                init_device_contexts_default()?;
                ctx.devices
                    .take()
                    .ok_or(device_error(device_id, "Failed to initialize context"))?
            }
        };
        let mut lease = MapLease {
            cell: &ctx.devices,
            map: Some(map),
        };
        f(lease
            .map
            .as_mut()
            .expect("leased map is present until drop"))
    })
}

/// Ensures `device_id` has a context in `map`, creating one with the default
/// policy on first use.
fn ensure_device_context(
    map: &mut HashMap<usize, AsyncDeviceContext>,
    device_id: usize,
) -> Result<(), DeviceError> {
    if !map.contains_key(&device_id) {
        init_with_default_policy(map, device_id)?;
    }
    Ok(())
}

pub fn with_global_device_context<F, R>(device_id: usize, f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&AsyncDeviceContext) -> R,
{
    with_context_map(device_id, |map| {
        ensure_device_context(map, device_id)?;
        let device_context = map
            .get(&device_id)
            .ok_or(device_error(device_id, "Failed to get context"))?;
        Ok(f(device_context))
    })
}

pub fn with_global_device_context_mut<F, R>(device_id: usize, f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&mut AsyncDeviceContext) -> R,
{
    with_context_map(device_id, |map| {
        ensure_device_context(map, device_id)?;
        let device_context = map
            .get_mut(&device_id)
            .ok_or(device_error(device_id, "Failed to get context"))?;
        Ok(f(device_context))
    })
}

/// Run a closure with a reference to the scheduling policy for `device_id`.
pub fn with_device_policy<F, R>(device_id: usize, f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&Arc<dyn SchedulingPolicy>) -> R,
{
    with_global_device_context(device_id, |device_context| f(&device_context.policy))
}

/// Get a cloned `Arc` of the scheduling policy for `device_id`.
///
/// Useful when you need to schedule operations on a specific device outside the
/// default `.await` / `.sync()` path.
pub fn global_policy(device_id: usize) -> Result<Arc<dyn SchedulingPolicy>, DeviceError> {
    with_global_device_context(device_id, |device_context| device_context.policy.clone())
}

pub unsafe fn with_deallocator_stream<F, R>(device_id: usize, f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&Arc<Stream>) -> R,
{
    with_global_device_context(device_id, |device_context| {
        f(&device_context.deallocator_stream)
    })
}

/// Run a closure with a reference to the [`Device`] for `device_id`.
pub fn with_device<F, R>(device_id: usize, f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&Arc<Device>) -> R,
{
    let device = get_or_init_device(device_id)?;
    Ok(f(&device))
}

// Default device policy.

/// Change the default GPU device for the current thread.
///
/// All subsequent `.sync()`, `.await`, and `with_default_device_policy` calls on this
/// thread will target `default_device_id`. The context for that device is lazily created
/// with the default round-robin policy if it doesn't already exist.
///
/// # Multi-GPU Example
///
/// ```rust,ignore
/// // Thread dedicated to device 1:
/// set_default_device(1);
/// let tensor = api::zeros(&[1024, 1024]).await; // runs on GPU 1
/// ```
pub fn set_default_device(default_device_id: usize) {
    DEVICE_CONTEXTS.with(|ctx| {
        ctx.default_device.set(default_device_id);
    })
}

/// Set a custom memory pool for the given device **on the current thread**.
///
/// Subsequent allocations on this device will use the given pool instead of the
/// default pool. The pool is resolved at scheduling time (`.sync()`, `.await`,
/// `.schedule()`, `.sync_on()`, `.async_on()`) and carried on
/// [`ExecutionContext`](crate::device_operation::ExecutionContext), so it also
/// applies to futures that are later polled on other threads.
///
/// # Thread-locality
///
/// `AsyncDeviceContext` — and therefore the pool registration — lives in a
/// `thread_local!`. Calling `set_device_pool(0, pool)` on thread A does **not**
/// affect allocations scheduled by thread B on device 0.
///
/// If you build a `DeviceFuture` on one thread and move it to another, the pool
/// travels with the future via its `ExecutionContext` snapshot — the destination
/// thread does not need its own `set_device_pool` call. But if thread B
/// independently creates ops via `.sync()`/`.await`, those ops see thread B's
/// pool (typically `None` unless B also called `set_device_pool`).
///
/// Multi-threaded workers that want a shared pool should each call
/// `set_device_pool` during their initialization.
///
/// # Errors
///
/// Returns [`DeviceError::Context`](crate::error::DeviceError::Context) if
/// `pool` was created on a different device than `device_id`.
pub fn set_device_pool(device_id: usize, pool: Arc<MemPool>) -> Result<(), DeviceError> {
    let pool_device = pool.device().ordinal();
    device_assert(
        device_id,
        pool_device == device_id,
        &format!("pool belongs to device {pool_device}, expected device {device_id}"),
    )?;
    with_global_device_context_mut(device_id, |device_context| {
        device_context.pool = Some(pool);
    })
}

/// Clear the custom memory pool for the given device **on the current thread**,
/// reverting to the default pool.
///
/// Only affects the calling thread's pool registration; see
/// [`set_device_pool`] for the full thread-locality contract. In-flight
/// `DeviceFuture`s that already captured the pool are unaffected (the pool is
/// kept alive via `Arc` until those futures complete).
pub fn clear_device_pool(device_id: usize) -> Result<(), DeviceError> {
    with_global_device_context_mut(device_id, |device_context| {
        device_context.pool = None;
    })
}

/// Returns the custom memory pool registered for the given device **on the
/// current thread**, if any.
///
/// Returns `Ok(None)` when the calling thread has not registered a pool, even
/// if another thread has done so. See [`set_device_pool`] for thread-locality.
pub fn get_device_pool(device_id: usize) -> Result<Option<Arc<MemPool>>, DeviceError> {
    with_global_device_context(device_id, |device_context| device_context.pool.clone())
}

/// Resolve the custom memory pool associated with the device that owns `stream`.
///
/// Errors from the device-context lookup are downgraded to `None`; this is the
/// single choke-point for that decision so callers don't each re-derive it.
pub fn pool_for_stream(stream: &Arc<Stream>) -> Option<Arc<MemPool>> {
    get_device_pool(stream.device().ordinal()).ok().flatten()
}

/// Run a closure with the scheduling policy of the current thread's default device.
///
/// This is the function called internally by
/// [`DeviceOp::sync()`](crate::device_operation::DeviceOp::sync) and by the
/// [`IntoFuture`](std::future::IntoFuture) implementation to schedule operations
/// when no explicit device is given.
pub fn with_default_device_policy<F, R>(f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&Arc<dyn SchedulingPolicy>) -> R,
{
    let default_device = get_default_device();
    with_global_device_context(default_device, |device_context| f(&device_context.policy))
}

// Kernel operations — compile, cache, and retrieve GPU kernels.

/// Load a compiled CUDA module from a `.cubin` file.
pub fn load_module_from_file(filename: &str, device_id: usize) -> Result<Arc<Module>, DeviceError> {
    with_device(device_id, |device| {
        let module = device.load_module_from_file(filename)?;
        Ok(module)
    })?
}

/// Load a compiled CUDA module from an in-memory **cubin** image. Not for PTX
/// (see [`cuda_core::Device::load_module_from_bytes`]); use
/// [`load_module_from_ptx`] for that.
///
/// # Safety
///
/// Same contract as [`cuda_core::Device::load_module_from_bytes`]: `image`
/// must be a complete, well-formed cubin. The driver dereferences the offsets
/// declared in the image header, so a truncated or malformed image is read
/// past the end of the slice.
pub unsafe fn load_module_from_bytes(
    image: &[u8],
    device_id: usize,
) -> Result<Arc<Module>, DeviceError> {
    with_device(device_id, |device| {
        // SAFETY: forwarded verbatim from this function's own contract.
        let module = unsafe { device.load_module_from_bytes(image) }?;
        Ok(module)
    })?
}

/// JIT-compile a PTX string into a CUDA module for the given device.
pub fn load_module_from_ptx(ptx_src: &str, device_id: usize) -> Result<Arc<Module>, DeviceError> {
    with_device(device_id, |device| {
        let module = device.load_module_from_ptx_src(ptx_src)?;
        Ok(module)
    })?
}

#[cfg(test)]
mod context_map_tests {
    //! Host-only tests for the thread-local context map's lifecycle. They only
    //! need the map itself (no device is ever created successfully), so they
    //! run on CPU-only CI; the GPU-backed variants live in
    //! `tests/pool_allocation.rs`.

    use super::*;

    /// Whether this thread's context map exists, without consuming it.
    fn context_map_is_initialized() -> bool {
        DEVICE_CONTEXTS.with(|ctx| {
            let map = ctx.devices.take();
            let initialized = map.is_some();
            ctx.devices.set(map);
            initialized
        })
    }

    fn on_fresh_thread<F: FnOnce() + Send + 'static>(f: F) {
        std::thread::spawn(f).join().expect("test thread panicked");
    }

    /// A second `init_device_contexts` is rejected *and* the first map stays
    /// alive. The old `replace(None).is_none()` check consumed the live map
    /// while reporting the error.
    #[test]
    fn rejected_reinit_preserves_context_map() {
        on_fresh_thread(|| {
            init_device_contexts(0, 1).expect("first init");
            assert!(context_map_is_initialized());
            let err = init_device_contexts(0, 1).unwrap_err();
            assert!(matches!(err, DeviceError::Context { .. }), "got {err:?}");
            assert!(
                context_map_is_initialized(),
                "rejected re-init discarded the live context map"
            );
        });
    }

    /// A lookup that fails after the map was leased out of the `Cell` (device
    /// 9999 cannot be created) must put the map back instead of dropping it.
    #[test]
    fn failed_lookup_preserves_context_map() {
        on_fresh_thread(|| {
            init_device_contexts(0, 1).expect("init");
            assert!(with_global_device_context(9999, |_| ()).is_err());
            assert!(
                context_map_is_initialized(),
                "failed lookup discarded the live context map"
            );
            assert!(with_global_device_context_mut(9999, |_| ()).is_err());
            assert!(context_map_is_initialized());
        });
    }
}
