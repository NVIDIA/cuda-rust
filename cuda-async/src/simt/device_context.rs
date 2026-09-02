/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Thread-local GPU device state, kernel cache, and scheduling policy management.
//!
//! Each thread maintains a set of [`AsyncDeviceContext`]s (one per device) via
//! the `DEVICE_CONTEXTS` thread-local. A context bundles:
//!
//! * A [`CudaContext`] for driver API calls.
//! * A [`GlobalSchedulingPolicy`] for stream selection.
//! * A dedicated deallocator stream for async `cuMemFreeAsync`.
//! * A kernel function cache keyed by [`FunctionKey`] hashes.
//!
//! Most users interact through the convenience functions
//! ([`with_default_device_policy`], [`with_cuda_context`], etc.) rather than
//! touching the thread-local directly.
//!
//! [`CudaContext`]: cuda_core::CudaContext

use crate::simt::error::{device_assert, device_error, DeviceError};
use crate::simt::scheduling_policies::{
    GlobalSchedulingPolicy, SchedulingPolicy, StreamPoolRoundRobin,
};
use cuda_core::{CudaContext, CudaFunction, CudaModule, CudaStream};
use rustc_hash::FxHashMap;
use std::cell::Cell;
use std::collections::hash_map::Entry;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

/// Default CUDA device ordinal used when no explicit device is specified.
pub const DEFAULT_DEVICE_ID: usize = 0;

/// Default number of devices initialized by `init_device_contexts_default`.
pub const DEFAULT_NUM_DEVICES: usize = 1;

/// Default number of streams in the [`StreamPoolRoundRobin`] policy.
pub const DEFAULT_ROUND_ROBIN_STREAM_POOL_SIZE: usize = 4;

/// Trait for types that uniquely identify a compiled kernel function.
///
/// The hash is used as the lookup key in the per-device function cache.
pub trait FunctionKey: Hash {
    /// Returns a hex-encoded hash string suitable for use as a cache key.
    fn get_hash_string(&self) -> String {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

/// Cached mapping from function hash keys to loaded `(module, function)` pairs.
type DeviceFunctions = FxHashMap<String, (Arc<CudaModule>, Arc<CudaFunction>)>;

/// Per-device state: CUDA context, scheduling policy, deallocator stream, and
/// compiled-kernel cache.
pub struct AsyncDeviceContext {
    /// Device ordinal.
    #[allow(dead_code)]
    device_id: usize,
    /// Primary CUDA context for this device.
    context: Arc<CudaContext>,
    /// Dedicated stream for async memory deallocation (see [`DeviceBox`]).
    ///
    /// [`DeviceBox`]: crate::simt::device_box::DeviceBox
    deallocator_stream: Arc<CudaStream>,
    /// Scheduling policy shared (via `Arc`) with all operations on this device.
    policy: Arc<GlobalSchedulingPolicy>,
    /// Cache of loaded kernel functions, keyed by [`FunctionKey`] hashes.
    functions: DeviceFunctions,
}

/// Container for all per-device contexts on a single thread.
///
/// Uses interior mutability ([`Cell`]) because it lives inside a
/// `thread_local!` and must be borrowed without `&mut`.
pub struct AsyncDeviceContexts {
    /// Currently selected default device ordinal.
    default_device: Cell<usize>,
    /// Lazily initialized map of device ordinal to context.
    devices: Cell<Option<FxHashMap<usize, AsyncDeviceContext>>>,
}

// Thread-local storage for per-device CUDA state.
// Lazily initialized on first access. Each thread gets its own independent
// set of device contexts, streams, and kernel caches.
thread_local!(static DEVICE_CONTEXTS: AsyncDeviceContexts = const {
    AsyncDeviceContexts {
        default_device: Cell::new(DEFAULT_DEVICE_ID),
        devices: Cell::new(None),
    }
});

/// Returns the default device ordinal for the current thread.
pub fn get_default_device() -> usize {
    DEVICE_CONTEXTS.with(|ctx| ctx.default_device.get())
}

/// Initializes the thread-local device context map with capacity for
/// `num_devices` devices and sets `default_device_id` as the default.
///
/// Must be called at most once per thread. Returns an error if already
/// initialized.
pub fn init_device_contexts(
    default_device_id: usize,
    num_devices: usize,
) -> Result<(), DeviceError> {
    DEVICE_CONTEXTS.with(|ctx| {
        let devices = ctx.devices.take();
        let is_uninitialized = devices.is_none();
        ctx.devices.set(devices);
        device_assert(
            default_device_id,
            is_uninitialized,
            "Context already initialized.",
        )
    })?;
    let devices = FxHashMap::with_capacity_and_hasher(num_devices, Default::default());
    DEVICE_CONTEXTS.with(|ctx| {
        ctx.default_device.set(default_device_id);
        ctx.devices.set(Some(devices));
    });
    Ok(())
}

/// Initializes the thread-local device contexts with default parameters.
fn init_device_contexts_default() -> Result<(), DeviceError> {
    let default_device = get_default_device();
    init_device_contexts(default_device, DEFAULT_NUM_DEVICES)
}

/// Creates a new [`AsyncDeviceContext`] for `device_id` with the given policy.
///
/// Initializes the CUDA context, the scheduling policy's stream pool, and a
/// dedicated deallocator stream.
pub fn new_device_context(
    device_id: usize,
    mut policy: GlobalSchedulingPolicy,
) -> Result<AsyncDeviceContext, DeviceError> {
    let context = CudaContext::new(device_id)?;
    policy.init(&context)?;
    let deallocator_stream = context.new_stream()?;
    Ok(AsyncDeviceContext {
        device_id,
        context,
        deallocator_stream,
        policy: Arc::new(policy),
        functions: FxHashMap::default(),
    })
}

/// Inserts a new device context into `hashmap`. Errors if the device is
/// already present.
fn init_device(
    hashmap: &mut FxHashMap<usize, AsyncDeviceContext>,
    device_id: usize,
    policy: GlobalSchedulingPolicy,
) -> Result<(), DeviceError> {
    let device_context = new_device_context(device_id, policy)?;
    let pred = hashmap.insert(device_id, device_context).is_none();
    device_assert(device_id, pred, "Device is already initialized.")
}

/// Initializes a device with the default round-robin policy.
fn init_with_default_policy(
    hashmap: &mut FxHashMap<usize, AsyncDeviceContext>,
    device_id: usize,
) -> Result<(), DeviceError> {
    let policy =
        unsafe { StreamPoolRoundRobin::new(device_id, DEFAULT_ROUND_ROBIN_STREAM_POOL_SIZE) };
    init_device(
        hashmap,
        device_id,
        GlobalSchedulingPolicy::RoundRobin(policy),
    )
}

/// Checks the thread's device map out of the thread-local, runs `f` on it,
/// and puts it back **whether or not `f` succeeded**.
///
/// The map lives in a `Cell<Option<..>>`, so it is moved out for the duration
/// of the call. An early `?` between the take and the put-back used to drop
/// the map on the floor, destroying every device context this thread had
/// initialized (contexts, stream pools, kernel cache) because one lookup of
/// an unrelated device failed; the next access then rebuilt everything from
/// scratch.
fn with_device_map<R>(
    device_id: usize,
    f: impl FnOnce(&mut FxHashMap<usize, AsyncDeviceContext>) -> Result<R, DeviceError>,
) -> Result<R, DeviceError> {
    DEVICE_CONTEXTS.with(|ctx| {
        let mut hashmap = match ctx.devices.take() {
            Some(hashmap) => hashmap,
            None => {
                // Nothing is checked out yet, so a failure here loses nothing.
                init_device_contexts_default()?;
                ctx.devices
                    .take()
                    .ok_or_else(|| device_error(device_id, "Failed to initialize context"))?
            }
        };
        let result = f(&mut hashmap);
        ctx.devices.replace(Some(hashmap));
        result
    })
}

/// Borrows the thread-local [`AsyncDeviceContext`] for `device_id` immutably.
///
/// Lazily initializes the context map and the specific device if needed.
fn with_global_device_context<F, R>(device_id: usize, f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&AsyncDeviceContext) -> R,
{
    with_device_map(device_id, |hashmap| {
        if !hashmap.contains_key(&device_id) {
            init_with_default_policy(hashmap, device_id)?;
        }
        let device_context = hashmap
            .get(&device_id)
            .ok_or_else(|| device_error(device_id, "Failed to get context"))?;
        Ok(f(device_context))
    })
}

/// Borrows the thread-local [`AsyncDeviceContext`] for `device_id` mutably.
///
/// Lazily initializes the context map and the specific device if needed.
fn with_global_device_context_mut<F, R>(device_id: usize, f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&mut AsyncDeviceContext) -> R,
{
    with_device_map(device_id, |hashmap| {
        if !hashmap.contains_key(&device_id) {
            init_with_default_policy(hashmap, device_id)?;
        }
        let device_context = hashmap
            .get_mut(&device_id)
            .ok_or_else(|| device_error(device_id, "Failed to get context"))?;
        Ok(f(device_context))
    })
}

/// Provides the default device's scheduling policy to `f`.
///
/// This is the primary entry point used by [`IntoFuture`] impls on
/// `DeviceOperation` combinators.
///
/// [`IntoFuture`]: std::future::IntoFuture
pub fn with_default_device_policy<F, R>(f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&Arc<GlobalSchedulingPolicy>) -> R,
{
    let default_device = get_default_device();
    with_global_device_context(default_device, |dc| f(&dc.policy))
}

/// Provides the deallocator stream for `device_id` to `f`.
///
/// # Safety
///
/// The stream must only be used for `cuMemFreeAsync` calls. Enqueuing other
/// work on this stream may interfere with async deallocation ordering.
pub unsafe fn with_deallocator_stream<F, R>(device_id: usize, f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&Arc<CudaStream>) -> R,
{
    with_global_device_context(device_id, |dc| f(&dc.deallocator_stream))
}

/// Provides the CUDA context for `device_id` to `f`.
pub fn with_cuda_context<F, R>(device_id: usize, f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&Arc<CudaContext>) -> R,
{
    with_global_device_context(device_id, |dc| f(&dc.context))
}

/// Sets the default device ordinal for the current thread.
pub fn set_default_device(default_device_id: usize) {
    DEVICE_CONTEXTS.with(|ctx| {
        ctx.default_device.set(default_device_id);
    })
}

/// Loads a CUDA module from a file (`.cubin`, `.fatbin`, or `.ptx`) on
/// `device_id`.
pub fn load_module_from_file(
    filename: &str,
    device_id: usize,
) -> Result<Arc<CudaModule>, DeviceError> {
    with_cuda_context(device_id, |cuda_ctx| {
        let module = cuda_ctx.load_module_from_file(filename)?;
        Ok(module)
    })?
}

/// Loads a CUDA module from PTX source text on `device_id`.
pub fn load_module_from_ptx(
    ptx_src: &str,
    device_id: usize,
) -> Result<Arc<CudaModule>, DeviceError> {
    with_cuda_context(device_id, |cuda_ctx| {
        let module = cuda_ctx.load_module_from_ptx_src(ptx_src)?;
        Ok(module)
    })?
}

/// Inserts a compiled function into the per-device kernel cache.
///
/// The function is keyed by the hash of `func_key`. Returns an error if a
/// function is already registered under the same key (a duplicate insert or
/// a hash collision); the existing entry is left in place rather than
/// replaced, so a caller that ignores the error still gets the function it
/// registered first from [`get_cuda_function`].
pub fn insert_cuda_function(
    device_id: usize,
    func_key: &impl FunctionKey,
    value: (Arc<CudaModule>, Arc<CudaFunction>),
) -> Result<(), DeviceError> {
    with_global_device_context_mut(device_id, |dc| {
        match dc.functions.entry(func_key.get_hash_string()) {
            Entry::Occupied(_) => Err(device_error(device_id, "Unexpected cache key collision.")),
            Entry::Vacant(slot) => {
                slot.insert(value);
                Ok(())
            }
        }
    })?
}

/// Retrieves a compiled [`CudaFunction`] from the per-device kernel cache.
///
/// Returns an error if no function has been registered under `func_key`.
pub fn get_cuda_function(
    device_id: usize,
    func_key: &impl FunctionKey,
) -> Result<Arc<CudaFunction>, DeviceError> {
    with_global_device_context(device_id, |dc| {
        let key = func_key.get_hash_string();
        let (_module, function) = dc
            .functions
            .get(&key)
            .ok_or_else(|| device_error(device_id, "Failed to get cuda function."))?;
        Ok(Arc::clone(function))
    })?
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_duplicate_init_error(result: Result<(), DeviceError>) {
        assert!(matches!(
            result,
            Err(DeviceError::Context {
                device_id: 0,
                message,
            }) if message == "Context already initialized."
        ));
    }

    #[test]
    fn duplicate_init_preserves_existing_device_contexts() {
        std::thread::spawn(|| {
            init_device_contexts(0, 1).expect("initial context initialization should succeed");

            assert_duplicate_init_error(init_device_contexts(0, 1));
            assert_duplicate_init_error(init_device_contexts(0, 1));
        })
        .join()
        .expect("test thread should not panic");
    }

    /// Runs `body` on a fresh thread (its own thread-local device map) if a
    /// GPU is available; otherwise reports a skip.
    fn with_gpu_thread(body: impl FnOnce() + Send + 'static) {
        if let Err(error) = CudaContext::new(0) {
            eprintln!("SKIPPED: CUDA unavailable ({error:?})");
            return;
        }
        std::thread::spawn(body)
            .join()
            .expect("test thread should not panic");
    }

    /// A failed lookup of one device must not destroy the contexts already
    /// initialized for other devices on this thread. Device 0's context is
    /// the same `Arc` before and after a request for a device that does not
    /// exist.
    #[test]
    fn failed_device_lookup_keeps_the_other_contexts() {
        with_gpu_thread(|| {
            let before = with_cuda_context(0, Arc::clone).expect("device 0 initializes");

            let bogus = with_cuda_context(usize::MAX, Arc::clone);
            assert!(bogus.is_err(), "a nonexistent device must fail: {bogus:?}");

            let after = with_cuda_context(0, Arc::clone).expect("device 0 is still there");
            assert!(
                Arc::ptr_eq(&before, &after),
                "device 0's context must survive the failed lookup"
            );
        });
    }

    /// A second insert under an existing key must fail *without* replacing
    /// the entry: the first function registered stays the one the cache
    /// returns.
    #[test]
    fn insert_cuda_function_keeps_the_existing_entry_on_collision() {
        const PTX: &str = r#"
.version 7.8
.target sm_75
.address_size 64
.visible .entry first() { ret; }
.visible .entry second() { ret; }
"#;

        #[derive(Hash)]
        struct Key;
        impl FunctionKey for Key {}

        with_gpu_thread(|| {
            let module = load_module_from_ptx(PTX, 0).expect("JIT the test PTX");
            let first = Arc::new(module.load_function("first").expect("first"));
            let second = Arc::new(module.load_function("second").expect("second"));

            insert_cuda_function(0, &Key, (Arc::clone(&module), Arc::clone(&first)))
                .expect("first insert");
            let collision = insert_cuda_function(0, &Key, (Arc::clone(&module), second));
            assert!(collision.is_err(), "duplicate key must be reported");

            let cached = get_cuda_function(0, &Key).expect("the key is still registered");
            assert!(
                Arc::ptr_eq(&cached, &first),
                "the colliding insert must not have replaced the first function"
            );
        });
    }

    /// `.sync()` must run the operation outside the thread-local borrow: an
    /// operation that itself touches the device context (here, reads device
    /// 0's `CudaContext`) must see the same context as the caller, not a
    /// freshly rebuilt one.
    #[test]
    fn sync_does_not_rebuild_the_device_map_when_reentered() {
        use crate::simt::device_operation::{empty, value, DeviceOperation};

        with_gpu_thread(|| {
            let outer = with_cuda_context(0, |ctx| Arc::as_ptr(ctx) as usize).expect("device 0");
            let inner = empty(|| {
                value(
                    with_cuda_context(0, |ctx| Arc::as_ptr(ctx) as usize)
                        .expect("device 0 from inside the operation"),
                )
            })
            .sync()
            .expect("nested operation runs");
            assert_eq!(
                outer, inner,
                "the operation must observe the caller's device context"
            );
        });
    }
}
