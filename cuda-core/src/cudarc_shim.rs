/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Portions of this file are copyright per https://github.com/chelsea0x3b/cudarc
 */

//\! Low-level CUDA driver API wrappers.
//\!
//\! Thin `pub(crate)` modules around `cuda_bindings` calls. The public API
//\! lives in [`crate::runtime`].
//\!
//\! CUDA driver flag bindings have platform-dependent integer types, so FFI
//\! calls cast them as `_`.

use crate::error::*;

/// Low-level primary context retain/release operations.
#[allow(dead_code)]
pub(crate) mod primary_ctx {

    use super::{DriverError, IntoResult};
    use std::mem::MaybeUninit;

    /// Retains the primary context for the given device, incrementing its reference count.
    ///
    /// # Safety
    /// `dev` must be a valid CUDA device handle.
    pub unsafe fn retain(
        dev: cuda_bindings::CUdevice,
    ) -> Result<cuda_bindings::CUcontext, DriverError> {
        let mut ctx = MaybeUninit::uninit();
        cuda_bindings::cuDevicePrimaryCtxRetain(ctx.as_mut_ptr(), dev).result()?;
        Ok(ctx.assume_init())
    }

    /// Releases the primary context for the given device.
    ///
    /// # Safety
    /// Must be paired with a prior `retain` call.
    pub unsafe fn release(dev: cuda_bindings::CUdevice) -> Result<(), DriverError> {
        cuda_bindings::cuDevicePrimaryCtxRelease_v2(dev).result()
    }
}

/// Low-level device query operations.
#[allow(dead_code)]
pub(crate) mod device {

    use super::{DriverError, IntoResult};
    use std::{
        ffi::{c_int, CStr},
        mem::MaybeUninit,
        string::String,
    };

    /// Returns the device handle for the given ordinal.
    pub fn get(ordinal: c_int) -> Result<cuda_bindings::CUdevice, DriverError> {
        let mut dev = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuDeviceGet(dev.as_mut_ptr(), ordinal).result()?;
            Ok(dev.assume_init())
        }
    }

    /// Returns the number of CUDA-capable devices.
    pub fn get_count() -> Result<c_int, DriverError> {
        let mut count = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuDeviceGetCount(count.as_mut_ptr()).result()?;
            Ok(count.assume_init())
        }
    }

    /// Returns the total memory in bytes on the device.
    ///
    /// # Safety
    /// `dev` must be a valid device handle.
    pub unsafe fn total_mem(dev: cuda_bindings::CUdevice) -> Result<usize, DriverError> {
        let mut bytes = MaybeUninit::uninit();
        cuda_bindings::cuDeviceTotalMem_v2(bytes.as_mut_ptr(), dev).result()?;
        Ok(bytes.assume_init())
    }

    /// Queries a device attribute value.
    ///
    /// # Safety
    /// `dev` must be a valid device handle.
    pub unsafe fn get_attribute(
        dev: cuda_bindings::CUdevice,
        attrib: cuda_bindings::CUdevice_attribute,
    ) -> Result<i32, DriverError> {
        let mut value = MaybeUninit::uninit();
        cuda_bindings::cuDeviceGetAttribute(value.as_mut_ptr(), attrib, dev).result()?;
        Ok(value.assume_init())
    }

    /// Returns the device name as a string.
    pub fn get_name(dev: cuda_bindings::CUdevice) -> Result<String, DriverError> {
        const BUF_SIZE: usize = 128;
        let mut buf = [0u8; BUF_SIZE];
        unsafe {
            cuda_bindings::cuDeviceGetName(buf.as_mut_ptr() as _, BUF_SIZE as _, dev).result()?;
        }
        let name = CStr::from_bytes_until_nul(&buf).expect("No null byte was present");
        Ok(String::from_utf8_lossy(name.to_bytes()).into())
    }

    /// Returns the UUID of the device.
    pub fn get_uuid(dev: cuda_bindings::CUdevice) -> Result<cuda_bindings::CUuuid, DriverError> {
        let id: cuda_bindings::CUuuid;
        unsafe {
            let mut uuid = MaybeUninit::uninit();
            cuda_bindings::cuDeviceGetUuid_v2(uuid.as_mut_ptr(), dev).result()?;
            id = uuid.assume_init();
        }
        Ok(id)
    }
}

/// Low-level function attribute operations.
#[allow(dead_code)]
pub(crate) mod function {

    use super::{DriverError, IntoResult};

    /// Sets a function attribute value.
    ///
    /// # Safety
    /// `f` must be a valid function handle.
    pub unsafe fn set_function_attribute(
        f: cuda_bindings::CUfunction,
        attribute: cuda_bindings::CUfunction_attribute_enum,
        value: i32,
    ) -> Result<(), DriverError> {
        unsafe {
            cuda_bindings::cuFuncSetAttribute(f, attribute, value).result()?;
        }
        Ok(())
    }

    /// Sets the preferred cache configuration for a function.
    ///
    /// # Safety
    /// `f` must be a valid function handle.
    pub unsafe fn set_function_cache_config(
        f: cuda_bindings::CUfunction,
        attribute: cuda_bindings::CUfunc_cache_enum,
    ) -> Result<(), DriverError> {
        unsafe {
            cuda_bindings::cuFuncSetCacheConfig(f, attribute).result()?;
        }
        Ok(())
    }
}

/// Low-level CUDA context management operations.
#[allow(dead_code)]
pub(crate) mod ctx {
    use super::{DriverError, IntoResult};
    use std::mem::MaybeUninit;

    /// Sets the current CUDA context for the calling thread.
    ///
    /// # Safety
    /// `ctx` must be a valid context handle.
    pub unsafe fn set_current(ctx: cuda_bindings::CUcontext) -> Result<(), DriverError> {
        cuda_bindings::cuCtxSetCurrent(ctx).result()
    }

    /// Returns the CUDA context bound to the calling thread, or `None`.
    pub fn get_current() -> Result<Option<cuda_bindings::CUcontext>, DriverError> {
        let mut ctx = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuCtxGetCurrent(ctx.as_mut_ptr()).result()?;
            let ctx: cuda_bindings::CUcontext = ctx.assume_init();
            if ctx.is_null() {
                Ok(None)
            } else {
                Ok(Some(ctx))
            }
        }
    }

    /// Sets flags on the current context.
    pub fn set_flags(flags: cuda_bindings::CUctx_flags) -> Result<(), DriverError> {
        unsafe { cuda_bindings::cuCtxSetFlags(flags as _).result() }
    }

    /// Blocks until all work in the current context is complete.
    pub fn synchronize() -> Result<(), DriverError> {
        unsafe { cuda_bindings::cuCtxSynchronize() }.result()
    }
}

/// Low-level CUDA stream operations.
#[allow(dead_code)]
pub(crate) mod stream {
    use super::{DriverError, IntoResult};
    use std::ffi::c_void;
    use std::mem::MaybeUninit;

    /// The kind of CUDA stream to create.
    pub enum StreamKind {
        /// > Default stream creation flag.
        Default,

        /// > Specifies that work running in the created stream
        /// > may run concurrently with work in stream 0 (the NULL stream),
        /// > and that the created stream should perform no implicit
        /// > synchronization with stream 0.
        NonBlocking,
    }

    impl StreamKind {
        fn flags(self) -> cuda_bindings::CUstream_flags {
            match self {
                Self::Default => cuda_bindings::CUstream_flags_enum_CU_STREAM_DEFAULT,
                Self::NonBlocking => cuda_bindings::CUstream_flags_enum_CU_STREAM_NON_BLOCKING,
            }
        }
    }

    /// Returns the null (default) stream handle.
    pub fn null() -> cuda_bindings::CUstream {
        std::ptr::null_mut()
    }

    /// Creates a new CUDA stream of the given kind.
    pub fn create(kind: StreamKind) -> Result<cuda_bindings::CUstream, DriverError> {
        let mut stream = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuStreamCreate(stream.as_mut_ptr(), kind.flags() as _).result()?;
            Ok(stream.assume_init())
        }
    }

    /// Blocks until all work on the stream is complete.
    ///
    /// # Safety
    /// `stream` must be a valid stream handle.
    pub unsafe fn synchronize(stream: cuda_bindings::CUstream) -> Result<(), DriverError> {
        cuda_bindings::cuStreamSynchronize(stream).result()
    }

    /// Queries stream completion without blocking: `Ok(true)` when all prior
    /// work has completed, `Ok(false)` when work is still in flight
    /// (`CUDA_ERROR_NOT_READY`).
    ///
    /// # Safety
    /// `stream` must be a valid stream handle.
    pub unsafe fn query(stream: cuda_bindings::CUstream) -> Result<bool, DriverError> {
        match cuda_bindings::cuStreamQuery(stream) {
            cuda_bindings::cudaError_enum_CUDA_SUCCESS => Ok(true),
            cuda_bindings::cudaError_enum_CUDA_ERROR_NOT_READY => Ok(false),
            code => {
                code.result()?;
                unreachable!("non-error CUresult handled above")
            }
        }
    }

    /// Destroys a CUDA stream.
    ///
    /// # Safety
    /// `stream` must be valid and not in use.
    pub unsafe fn destroy(stream: cuda_bindings::CUstream) -> Result<(), DriverError> {
        cuda_bindings::cuStreamDestroy_v2(stream).result()
    }

    /// Makes a stream wait on an event.
    ///
    /// # Safety
    /// Both handles must be valid.
    pub unsafe fn wait_event(
        stream: cuda_bindings::CUstream,
        event: cuda_bindings::CUevent,
        flags: cuda_bindings::CUevent_wait_flags,
    ) -> Result<(), DriverError> {
        cuda_bindings::cuStreamWaitEvent(stream, event, flags as _).result()
    }

    /// Attaches memory to a stream for managed memory visibility.
    ///
    /// # Safety
    /// `dptr` must be a valid managed memory pointer.
    pub unsafe fn attach_mem_async(
        stream: cuda_bindings::CUstream,
        dptr: cuda_bindings::CUdeviceptr,
        num_bytes: usize,
        flags: cuda_bindings::CUmemAttach_flags,
    ) -> Result<(), DriverError> {
        cuda_bindings::cuStreamAttachMemAsync(stream, dptr, num_bytes, flags as _).result()
    }

    /// Enqueues a host function callback on the stream.
    ///
    /// # Safety
    /// `func` and `arg` must remain valid until the callback executes.
    pub unsafe fn launch_host_function(
        stream: cuda_bindings::CUstream,
        func: unsafe extern "C" fn(*mut ::core::ffi::c_void),
        arg: *mut c_void,
    ) -> Result<(), DriverError> {
        cuda_bindings::cuLaunchHostFunc(stream, Some(func), arg).result()
    }

    /// Enqueues a host function on the stream with an explicit sync mode.
    ///
    /// `sync_mode` is `CU_HOST_TASK_BLOCKING` or `CU_HOST_TASK_SPINWAIT`;
    /// spin-wait keeps the driver's host-task thread hot for lower callback
    /// latency at the cost of a busy core.
    ///
    /// # Safety
    /// `func` and `arg` must remain valid until the callback executes.
    pub unsafe fn launch_host_function_v2(
        stream: cuda_bindings::CUstream,
        func: unsafe extern "C" fn(*mut ::core::ffi::c_void),
        arg: *mut c_void,
        sync_mode: ::core::ffi::c_uint,
    ) -> Result<(), DriverError> {
        cuda_bindings::cu_launch_host_func(stream, Some(func), arg, sync_mode).result()
    }

    /// Begins stream capture for graph construction.
    ///
    /// # Safety
    /// `stream` must be valid and not already capturing.
    pub unsafe fn begin_capture(
        stream: cuda_bindings::CUstream,
        mode: cuda_bindings::CUstreamCaptureMode,
    ) -> Result<(), DriverError> {
        cuda_bindings::cuStreamBeginCapture_v2(stream, mode).result()
    }

    /// Ends stream capture and returns the captured graph.
    ///
    /// # Safety
    /// `stream` must be in a capturing state.
    pub unsafe fn end_capture(
        stream: cuda_bindings::CUstream,
    ) -> Result<cuda_bindings::CUgraph, DriverError> {
        let mut graph = MaybeUninit::uninit();
        cuda_bindings::cuStreamEndCapture(stream, graph.as_mut_ptr()).result()?;
        Ok(graph.assume_init())
    }

    /// Queries whether the stream is currently capturing.
    ///
    /// # Safety
    /// `stream` must be a valid stream handle.
    pub unsafe fn is_capturing(
        stream: cuda_bindings::CUstream,
    ) -> Result<cuda_bindings::CUstreamCaptureStatus, DriverError> {
        let mut status = MaybeUninit::uninit();
        cuda_bindings::cuStreamIsCapturing(stream, status.as_mut_ptr()).result()?;
        Ok(status.assume_init())
    }
}

/// Low-level CUDA module load/unload and function lookup operations.
#[allow(dead_code)]
pub(crate) mod module {
    use super::{DriverError, IntoResult};
    use core::ffi::c_void;
    use std::ffi::CString;
    use std::mem::MaybeUninit;

    /// Loads a CUDA module from a file path.
    pub fn load(filename: &str) -> Result<cuda_bindings::CUmodule, DriverError> {
        let c_str = CString::new(filename).unwrap();
        let fname_ptr = c_str.as_c_str().as_ptr();
        let mut module = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuModuleLoad(module.as_mut_ptr(), fname_ptr).result()?;
            Ok(module.assume_init())
        }
    }

    /// Loads a CUDA module from a PTX source string.
    ///
    /// # Safety
    /// The PTX source must be valid.
    pub unsafe fn load_ptx_str(src_str: &str) -> Result<cuda_bindings::CUmodule, DriverError> {
        let mut module = MaybeUninit::uninit();
        let c_str = CString::new(src_str).unwrap();
        let module_res =
            cuda_bindings::cuModuleLoadData(module.as_mut_ptr(), c_str.as_ptr() as *const _);
        (module_res, module).result()
    }

    /// Loads a CUDA module from a raw data image pointer.
    ///
    /// # Safety
    /// `image` must point to valid module data (PTX or cubin).
    pub unsafe fn load_data(image: *const c_void) -> Result<cuda_bindings::CUmodule, DriverError> {
        let mut module = MaybeUninit::uninit();
        cuda_bindings::cuModuleLoadData(module.as_mut_ptr(), image).result()?;
        Ok(module.assume_init())
    }

    /// Looks up a device function by name within a module.
    ///
    /// # Safety
    /// `module` must be a valid, loaded module handle.
    pub unsafe fn get_function(
        module: cuda_bindings::CUmodule,
        name: &str,
    ) -> Result<cuda_bindings::CUfunction, DriverError> {
        let name = CString::new(name).unwrap();
        let name_ptr = name.as_c_str().as_ptr();
        let mut func = MaybeUninit::uninit();
        let res = cuda_bindings::cuModuleGetFunction(func.as_mut_ptr(), module, name_ptr);
        (res, func).result()
    }

    /// Unloads a CUDA module.
    ///
    /// # Safety
    /// `module` must be valid and all functions from it must no longer be in use.
    pub unsafe fn unload(module: cuda_bindings::CUmodule) -> Result<(), DriverError> {
        cuda_bindings::cuModuleUnload(module).result()
    }
}

/// Low-level CUDA event operations.
#[allow(dead_code)]
pub(crate) mod event {
    use super::{DriverError, IntoResult};
    use std::mem::MaybeUninit;

    /// Creates a new CUDA event with the given flags.
    pub fn create(
        flags: cuda_bindings::CUevent_flags,
    ) -> Result<cuda_bindings::CUevent, DriverError> {
        let mut event = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuEventCreate(event.as_mut_ptr(), flags as _).result()?;
            Ok(event.assume_init())
        }
    }

    /// Records an event on a stream.
    ///
    /// # Safety
    /// Both `event` and `stream` must be valid handles.
    pub unsafe fn record(
        event: cuda_bindings::CUevent,
        stream: cuda_bindings::CUstream,
    ) -> Result<(), DriverError> {
        unsafe { cuda_bindings::cuEventRecord(event, stream).result() }
    }

    /// Returns elapsed time in milliseconds between two recorded events.
    ///
    /// # Safety
    /// Both events must have been recorded and completed.
    pub unsafe fn elapsed(
        start: cuda_bindings::CUevent,
        end: cuda_bindings::CUevent,
    ) -> Result<f32, DriverError> {
        let mut ms: f32 = 0.0;
        unsafe {
            cuda_bindings::cuEventElapsedTime_v2((&mut ms) as *mut _, start, end).result()?;
        }
        Ok(ms)
    }

    /// Queries whether an event has completed. Returns `Ok` if complete.
    ///
    /// # Safety
    /// `event` must be a valid event handle.
    pub unsafe fn query(event: cuda_bindings::CUevent) -> Result<(), DriverError> {
        unsafe { cuda_bindings::cuEventQuery(event).result() }
    }

    /// Blocks until the event has been recorded.
    ///
    /// # Safety
    /// `event` must be a valid event handle.
    pub unsafe fn synchronize(event: cuda_bindings::CUevent) -> Result<(), DriverError> {
        unsafe { cuda_bindings::cuEventSynchronize(event).result() }
    }

    /// Destroys a CUDA event.
    ///
    /// # Safety
    /// `event` must be valid and not in use by any stream.
    pub unsafe fn destroy(event: cuda_bindings::CUevent) -> Result<(), DriverError> {
        cuda_bindings::cuEventDestroy_v2(event).result()
    }
}

/// Low-level CUDA memory pool operations.
pub(crate) mod pool {
    use super::{DriverError, IntoResult};
    use std::mem::MaybeUninit;

    /// Creates a new memory pool with the given properties.
    ///
    /// # Safety
    /// `props` must describe a valid pool configuration for an available device.
    pub unsafe fn create(
        props: &cuda_bindings::CUmemPoolProps,
    ) -> Result<cuda_bindings::CUmemoryPool, DriverError> {
        let mut pool = MaybeUninit::uninit();
        cuda_bindings::cuMemPoolCreate(pool.as_mut_ptr(), props as *const _ as *mut _).result()?;
        Ok(pool.assume_init())
    }

    /// Destroys a memory pool.
    ///
    /// # Safety
    /// `pool` must be valid and all allocations from it must have been freed.
    pub unsafe fn destroy(pool: cuda_bindings::CUmemoryPool) -> Result<(), DriverError> {
        cuda_bindings::cuMemPoolDestroy(pool).result()
    }

    /// Returns the default memory pool for the given device.
    ///
    /// # Safety
    /// `device` must be a valid device ordinal.
    pub unsafe fn get_default(
        device: cuda_bindings::CUdevice,
    ) -> Result<cuda_bindings::CUmemoryPool, DriverError> {
        let mut pool = MaybeUninit::uninit();
        cuda_bindings::cuDeviceGetDefaultMemPool(pool.as_mut_ptr(), device).result()?;
        Ok(pool.assume_init())
    }

    /// Sets the release threshold for a memory pool.
    ///
    /// # Safety
    /// `pool` must be a valid pool handle.
    pub unsafe fn set_release_threshold(
        pool: cuda_bindings::CUmemoryPool,
        threshold: u64,
    ) -> Result<(), DriverError> {
        cuda_bindings::cuMemPoolSetAttribute(
            pool,
            cuda_bindings::CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            &threshold as *const _ as *mut _,
        )
        .result()
    }

    /// Reads a `u64`-valued pool attribute.
    ///
    /// # Safety
    /// `pool` must be a valid handle. `attr` must be one whose value type is
    /// `cuuint64_t` — the four `*_MEM_CURRENT` / `*_MEM_HIGH` accounting
    /// attributes and `RELEASE_THRESHOLD`.
    pub unsafe fn get_attribute_u64(
        pool: cuda_bindings::CUmemoryPool,
        attr: cuda_bindings::CUmemPool_attribute,
    ) -> Result<u64, DriverError> {
        let mut value: u64 = 0;
        cuda_bindings::cuMemPoolGetAttribute(pool, attr, &mut value as *mut _ as *mut _)
            .result()?;
        Ok(value)
    }

    /// Resets a high-watermark attribute by writing 0. The driver rejects any
    /// other value for these attributes.
    ///
    /// # Safety
    /// `pool` must be valid. `attr` must be `USED_MEM_HIGH` or
    /// `RESERVED_MEM_HIGH`.
    pub unsafe fn reset_high_watermark(
        pool: cuda_bindings::CUmemoryPool,
        attr: cuda_bindings::CUmemPool_attribute,
    ) -> Result<(), DriverError> {
        let zero: u64 = 0;
        cuda_bindings::cuMemPoolSetAttribute(pool, attr, &zero as *const _ as *mut _).result()
    }

    /// Allocates device memory from a specific pool asynchronously.
    ///
    /// # Safety
    /// `pool` and `stream` must be valid handles.
    pub unsafe fn malloc_from_pool_async(
        pool: cuda_bindings::CUmemoryPool,
        stream: cuda_bindings::CUstream,
        num_bytes: usize,
    ) -> Result<cuda_bindings::CUdeviceptr, DriverError> {
        let mut dev_ptr = MaybeUninit::uninit();
        cuda_bindings::cuMemAllocFromPoolAsync(dev_ptr.as_mut_ptr(), num_bytes, pool, stream)
            .result()?;
        Ok(dev_ptr.assume_init())
    }
}

/// Low-level CUDA memory allocation, transfer, and management operations.
#[allow(dead_code)]
pub(crate) mod memory {

    use crate::sys::{self};
    use std::ffi::{c_uchar, c_uint, c_void};
    use std::mem::MaybeUninit;

    use crate::error::*;

    /// Allocates device memory asynchronously on the given stream.
    ///
    /// # Safety
    /// `stream` must be a valid stream handle.
    pub unsafe fn malloc_async(
        stream: sys::CUstream,
        num_bytes: usize,
    ) -> Result<sys::CUdeviceptr, DriverError> {
        let mut dev_ptr = MaybeUninit::uninit();
        sys::cuMemAllocAsync(dev_ptr.as_mut_ptr(), num_bytes, stream).result()?;
        Ok(dev_ptr.assume_init())
    }

    /// Allocates device memory synchronously.
    ///
    /// # Safety
    /// A valid CUDA context must be current.
    pub unsafe fn malloc_sync(num_bytes: usize) -> Result<sys::CUdeviceptr, DriverError> {
        let mut dev_ptr = MaybeUninit::uninit();
        sys::cuMemAlloc_v2(dev_ptr.as_mut_ptr(), num_bytes).result()?;
        Ok(dev_ptr.assume_init())
    }

    /// Allocates managed (unified) memory accessible from both host and device.
    ///
    /// # Safety
    /// A valid CUDA context must be current.
    pub unsafe fn malloc_managed(
        num_bytes: usize,
        flags: sys::CUmemAttach_flags,
    ) -> Result<sys::CUdeviceptr, DriverError> {
        let mut dev_ptr = MaybeUninit::uninit();
        sys::cuMemAllocManaged(dev_ptr.as_mut_ptr(), num_bytes, flags as _).result()?;
        Ok(dev_ptr.assume_init())
    }

    /// Allocates page-locked host memory.
    ///
    /// # Safety
    /// A valid CUDA context must be current.
    pub unsafe fn malloc_host(num_bytes: usize, flags: c_uint) -> Result<*mut c_void, DriverError> {
        let mut host_ptr = MaybeUninit::uninit();
        sys::cuMemHostAlloc(host_ptr.as_mut_ptr(), num_bytes, flags).result()?;
        Ok(host_ptr.assume_init())
    }

    /// Frees page-locked host memory allocated by `malloc_host`.
    ///
    /// # Safety
    /// `host_ptr` must have been allocated with `malloc_host`.
    pub unsafe fn free_host(host_ptr: *mut c_void) -> Result<(), DriverError> {
        sys::cuMemFreeHost(host_ptr).result()
    }

    /// Advises the CUDA runtime about the expected access pattern for managed memory.
    ///
    /// # Safety
    /// `dptr` must be a valid managed memory pointer.
    pub unsafe fn mem_advise(
        dptr: sys::CUdeviceptr,
        num_bytes: usize,
        advice: sys::CUmem_advise,
        location: sys::CUmemLocation,
    ) -> Result<(), DriverError> {
        sys::cuMemAdvise_v2(dptr, num_bytes, advice, location).result()
    }

    /// Asynchronously prefetches managed memory to the specified location.
    ///
    /// # Safety
    /// `dptr` must be valid managed memory; `stream` must be valid.
    pub unsafe fn mem_prefetch_async(
        dptr: sys::CUdeviceptr,
        num_bytes: usize,
        location: sys::CUmemLocation,
        stream: sys::CUstream,
    ) -> Result<(), DriverError> {
        sys::cuMemPrefetchAsync_v2(dptr, num_bytes, location, 0, stream).result()
    }

    /// Frees device memory asynchronously on the given stream.
    ///
    /// # Safety
    /// `dptr` must have been allocated with `malloc_async` and must not be used after this call.
    pub unsafe fn free_async(
        dptr: sys::CUdeviceptr,
        stream: sys::CUstream,
    ) -> Result<(), DriverError> {
        sys::cuMemFreeAsync(dptr, stream).result()
    }

    /// Frees device memory synchronously.
    ///
    /// # Safety
    /// `dptr` must be a valid device pointer not in use.
    pub unsafe fn free_sync(dptr: sys::CUdeviceptr) -> Result<(), DriverError> {
        sys::cuMemFree_v2(dptr).result()
    }

    /// Frees device memory synchronously (alias for `free_sync`).
    ///
    /// # Safety
    /// `device_ptr` must be a valid device pointer not in use.
    pub unsafe fn memory_free(device_ptr: sys::CUdeviceptr) -> Result<(), DriverError> {
        sys::cuMemFree_v2(device_ptr).result()
    }

    /// Asynchronously sets device memory to a byte value.
    ///
    /// # Safety
    /// `dptr` must be valid device memory with at least `num_bytes` capacity.
    pub unsafe fn memset_d8_async(
        dptr: sys::CUdeviceptr,
        uc: c_uchar,
        num_bytes: usize,
        stream: sys::CUstream,
    ) -> Result<(), DriverError> {
        sys::cuMemsetD8Async(dptr, uc, num_bytes, stream).result()
    }

    /// Synchronously sets device memory to a byte value.
    ///
    /// # Safety
    /// `dptr` must be valid device memory with at least `num_bytes` capacity.
    pub unsafe fn memset_d8_sync(
        dptr: sys::CUdeviceptr,
        uc: c_uchar,
        num_bytes: usize,
    ) -> Result<(), DriverError> {
        sys::cuMemsetD8_v2(dptr, uc, num_bytes).result()
    }

    /// Asynchronously copies bytes from host to device memory.
    ///
    /// # Safety
    /// `src` and `dst` must be valid with sufficient capacity; `stream` must be valid.
    pub unsafe fn memcpy_htod_async<T>(
        dst: sys::CUdeviceptr,
        src: *const T,
        num_bytes: usize,
        stream: sys::CUstream,
    ) -> Result<(), DriverError> {
        sys::cuMemcpyHtoDAsync_v2(dst, src as *const _, num_bytes, stream).result()
    }

    /// Synchronously copies a host slice to device memory.
    ///
    /// # Safety
    /// `dst` must have capacity for the full slice.
    pub unsafe fn memcpy_htod_sync<T>(dst: sys::CUdeviceptr, src: &[T]) -> Result<(), DriverError> {
        sys::cuMemcpyHtoD_v2(dst, src.as_ptr() as *const _, std::mem::size_of_val(src)).result()
    }

    /// Asynchronously copies bytes from device to host memory.
    ///
    /// # Safety
    /// `dst` and `src` must be valid with sufficient capacity; `stream` must be valid.
    pub unsafe fn memcpy_dtoh_async<T>(
        dst: *mut T,
        src: sys::CUdeviceptr,
        num_bytes: usize,
        stream: sys::CUstream,
    ) -> Result<(), DriverError> {
        sys::cuMemcpyDtoHAsync_v2(dst as *mut _, src, num_bytes, stream).result()
    }

    /// Synchronously copies device memory into a host slice.
    ///
    /// # Safety
    /// `src` must have at least as many bytes as `dst`.
    pub unsafe fn memcpy_dtoh_sync<T>(
        dst: &mut [T],
        src: sys::CUdeviceptr,
    ) -> Result<(), DriverError> {
        sys::cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut _, src, std::mem::size_of_val(dst)).result()
    }

    /// Asynchronously copies bytes between device memory regions.
    ///
    /// # Safety
    /// Both pointers must be valid with sufficient capacity; `stream` must be valid.
    pub unsafe fn memcpy_dtod_async(
        dst: sys::CUdeviceptr,
        src: sys::CUdeviceptr,
        num_bytes: usize,
        stream: sys::CUstream,
    ) -> Result<(), DriverError> {
        sys::cuMemcpyDtoDAsync_v2(dst, src, num_bytes, stream).result()
    }

    /// Synchronously copies bytes between device memory regions.
    ///
    /// # Safety
    /// Both pointers must be valid with sufficient capacity.
    pub unsafe fn memcpy_dtod_sync(
        dst: sys::CUdeviceptr,
        src: sys::CUdeviceptr,
        num_bytes: usize,
    ) -> Result<(), DriverError> {
        sys::cuMemcpyDtoD_v2(dst, src, num_bytes).result()
    }

    /// Returns `(free, total)` bytes of device memory for the current context.
    pub fn mem_get_info() -> Result<(usize, usize), DriverError> {
        let mut free = 0;
        let mut total = 0;
        unsafe { sys::cuMemGetInfo_v2(&mut free as *mut _, &mut total as *mut _) }.result()?;
        Ok((free, total))
    }
}
