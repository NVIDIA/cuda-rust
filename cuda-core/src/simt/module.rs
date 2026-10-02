/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA module and function management (RAII, PTX/cubin loading).
//!
//! A [`CudaModule`] wraps a `CUmodule` loaded from PTX source or a cubin file.
//! [`CudaFunction`] extracts a kernel entry point from a loaded module by
//! symbol name. Both types are reference-counted and tie their lifetime to the
//! parent [`CudaContext`] / [`CudaModule`] respectively.
//!
//! # Typical workflow
//!
//! ```ignore
//! let ctx = CudaContext::new(0)?;
//! let module = ctx.load_module_from_ptx_src(ptx)?;
//! let kernel = module.load_function("my_kernel")?;
//! ```
//!
//! # Raw CUDA interop
//!
//! Most users should load kernels with [`CudaModule::load_function`] and
//! launch them through cuda-oxide's typed launch helpers. Some CUDA-adjacent
//! libraries need the underlying driver handle to inspect or register
//! module-scope device state before launch. For those cases,
//! [`CudaModule::cu_module`] exposes a non-owning raw `CUmodule` handle under
//! an explicit `unsafe` contract.

use crate::error::{DriverError, IntoResult};
use crate::simt::context::CudaContext;
use std::borrow::Cow;
use std::ffi::{c_void, CString};
use std::mem::MaybeUninit;
use std::sync::Arc;

/// An RAII wrapper around a `CUmodule` handle.
///
/// Holds an `Arc<CudaContext>` to ensure the context outlives the module.
/// Unloaded automatically via `cuModuleUnload` on [`Drop`].
#[derive(Debug)]
pub struct CudaModule {
    /// Raw CUDA module handle.
    pub(crate) cu_module: cuda_bindings::CUmodule,
    /// Owning context. Kept alive for the lifetime of this module.
    pub(crate) ctx: Arc<CudaContext>,
}

/// # Safety
///
/// `CUmodule` handles are not thread-local. The CUDA driver permits querying
/// functions from a module on any thread, provided the owning context is bound.
unsafe impl Send for CudaModule {}
/// See [`Send`] impl.
unsafe impl Sync for CudaModule {}

/// Unloads the module on drop.
///
/// Binds the context to the current thread first (required by
/// `cuModuleUnload`). Errors are recorded on the context rather than
/// panicking.
impl Drop for CudaModule {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        self.ctx
            .record_err(unsafe { cuda_bindings::cuModuleUnload(self.cu_module).result() });
    }
}

impl CudaContext {
    /// JIT-compiles PTX source and loads the resulting module into this
    /// context.
    ///
    /// `ptx_src` must be a valid, null-terminator-free PTX string. The driver
    /// performs JIT compilation targeting the current device architecture.
    ///
    /// # Panics
    ///
    /// Panics if `ptx_src` contains interior null bytes.
    pub fn load_module_from_ptx_src(
        self: &Arc<Self>,
        ptx_src: &str,
    ) -> Result<Arc<CudaModule>, DriverError> {
        self.bind_to_thread()?;
        let c_src = CString::new(ptx_src).unwrap();
        let cu_module = unsafe {
            let mut cu_module = MaybeUninit::uninit();
            cuda_bindings::cuModuleLoadData(cu_module.as_mut_ptr(), c_src.as_ptr() as *const _)
                .result()?;
            cu_module.assume_init()
        };
        Ok(Arc::new(CudaModule {
            cu_module,
            ctx: self.clone(),
        }))
    }

    /// Loads a CUDA module from an in-memory image.
    ///
    /// `image` may be PTX source bytes, a cubin, or a fatbin. PTX text is
    /// null-terminated before it is passed to the CUDA driver; binary module
    /// images tolerate the trailing byte because their own headers describe
    /// their size.
    pub fn load_module_from_image(
        self: &Arc<Self>,
        image: &[u8],
    ) -> Result<Arc<CudaModule>, DriverError> {
        self.bind_to_thread()?;
        let image = null_terminated_image(image);
        let cu_module = unsafe {
            let mut cu_module = MaybeUninit::uninit();
            cuda_bindings::cuModuleLoadData(cu_module.as_mut_ptr(), image.as_ptr() as *const _)
                .result()?;
            cu_module.assume_init()
        };
        Ok(Arc::new(CudaModule {
            cu_module,
            ctx: self.clone(),
        }))
    }

    /// Loads a module from a cubin or PTX file on disk.
    ///
    /// `filename` is the filesystem path. The driver selects the loader based
    /// on file contents (PTX text or cubin ELF).
    ///
    /// # Panics
    ///
    /// Panics if `filename` contains interior null bytes.
    pub fn load_module_from_file(
        self: &Arc<Self>,
        filename: &str,
    ) -> Result<Arc<CudaModule>, DriverError> {
        self.bind_to_thread()?;
        let c_str = CString::new(filename).unwrap();
        let mut cu_module = MaybeUninit::uninit();
        let cu_module = unsafe {
            cuda_bindings::cuModuleLoad(cu_module.as_mut_ptr(), c_str.as_ptr()).result()?;
            cu_module.assume_init()
        };
        Ok(Arc::new(CudaModule {
            cu_module,
            ctx: self.clone(),
        }))
    }
}

fn null_terminated_image(image: &[u8]) -> Cow<'_, [u8]> {
    if image.last() == Some(&0) {
        Cow::Borrowed(image)
    } else {
        let mut owned = Vec::with_capacity(image.len() + 1);
        owned.extend_from_slice(image);
        owned.push(0);
        Cow::Owned(owned)
    }
}

/// A handle to a device kernel entry point within a loaded [`CudaModule`].
///
/// Holds an `Arc<CudaModule>` so the module (and transitively the context)
/// remains loaded for the lifetime of this handle. Cloning is cheap (just an
/// `Arc` bump).
#[derive(Debug, Clone)]
pub struct CudaFunction {
    /// Raw CUDA function handle.
    pub(crate) cu_function: cuda_bindings::CUfunction,
    /// Owning module. Prevents unloading while this function handle exists.
    #[allow(unused)]
    pub(crate) module: Arc<CudaModule>,
}

/// # Safety
///
/// `CUfunction` handles are derived from a `CUmodule` and valid in any thread
/// that has the owning context bound.
unsafe impl Send for CudaFunction {}
/// See [`Send`] impl.
unsafe impl Sync for CudaFunction {}

impl CudaModule {
    /// Returns the parent [`CudaContext`].
    ///
    /// This is mainly useful when interoperating with raw CUDA driver APIs:
    /// call [`CudaContext::bind_to_thread`] on this context before passing raw
    /// module/function handles to APIs that require the owning context to be
    /// current on the calling host thread.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Returns the raw `CUmodule` handle owned by this wrapper.
    ///
    /// This is an escape hatch for CUDA driver interop libraries that need to
    /// inspect or register a loaded module directly. For example, NVSHMEM uses
    /// module-level state and may need the raw `CUmodule` before launching a
    /// kernel that calls into NVSHMEM device code.
    ///
    /// The returned handle is copied by value, but it is non-owning.
    /// cuda-oxide still owns the module and will unload it when the last
    /// [`Arc`] owning this module is dropped.
    ///
    /// # Safety
    ///
    /// - The returned handle is valid only while this [`CudaModule`] remains
    ///   alive. If the handle is stored outside the immediate call, the caller
    ///   must keep an [`Arc`] to this module alive for at least as long.
    /// - The caller must not unload the module through the raw handle, transfer
    ///   ownership of it, or pass it to any API that may invalidate module,
    ///   function, or global handles owned by cuda-oxide.
    /// - Before passing the handle to CUDA driver APIs or interop libraries
    ///   that make driver calls, the caller must ensure this module's owning
    ///   context is current on the calling host thread, for example with
    ///   [`CudaModule::context`] followed by
    ///   [`CudaContext::bind_to_thread`].
    /// - Any foreign library that retains this handle must obey the same
    ///   lifetime and context-current requirements. cuda-oxide cannot enforce
    ///   those requirements once the raw handle leaves Rust's type system.
    pub unsafe fn cu_module(&self) -> cuda_bindings::CUmodule {
        self.cu_module
    }

    /// Looks up a kernel entry point by `fn_name` in this module.
    ///
    /// The returned [`CudaFunction`] holds an `Arc` back to this module,
    /// preventing unloading while the handle is live.
    ///
    /// This method first binds the module's owning context to the calling
    /// thread, then performs `cuModuleGetFunction`. That makes it safe to look
    /// up functions from any host thread, provided the module and its context
    /// are still alive.
    ///
    /// # Errors
    ///
    /// Returns an error if binding the module's context fails or if
    /// `cuModuleGetFunction` cannot resolve `fn_name` in this module.
    ///
    /// # Panics
    ///
    /// Panics if `fn_name` contains interior null bytes.
    pub fn load_function(self: &Arc<Self>, fn_name: &str) -> Result<CudaFunction, DriverError> {
        self.ctx.bind_to_thread()?;
        let c_name = CString::new(fn_name).unwrap();
        let cu_function = unsafe {
            let mut cu_function = MaybeUninit::uninit();
            cuda_bindings::cuModuleGetFunction(
                cu_function.as_mut_ptr(),
                self.cu_module,
                c_name.as_ptr(),
            )
            .result()?;
            cu_function.assume_init()
        };
        Ok(CudaFunction {
            cu_function,
            module: self.clone(),
        })
    }
}

/// A resolved handle to a `#[constant]` device global. Macro-generated
/// `set_<name>` methods resolve these lazily on first use and cache the
/// handle on the `LoadedModule` struct. Callers pass `size_of::<T>()` on
/// every write; correctness depends on the resolver asserting that the
/// driver-reported size matches the host-side type.
#[derive(Clone, Copy, Debug)]
pub struct ConstantHandle {
    pub(crate) dptr: cuda_bindings::CUdeviceptr,
}

impl ConstantHandle {
    /// Construct from a raw device pointer. Used by macro-generated
    /// `LoadedModule` initializers after [`CudaModule::get_global`] has
    /// resolved the symbol and the size has been asserted against
    /// `size_of::<T>()`.
    ///
    /// # Safety
    ///
    /// `dptr` must point to at least `size_of::<T>()` bytes of constant
    /// memory in a still-loaded module.
    pub unsafe fn from_raw(dptr: cuda_bindings::CUdeviceptr) -> Self {
        Self { dptr }
    }
}

impl ConstantHandle {
    /// Stream-ordered `cuMemcpyHtoDAsync` from `src` (`num_bytes` of host
    /// memory) into the device global.
    ///
    /// # Safety
    ///
    /// - `src` must point to at least `num_bytes` of readable host memory.
    /// - The bytes must have a layout compatible with the device-side type.
    pub unsafe fn write_async(
        &self,
        stream: &crate::CudaStream,
        src: *const u8,
        num_bytes: usize,
    ) -> Result<(), DriverError> {
        stream.context().bind_to_thread()?;
        unsafe {
            crate::simt::memory::memcpy_htod_async(self.dptr, src, num_bytes, stream.cu_stream())
        }
    }

    /// Stream-ordered `cuMemcpyHtoDAsync` from owned host bytes into the
    /// device global.
    ///
    /// The bytes are kept alive until the stream reaches a host callback
    /// enqueued after the copy. This makes safe setters sound even when the
    /// caller passes a temporary such as `&3.0`.
    pub fn write_async_staged(
        &self,
        stream: &crate::CudaStream,
        bytes: Box<[MaybeUninit<u8>]>,
    ) -> Result<(), DriverError> {
        stream.context().bind_to_thread()?;
        let num_bytes = bytes.len();
        if num_bytes == 0 {
            return Ok(());
        }

        unsafe {
            crate::simt::memory::memcpy_htod_async(
                self.dptr,
                bytes.as_ptr() as *const u8,
                num_bytes,
                stream.cu_stream(),
            )?;
        }

        unsafe extern "C" fn drop_staged_bytes(callback: *mut c_void) {
            drop(unsafe {
                Box::<Box<[MaybeUninit<u8>]>>::from_raw(callback as *mut Box<[MaybeUninit<u8>]>)
            });
        }

        let callback_data = Box::into_raw(Box::new(bytes)) as *mut c_void;
        let callback_result = unsafe {
            cuda_bindings::cuLaunchHostFunc(
                stream.cu_stream(),
                Some(drop_staged_bytes),
                callback_data,
            )
        }
        .result();

        if let Err(err) = callback_result {
            let staged = unsafe {
                Box::<Box<[MaybeUninit<u8>]>>::from_raw(
                    callback_data as *mut Box<[MaybeUninit<u8>]>,
                )
            };
            if let Err(sync_err) = stream.synchronize() {
                Box::leak(staged);
                return Err(sync_err);
            }
            drop(staged);
            return Err(err);
        }

        Ok(())
    }

    /// Synchronous `cuMemcpyHtoD` from `src` into the device global. Blocks
    /// the calling thread.
    ///
    /// # Safety
    ///
    /// Same contract as [`write_async`](Self::write_async).
    pub unsafe fn write_blocking(
        &self,
        module: &Arc<CudaModule>,
        src: *const u8,
        num_bytes: usize,
    ) -> Result<(), DriverError> {
        unsafe { module.copy_bytes_to_device_global_sync(self.dptr, src, num_bytes) }
    }
}

impl CudaModule {
    /// Resolves a device global by name and returns its device pointer and
    /// size in bytes.
    ///
    /// Used to find `__constant__`-style globals (and other module-scope
    /// device symbols) so the host can populate them via `cuMemcpyHtoD`.
    /// The returned size is what the driver recorded for the symbol — host
    /// code should assert it matches the expected element size before
    /// copying.
    ///
    /// Binds the owning context to the calling thread first.
    ///
    /// # Errors
    ///
    /// Returns an error if binding fails or if `name` cannot be resolved
    /// in this module.
    ///
    /// # Panics
    ///
    /// Panics if `name` contains interior null bytes.
    pub fn get_global(
        self: &Arc<Self>,
        name: &str,
    ) -> Result<(cuda_bindings::CUdeviceptr, usize), DriverError> {
        self.ctx.bind_to_thread()?;
        let c_name = CString::new(name).unwrap();
        let mut dptr = MaybeUninit::<cuda_bindings::CUdeviceptr>::uninit();
        let mut size = MaybeUninit::<usize>::uninit();
        unsafe {
            cuda_bindings::cuModuleGetGlobal_v2(
                dptr.as_mut_ptr(),
                size.as_mut_ptr(),
                self.cu_module,
                c_name.as_ptr(),
            )
            .result()?;
            Ok((dptr.assume_init(), size.assume_init()))
        }
    }

    /// Synchronously copies `num_bytes` from host memory at `src` into the
    /// device global at `dptr`.
    ///
    /// Intended for populating `#[constant]` statics from the macro-generated
    /// `set_<name>_blocking` methods on `LoadedModule`. The caller is expected to have
    /// already resolved `dptr` via [`get_global`](Self::get_global) and
    /// verified that the driver-reported size matches the host type's size.
    ///
    /// # Safety
    ///
    /// - `dptr` must be a valid device pointer with at least `num_bytes` of
    ///   accessible storage.
    /// - `src` must point to at least `num_bytes` of readable host memory.
    /// - The device-side type at `dptr` and the host bytes at `src` must
    ///   have compatible layout.
    pub unsafe fn copy_bytes_to_device_global_sync(
        self: &Arc<Self>,
        dptr: cuda_bindings::CUdeviceptr,
        src: *const u8,
        num_bytes: usize,
    ) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { crate::simt::memory::memcpy_htod_sync(dptr, src, num_bytes) }
    }
}

impl CudaFunction {
    fn attribute(
        &self,
        attribute: cuda_bindings::CUfunction_attribute,
    ) -> Result<u32, DriverError> {
        self.context().bind_to_thread()?;
        let mut value = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuFuncGetAttribute(value.as_mut_ptr(), attribute, self.cu_function)
                .result()?;
            u32::try_from(value.assume_init())
                .map_err(|_| DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE))
        }
    }

    /// Returns the context that owns this function.
    pub fn context(&self) -> &Arc<CudaContext> {
        self.module.context()
    }

    /// Queries the largest thread block accepted by this function.
    pub fn max_threads_per_block(&self) -> Result<u32, DriverError> {
        self.attribute(
            cuda_bindings::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        )
    }

    /// Queries this function's statically allocated shared memory per block.
    pub fn static_shared_memory_bytes(&self) -> Result<u32, DriverError> {
        self.attribute(cuda_bindings::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
    }

    /// Queries the currently configured dynamic shared-memory maximum.
    pub fn max_dynamic_shared_memory_bytes(&self) -> Result<u32, DriverError> {
        self.attribute(
            cuda_bindings::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        )
    }

    /// Queries the number of registers each thread of this function uses.
    ///
    /// This is the `registers` figure `ptxas -v` prints at compile time, read
    /// back from the loaded module instead. Register pressure is what caps
    /// occupancy on most kernels, so this is the number to check first when a
    /// launch runs at fewer blocks per SM than expected.
    pub fn num_registers(&self) -> Result<u32, DriverError> {
        self.attribute(cuda_bindings::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_NUM_REGS)
    }

    /// Queries the per-thread local memory (frame) size of this function, in
    /// bytes.
    ///
    /// This is the per-thread quantity that
    /// [`CudaContext::set_stack_size`](crate::simt::context::CudaContext::set_stack_size)
    /// budgets for: the driver reserves the stack limit for every resident
    /// thread on the device, so a kernel with a large frame can reserve
    /// device memory far in excess of anything it allocates explicitly.
    pub fn local_size_bytes(&self) -> Result<u32, DriverError> {
        self.attribute(cuda_bindings::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)
    }

    /// Queries the user-declared constant memory this function requires, in
    /// bytes.
    ///
    /// Covers only constant memory the kernel declares; it excludes the
    /// driver's own kernel parameter and system constant banks.
    pub fn const_size_bytes(&self) -> Result<u32, DriverError> {
        self.attribute(cuda_bindings::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES)
    }

    /// Queries a cluster shape compiled into this function, if present.
    ///
    /// CUDA requires the three required-cluster attributes to be either all
    /// zero or all positive. A partial tuple is treated as an invalid driver
    /// response rather than silently normalizing it.
    pub fn required_cluster_dimensions(&self) -> Result<Option<(u32, u32, u32)>, DriverError> {
        let required = (
            self.attribute(
                cuda_bindings::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH,
            )?,
            self.attribute(
                cuda_bindings::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT,
            )?,
            self.attribute(
                cuda_bindings::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH,
            )?,
        );
        match required {
            (0, 0, 0) => Ok(None),
            (x, y, z) if x != 0 && y != 0 && z != 0 => Ok(Some(required)),
            _ => Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
            )),
        }
    }

    /// Opts this function into a larger dynamic shared-memory allocation.
    ///
    /// Typed launch preparation calls this at most once, and only after
    /// checking static plus dynamic memory against the device opt-in limit.
    pub(crate) fn set_max_dynamic_shared_memory_bytes(
        &self,
        bytes: u32,
    ) -> Result<(), DriverError> {
        self.context().bind_to_thread()?;
        let bytes = i32::try_from(bytes)
            .map_err(|_| DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE))?;
        unsafe {
            cuda_bindings::cuFuncSetAttribute(
                self.cu_function,
                cuda_bindings::CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                bytes,
            )
        }
        .result()
    }

    /// Computes the maximum active blocks per streaming multiprocessor for a
    /// concrete non-cluster launch shape.
    pub fn max_active_blocks_per_multiprocessor(
        &self,
        block_threads: u32,
        dynamic_shared_memory_bytes: u32,
    ) -> Result<u32, DriverError> {
        self.context().bind_to_thread()?;
        let block_threads = i32::try_from(block_threads)
            .map_err(|_| DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE))?;
        let mut blocks = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuOccupancyMaxActiveBlocksPerMultiprocessor(
                blocks.as_mut_ptr(),
                self.cu_function,
                block_threads,
                dynamic_shared_memory_bytes as usize,
            )
            .result()?;
            u32::try_from(blocks.assume_init())
                .map_err(|_| DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE))
        }
    }

    /// Queries the maximum cluster size for this function and launch shape on
    /// the current device.
    ///
    /// CUDA documents that `cuOccupancyMaxPotentialClusterSize` ignores any
    /// cluster-dimension attribute in the supplied launch configuration, so
    /// this query intentionally supplies only grid, block, and dynamic shared
    /// memory. It respects compile-time cluster launch bounds and any function
    /// opt-in to non-portable cluster sizes.
    pub fn max_potential_cluster_size(
        &self,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        dynamic_shared_memory_bytes: u32,
    ) -> Result<u32, DriverError> {
        self.context().bind_to_thread()?;
        let config = cuda_bindings::CUlaunchConfig_st {
            gridDimX: grid_dim.0,
            gridDimY: grid_dim.1,
            gridDimZ: grid_dim.2,
            blockDimX: block_dim.0,
            blockDimY: block_dim.1,
            blockDimZ: block_dim.2,
            sharedMemBytes: dynamic_shared_memory_bytes,
            hStream: std::ptr::null_mut(),
            attrs: std::ptr::null_mut(),
            numAttrs: 0,
        };
        let mut cluster_size = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuOccupancyMaxPotentialClusterSize(
                cluster_size.as_mut_ptr(),
                self.cu_function,
                &config,
            )
            .result()?;
            u32::try_from(cluster_size.assume_init())
                .map_err(|_| DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE))
        }
    }

    /// Queries how many clusters with the exact requested shape can be active
    /// on the target device.
    ///
    /// Unlike [`max_potential_cluster_size`](Self::max_potential_cluster_size),
    /// this passes `cluster_dim` as a launch attribute. CUDA therefore checks
    /// the concrete shape and any compiled required-cluster dimensions.
    pub fn max_active_clusters(
        &self,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        dynamic_shared_memory_bytes: u32,
        cluster_dim: (u32, u32, u32),
    ) -> Result<u32, DriverError> {
        self.context().bind_to_thread()?;

        // CUlaunchAttribute_st is opaque in the generated CUDA 13.2+
        // bindings. Its C layout stores the id at offset 0 and the value union
        // at offset 8; clusterDim.x/y/z occupy the first three u32 values in
        // that union. This matches the launch helpers in cuda-core's root.
        let mut cluster_attribute: cuda_bindings::CUlaunchAttribute_st =
            unsafe { std::mem::zeroed() };
        unsafe {
            let base = &mut cluster_attribute as *mut _ as *mut u8;
            (base as *mut u32).write(
                cuda_bindings::CUlaunchAttributeID_enum_CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION,
            );
            let dimensions = base.add(8) as *mut u32;
            dimensions.write(cluster_dim.0);
            dimensions.add(1).write(cluster_dim.1);
            dimensions.add(2).write(cluster_dim.2);
        }

        let config = cuda_bindings::CUlaunchConfig_st {
            gridDimX: grid_dim.0,
            gridDimY: grid_dim.1,
            gridDimZ: grid_dim.2,
            blockDimX: block_dim.0,
            blockDimY: block_dim.1,
            blockDimZ: block_dim.2,
            sharedMemBytes: dynamic_shared_memory_bytes,
            hStream: std::ptr::null_mut(),
            attrs: &mut cluster_attribute,
            numAttrs: 1,
        };
        let mut active_clusters = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuOccupancyMaxActiveClusters(
                active_clusters.as_mut_ptr(),
                self.cu_function,
                &config,
            )
            .result()?;
            u32::try_from(active_clusters.assume_init())
                .map_err(|_| DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE))
        }
    }

    /// Returns the raw `CUfunction` handle.
    ///
    /// # Safety
    ///
    /// The returned handle is copied by value, but it is non-owning. It is
    /// invalidated if the parent [`CudaModule`] is dropped.
    ///
    /// Because [`CudaFunction`] holds an [`Arc`] to its parent module, the
    /// module cannot be unloaded while `self` is alive. If the raw handle is
    /// stored outside the immediate call, the caller must keep this
    /// [`CudaFunction`] or another [`Arc`] owning the parent module alive for
    /// at least as long as the raw handle is used.
    pub unsafe fn cu_function(&self) -> cuda_bindings::CUfunction {
        self.cu_function
    }
}
