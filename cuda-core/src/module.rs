/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Module globals and function resource queries.

pub use crate::runtime::{Function, Module};
use crate::{Context, DriverError, IntoResult};
use std::{
    borrow::Cow,
    ffi::{c_void, CString},
    mem::MaybeUninit,
    sync::Arc,
};

impl Context {
    /// Loads a CUDA module from an in-memory image.
    ///
    /// `image` may be PTX source bytes, a cubin, or a fatbin. PTX text is
    /// null-terminated before it is passed to the CUDA driver; binary module
    /// images tolerate the trailing byte because their own headers describe
    /// their size.
    pub fn load_module_from_image(
        self: &Arc<Self>,
        image: &[u8],
    ) -> Result<Arc<Module>, DriverError> {
        self.bind_to_thread()?;
        let image = null_terminated_image(image);
        let cu_module = unsafe {
            let mut cu_module = MaybeUninit::uninit();
            cuda_bindings::cuModuleLoadData(cu_module.as_mut_ptr(), image.as_ptr() as *const _)
                .result()?;
            cu_module.assume_init()
        };
        Ok(Arc::new(Module {
            cu_module,
            context: self.clone(),
            owned: true,
        }))
    }
}
impl Module {
    /// Returns the parent [`Context`].
    ///
    /// This is mainly useful when interoperating with raw CUDA driver APIs:
    /// call [`Context::bind_to_thread`] on this context before passing raw
    /// module/function handles to APIs that require the owning context to be
    /// current on the calling host thread.
    pub fn context(&self) -> &Arc<Context> {
        &self.context
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
    /// `LoadedModule` initializers after [`Module::get_global`] has
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
        stream: &crate::Stream,
        src: *const u8,
        num_bytes: usize,
    ) -> Result<(), DriverError> {
        stream.context().bind_to_thread()?;
        unsafe { crate::memory::memcpy_htod_async(self.dptr, src, num_bytes, stream.cu_stream()) }
    }

    /// Stream-ordered `cuMemcpyHtoDAsync` from owned host bytes into the
    /// device global.
    ///
    /// The bytes are kept alive until the stream reaches a host callback
    /// enqueued after the copy. This makes safe setters sound even when the
    /// caller passes a temporary such as `&3.0`.
    pub fn write_async_staged(
        &self,
        stream: &crate::Stream,
        bytes: Box<[MaybeUninit<u8>]>,
    ) -> Result<(), DriverError> {
        stream.context().bind_to_thread()?;
        let num_bytes = bytes.len();
        if num_bytes == 0 {
            return Ok(());
        }

        unsafe {
            crate::memory::memcpy_htod_async(
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
        module: &Arc<Module>,
        src: *const u8,
        num_bytes: usize,
    ) -> Result<(), DriverError> {
        unsafe { module.copy_bytes_to_device_global_sync(self.dptr, src, num_bytes) }
    }
}

impl Module {
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
        self.context.bind_to_thread()?;
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
        self.context.bind_to_thread()?;
        unsafe { crate::memory::memcpy_htod_sync(dptr, src, num_bytes) }
    }
}

impl Function {
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
    pub fn context(&self) -> &Arc<Context> {
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
    /// [`Context::set_stack_size`](crate::context::Context::set_stack_size)
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
