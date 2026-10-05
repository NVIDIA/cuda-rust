/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! High-level wrappers around CUDA driver API functions.
//!
//! Provides safe(r) Rust interfaces for initialization, kernel launch, memory
//! operations, device queries, and random number generation.

pub use cuda_bindings as sys;
use cuda_bindings::{
    cuDeviceGetAttribute, CUdevice, CUdevice_attribute,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
};
use std::ffi::{c_int, c_uint, c_void};
use std::mem::{self, MaybeUninit};
use std::sync::Arc;

use crate::error::*;
use crate::runtime::Stream;

/// Initializes the CUDA driver API. Must be called before any other driver call.
///
/// # Safety
/// Caller must ensure CUDA is available and `flags` is valid (typically `0`).
pub unsafe fn init(flags: c_uint) -> Result<(), DriverError> {
    cuda_bindings::cuInit(flags).result()
}

/// Returns the API version associated with the given CUDA context.
///
/// # Safety
/// `ctx` must be a valid CUDA context handle.
pub unsafe fn api_version(ctx: cuda_bindings::CUcontext) -> Result<c_uint, DriverError> {
    let mut api_version = 0 as c_uint;
    unsafe { cuda_bindings::cuCtxGetApiVersion(ctx, &mut api_version) }.result()?;
    Ok(api_version)
}

/// Launches a CUDA kernel with the given grid/block dimensions and parameters.
///
/// # Safety
/// `f`, `stream`, and all pointers in `kernel_params` must be valid.
#[inline]
pub unsafe fn launch_kernel(
    f: cuda_bindings::CUfunction,
    grid_dim: (c_uint, c_uint, c_uint),
    block_dim: (c_uint, c_uint, c_uint),
    shared_mem_bytes: c_uint,
    stream: cuda_bindings::CUstream,
    kernel_params: &mut [*mut c_void],
) -> Result<(), DriverError> {
    cuda_bindings::cuLaunchKernel(
        f,
        grid_dim.0,
        grid_dim.1,
        grid_dim.2,
        block_dim.0,
        block_dim.1,
        block_dim.2,
        shared_mem_bytes,
        stream,
        kernel_params.as_mut_ptr(),
        std::ptr::null_mut(),
    )
    .result()
}

/// Launch with programmatic stream serialization enabled for this kernel.
/// The driver entry point is resolved at runtime; a missing entry point is
/// reported as unsupported. Ordinary launches keep using [`launch_kernel`].
///
/// # Safety
/// The ordinary launch contract applies. Additionally, this kernel must wait
/// for predecessor completion before dependent accesses, and work before that
/// wait must not race predecessor work. Neither kernel may require overlap.
/// All resources must outlive their last use by either kernel.
#[inline]
pub unsafe fn launch_kernel_pdl(
    f: cuda_bindings::CUfunction,
    grid_dim: (c_uint, c_uint, c_uint),
    block_dim: (c_uint, c_uint, c_uint),
    shared_mem_bytes: c_uint,
    stream: cuda_bindings::CUstream,
    kernel_params: &mut [*mut c_void],
) -> Result<(), DriverError> {
    let mut attribute = programmatic_launch_attribute();
    let config = cuda_bindings::CUlaunchConfig_st {
        gridDimX: grid_dim.0,
        gridDimY: grid_dim.1,
        gridDimZ: grid_dim.2,
        blockDimX: block_dim.0,
        blockDimY: block_dim.1,
        blockDimZ: block_dim.2,
        sharedMemBytes: shared_mem_bytes,
        hStream: stream,
        attrs: &mut attribute,
        numAttrs: 1,
    };
    // SAFETY: the caller supplies valid launch handles/arguments. The driver
    // copies config, attributes, and parameter values before returning.
    let result = unsafe {
        cuda_bindings::cuLaunchKernelEx(
            &config,
            f,
            kernel_params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    if result == cuda_bindings::cudaError_enum_CUDA_ERROR_NOT_FOUND {
        // There is no legacy launch with the requested overlap semantics.
        return Err(DriverError(
            cuda_bindings::cudaError_enum_CUDA_ERROR_NOT_SUPPORTED,
        ));
    }
    result.result()
}

fn programmatic_launch_attribute() -> cuda_bindings::CUlaunchAttribute_st {
    // The bindings intentionally keep this union opaque across CTK versions.
    // CUDA's stable layout is id:u32 at 0, padding at 4, value union at 8.
    // programmaticStreamSerializationAllowed is an int at union offset zero.
    const {
        assert!(std::mem::size_of::<cuda_bindings::CUlaunchAttribute_st>() >= 12);
    }
    // SAFETY: all-zero is a valid bit pattern for this C attribute storage.
    let mut attribute = unsafe { std::mem::zeroed::<cuda_bindings::CUlaunchAttribute_st>() };
    // SAFETY: writes stay within the checked storage; unaligned writes avoid
    // depending on bindgen's opaque wrapper alignment.
    unsafe {
        let base = (&mut attribute as *mut cuda_bindings::CUlaunchAttribute_st).cast::<u8>();
        base.cast::<u32>().write_unaligned(
            cuda_bindings::CUlaunchAttributeID_enum_CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        );
        base.add(8).cast::<i32>().write_unaligned(1);
    }
    attribute
}

#[cfg(test)]
mod pdl_attribute_tests {
    #[test]
    fn programmatic_launch_attribute_has_driver_abi_layout() {
        let attribute = super::programmatic_launch_attribute();
        // SAFETY: the helper initializes every byte of the opaque C storage.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                (&attribute as *const cuda_bindings::CUlaunchAttribute_st).cast::<u8>(),
                std::mem::size_of_val(&attribute),
            )
        };
        let id = cuda_bindings::CUlaunchAttributeID_enum_CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        assert_eq!(&bytes[..4], &id.to_ne_bytes());
        assert_eq!(&bytes[8..12], &1i32.to_ne_bytes());
        assert!(bytes[4..8].iter().chain(&bytes[12..]).all(|b| *b == 0));
    }
}

/// Asynchronously allocates `num_bytes` of device memory on the given stream.
///
/// Driver failures (out of memory included) come back as `Err` for the
/// caller to handle; the returned pointer becomes valid once the allocation
/// executes in stream order.
///
/// # Safety
/// `stream` must be a valid, non-destroyed CUDA stream.
pub unsafe fn malloc_async(
    num_bytes: usize,
    stream: &Arc<Stream>,
) -> Result<sys::CUdeviceptr, DriverError> {
    crate::cudarc_shim::memory::malloc_async(stream.cu_stream(), num_bytes)
}

/// Asynchronously allocates `num_bytes` of device memory from a specific pool on the given stream.
///
/// Driver failures come back as `Err`, as for [`malloc_async`].
///
/// # Safety
/// `stream` must be a valid, non-destroyed CUDA stream. `pool` must be a valid memory pool.
pub unsafe fn malloc_from_pool_async(
    num_bytes: usize,
    pool: &Arc<crate::MemPool>,
    stream: &Arc<Stream>,
) -> Result<sys::CUdeviceptr, DriverError> {
    crate::cudarc_shim::pool::malloc_from_pool_async(pool.cu_pool(), stream.cu_stream(), num_bytes)
}

/// Asynchronously sets `num_bytes` bytes at `dptr` to `value` on `stream`.
///
/// # Safety
/// `dptr` must point to a device allocation of at least `num_bytes` bytes
/// that stays valid until the operation completes on `stream`.
pub unsafe fn memset_d8_async(
    dptr: sys::CUdeviceptr,
    value: u8,
    num_bytes: usize,
    stream: &Arc<Stream>,
) -> Result<(), DriverError> {
    crate::cudarc_shim::memory::memset_d8_async(dptr, value, num_bytes, stream.cu_stream())
}

/// Asynchronously frees device memory on the given stream.
///
/// # Safety
/// `dptr` must have been allocated with [`malloc_async`] or
/// [`malloc_from_pool_async`], `stream` must be ordered after every use of
/// the allocation, and `dptr` must not be used after this call.
pub unsafe fn free_async(dptr: sys::CUdeviceptr, stream: &Arc<Stream>) -> Result<(), DriverError> {
    crate::cudarc_shim::memory::free_async(dptr, stream.cu_stream())
}

/// Asynchronously copies `num_elements` of type `T` from host to device memory.
///
/// # Safety
/// `src` must point to at least `num_elements` valid elements; `dst` must have sufficient capacity.
pub unsafe fn memcpy_htod_async<T>(
    dst: sys::CUdeviceptr,
    src: *const T,
    num_elements: usize,
    stream: &Arc<Stream>,
) -> Result<(), DriverError> {
    let num_bytes = num_elements * mem::size_of::<T>();
    unsafe {
        crate::cudarc_shim::memory::memcpy_htod_async(dst, src, num_bytes, stream.cu_stream())
    }
}

/// Asynchronously copies `num_elements` of type `T` from device to host memory.
///
/// # Safety
/// `dst` must point to at least `num_elements` writable elements; `src` must be valid device memory.
pub unsafe fn memcpy_dtoh_async<T>(
    dst: *mut T,
    src: sys::CUdeviceptr,
    num_elements: usize,
    stream: &Arc<Stream>,
) -> Result<(), DriverError> {
    let num_bytes = num_elements * mem::size_of::<T>();
    unsafe {
        crate::cudarc_shim::memory::memcpy_dtoh_async(dst, src, num_bytes, stream.cu_stream())
    }
}

/// Asynchronously copies `num_elements` of type `T` between device memory regions.
///
/// # Safety
/// Both `dst` and `src` must be valid device pointers with sufficient capacity.
pub unsafe fn memcpy_dtod_async<T>(
    dst: sys::CUdeviceptr,
    src: sys::CUdeviceptr,
    num_elements: usize,
    stream: &Arc<Stream>,
) -> Result<(), DriverError> {
    let num_bytes = num_elements * mem::size_of::<T>();
    unsafe {
        crate::cudarc_shim::memory::memcpy_dtod_async(dst, src, num_bytes, stream.cu_stream())
    }
}

/// Wrappers around the cuRAND random number generation library.
pub mod curand {
    // TODO (hme): Probably move this into its own file at some point.

    use crate::runtime::Stream;
    use cuda_bindings::{
        curandCreateGenerator, curandDestroyGenerator, curandGenerateNormal,
        curandGenerateNormalDouble, curandGenerateUniform, curandGenerateUniformDouble,
        curandGenerator_t, curandRngType_CURAND_RNG_PSEUDO_DEFAULT,
        curandSetPseudoRandomGeneratorSeed, curandSetStream, CUdeviceptr,
    };
    use std::ffi::c_ulonglong;
    use std::mem::MaybeUninit;
    use std::sync::Arc;

    /// Creates a new pseudo-random number generator with default RNG type.
    ///
    /// # Safety
    /// cuRAND library must be available.
    pub unsafe fn get_rng() -> curandGenerator_t {
        let mut curand_gen_uninited: MaybeUninit<curandGenerator_t> = MaybeUninit::uninit();
        let curand_rng_type = curandRngType_CURAND_RNG_PSEUDO_DEFAULT;
        assert!(curandCreateGenerator(curand_gen_uninited.as_mut_ptr(), curand_rng_type) == 0);
        curand_gen_uninited.assume_init()
    }

    /// Sets the seed for a pseudo-random number generator.
    ///
    /// # Safety
    /// `gen` must be a valid cuRAND generator handle.
    pub unsafe fn set_seed(gen: curandGenerator_t, seed: u64) {
        assert!(curandSetPseudoRandomGeneratorSeed(gen, c_ulonglong::from(seed)) == 0);
    }

    /// Generates normally distributed `f32` values into device memory.
    ///
    /// # Safety
    /// `dptr` must be valid device memory with capacity for `num_elements` floats.
    pub unsafe fn generate_normal_f32(
        curand_gen: curandGenerator_t,
        dptr: CUdeviceptr,
        num_elements: usize,
        mean: f32,
        std: f32,
    ) {
        assert!(curandGenerateNormal(curand_gen, dptr as *mut f32, num_elements, mean, std) == 0);
    }

    /// RAII wrapper around a cuRAND pseudo-random number generator.
    pub struct RNG {
        curand_gen: curandGenerator_t,
    }

    impl RNG {
        /// Creates a new RNG, optionally seeded.
        ///
        /// A fresh generator launches its kernels on the **legacy default
        /// stream**: each `generate_*` call is ordered with that stream, not
        /// with the stream that allocated or will consume the destination
        /// buffer. A buffer allocated with `cuMemAllocAsync` on a non-blocking
        /// stream may therefore not exist yet when the generator writes to it,
        /// and a consumer on that stream may read before the write lands. Use
        /// [`new_on_stream`](Self::new_on_stream) (or
        /// [`set_stream`](Self::set_stream)) to bind generation to the stream
        /// that owns the buffer.
        ///
        /// # Safety
        /// cuRAND library must be available.
        pub unsafe fn new(seed: Option<u64>) -> Self {
            let curand_gen = get_rng();
            if let Some(seed) = seed {
                set_seed(curand_gen, seed);
            }
            Self { curand_gen }
        }

        /// Creates a new RNG whose kernels run on `stream`, optionally seeded.
        ///
        /// The stream is bound before the seed is applied, so the generator's
        /// state setup is stream-ordered too. Equivalent to
        /// [`new`](Self::new) followed by [`set_stream`](Self::set_stream).
        ///
        /// # Safety
        /// cuRAND library must be available, and `stream` must be a valid,
        /// non-destroyed stream whose device context is current on the
        /// calling thread.
        pub unsafe fn new_on_stream(seed: Option<u64>, stream: &Arc<Stream>) -> Self {
            let rng = Self {
                curand_gen: get_rng(),
            };
            rng.set_stream(stream);
            if let Some(seed) = seed {
                set_seed(rng.curand_gen, seed);
            }
            rng
        }

        /// Binds this generator's kernel launches to `stream`
        /// (`curandSetStream`). Every later `generate_*` call is ordered on
        /// `stream`; until this is called the generator uses the legacy
        /// default stream (see [`new`](Self::new)). Like the other calls in
        /// this module, a non-success cuRAND status is an assertion failure.
        ///
        /// # Safety
        /// `stream` must be a valid, non-destroyed stream on the device whose
        /// context is current on the calling thread.
        pub unsafe fn set_stream(&self, stream: &Arc<Stream>) {
            assert!(curandSetStream(self.curand_gen, stream.cu_stream()) == 0);
        }

        /// Generates normally distributed `f32` values into device memory.
        ///
        /// # Safety
        /// `dptr` must be valid device memory with capacity for `num_elements` floats.
        pub unsafe fn generate_normal_f32(
            &self,
            dptr: CUdeviceptr,
            num_elements: usize,
            mean: f32,
            std: f32,
        ) {
            assert!(
                curandGenerateNormal(self.curand_gen, dptr as *mut f32, num_elements, mean, std)
                    == 0
            );
        }

        /// Generates normally distributed `f64` values into device memory.
        ///
        /// # Safety
        /// `dptr` must be valid device memory with capacity for `num_elements` doubles.
        pub unsafe fn generate_normal_f64(
            &self,
            dptr: CUdeviceptr,
            num_elements: usize,
            mean: f64,
            std: f64,
        ) {
            assert!(
                curandGenerateNormalDouble(
                    self.curand_gen,
                    dptr as *mut f64,
                    num_elements,
                    mean,
                    std
                ) == 0
            );
        }

        /// Generates uniformly distributed `f32` values in `[0, 1)` into device memory.
        ///
        /// # Safety
        /// `dptr` must be valid device memory with capacity for `num_elements` floats.
        pub unsafe fn generate_uniform_f32(&self, dptr: CUdeviceptr, num_elements: usize) {
            assert!(curandGenerateUniform(self.curand_gen, dptr as *mut f32, num_elements) == 0);
        }

        /// Generates uniformly distributed `f64` values in `[0, 1)` into device memory.
        ///
        /// # Safety
        /// `dptr` must be valid device memory with capacity for `num_elements` doubles.
        pub unsafe fn generate_uniform_f64(&self, dptr: CUdeviceptr, num_elements: usize) {
            assert!(
                curandGenerateUniformDouble(self.curand_gen, dptr as *mut f64, num_elements) == 0
            );
        }
    }

    impl Drop for RNG {
        fn drop(&mut self) {
            unsafe { assert!(curandDestroyGenerator(self.curand_gen) == 0) };
        }
    }

    /// Trait for types that support cuRAND normal distribution generation.
    pub trait RandNormal: Sized + Send {
        /// Generate normally distributed values into device memory.
        ///
        /// # Safety
        /// `dptr` must be valid device memory with capacity for `len` elements.
        unsafe fn generate_normal(rng: &RNG, dptr: CUdeviceptr, len: usize, mean: Self, std: Self);
    }

    impl RandNormal for f32 {
        unsafe fn generate_normal(rng: &RNG, dptr: CUdeviceptr, len: usize, mean: f32, std: f32) {
            rng.generate_normal_f32(dptr, len, mean, std);
        }
    }

    impl RandNormal for f64 {
        unsafe fn generate_normal(rng: &RNG, dptr: CUdeviceptr, len: usize, mean: f64, std: f64) {
            rng.generate_normal_f64(dptr, len, mean, std);
        }
    }

    /// Trait for types that support cuRAND uniform distribution generation.
    pub trait RandUniform: Sized + Send {
        /// Generate uniformly distributed values in `[0, 1)` into device memory.
        ///
        /// # Safety
        /// `dptr` must be valid device memory with capacity for `len` elements.
        unsafe fn generate_uniform(rng: &RNG, dptr: CUdeviceptr, len: usize);
    }

    impl RandUniform for f32 {
        unsafe fn generate_uniform(rng: &RNG, dptr: CUdeviceptr, len: usize) {
            rng.generate_uniform_f32(dptr, len);
        }
    }

    impl RandUniform for f64 {
        unsafe fn generate_uniform(rng: &RNG, dptr: CUdeviceptr, len: usize) {
            rng.generate_uniform_f64(dptr, len);
        }
    }
}

unsafe fn get_device_attribute(
    device: CUdevice,
    device_attr: CUdevice_attribute,
) -> Result<i32, DriverError> {
    let mut result: MaybeUninit<c_int> = MaybeUninit::uninit();
    // `result` is only read (by `IntoResult`) when the driver reported success.
    (
        cuDeviceGetAttribute(result.as_mut_ptr(), device_attr, device),
        result,
    )
        .result()
}

/// Returns the device clock rate in kHz.
///
/// # Safety
/// `device` must be a valid CUDA device handle.
pub unsafe fn get_device_clock_rate(device: CUdevice) -> Result<i32, DriverError> {
    get_device_attribute(
        device,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
    )
}

/// Returns the device's compute capability as an `sm_<major><minor>` name
/// (e.g. `sm_120`), the form the Tile compiler takes as its target.
///
/// # Safety
/// `device` must be a valid CUDA device handle.
pub unsafe fn get_device_sm_name(device: CUdevice) -> Result<String, DriverError> {
    let major = get_device_attribute(
        device,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    )?;
    let minor = get_device_attribute(
        device,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    )?;
    Ok(format!("sm_{major}{minor}"))
}
