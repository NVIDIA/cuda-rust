/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA Virtual Memory Management (VMM) API wrappers.
//!
//! The VMM APIs provide fine-grained control over physical memory allocation,
//! virtual address reservation, and mapping. Unlike `cuMemAlloc`, which bundles
//! all three steps, VMM separates them so that physical memory from one device
//! can be mapped into another device's virtual address space -- the foundation
//! for P2P symmetric heaps.
//!
//! All handle types are RAII: `PhysicalAllocation` releases via `cuMemRelease`,
//! `VirtualReservation` frees via `cuMemAddressFree`, and `Mapping` unmaps
//! via `cuMemUnmap`. Drop order matters -- mappings must be dropped before the
//! physical allocation or virtual reservation they reference.
//!
//! The module also wraps CUDA's multicast objects (`cuMulticast*`, CUDA
//! 12.1+): `MulticastObject` (compiled out on pre-12.1 toolkits, see the
//! build script probe) builds an NVLink SHARP (NVLS) team whose
//! mapped VA ranges respond to device-side `multimem.*` instructions with
//! switch-side reduction and broadcast across every bound GPU. See
//! `tests/vmm_multicast.rs` and the `nvls_all_reduce` example.

use crate::error::{DriverError, IntoResult};
use cuda_bindings::CUdeviceptr;
use std::mem::MaybeUninit;

/// Sets the device ordinal on a `CUmemLocation_st`.
///
/// CUDA 13.2 wraps `id` in an anonymous union (`__bindgen_anon_1.id`), while
/// older versions expose it directly. The memory layout is identical -- `id` is
/// always at offset 4 (after the `type_` enum). Writing via pointer works across
/// both layouts.
unsafe fn set_mem_location_device(
    loc: &mut cuda_bindings::CUmemLocation_st,
    device: cuda_bindings::CUdevice,
) {
    loc.type_ = cuda_bindings::CUmemLocationType_enum_CU_MEM_LOCATION_TYPE_DEVICE;
    unsafe {
        let base = loc as *mut _ as *mut u8;
        (base.add(4) as *mut i32).write(device);
    }
}

/// A physical memory allocation created by `cuMemCreate`.
///
/// Owns the underlying `CUmemGenericAllocationHandle`. The allocation lives on
/// a specific device and can be mapped into any device's VA space that has been
/// granted access.
///
/// Dropping this releases the physical memory. All `Mapping`s referencing this
/// allocation must be dropped first.
pub struct PhysicalAllocation {
    handle: cuda_bindings::CUmemGenericAllocationHandle,
    size: usize,
    device: cuda_bindings::CUdevice,
}

impl PhysicalAllocation {
    /// Allocates `size` bytes of physical memory on `device`.
    ///
    /// `size` must be a multiple of the allocation granularity for the device
    /// (query via [`allocation_granularity`]).
    pub fn new(device: cuda_bindings::CUdevice, size: usize) -> Result<Self, DriverError> {
        let mut prop: cuda_bindings::CUmemAllocationProp_st = unsafe { std::mem::zeroed() };
        prop.type_ = cuda_bindings::CUmemAllocationType_enum_CU_MEM_ALLOCATION_TYPE_PINNED;
        unsafe { set_mem_location_device(&mut prop.location, device) };

        let mut handle = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuMemCreate(handle.as_mut_ptr(), size, &prop, 0).result()?;
            Ok(Self {
                handle: handle.assume_init(),
                size,
                device,
            })
        }
    }

    /// Returns the raw `CUmemGenericAllocationHandle`.
    pub fn handle(&self) -> cuda_bindings::CUmemGenericAllocationHandle {
        self.handle
    }

    /// Returns the allocation size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the device this allocation lives on.
    pub fn device(&self) -> cuda_bindings::CUdevice {
        self.device
    }
}

impl Drop for PhysicalAllocation {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_bindings::cuMemRelease(self.handle).result();
        }
    }
}

/// A reserved virtual address range created by `cuMemAddressReserve`.
///
/// Owns a contiguous VA range `[base, base + size)`. Physical memory can be
/// mapped into this range via [`Mapping::new`]. The range is freed on drop.
///
/// All `Mapping`s within this range must be dropped before the reservation.
pub struct VirtualReservation {
    base: CUdeviceptr,
    size: usize,
}

impl VirtualReservation {
    /// Reserves `size` bytes of virtual address space.
    ///
    /// The driver chooses the base address. `size` must be a multiple of the
    /// allocation granularity. `alignment` can be 0 to let the driver choose.
    pub fn new(size: usize, alignment: usize) -> Result<Self, DriverError> {
        let mut base = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuMemAddressReserve(base.as_mut_ptr(), size, alignment, 0, 0)
                .result()?;
            Ok(Self {
                base: base.assume_init(),
                size,
            })
        }
    }

    /// Returns the base device pointer of the reserved range.
    pub fn base(&self) -> CUdeviceptr {
        self.base
    }

    /// Returns the reserved size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for VirtualReservation {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_bindings::cuMemAddressFree(self.base, self.size).result();
        }
    }
}

/// A mapping of physical memory into a virtual address range.
///
/// Created by [`Mapping::new`], which calls `cuMemMap` to bind a
/// `PhysicalAllocation` (or a portion of it) to a region within a
/// `VirtualReservation`. Dropped via `cuMemUnmap`.
pub struct Mapping {
    va: CUdeviceptr,
    size: usize,
}

impl Mapping {
    /// Maps `size` bytes of `phys` at `offset` into virtual address `va`.
    ///
    /// `va` must lie within a `VirtualReservation`. `offset` is the byte
    /// offset into the physical allocation (typically 0 for full mappings).
    /// `size` must be a multiple of the allocation granularity.
    pub fn new(
        va: CUdeviceptr,
        size: usize,
        phys: &PhysicalAllocation,
        offset: usize,
    ) -> Result<Self, DriverError> {
        unsafe {
            cuda_bindings::cuMemMap(va, size, offset, phys.handle(), 0).result()?;
        }
        Ok(Self { va, size })
    }

    /// Maps `size` bytes of a multicast object at `offset` into virtual
    /// address `va`.
    ///
    /// The resulting VA is a *multicast* view: `multimem.*` PTX instructions
    /// issued against it operate on every copy bound to the object (see
    /// [`MulticastObject`]). Like [`Mapping::new`], the mapping is not
    /// accessible until [`set_access`] grants the accessing device permission.
    ///
    /// All devices must have been added to the multicast object (via
    /// [`MulticastObject::add_device`]) before mapping it.
    #[cfg(cuda_has_multicast)]
    pub fn new_multicast(
        va: CUdeviceptr,
        size: usize,
        multicast: &MulticastObject,
        offset: usize,
    ) -> Result<Self, DriverError> {
        unsafe {
            cuda_bindings::cuMemMap(va, size, offset, multicast.handle(), 0).result()?;
        }
        Ok(Self { va, size })
    }

    /// Returns the virtual address this mapping occupies.
    pub fn va(&self) -> CUdeviceptr {
        self.va
    }

    /// Returns the mapped size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_bindings::cuMemUnmap(self.va, self.size).result();
        }
    }
}

/// Sets read/write access on a virtual address range for one or more devices.
///
/// After calling `cuMemMap`, the mapping is not yet accessible. This function
/// grants the specified `devices` read/write permission on the range
/// `[va, va + size)`.
///
/// Typically called once after all mappings within a reservation are established.
pub fn set_access(
    va: CUdeviceptr,
    size: usize,
    devices: &[cuda_bindings::CUdevice],
) -> Result<(), DriverError> {
    let descs: Vec<cuda_bindings::CUmemAccessDesc_st> = devices
        .iter()
        .map(|&dev| {
            let mut desc: cuda_bindings::CUmemAccessDesc_st = unsafe { std::mem::zeroed() };
            unsafe { set_mem_location_device(&mut desc.location, dev) };
            desc.flags = cuda_bindings::CUmemAccess_flags_enum_CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            desc
        })
        .collect();

    unsafe { cuda_bindings::cuMemSetAccess(va, size, descs.as_ptr(), descs.len()) }.result()
}

/// Queries the minimum allocation granularity for VMM operations on `device`.
///
/// All sizes passed to [`PhysicalAllocation::new`], [`VirtualReservation::new`],
/// and [`Mapping::new`] must be multiples of this value.
pub fn allocation_granularity(device: cuda_bindings::CUdevice) -> Result<usize, DriverError> {
    let mut prop: cuda_bindings::CUmemAllocationProp_st = unsafe { std::mem::zeroed() };
    prop.type_ = cuda_bindings::CUmemAllocationType_enum_CU_MEM_ALLOCATION_TYPE_PINNED;
    unsafe { set_mem_location_device(&mut prop.location, device) };

    let mut granularity = MaybeUninit::uninit();
    unsafe {
        cuda_bindings::cuMemGetAllocationGranularity(
            granularity.as_mut_ptr(),
            &prop,
            cuda_bindings::CUmemAllocationGranularity_flags_enum_CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )
        .result()?;
        Ok(granularity.assume_init())
    }
}

/// Rounds `size` up to the nearest multiple of `granularity`.
pub fn align_size(size: usize, granularity: usize) -> usize {
    (size + granularity - 1) & !(granularity - 1)
}

/// Granularity flavor for [`multicast_granularity`].
#[cfg(cuda_has_multicast)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MulticastGranularity {
    /// Minimum required granularity for multicast sizes and offsets.
    Minimum,
    /// Recommended granularity for best performance.
    Recommended,
}

#[cfg(cuda_has_multicast)]
impl MulticastGranularity {
    fn to_flag(self) -> cuda_bindings::CUmulticastGranularity_flags {
        match self {
            MulticastGranularity::Minimum => {
                cuda_bindings::CUmulticastGranularity_flags_enum_CU_MULTICAST_GRANULARITY_MINIMUM
            }
            MulticastGranularity::Recommended => {
                cuda_bindings::CUmulticastGranularity_flags_enum_CU_MULTICAST_GRANULARITY_RECOMMENDED
            }
        }
    }
}

/// Fills a `CUmulticastObjectProp_st` for a single-process team of
/// `num_devices` GPUs binding up to `size` bytes each.
///
/// `handleTypes` is left at 0: the object cannot be exported to other
/// processes. Multi-process teams (exporting the handle over a POSIX file
/// descriptor or fabric handle) are out of scope for these wrappers.
#[cfg(cuda_has_multicast)]
fn multicast_prop(num_devices: u32, size: usize) -> cuda_bindings::CUmulticastObjectProp_st {
    let mut prop: cuda_bindings::CUmulticastObjectProp_st = unsafe { std::mem::zeroed() };
    prop.numDevices = num_devices;
    prop.size = size;
    prop
}

/// Returns whether `device` supports switch multicast and reduction
/// operations (`CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`).
///
/// Multicast requires an NVLink-switch-connected system (e.g. HGX/DGX
/// H100 or B200) and CUDA 12.1+.
#[cfg(cuda_has_multicast)]
pub fn multicast_supported(device: cuda_bindings::CUdevice) -> Result<bool, DriverError> {
    let mut value = MaybeUninit::uninit();
    let status = unsafe {
        cuda_bindings::cuDeviceGetAttribute(
            value.as_mut_ptr(),
            cuda_bindings::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
            device,
        )
    };
    // Drivers older than CUDA 12.1 do not know this attribute and answer
    // CUDA_ERROR_INVALID_VALUE. That means "no multicast on this system",
    // not a failure, so callers can probe-and-skip uniformly.
    if status == cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE {
        return Ok(false);
    }
    status.result()?;
    Ok(unsafe { value.assume_init() } != 0)
}

/// Built against a pre-12.1 CUDA toolkit whose headers lack the multicast
/// API: no device can be used for multicast through these wrappers.
#[cfg(not(cuda_has_multicast))]
pub fn multicast_supported(_device: cuda_bindings::CUdevice) -> Result<bool, DriverError> {
    Ok(false)
}

/// Queries the multicast size/offset granularity for a team of
/// `num_devices` GPUs binding `size` bytes each.
///
/// All sizes and offsets passed to [`MulticastObject::new`] and
/// [`MulticastObject::bind_mem`] must be multiples of the
/// [`MulticastGranularity::Minimum`] value; use
/// [`MulticastGranularity::Recommended`] for best performance.
#[cfg(cuda_has_multicast)]
pub fn multicast_granularity(
    num_devices: u32,
    size: usize,
    granularity: MulticastGranularity,
) -> Result<usize, DriverError> {
    let prop = multicast_prop(num_devices, size);
    let mut value = MaybeUninit::uninit();
    unsafe {
        cuda_bindings::cuMulticastGetGranularity(value.as_mut_ptr(), &prop, granularity.to_flag())
            .result()?;
        Ok(value.assume_init())
    }
}

/// A multicast object created by `cuMulticastCreate`: one virtual "team"
/// handle backed by up to one physical allocation per participating GPU.
///
/// Once every team device is added ([`add_device`](Self::add_device)) and has
/// bound physical memory ([`bind_mem`](Self::bind_mem)), the object can be
/// mapped into each device's VA space with [`Mapping::new_multicast`].
/// Device-side `multimem.ld_reduce` / `multimem.st` / `multimem.red`
/// instructions issued against that mapping operate on *all* bound copies at
/// once -- the NVSwitch performs the reduction/broadcast in the fabric
/// (NVLink SHARP, the mechanism behind NCCL's NVLS algorithm).
///
/// Lifecycle rules, in order:
/// 1. [`MulticastObject::new`] with the final team size.
/// 2. [`add_device`](Self::add_device) exactly `num_devices` times.
/// 3. [`bind_mem`](Self::bind_mem) per device (after ALL devices are added).
/// 4. [`Mapping::new_multicast`] + [`set_access`] per device.
///
/// Teardown reverses it: mappings first, then bindings ([`MulticastBinding`]),
/// then this object, then the physical allocations.
///
/// Dropping releases the handle via `cuMemRelease` (the documented release
/// path for multicast objects).
#[cfg(cuda_has_multicast)]
pub struct MulticastObject {
    handle: cuda_bindings::CUmemGenericAllocationHandle,
    size: usize,
    num_devices: u32,
}

#[cfg(cuda_has_multicast)]
impl MulticastObject {
    /// Creates a multicast object for a team of `num_devices` GPUs binding
    /// up to `size` bytes each.
    ///
    /// `size` must be a multiple of the minimum multicast granularity
    /// (query via [`multicast_granularity`]). The object is single-process
    /// only (no exportable handle types).
    pub fn new(num_devices: u32, size: usize) -> Result<Self, DriverError> {
        let prop = multicast_prop(num_devices, size);
        let mut handle = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuMulticastCreate(handle.as_mut_ptr(), &prop).result()?;
            Ok(Self {
                handle: handle.assume_init(),
                size,
                num_devices,
            })
        }
    }

    /// Adds `device` to the multicast team.
    ///
    /// Must be called exactly [`num_devices`](Self::num_devices) times, once
    /// per device, before any memory is bound.
    pub fn add_device(&self, device: cuda_bindings::CUdevice) -> Result<(), DriverError> {
        unsafe { cuda_bindings::cuMulticastAddDevice(self.handle, device).result() }
    }

    /// Binds `size` bytes of `phys` (starting at `mem_offset`) to this
    /// multicast object at `mc_offset`.
    ///
    /// The bound device is taken from the physical allocation. All offsets
    /// and `size` must be multiples of the minimum multicast granularity,
    /// and every team device must already have been added. A CUDA context
    /// for the bound device must be current.
    ///
    /// The returned [`MulticastBinding`] unbinds on drop; it must be dropped
    /// before this object and before `phys`.
    pub fn bind_mem(
        &self,
        mc_offset: usize,
        phys: &PhysicalAllocation,
        mem_offset: usize,
        size: usize,
    ) -> Result<MulticastBinding, DriverError> {
        unsafe {
            cuda_bindings::cuMulticastBindMem(
                self.handle,
                mc_offset,
                phys.handle(),
                mem_offset,
                size,
                0,
            )
            .result()?;
        }
        Ok(MulticastBinding {
            mc_handle: self.handle,
            device: phys.device(),
            mc_offset,
            size,
        })
    }

    /// Returns the raw multicast `CUmemGenericAllocationHandle`.
    pub fn handle(&self) -> cuda_bindings::CUmemGenericAllocationHandle {
        self.handle
    }

    /// Returns the per-device bind capacity in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the team size this object was created for.
    pub fn num_devices(&self) -> u32 {
        self.num_devices
    }
}

#[cfg(cuda_has_multicast)]
impl Drop for MulticastObject {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_bindings::cuMemRelease(self.handle).result();
        }
    }
}

/// One device's physical memory bound into a [`MulticastObject`].
///
/// Created by [`MulticastObject::bind_mem`]; unbinds via `cuMulticastUnbind`
/// on drop. Must be dropped before the multicast object and before the
/// physical allocation it binds.
#[cfg(cuda_has_multicast)]
pub struct MulticastBinding {
    mc_handle: cuda_bindings::CUmemGenericAllocationHandle,
    device: cuda_bindings::CUdevice,
    mc_offset: usize,
    size: usize,
}

#[cfg(cuda_has_multicast)]
impl MulticastBinding {
    /// Returns the device whose memory is bound.
    pub fn device(&self) -> cuda_bindings::CUdevice {
        self.device
    }
}

#[cfg(cuda_has_multicast)]
impl Drop for MulticastBinding {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_bindings::cuMulticastUnbind(
                self.mc_handle,
                self.device,
                self.mc_offset,
                self.size,
            )
            .result();
        }
    }
}
