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
//! All handle types are RAII: `PhysicalAllocation` releases its handle via
//! `cuMemRelease`, `VirtualReservation` frees via `cuMemAddressFree`, and
//! `Mapping` unmaps via `cuMemUnmap`. A `Mapping<'a>` borrows both the
//! reservation it lives in and the memory it maps, so the teardown order
//! (mapping first, then reservation and allocation) is checked by the
//! compiler: a reservation cannot be freed while a mapping still points into
//! it, which `Drop` could not report. CUDA does keep a physical allocation
//! alive while mappings reference it, so the borrow of the allocation is
//! about determinism rather than a driver requirement.
//!
//! Unmapping is **not stream-ordered**: `cuMemUnmap` acts on the host
//! timeline without waiting for enqueued work, so a mapping must not be
//! dropped while a kernel or copy that touches its range is in flight (see
//! [`Mapping`]).
//!
//! The module also wraps CUDA's multicast objects (`cuMulticast*`, CUDA
//! 12.1+): `MulticastObject` builds an NVLink SHARP (NVLS) team whose
//! mapped VA ranges respond to device-side `multimem.*` instructions with
//! switch-side reduction and broadcast across every bound GPU. The host-side
//! plumbing is exercised by `tests/vmm_multicast.rs`; the switch-side
//! semantics need device code and are not tested in this crate.

use crate::error::{DriverError, IntoResult};
use cuda_bindings::CUdeviceptr;
use std::mem::MaybeUninit;

/// Sets the device ordinal on a `CUmemLocation_st`.
///
/// The generated bindings hide the CUDA-version-specific layout of the `id`
/// field behind [`cuda_bindings::set_mem_location_id`].
fn set_mem_location_device(
    loc: &mut cuda_bindings::CUmemLocation_st,
    device: cuda_bindings::CUdevice,
) {
    loc.type_ = cuda_bindings::CUmemLocationType_enum_CU_MEM_LOCATION_TYPE_DEVICE;
    cuda_bindings::set_mem_location_id(loc, device);
}

/// A physical memory allocation created by `cuMemCreate`.
///
/// Owns the underlying `CUmemGenericAllocationHandle`. The allocation lives on
/// a specific device and can be mapped into any device's VA space that has been
/// granted access.
///
/// Dropping this releases the handle reference. CUDA defers freeing the
/// physical memory while mappings still reference it, so existing mappings
/// remain valid.
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
        set_mem_location_device(&mut prop.location, device);

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
/// All `Mapping`s within this range must be dropped before the reservation for
/// deterministic, leak-free cleanup. If the driver rejects an attempt to free
/// a still-mapped range, `Drop` cannot report that error.
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
/// Created by [`Mapping::new`] (`cuMemMap` of a [`PhysicalAllocation`]) or
/// [`Mapping::new_multicast`] (of a [`MulticastObject`]); dropped via
/// `cuMemUnmap`. The mapping borrows the [`VirtualReservation`] it lives in
/// and the memory it maps for its whole lifetime, so it cannot outlive either.
/// Freeing a reservation that is still mapped is exactly the teardown mistake
/// `Drop` cannot report, and the borrow turns it into a compile error:
///
/// ```compile_fail,E0505
/// # use cuda_core::vmm::{Mapping, PhysicalAllocation, VirtualReservation};
/// # fn demo(va: VirtualReservation, phys: PhysicalAllocation) -> Result<(), cuda_core::DriverError> {
/// let mapping = Mapping::new(&va, 0, &phys, 0, va.size())?;
/// drop(va); // the range is still mapped
/// drop(mapping);
/// # Ok(())
/// # }
/// ```
///
/// # Unmapping is not stream-ordered
///
/// `cuMemUnmap` takes effect on the host timeline; it does not wait for work
/// already enqueued on any stream. Dropping a `Mapping` while a kernel or a
/// copy that touches its range is still in flight is a device-side
/// use-after-unmap: the access faults (`CUDA_ERROR_ILLEGAL_ADDRESS`) or, if
/// the range has been remapped meanwhile, lands in someone else's memory.
/// Synchronize every stream that uses the range before the mapping drops;
/// nothing here does it for you.
pub struct Mapping<'a> {
    va: CUdeviceptr,
    size: usize,
    /// The range `va` lies in; borrowed so the mapping is unmapped first.
    reservation: &'a VirtualReservation,
    /// What `va` is backed by; borrowed so it outlives the mapping.
    backing: MappingBacking<'a>,
}

/// The memory a [`Mapping`] maps.
enum MappingBacking<'a> {
    Physical(&'a PhysicalAllocation),
    Multicast(&'a MulticastObject),
}

/// Bounds a mapping of `size` bytes at `va_offset` into `reservation` and at
/// `backing_offset` into a backing of `backing_size` bytes, returning the
/// mapped address.
///
/// The driver cannot check the reservation bound: VA ranges are process-wide,
/// so a `cuMemMap` past the end of this reservation would succeed if it
/// happened to land in another one.
fn mapping_range(
    reservation: &VirtualReservation,
    va_offset: usize,
    backing_size: usize,
    backing_offset: usize,
    size: usize,
) -> Result<CUdeviceptr, DriverError> {
    let invalid = || DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE);
    let va_end = va_offset.checked_add(size).ok_or_else(invalid)?;
    let backing_end = backing_offset.checked_add(size).ok_or_else(invalid)?;
    if va_end > reservation.size() || backing_end > backing_size {
        return Err(invalid());
    }
    let va_offset = CUdeviceptr::try_from(va_offset).map_err(|_| invalid())?;
    reservation
        .base()
        .checked_add(va_offset)
        .ok_or_else(invalid)
}

impl<'a> Mapping<'a> {
    /// Maps `size` bytes of `phys`, starting `phys_offset` bytes into it, at
    /// `va_offset` bytes into `reservation`.
    ///
    /// Both offsets and `size` must be multiples of the allocation
    /// granularity; the mapping is not accessible until [`set_access`] grants
    /// a device permission on it.
    ///
    /// CUDA retains the physical memory while the mapping exists, so the
    /// borrow of `phys` is about deterministic teardown, not validity.
    ///
    /// # Errors
    ///
    /// `CUDA_ERROR_INVALID_VALUE` when the range does not fit in
    /// `reservation` or in `phys` (checked before the driver call), and
    /// whatever `cuMemMap` reports otherwise.
    pub fn new(
        reservation: &'a VirtualReservation,
        va_offset: usize,
        phys: &'a PhysicalAllocation,
        phys_offset: usize,
        size: usize,
    ) -> Result<Self, DriverError> {
        let va = mapping_range(reservation, va_offset, phys.size(), phys_offset, size)?;
        unsafe {
            cuda_bindings::cuMemMap(va, size, phys_offset, phys.handle(), 0).result()?;
        }
        Ok(Self {
            va,
            size,
            reservation,
            backing: MappingBacking::Physical(phys),
        })
    }

    /// Maps `size` bytes of `multicast`, starting `mc_offset` bytes into it,
    /// at `va_offset` bytes into `reservation`.
    ///
    /// The resulting VA is a *multicast* view: `multimem.*` PTX instructions
    /// issued against it operate on every copy bound to the object (see
    /// [`MulticastObject`]). Like [`Mapping::new`], the mapping is not
    /// accessible until [`set_access`] grants the accessing device permission.
    ///
    /// All devices must have been added to the multicast object (via
    /// [`MulticastObject::add_device`]) before mapping it.
    ///
    /// # Errors
    ///
    /// As [`Mapping::new`], with the object's per-device size as the bound.
    pub fn new_multicast(
        reservation: &'a VirtualReservation,
        va_offset: usize,
        multicast: &'a MulticastObject,
        mc_offset: usize,
        size: usize,
    ) -> Result<Self, DriverError> {
        let va = mapping_range(reservation, va_offset, multicast.size(), mc_offset, size)?;
        unsafe {
            cuda_bindings::cuMemMap(va, size, mc_offset, multicast.handle(), 0).result()?;
        }
        Ok(Self {
            va,
            size,
            reservation,
            backing: MappingBacking::Multicast(multicast),
        })
    }

    /// Returns the virtual address this mapping occupies.
    pub fn va(&self) -> CUdeviceptr {
        self.va
    }

    /// Returns the mapped size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the reservation this mapping lies in.
    pub fn reservation(&self) -> &'a VirtualReservation {
        self.reservation
    }

    /// Returns the physical allocation this mapping maps, or `None` for a
    /// multicast view.
    pub fn physical(&self) -> Option<&'a PhysicalAllocation> {
        match self.backing {
            MappingBacking::Physical(phys) => Some(phys),
            MappingBacking::Multicast(_) => None,
        }
    }

    /// Returns the multicast object this mapping is a view of, or `None` for
    /// a mapping of physical memory.
    pub fn multicast(&self) -> Option<&'a MulticastObject> {
        match self.backing {
            MappingBacking::Multicast(multicast) => Some(multicast),
            MappingBacking::Physical(_) => None,
        }
    }
}

impl Drop for Mapping<'_> {
    /// `cuMemUnmap`, immediately; see the type docs for why in-flight work on
    /// the range must be complete first.
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
            set_mem_location_device(&mut desc.location, dev);
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
    set_mem_location_device(&mut prop.location, device);

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
    assert!(granularity != 0, "granularity must be nonzero");
    let remainder = size % granularity;
    if remainder == 0 {
        size
    } else {
        size.checked_add(granularity - remainder)
            .expect("aligned size overflows usize")
    }
}

/// Granularity flavor for [`multicast_granularity`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MulticastGranularity {
    /// Minimum required granularity for multicast sizes and offsets.
    Minimum,
    /// Recommended granularity for best performance.
    Recommended,
}

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

/// Queries the multicast size/offset granularity for a team of
/// `num_devices` GPUs binding `size` bytes each.
///
/// All sizes and offsets passed to [`MulticastObject::new`] and
/// [`MulticastObject::bind_mem`] must be multiples of the
/// [`MulticastGranularity::Minimum`] value; use
/// [`MulticastGranularity::Recommended`] for best performance.
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
/// then this object, then the physical allocations. Mappings and bindings
/// borrow this object, so the compiler enforces that they go first; the
/// physical allocations are not borrowed by the bindings (CUDA keeps bound
/// memory alive), so releasing them last stays a convention.
///
/// Dropping releases the handle via `cuMemRelease` (the documented release
/// path for multicast objects).
pub struct MulticastObject {
    handle: cuda_bindings::CUmemGenericAllocationHandle,
    size: usize,
    num_devices: u32,
}

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
    /// The returned [`MulticastBinding`] unbinds on drop. It borrows this
    /// object, so it cannot outlive it; drop it before `phys` as well.
    pub fn bind_mem(
        &self,
        mc_offset: usize,
        phys: &PhysicalAllocation,
        mem_offset: usize,
        size: usize,
    ) -> Result<MulticastBinding<'_>, DriverError> {
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
            multicast: self,
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
/// on drop. It borrows the multicast object, so it cannot outlive it (an
/// unbind against a released object handle would be a wild handle); drop it
/// before the physical allocation it binds as well.
pub struct MulticastBinding<'a> {
    /// The object the memory is bound into; borrowed so the unbind runs
    /// while the handle is still valid.
    multicast: &'a MulticastObject,
    device: cuda_bindings::CUdevice,
    mc_offset: usize,
    size: usize,
}

impl<'a> MulticastBinding<'a> {
    /// Returns the device whose memory is bound.
    pub fn device(&self) -> cuda_bindings::CUdevice {
        self.device
    }

    /// Returns the multicast object the memory is bound into.
    pub fn multicast(&self) -> &'a MulticastObject {
        self.multicast
    }
}

impl Drop for MulticastBinding<'_> {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_bindings::cuMulticastUnbind(
                self.multicast.handle(),
                self.device,
                self.mc_offset,
                self.size,
            )
            .result();
        }
    }
}
