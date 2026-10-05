# Virtual Memory Management and Peer Access

CUDA Virtual Memory Management (VMM) separates physical memory allocation,
virtual address reservation, mapping, and access permissions into independent
operations.

This is lower-level than `DeviceBuffer`, which allocates memory and returns a
ready-to-use device pointer in one operation. VMM is useful when an application
needs explicit control over where physical memory is mapped or which GPUs may
access it.

Peer-to-peer access (P2P) is a related but separate mechanism. It allows one
CUDA context to access memory associated with another GPU directly over a
supported interconnect such as NVLink or PCIe.

This chapter covers:

* Querying whether two GPUs support peer access.
* Enabling and disabling directional peer access.
* Allocating physical device memory with VMM.
* Reserving virtual address ranges.
* Mapping physical allocations into those ranges.
* Granting mappings to one or more devices.
* Mapping one physical allocation into multiple GPU-visible address ranges.
* Synchronization and resource destruction order.

:::{seealso}
The complete integration test is available in
[`cuda-core/tests/simt_vmm_p2p.rs` in cutile-rs](https://github.com/NVlabs/cutile-rs/blob/main/cuda-core/tests/simt_vmm_p2p.rs).

The corresponding implementations are:

* [`cuda-core/src/simt/vmm.rs` in cutile-rs](https://github.com/NVlabs/cutile-rs/blob/main/cuda-core/src/simt/vmm.rs)
* [`cuda-core/src/simt/peer.rs` in cutile-rs](https://github.com/NVlabs/cutile-rs/blob/main/cuda-core/src/simt/peer.rs)
:::

## VMM and P2P solve different problems

VMM controls the relationship between physical memory and virtual addresses:

```text
Physical allocation
        |
        v
Virtual address reservation
        |
        v
Mapping
        |
        v
Per-device access permissions
```

P2P controls whether a CUDA context can access memory associated with another
device:

```text
Context on GPU 0
        |
        | enable_peer_access(&gpu0, &gpu1)
        v
Memory accessible through GPU 1
```

A multi-GPU application may use both mechanisms:

1. Allocate physical memory on one device.
2. Reserve virtual address ranges.
3. Map the physical allocation into those ranges.
4. Grant both devices access to the mappings.
5. Enable peer access in the required directions.

Neither operation implies the other. A VMM mapping still requires explicit
access permissions, and peer access must be enabled when a context needs direct
access to memory associated with a peer context.

## Hardware and topology requirements

Peer access depends on the physical GPU topology, the CUDA driver, and the
platform configuration. Two GPUs in the same machine are not necessarily able
to access each other directly.

Always query support before enabling peer access:

```rust
use cuda_core::simt::peer;
use cuda_core::{CudaContext, DriverError};

fn check_peer_access() -> Result<(), DriverError> {
    let ctx0 = CudaContext::new(0)?;
    let ctx1 = CudaContext::new(1)?;

    if peer::can_access_peer(&ctx0, &ctx1)? {
        println!("GPU 0 can access GPU 1");
    } else {
        println!("GPU 0 cannot access GPU 1");
    }

    Ok(())
}
```

`can_access_peer(from, to)` is only a capability query. It does not change
either context.

:::{note}
Peer access is directional. Support from GPU 0 to GPU 1 should not be treated
as proof that the reverse direction is enabled.
:::

## Enabling peer access

Use `peer::enable_peer_access` to allow one context to access memory associated
with another device:

```rust
use cuda_core::simt::peer;
use cuda_core::{CudaContext, DriverError};

fn enable_bidirectional_peer_access() -> Result<(), DriverError> {
    let ctx0 = CudaContext::new(0)?;
    let ctx1 = CudaContext::new(1)?;

    if !peer::can_access_peer(&ctx0, &ctx1)? {
        return Ok(());
    }

    peer::enable_peer_access(&ctx0, &ctx1)?;
    peer::enable_peer_access(&ctx1, &ctx0)?;

    // Both contexts may now initiate peer accesses.

    peer::disable_peer_access(&ctx0, &ctx1)?;
    peer::disable_peer_access(&ctx1, &ctx0)?;

    Ok(())
}
```

The direction is determined by the argument order:

| Call                               | Effect                                  |
| :--------------------------------- | :-------------------------------------- |
| `enable_peer_access(&ctx0, &ctx1)` | GPU 0's context may access GPU 1 memory |
| `enable_peer_access(&ctx1, &ctx0)` | GPU 1's context may access GPU 0 memory |
| Both calls                         | Bidirectional peer access               |

Calling `enable_peer_access` when access is already enabled succeeds.
Likewise, disabling access that is not currently enabled is treated as a
successful no-op.

:::{warning}
Complete all operations using peer memory before disabling peer access.
Disabling access while kernels or copies are still using peer-visible memory
can invalidate those operations.
:::

## The VMM lifecycle

A VMM allocation has four explicit stages:

1. Query the device's allocation granularity.
2. Allocate physical memory.
3. Reserve a virtual address range and map the physical memory into it.
4. Grant one or more devices access to the mapping.

The main APIs are:

| API                                         | Purpose                                   |
| :------------------------------------------ | :---------------------------------------- |
| `vmm::allocation_granularity(device)`       | Query the required allocation granularity |
| `vmm::align_size(size, granularity)`        | Round a size to that granularity          |
| `PhysicalAllocation::new(device, size)`     | Allocate physical memory                  |
| `VirtualReservation::new(size, alignment)`  | Reserve a virtual address range           |
| `Mapping::new(va, size, &physical, offset)` | Map physical memory into a VA range       |
| `vmm::set_access(va, size, devices)`        | Grant devices access to the mapped range  |

## Allocation granularity

CUDA VMM operations require allocation sizes and mapping sizes to be multiples
of the device's allocation granularity.

Use `allocation_granularity` and `align_size` before allocating:

```rust
use cuda_core::simt::vmm;
use cuda_core::{CudaContext, DriverError};

fn aligned_allocation_size(requested: usize) -> Result<usize, DriverError> {
    let ctx = CudaContext::new(0)?;
    let granularity = vmm::allocation_granularity(ctx.cu_device())?;

    Ok(vmm::align_size(requested, granularity))
}
```

Do not assume that a page-sized value such as 4 KiB satisfies the CUDA VMM
granularity requirement. The required value is device-dependent.

## Single-GPU VMM mapping

The following example allocates physical memory, maps it into a virtual address
range, writes data through the mapping, and reads it back.

```rust
use cuda_core::simt::vmm;
use cuda_core::{CudaContext, DriverError};

fn single_gpu_vmm_roundtrip() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let granularity = vmm::allocation_granularity(ctx.cu_device())?;
    let allocation_size = vmm::align_size(4096, granularity);

    // Stage 1: allocate physical memory.
    let physical =
        vmm::PhysicalAllocation::new(ctx.cu_device(), allocation_size)?;

    // Stage 2: reserve virtual address space.
    let reservation =
        vmm::VirtualReservation::new(allocation_size, 0)?;

    // Stage 3: map the physical allocation into the reserved range.
    let mapping = vmm::Mapping::new(
        reservation.base(),
        allocation_size,
        &physical,
        0,
    )?;

    // Stage 4: grant GPU 0 read/write access to the mapping.
    vmm::set_access(
        reservation.base(),
        allocation_size,
        &[ctx.cu_device()],
    )?;

    let input: Vec<u8> = (0..64).map(|i| (i * 3 + 7) as u8).collect();

    unsafe {
        cuda_core::simt::memory::memcpy_htod_async(
            reservation.base(),
            input.as_ptr(),
            input.len(),
            stream.cu_stream(),
        )?;
    }

    ctx.synchronize()?;

    let mut output = vec![0u8; input.len()];

    unsafe {
        cuda_core::simt::memory::memcpy_dtoh_async(
            output.as_mut_ptr(),
            reservation.base(),
            output.len(),
            stream.cu_stream(),
        )?;
    }

    ctx.synchronize()?;

    assert_eq!(output, input);

    // Resources must be destroyed in dependency order.
    drop(mapping);
    drop(reservation);
    drop(physical);

    Ok(())
}
```

`Mapping::new` does not by itself make the mapped address accessible. The
mapping remains unusable until `vmm::set_access` grants access to at least one
device.

## Mapping one physical allocation for two GPUs

VMM allows one physical allocation to be mapped into multiple virtual address
ranges.

The following pattern:

* Allocates physical memory on GPU 0.
* Creates one virtual address mapping for GPU 0.
* Creates another virtual address mapping for GPU 1.
* Grants both devices access to both mappings.
* Writes through GPU 0's mapping.
* Reads the same physical memory through GPU 1's mapping.

```rust
use cuda_core::simt::{peer, vmm};
use cuda_core::{CudaContext, DriverError, IntoResult};
use std::mem::MaybeUninit;

fn gpu_count() -> Result<usize, DriverError> {
    unsafe {
        cuda_core::init(0)?;
    }

    let mut count = MaybeUninit::uninit();

    unsafe {
        cuda_core::sys::cuDeviceGetCount(count.as_mut_ptr()).result()?;
        Ok(count.assume_init() as usize)
    }
}

fn cross_gpu_vmm_roundtrip() -> Result<(), DriverError> {
    if gpu_count()? < 2 {
        println!("This example requires at least two CUDA devices.");
        return Ok(());
    }

    let ctx0 = CudaContext::new(0)?;
    let ctx1 = CudaContext::new(1)?;

    if !peer::can_access_peer(&ctx0, &ctx1)? {
        println!("The selected devices do not support peer access.");
        return Ok(());
    }

    peer::enable_peer_access(&ctx0, &ctx1)?;
    peer::enable_peer_access(&ctx1, &ctx0)?;

    let granularity =
        vmm::allocation_granularity(ctx0.cu_device())?;
    let allocation_size =
        vmm::align_size(4096, granularity);

    // The physical memory is allocated only once, on GPU 0.
    let physical =
        vmm::PhysicalAllocation::new(
            ctx0.cu_device(),
            allocation_size,
        )?;

    // GPU 0 virtual address range.
    let reservation0 =
        vmm::VirtualReservation::new(allocation_size, 0)?;

    let mapping0 = vmm::Mapping::new(
        reservation0.base(),
        allocation_size,
        &physical,
        0,
    )?;

    vmm::set_access(
        reservation0.base(),
        allocation_size,
        &[ctx0.cu_device(), ctx1.cu_device()],
    )?;

    // GPU 1 virtual address range backed by the same physical allocation.
    let reservation1 =
        vmm::VirtualReservation::new(allocation_size, 0)?;

    let mapping1 = vmm::Mapping::new(
        reservation1.base(),
        allocation_size,
        &physical,
        0,
    )?;

    vmm::set_access(
        reservation1.base(),
        allocation_size,
        &[ctx0.cu_device(), ctx1.cu_device()],
    )?;

    let input: Vec<u32> = (0..256).collect();
    let byte_length =
        input.len() * std::mem::size_of::<u32>();

    let stream0 = ctx0.default_stream();
    ctx0.bind_to_thread()?;

    unsafe {
        cuda_core::simt::memory::memcpy_htod_async(
            reservation0.base(),
            input.as_ptr(),
            byte_length,
            stream0.cu_stream(),
        )?;
    }

    ctx0.synchronize()?;

    let mut output = vec![0u32; input.len()];
    let stream1 = ctx1.default_stream();
    ctx1.bind_to_thread()?;

    unsafe {
        cuda_core::simt::memory::memcpy_dtoh_async(
            output.as_mut_ptr(),
            reservation1.base(),
            byte_length,
            stream1.cu_stream(),
        )?;
    }

    ctx1.synchronize()?;

    assert_eq!(output, input);

    // Destroy mappings before their reservations and backing allocation.
    drop(mapping1);
    drop(mapping0);
    drop(reservation1);
    drop(reservation0);
    drop(physical);

    peer::disable_peer_access(&ctx0, &ctx1)?;
    peer::disable_peer_access(&ctx1, &ctx0)?;

    Ok(())
}
```

## Virtual addresses are not guaranteed to match

`VirtualReservation::new(size, 0)` asks the CUDA driver to select a virtual
address. Two reservations backed by the same physical allocation may therefore
have different base addresses:

```text
GPU 0 VA: 0x0000_7f10_0000_0000
                         \
                          same physical allocation
                         /
GPU 1 VA: 0x0000_7e80_0000_0000
```

The current API supports shared physical backing, but it does not guarantee
that every GPU observes that memory at the same numerical virtual address.

Code must use the `base()` returned by the reservation associated with the
current access path.

:::{warning}
Do not describe this pattern as a complete symmetric heap. A symmetric heap
requires equivalent virtual addresses across participating devices, which the
current driver-selected reservation API does not guarantee.
:::

## Ownership and destruction order

The VMM wrapper types own CUDA resources through RAII:

| Type                 | CUDA resource                      | Drop operation     |
| :------------------- | :--------------------------------- | :----------------- |
| `PhysicalAllocation` | Generic physical allocation handle | `cuMemRelease`     |
| `VirtualReservation` | Reserved virtual address range     | `cuMemAddressFree` |
| `Mapping`            | Physical-to-virtual mapping        | `cuMemUnmap`       |

The dependency order is:

```text
PhysicalAllocation   VirtualReservation
        ^                    ^
         \                  /
          \                /
               Mapping
```

A `Mapping` consumes one `PhysicalAllocation` and one `VirtualReservation`,
so both must outlive it.

Destroy resources in the reverse order:

```rust
ctx0.synchronize()?;
ctx1.synchronize()?;

drop(mapping1);
drop(mapping0);

drop(reservation1);
drop(reservation0);

drop(physical);
```

The complete teardown sequence for a multi-GPU operation is:

1. Stop submitting operations that use the mappings.
2. Synchronize every stream or context using them.
3. Drop all `Mapping` values.
4. Drop all `VirtualReservation` values.
5. Drop the `PhysicalAllocation`.
6. Disable peer access if it is no longer needed.

:::{warning}
Rust does not encode the mapping dependencies in the type system. A `Mapping`
does not borrow its `PhysicalAllocation` or `VirtualReservation` for its full
lifetime. The application is responsible for preserving the required drop
order.
:::

## Do not wrap VMM pointers in an owning buffer

A VMM virtual address is owned by its `Mapping`, `VirtualReservation`, and
`PhysicalAllocation`.

Do not construct an owning `DeviceBuffer` from that pointer unless the API
explicitly defines a non-owning conversion. An owning buffer could attempt to
release the pointer using `cuMemFree`, while the VMM resources expect
`cuMemUnmap`, `cuMemAddressFree`, and `cuMemRelease`.

Pass the raw `CUdeviceptr` only to APIs that accept non-owning device
addresses.

## Error handling and unsupported systems

A portable multi-GPU application should treat these conditions separately:

| Condition                         | Recommended behavior                      |
| :-------------------------------- | :---------------------------------------- |
| Fewer than two devices            | Skip the P2P path                         |
| `can_access_peer` returns `false` | Use host-staged transfers or another path |
| VMM allocation fails              | Report the CUDA driver error              |
| Access permission setup fails     | Destroy mappings and fall back            |
| Peer access is already enabled    | Continue                                  |
| Peer access is already disabled   | Continue                                  |

The cuda-oxide integration test returns early when the machine has fewer than
two GPUs or when the selected topology does not support P2P. This allows the
test suite to run on single-GPU development systems while still exercising the
feature on compatible multi-GPU hosts.

## API summary

### VMM

| API                                        | Description                                  |
| :----------------------------------------- | :------------------------------------------- |
| `vmm::allocation_granularity(device)`      | Required size granularity for VMM operations |
| `vmm::align_size(size, granularity)`       | Round a requested size upward                |
| `PhysicalAllocation::new(device, size)`    | Allocate physical memory on a device         |
| `PhysicalAllocation::handle()`             | Return the raw allocation handle             |
| `PhysicalAllocation::size()`               | Return the physical allocation size          |
| `VirtualReservation::new(size, alignment)` | Reserve a virtual address range              |
| `VirtualReservation::base()`               | Return the reserved base address             |
| `VirtualReservation::size()`               | Return the reserved range size               |
| `Mapping::new(va, size, physical, offset)` | Map physical memory into a virtual range     |
| `Mapping::va()`                            | Return the mapped virtual address            |
| `Mapping::size()`                          | Return the mapped size                       |
| `vmm::set_access(va, size, devices)`       | Grant read/write access to selected devices  |

### P2P

| API                                   | Description                     |
| :------------------------------------ | :------------------------------ |
| `peer::can_access_peer(from, to)`     | Query topology support          |
| `peer::enable_peer_access(from, to)`  | Enable directional peer access  |
| `peer::disable_peer_access(from, to)` | Disable directional peer access |

## Next steps

For production multi-GPU code, add an explicit fallback path for systems where
P2P is unavailable. A common fallback copies data through pinned host memory
instead of assuming that every multi-GPU topology supports direct access.
