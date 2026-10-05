# cluster

## Thread Block Clusters - Hopper (sm_90+) Cooperative Groups

Demonstrates Thread Block Clusters, a Hopper feature that enables direct shared memory access between blocks. Multiple blocks form a cluster that can share data without going through global memory.

## What This Example Does

1. **test_cluster_compile_time**: Compile-time cluster configuration with `#[cluster_launch]`
2. **test_cluster_intrinsics**: Cluster special registers (ctaid, nctaid, rank, size)
3. **test_cluster_sync**: Cluster-wide synchronization
4. **test_dsmem_ring_exchange**: Distributed shared memory read through `map_shared_rank` + dereference
5. **test_dsmem_reduction**: All-to-one reduction using `dsmem_read_u32`
6. **test_dsmem_mapped_store**: Distributed shared memory write through `map_shared_rank_mut` + dereference

## Key Concepts Demonstrated

### Compile-Time Cluster Configuration

```rust
#[kernel]
#[cluster_launch(4, 1, 1)]  // 4 blocks per cluster
pub fn test_cluster_compile_time(mut output: DisjointSlice<u32>) {
    let my_rank = cluster::block_rank();      // 0-3 within cluster
    let cluster_size = cluster::cluster_size();  // 4

    // Write cluster info: high 16 bits = rank, low 16 bits = cluster_size
    if thread::threadIdx_x() == 0 {
        let idx = my_rank as usize;
        if idx < output.len() {
            let value = ((my_rank as u32) << 16) | (cluster_size as u32);
            unsafe { *output.get_unchecked_mut(idx) = value };
        }
    }
}
```

Generates PTX:

```ptx
.entry test_cluster_compile_time
    .explicitcluster
    .reqnctapercluster 4, 1, 1
```

### Cluster Intrinsics

```rust
// Position within cluster
let ctaid_x = cluster::cluster_ctaidX();   // Block's X in cluster
let ctaid_y = cluster::cluster_ctaidY();
let ctaid_z = cluster::cluster_ctaidZ();

// Cluster dimensions
let nctaid_x = cluster::cluster_nctaidX();  // Cluster size in X
let nctaid_y = cluster::cluster_nctaidY();
let nctaid_z = cluster::cluster_nctaidZ();

// Derived values
let rank = cluster::block_rank();       // Linear block index in cluster
let size = cluster::cluster_size();     // Total blocks in cluster
```

### Cluster Synchronization

```rust
// Write to local shared memory
SHMEM[0] = my_rank * 100 + 42;
thread::sync_threads();

// Synchronize ENTIRE cluster (all 4 blocks)
cluster::cluster_sync();

// Now all blocks have written their data
```

### Distributed Shared Memory (DSMEM)

`map_shared_rank` maps a CTA-shared pointer to the same offset in another rank. The mapped result is represented as LLVM `addrspace(7)`, so an ordinary Rust dereference lowers through cluster shared memory.

```rust
// Each block writes to its own shared memory
SHMEM[0] = 1000 + my_rank;  // Block 0: 1000, Block 1: 1001, etc.
thread::sync_threads();
cluster::cluster_sync();

// Read ANOTHER block's shared memory through a mapped cluster-shared pointer.
let neighbor_rank = (my_rank + 1) % cluster_size;
let neighbor_ptr =
    cluster::map_shared_rank(addr_of!(SHMEM) as *const u32, neighbor_rank);
let neighbor_value = *neighbor_ptr;

// Block 0 reads 1001, Block 1 reads 1002, Block 2 reads 1003, Block 3 reads 1000
```

For a mutable mapping, the same address-space semantics apply:

```rust
let neighbor_ptr =
    cluster::map_shared_rank_mut(addr_of_mut!(SHMEM) as *mut u32, neighbor_rank);
*neighbor_ptr = 3000 + my_rank;
```

`dsmem_read_u32` remains available as a fixed-width convenience intrinsic. It combines the mapping and remote load in one inline-PTX block:

```rust
let neighbor_value =
    cluster::dsmem_read_u32(addr_of!(SHMEM) as *const u32, neighbor_rank);
```

Both approaches require the target CTA to remain live and require cluster synchronization appropriate to the data dependency.

## Build and Run

```bash
cargo oxide run cluster
```

## Expected Output

### On sm_90+ (Hopper, Blackwell):

```text
=== Thread Block Cluster Tests (sm_90+) ===

GPU Compute Capability: sm_100

=== Test 0: Compile-Time Cluster Configuration ===
Launching test_cluster_compile_time via cuLaunchKernelEx
  Grid: 4x1x1, Block: 32, Cluster: 4x1x1
Results (each block writes: (rank << 16) | cluster_size):
  Block 0: raw=0x00000004, rank=0, cluster_size=4
  ...
✓ Compile-time cluster config test PASSED

=== Test 2: Cluster Synchronization ===
Results: [42, 142, 242, 342]
✓ Cluster sync test PASSED

=== Test 3: DSMEM Ring Exchange (cluster launch) ===
Results (each block reads neighbor's value):
  Block 0: got 1001, expected 1001 ✓
  Block 1: got 1002, expected 1002 ✓
  Block 2: got 1003, expected 1003 ✓
  Block 3: got 1000, expected 1000 ✓
✓ DSMEM ring exchange PASSED

=== Test 4: DSMEM Reduction (cluster launch) ===
Result: 100, expected: 100
✓ DSMEM reduction PASSED

=== Test 5: DSMEM Mapped Remote Store (cluster launch) ===
Results (each block observes the write from its previous rank):
  Block 0: got 3003, expected 3003 ✓
  Block 1: got 3000, expected 3000 ✓
  Block 2: got 3001, expected 3001 ✓
  Block 3: got 3002, expected 3002 ✓
✓ DSMEM mapped remote store PASSED

🎉 All cluster + DSMEM tests PASSED!
```

### On Pre-Hopper (sm_80 and earlier):

```text
GPU Compute Capability: sm_86

⚠ WARNING: Thread Block Clusters require sm_90+ (Hopper)
⚠ Your GPU is sm_86. Tests may not run correctly.
⚠ Continuing anyway to verify PTX generation...
```

## Hardware Requirements

- **Required GPU**: Hopper H100, H200 or Blackwell B100, B200 (sm_90+)
- **CUDA Driver**: 12.0+ for cluster launch support
- **Cluster launch**: Uses `cuLaunchKernelEx` through the typed `#[cuda_module]` launch method

## Cluster vs Traditional CUDA

| Feature          | Traditional         | Cluster                    |
|------------------|---------------------|----------------------------|
| Shared Memory    | Per-block (private) | Distributed (accessible)   |
| Sync scope       | Block               | Cluster                    |
| Max cooperation  | 1024 threads        | 4+ blocks × 1024 threads   |
| Memory sharing   | Via global mem      | Direct DSMEM access        |

## DSMEM Use Cases

1. **Neighbor exchange**: Halo regions for stencils
2. **Reduction**: Combine results across blocks
3. **Producer-consumer**: Pipeline between blocks
4. **Load balancing**: Work stealing between blocks

## Cluster Intrinsic Reference

| Intrinsic                    | PTX                             | Description                                      |
|------------------------------|---------------------------------|--------------------------------------------------|
| `cluster_ctaidX/Y/Z()`       | `mov.u32 %r, %clusterctaid.x`   | Block position in cluster                        |
| `cluster_nctaidX/Y/Z()`      | `mov.u32 %r, %clusternctaid.x`  | Cluster dimensions                               |
| `block_rank()`               | Computed                        | Linear block index                               |
| `cluster_size()`             | Computed                        | Total blocks in cluster                          |
| `cluster_sync()`             | `barrier.cluster.sync.aligned`  | Cluster-wide barrier                             |
| `map_shared_rank(ptr, rank)` | `mapa.shared::cluster`          | `addrspace(7)` pointer to another block's SMEM   |
| `dsmem_read_u32(ptr, rank)`  | `mapa` + `ld.shared::cluster`   | Read u32 from other block's SMEM                 |

## Generated PTX

```ptx
// Cluster-enabled entry point
.entry test_cluster_compile_time
    .explicitcluster
    .reqnctapercluster 4, 1, 1
{
    // Read cluster position
    mov.u32 %r1, %cluster_ctaid.x;

    // Cluster-wide sync
    barrier.cluster.arrive.aligned;
    barrier.cluster.wait.aligned;

    // dsmem_read_u32: combined mapa + ld.shared::cluster
    {
        .reg .u64 %mapped;
        mapa.shared::cluster.u64 %mapped, %rd_local_shmem, %r_neighbor_rank;
        ld.shared::cluster.u32 %r_result, [%mapped];
    }
}
```

An ordinary dereference through `map_shared_rank` selects a cluster-shared load:

```ptx
mapa.shared::cluster.u64 %rd_mapped, %rd_local_shmem, %r_neighbor_rank;
ld.shared::cluster.u32 %r_result, [%rd_mapped];
```

A mutable mapped pointer similarly selects a cluster-shared store:

```ptx
mapa.shared::cluster.u64 %rd_mapped, %rd_local_shmem, %r_neighbor_rank;
st.shared::cluster.u32 [%rd_mapped], %r_value;
```

## Potential Errors

| Error                        | Cause                                    | Solution                                  |
|------------------------------|------------------------------------------|-------------------------------------------|
| `CUDA_ERROR_NOT_SUPPORTED`   | Pre-Hopper GPU                           | Use sm_90+ hardware                       |
| `CUDA_ERROR_INVALID_VALUE`   | Cluster dims > max                       | Check device limits                       |
| `CUDA_ERROR_ILLEGAL_ADDRESS` | Invalid mapped pointer or dead target CTA | Keep target CTA live and validate pointer |
| `CUDA_ERROR_LAUNCH_FAILED`   | Blocks exited during DSMEM               | Add `cluster_sync()` before return        |
| DSMEM read/write wrong value | Missing or incorrect synchronization     | Add the required cluster synchronization  |

## Cluster Configuration Options

| Method                       | When to Use                         |
|------------------------------|-------------------------------------|
| `#[cluster_launch(x,y,z)]`   | Compile-time fixed cluster size     |
| cuLaunchKernelEx             | Runtime-configurable cluster size   |

For most cases, compile-time configuration with `#[cluster_launch]` is simpler and ensures the kernel is correctly marked for cluster execution.
