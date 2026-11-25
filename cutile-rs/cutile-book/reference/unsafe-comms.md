# Unsafe GPU communication

The CUDA 13.5 communication surface provides peer-memory access and multicast
operations. Kernels use `cutile::core::*` for memory operations and ordering
types, and `cutile::tileir::*` for the new pointer classifications and multicast
operations. The host owns allocation, mapping, access permissions, and lifetime.

These are low-level unsafe APIs. They expose the synchronization primitives
needed to build communication protocols; they do not insert an inter-GPU
barrier automatically.

## Unicast: access one GPU's memory

`unsafe ptr_with_attr_unicast(pointer)` classifies a pointer as an NVLink peer
pointer. It preserves the address and does not allocate memory, enable peer
access, or grant permission. Reshape, broadcast, and offset operations preserve
the classification.

Use the classified pointer with the existing unsafe memory operations:

| Operation | Purpose | Ordering choices |
|---|---|---|
| `load_ptr_tko` | Read peer elements | Weak, Relaxed, Acquire |
| `store_ptr_tko` | Write peer elements | Weak, Relaxed, Release |
| `atomic_rmw_tko` | Atomic update, returning old values | Relaxed, Acquire, Release, AcqRel |
| `atomic_cas_tko` | Atomic compare-and-swap | Relaxed, Acquire, Release, AcqRel |

Views built over a unicast pointer retain its classification and support the
ordinary view memory operations, subject to their type and shape restrictions.
The concurrent unicast example exercises pointer loads, stores, and integer
atomic adds; it does not claim execution coverage of every atomic mode or type.

For operations coordinating different GPUs, choose `scope::System`. Device
scope covers the issuing device and is insufficient for an inter-GPU
release/acquire handshake. Atomics require peer-native atomic support. The
pointer's target must be accessible to the issuing GPU for the entire launch.

## Multicast: access corresponding memory on several GPUs

`unsafe ptr_with_attr_multicast(pointer)` requires an actual CUDA multicast
mapping. Ordinary allocations cannot become multicast allocations just by
adding this classification. Ordinary loads, stores and atomics reject multicast
pointers; build a partition view and use these operations instead:

| Operation | Effect | Ordering choices |
|---|---|---|
| `multimem_load_reduce_view_tko` | Read corresponding elements from all participants and return their reduction | Weak, Relaxed, Acquire |
| `multimem_reduce_view_tko` | Atomically combine the supplied tile into each participant's backing memory | Relaxed, Release |
| `multimem_store_view_tko` | Write the supplied tile into each participant's backing memory | Weak, Relaxed, Release |

Weak accesses assume no concurrent access to the addressed locations and omit the encoded
memory scope. Relaxed/acquire/release accesses require a scope. Use system scope
for the inter-GPU cases described here. Atomic reduction is atomic per accessed
element; a whole tile or all participants are not one atomic transaction.

| Reduction | Load-reduce types | Atomic-reduce types |
|---|---|---|
| `multimem::Add` | i32, i64, u64, f16, bf16, f32, f64 | Same |
| `multimem::Min`, `Max` | i32, i64, f16, bf16 | i32, i64 |
| `multimem::Umin`, `Umax` | i32, i64, u64, interpreted as unsigned | Same |
| `multimem::And`, `Or`, `Xor` | i32, i64, u64 | Same |

Rust `u64` uses the same Tile IR `i64` representation as Rust `i64`; 64-bit
addition has the same result bits for both types and wraps modulo 2^64.
Integer Min/Max use signed comparisons regardless of the Rust type; use
Umin/Umax for unsigned comparisons. Multicast stores support i32, i64, u64,
f16, bf16, f32, and f64. Floating-point tiles must occupy a multiple of four bytes; f16 and
bf16 views additionally require exactly one statically unit-stride dimension.
Each true `inbounds` entry is the caller's promise that the corresponding tile
coordinate is valid for every program instance.

## Synchronization contracts

The primitives are already present:

| Primitive | What it establishes |
|---|---|
| Release store or atomic, paired with an acquire load or atomic using system scope | Publishes preceding accesses to a peer that observes the signal |
| `Token`, `Some(token)`, `join_tokens` | Dependencies between accesses in one Tile program |
| `unsafe memory_fence_alias_tko(token)` | Alias-proxy ordering when accessing the same physical storage through different virtual aliases |

A token alone does not signal another GPU. An alias fence alone does not wait
for another GPU. A polling load must have the proper acquire ordering and its
returned token must feed dependent reads. Chain the producer's data operation
to its release signal as well; source-code order alone is insufficient.

For example, each GPU can write its contribution, release-store a sequence
number into its peer's inbox, acquire-load its own inbox until the expected
number arrives, and then read the contributions. A second exchange after the
reads prevents either producer from reusing the buffer too early. Monotonic
sequence numbers distinguish repeated uses of the same inbox.

The example includes explicit alias fences around transitions between ordinary
and multicast mappings. Both kernels launch before the host waits for either
one. A sequential pair of blocking launches would deadlock while the first
kernel waited for an unlaunched peer.

## Host mappings

The example uses the existing `cuda_core::vmm` wrappers:

| Wrapper / function | Responsibility |
|---|---|
| `PhysicalAllocation::new` | Allocate backing memory on a selected GPU |
| `VirtualReservation::new`, `Mapping::new` | Reserve and map an ordinary virtual address |
| `set_access` | Grant accessing devices read/write permission |
| `MulticastObject::new`, `add_device`, `bind_mem` | Create a multicast team and bind each participant's backing storage |
| `Mapping::new_multicast` | Map the multicast object into an address range |

Query and honor allocation/multicast granularity. Add every multicast
participant before binding backing memory. Retain mappings, bindings and buffers
until every submitted kernel completes. Unmap before releasing the associated
virtual address reservation. The example grants both devices access to each
ordinary mapping; the multicast mode additionally maps the multicast object.
The unicast mode does not require or create a multicast object.

## Support and runnable example

The current Rust branch enables both classifications on `sm_100`, `sm_103`,
and `sm_107` with Tile IR 13.5. Unicast requires an NVLink path; multicast also
requires an NVLink SHARP-capable NVSwitch. The selected `tileiras` must use a
matching 13.5 backend. See [CUDA and Tile IR compatibility](compatibility.md).

From `cutile-rs/`, with that toolchain activated:

```bash
cargo run -p cutile --example unsafe_comms -- unicast
cargo run -p cutile --example unsafe_comms -- multicast
```

The source is `cutile/examples/unsafe_comms.rs`. An optional second argument
sets the round count, for example `-- unicast 1024`. This is a correctness
example with one CTA per GPU, not a throughput benchmark or a reusable
collective abstraction.

Each mode runs 256 rounds by default and repeats with the launch order reversed:

- **Unicast:** both GPUs read peer inputs, write peer outputs, and atomically add
  into local and peer memory. The host checks every phase on both devices.
- **Multicast:** both GPUs load-reduce their contributions and atomically reduce
  into both backing allocations; the broadcast producer alternates each round.
  The host checks every phase and both final backing allocations.

The host precompiles both launches, initializes the inboxes, and poisons the
output snapshots. Between the first and last round, all coordination occurs in
the kernels. The polling loop is bounded and asserts if the peer fails to
signal. Unsupported hardware and execution failures are errors, not skips.

The hardware tests call the example's implementation directly:

```bash
cargo test -p cutile --test tile_ir_13_5_comms \
  concurrent_signaling_and_buffer_reuse -- --ignored --test-threads=1 --nocapture
```

CPU verification runs normally; assembler verification is separate and needs
no GPU:

```bash
cargo test -p cutile --test tile_ir_13_5_comms
CUTILE_BYTECODE_VERSION=13.5 cargo test -p cutile --test tile_ir_13_5_comms \
  assemble_concurrent -- --ignored
```
