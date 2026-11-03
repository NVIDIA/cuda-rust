# Overlapping Transfers and Compute

Pageable host memory is convenient, but CUDA may stage it through internal
page-locked memory before a DMA transfer. That extra staging work limits
asynchronous copies. `PinnedHostBuffer<T>` gives CUDA a stable host address for
direct transfer. True copy/compute overlap also requires asynchronous copies
on non-default streams.

Run the complete demo with:

```bash
cargo oxide run pinned_overlap
```

The example uses a cheap in-place increment kernel, so its timing is dominated
by data movement rather than computation.

## Choosing a copy API

The safe helper synchronizes the stream before returning. Use it when the host
does not need to do other work while the copy is in flight:

```rust
device.copy_to_pinned_host(&stream, &mut host)?;
// `host` is safe to read because the helper synchronized `stream`.
```

The async helper returns before the copy finishes. The caller owns the
synchronization and the buffer lifetime:

```rust
// SAFETY: `host` and `device` stay alive and are not reused until `done` fires.
unsafe { device.copy_to_pinned_host_async(&stream, &mut host)? };
let done = stream.record_event(None)?;
```

The caller must keep both buffers alive, must not read or mutate an in-flight
host buffer, and must wait for the stream work before dropping or reusing it.
The same rule applies to `copy_from_pinned_host_async`. Pre-allocate device
buffers and refill them; `from_pinned_host` still performs a synchronous device
allocation before enqueuing its copy.

## Rotating pinned staging buffers

Operations stay ordered within one stream, while independent streams can make
progress concurrently on hardware that supports concurrent copy and compute:

```text
stream 0: upload 0 -> kernel 0 -> download 0 -> upload 3 -> kernel 3
stream 1: upload 1 -> kernel 1 -> download 1 -> upload 4 -> kernel 4
stream 2: upload 2 -> kernel 2 -> download 2 -> upload 5 -> kernel 5
```

The example gives each stream a persistent `DeviceBuffer`, a pinned staging
buffer, and a completion event. Upload, kernel, and download are already
ordered by the stream, so the event is only needed to protect host-side slot
reuse. Before reusing a slot, the host waits for its event, copies out the
completed result, and then refills the staging buffer. The final stream join
drains all work before CUDA-owned memory is dropped.

The core enqueue sequence inside the example's loop is shown below; surrounding
setup and result handling are omitted:

```rust
// SAFETY: the slot buffers stay alive and are not reused until completion.
unsafe {
    devices[slot].copy_from_pinned_host_async(&streams[slot], &stagers[slot])?;
    module.increment(&streams[slot], config, &mut devices[slot])?;
    devices[slot].copy_to_pinned_host_async(&streams[slot], &mut stagers[slot])?;
}
completions[slot] = Some(streams[slot].record_event(None)?);
```

The complete setup, slot-reuse wait, and final drain are in the
[`run_overlapped` implementation](https://github.com/NVIDIA/cuda-rust/blob/main/cuda-oxide/crates/rustc-codegen-cuda/examples/pinned_overlap/src/main.rs).

## Measured impact

The bandwidth benchmark warms up three times, records ten CUDA-event samples,
and reports their median. The pipeline benchmark measures host wall time for
eight 32 MiB chunks. The serialized path allocates and synchronizes a pageable
device buffer for each chunk. The overlapped path pre-allocates three device
buffers and pinned staging buffers, then includes host staging copies, event
waits, and result copies in its wall time. The result is therefore an
end-to-end path comparison, not a pure copy-engine measurement.

On the reference RTX 4090 (`sm_89`) run:

| Pipeline | Time |
|---|---:|
| Serialized pageable | 174.996 ms |
| Overlapped pinned | 99.332 ms |
| Speedup | 1.76x |

The full transfer table is in the [example README](https://github.com/NVIDIA/cuda-rust/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/pinned_overlap).
Results vary with the GPU, PCIe topology, driver, and system load.

## Practical tradeoffs

Pinning consumes limited host memory, and copying pageable input into a staging
buffer is not free. Reuse a small ring of buffers and profile the copy engines
before increasing the number of streams.

For the lower-level stream model, see
[Scheduling and Streams](scheduling-and-streams.md). For the complete runnable
implementation, see the [`pinned_overlap` example](https://github.com/NVIDIA/cuda-rust/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/pinned_overlap).
