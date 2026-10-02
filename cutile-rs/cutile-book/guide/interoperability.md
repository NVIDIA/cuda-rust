# Interoperability

cuTile Rust can launch external CUDA kernels and borrow handles from other
libraries:

- **Integrating external PTX or CUBIN kernels** — for CUDA C++ kernels, cuda-oxide-generated PTX, or other CUDA module artifacts that you want to launch alongside cuTile kernels.
- **Borrowing foreign CUDA handles** — wrap a `CUcontext` / `CUstream` from another Rust binding crate (cudarc, Candle, hand-rolled FFI) so cuTile kernels can run on handles you already own.
- **Migrating from other tile DSLs** — conceptual mapping from Triton and cuTile Python.

External CUDA kernels participate in the same `DeviceOp` execution model as tile kernels — sharing streams, chaining with `.then()`, and avoiding unnecessary synchronization.

---

## Integrating External PTX or CUBIN Kernels

### From CUDA C++

Compile your CUDA C++ kernel to PTX (portable) or a `.cubin` (architecture-specific):

```bash
# PTX — portable across GPU architectures, JIT-compiled at load time.
nvcc -ptx -arch=compute_80 my_kernel.cu -o my_kernel.ptx

# cubin — pre-compiled for a single architecture, no JIT overhead.
nvcc -cubin -arch=sm_80 my_kernel.cu -o my_kernel.cubin
```

> **Architecture portability:** A `.cubin` file only runs on the exact SM architecture it was compiled for. Code compiled with `-arch=sm_80` will not load on an `sm_100` GPU. PTX avoids this problem — the CUDA driver JIT-compiles it for the target GPU at load time, at the cost of a one-time compilation delay. Prefer PTX unless you need to eliminate JIT overhead.

### From cuda-oxide

[cuda-oxide](https://github.com/NVlabs/cuda-oxide) is an NVlabs experimental Rust-to-CUDA compiler for SIMT-style kernels. Its documented output path is PTX: build or inspect the generated PTX with `cargo oxide build` or `cargo oxide pipeline`, then load that PTX with `load_module_from_ptx` and wrap the entry function in an `AsyncKernelLaunch` or typed `DeviceOp`.

If you need a CUBIN, compile the PTX through the normal CUDA toolchain as a separate step.

### Loading and Launching

Load the compiled module and get a handle to the entry function:

```rust
use cuda_async::device_context::{load_module_from_file, load_module_from_ptx};

// From cubin:
let module = load_module_from_file("my_kernel.cubin", device_id)?;

// Or from PTX (JIT-compiled at load time):
let ptx_src = include_str!("my_kernel.ptx");
let module = load_module_from_ptx(ptx_src, device_id)?;

let function = Arc::new(module.load_function("my_kernel_entry")?);
```

Launch the kernel via `AsyncKernelLaunch`, which is a `DeviceOp` wrapping the CUDA driver's kernel launch API:

```rust
use cuda_async::launch::AsyncKernelLaunch;
use cuda_core::LaunchConfig;

let mut launcher = AsyncKernelLaunch::new(function.clone());
launcher.push_arg(num_elements as u32);
launcher.push_arg(scale);
// SAFETY: input and output are valid device allocations with at least
// num_elements f32 elements. output is exclusively written; input is
// read-only. Both remain allocated until this operation completes.
unsafe {
    launcher
        .push_device_ptr(input.device_pointer().cu_deviceptr())
        .push_device_ptr(output.device_pointer().cu_deviceptr());
}
launcher.set_launch_config(LaunchConfig {
    grid_dim: ((num_elements as u32 + 255) / 256, 1, 1),
    block_dim: (256, 1, 1),
    shared_mem_bytes: 0,
});

// Execute as a DeviceOp — integrates with the async model.
launcher.await?;
```

Scalar arguments implementing `DType` use the safe `push_arg` method. Device
pointers require `unsafe { push_device_ptr() }`. The caller must check that
allocations are valid on the target GPU, large enough for the kernel, and free
of conflicting accesses. Argument order and types must match the kernel's
signature. Rust and the CUDA driver cannot validate these conditions; getting
them wrong can cause undefined behavior or GPU faults.

To prevent data races, use stream ordering: operations chained with `.then()` on the same stream execute in order and see each other's writes. Operations on different streams require explicit synchronization.

> **Why generated cuTile kernels don't require `unsafe`:** the `#[cutile::entry]` macro generates launchers that call `push_device_ptr` internally, but they do so safely because the framework controls both sides — device pointers come from framework-managed allocations, and the ownership model (`Partition` for exclusive access, `Arc<Tensor>` for shared reads) prevents aliasing at the type level. Custom kernels bypass this: you are pushing pointers the framework didn't allocate and can't track, so the safety burden falls on you.

---

## Wrapping as a Safe DeviceOp

Wrap a custom kernel launch in a struct that implements `DeviceOp`. The struct's typed fields enforce the correct argument signature; `unsafe` is confined to `execute`:

```rust
use cuda_async::device_context::with_default_device_policy;
use cuda_async::device_future::DeviceFuture;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_async::error::DeviceError;
use cuda_async::launch::AsyncKernelLaunch;
use cuda_async::scheduling_policies::SchedulingPolicy;
use cuda_core::{Function, LaunchConfig};
use std::future::IntoFuture;

pub struct ScaleKernel {
    function: Arc<Function>,
    n: u32,
    scale: f32,
    input: Arc<Tensor<f32>>,
    output: Tensor<f32>,
}

impl DeviceOp for ScaleKernel {
    type Output = (Arc<Tensor<f32>>, Tensor<f32>);

    // execute is unsafe because it enqueues async GPU work without
    // synchronizing — the returned tensors may still be in-flight.
    unsafe fn execute(
        self,
        ctx: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let mut launcher = AsyncKernelLaunch::new(self.function);
        launcher.push_arg(self.n);
        launcher.push_arg(self.scale);
        // SAFETY: input and output are framework-managed Tensor allocations.
        // input is shared (Arc, read-only); output is exclusively written.
        unsafe {
            launcher
                .push_device_ptr(self.input.device_pointer().cu_deviceptr())
                .push_device_ptr(self.output.device_pointer().cu_deviceptr());
        }
        launcher.set_launch_config(LaunchConfig {
            grid_dim: ((self.n + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        });
        unsafe { launcher.execute(ctx)? };
        Ok((self.input, self.output))
    }
}

// IntoFuture is a supertrait of DeviceOp. Every custom DeviceOp
// needs this boilerplate to enable `.await` and `.sync()`.
impl IntoFuture for ScaleKernel {
    type Output = Result<(Arc<Tensor<f32>>, Tensor<f32>), DeviceError>;
    type IntoFuture = DeviceFuture<(Arc<Tensor<f32>>, Tensor<f32>), ScaleKernel>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            Ok(DeviceFuture::scheduled(self, ExecutionContext::new(stream)))
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}
```

This is the same pattern the `#[cutile::entry]` macro uses to generate safe launchers for tile kernels — no `unsafe` at the call site. Once wrapped, the custom kernel composes with tile kernels through the usual `DeviceOp` combinators. This pipeline runs a tile add (`z = x + y`), then the wrapped scale kernel (`w = scale * z`):

```rust
let (z_part, _x, _y) =
    tile_add::add(z.partition([tile_size]), x.clone(), y.clone()).await?;
let z: Tensor<f32> = z_part.unpartition();

let w: Tensor<f32> = zeros(&[num_elements]).await?;
let (_z, w) = ScaleKernel {
    function: scale_function,
    n: num_elements as u32,
    scale,
    input: Arc::new(z),
    output: w,
}
.await?;
```

See [`interop.rs`](https://github.com/NVlabs/cutile-rs/blob/main/cutile-examples/examples/interop.rs) for a complete, runnable version.

---

## Low-Level Driver Access

For more direct control, use `with_context` to access the CUDA stream and issue driver API calls directly:

```rust
use cuda_async::device_operation::{with_context, value, DeviceOp};
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{malloc_async, memcpy_htod_async, free_async};

let host_data: Vec<f32> = vec![1.0; num_elements];
let num_bytes = num_elements * std::mem::size_of::<f32>();

// host_data is captured by reference — it must outlive the await so that
// the async memcpy can read from it until the stream synchronizes.
let op = with_context(|ctx: &ExecutionContext| {
    let stream = ctx.get_cuda_stream();

    let dptr = unsafe {
        let dptr = malloc_async(num_bytes, stream).expect("device allocation");
        memcpy_htod_async(dptr, host_data.as_ptr(), num_elements, stream)
            .expect("host-to-device copy");
        dptr
    };

    value(dptr)
});

let dptr = op.await?;

// Clean up: free the device memory on a stream.
with_context(move |ctx: &ExecutionContext| {
    unsafe { free_async(dptr, ctx.get_cuda_stream()) }.expect("device free");
    value(())
})
.await?;
```

Driver calls inside the `unsafe` block must satisfy the CUDA API's safety
requirements.

---

## Borrowing Foreign CUDA Handles

If your application already owns CUDA handles through another Rust binding crate — cudarc, Candle, or a hand-rolled `bindgen` wrapper — wrap them into a cuTile `Device`, `Stream`, `Module`, or `Function` without transferring ownership.

The `borrow_raw` constructors take raw C primitives (`*mut c_void` for opaque handles, `c_int` for `CUdevice`) rather than `cuda_bindings` typedefs, so no nominal-type mismatch between binding crates gets in the way:

```rust
use core::ffi::{c_int, c_void};
use cuda_core::{Device, Stream};

// `foreign` is a handle bundle from another Rust binding crate (cudarc,
// Candle, a bindgen wrapper, …). Its `CUcontext` / `CUstream` typedefs are
// nominally distinct from cuTile's, but the underlying C ABI is identical —
// cast at the boundary and the handles flow through unchanged.

// Safety: the caller guarantees the handles are valid, outlive the returned
// wrappers, and are not concurrently destroyed.
let device = unsafe {
    Device::borrow_raw(
        foreign.cu_ctx as *mut c_void,
        foreign.cu_device as c_int,
        foreign.ordinal,
    )
};
let stream = unsafe { Stream::borrow_raw(foreign.cu_stream as *mut c_void, &device) };

// cuTile kernels now run on the borrowed stream.
let result = my_kernel(out, x).sync_on(&stream)?;
```

Borrowed wrappers skip destruction on drop: `Stream` does not call `cuStreamDestroy`, `Module` does not unload, and `Device` does not release the primary context. The source framework retains full control over handle lifetimes.

`Module::borrow_raw` and `Function::borrow_raw` follow the same pattern for pre-compiled modules and already-resolved functions. See the `cudarc_interop` example in `cutile-examples/examples/` for an end-to-end walkthrough.

---

## Migrating from Other DSLs

**From Triton.** [Triton](https://triton-lang.org/) and cuTile Rust both let you write kernels in terms of tile-level operations. Many patterns that require explicit warp specialization in Triton are handled implicitly by the cuTile Rust compiler:

| Triton (manual) | cuTile Rust (automatic) |
|-----------------|------------------------|
| Assign producer warps to prefetch tiles from global → shared memory | Compiler generates shared memory staging for `load_tile` operations |
| Assign consumer warps to compute on shared memory tiles | Compiler maps tile arithmetic to Tensor Cores and registers |
| Software pipeline with `warp_specialize` in `tl.range` | Compiler uses Tensor Memory Accelerator (TMA) instructions for hardware-assisted pipelining on supported architectures |
| Manual `tl.dot` placement across warps | `mma()` maps directly to Tensor Core instructions; thread/warp assignment is compiler-managed |
| Tune `num_warps` and `num_stages` for occupancy | `occupancy` and `num_cta_in_cga` optimization hints guide the compiler |

For patterns that don't map to the tile model, compile the kernel with Triton and integrate it via `AsyncKernelLaunch` as described above. Since Triton outputs PTX, load it directly:

```rust
let module = load_module_from_ptx(triton_generated_ptx, device_id)?;
let function = Arc::new(module.load_function("gemm_kernel")?);
```

**From cuTile Python.** If you're familiar with [cuTile Python](https://docs.nvidia.com/cuda/cutile-python/), the kernel-side concepts map directly:

| cuTile Python | cuTile Rust |
|---------------|-------------|
| `@ct.kernel` | `#[cutile::entry()]` |
| `ct.load()` | `x.load_like(z)` |
| `ct.store()` | `tensor.store()` |
| `ct.bid(0)` | Implicit via partition |
| `ct.launch()` | Async operation + `.await` |

Both front-ends use the same underlying Tile IR compilation pipeline and generate equivalent GPU code; the difference is the host language and its type system.

---

Continue to [Debugging and Profiling](debugging-and-profiling.md) for troubleshooting. Interop kernels use the `DeviceOp` model described in [Device Operations](device-operations.md).
