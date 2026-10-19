# 9. Pointer Addition

Raw device pointers give kernels direct access to memory. This vector addition
example uses them with async execution, where the caller must keep the memory
alive until the kernel finishes.

> **Warning:** Working with raw pointers bypasses cutile's safety guarantees. Use them only when there is no other way to implement a kernel or further improve performance.

---

```rust
use std::future::IntoFuture;
use cutile;
use cutile::api::{arange, ones, zeros};
use cutile::tensor::{Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;

#[cutile::module]
mod my_module {

    use cutile::core::*;

    unsafe fn get_tensor<T: ElementType>(ptr: *mut T, len: i32) -> Tensor<T, { [-1] }> {
        let shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[len] };
        let strides: Array<{ [-1] }> = Array::<{ [-1] }> { dims: &[1i32] };
        let ptr_tile: PointerTile<*mut T, { [] }> = pointer_to_tile(ptr);
        let tensor = unsafe { make_tensor_view(ptr_tile, shape, strides, new_token_unordered()) };
        tensor
    }

    #[cutile::entry()]
    unsafe fn add_ptr<T: ElementType>(z_ptr: *mut T, x_ptr: *mut T, y_ptr: *mut T, len: i32) {
        let mut z_tensor: Tensor<T, { [-1] }> = unsafe { get_tensor(z_ptr, len) };
        let x_tensor: Tensor<T, { [-1] }> = unsafe { get_tensor(x_ptr, len) };
        let y_tensor: Tensor<T, { [-1] }> = unsafe { get_tensor(y_ptr, len) };
        let pid = program_id(0);
        let tile_shape = shape![4i32];
        let tile_x = x_tensor.partition(tile_shape).load([pid]);
        let tile_y = y_tensor.partition(tile_shape).load([pid]);
        z_tensor
            .partition_mut(tile_shape)
            .store(tile_x + tile_y, [pid]);
    }
}

use my_module::add_ptr;

#[tokio::main]
async fn main() -> Result<(), cutile::error::Error> {
    let len = 2usize.pow(5);
    let tile_size = 4usize;

    // Initialize tensors.
    let z: Tensor<f32> = zeros(&[len]).await?;
    let x: Tensor<f32> = ones(&[len]).await?;
    let y: Tensor<f32> = ones(&[len]).await?;

    // Extract device pointers.
    let z_ptr = z.device_pointer();
    let x_ptr = x.device_pointer();
    let y_ptr = y.device_pointer();

    // Prepare kernel launch. Note that, since we're passing in pointers, unsafe is required.
    let op = unsafe { add_ptr(z_ptr, x_ptr, y_ptr, len as i32) }.grid((
        (len / tile_size) as u32,
        1,
        1,
    ));

    // Spawn an asynchronous task to compute this operation.
    let op_handle = tokio::spawn(op.into_future());

    // Note that, while the device operates on these tensors, the programmer is not protected from UB!
    // We do not need the results. Device pointers are Copy (they are copied to device memory).
    let _ = op_handle.await.expect("Failed to execute tokio task.")?;

    let x_host = x.to_host_vec().await?;
    let y_host = y.to_host_vec().await?;
    let z_host = z.to_host_vec().await?;
    for i in 0..z_host.len() {
        let x_i: f32 = x_host[i];
        let y_i: f32 = y_host[i];
        let answer = x_i + y_i;
        println!("{} + {} = {}", x_i, y_i, z_host[i]);
        assert_eq!(answer, z_host[i], "{} != {} ?", answer, z_host[i]);
    }
    Ok(())
}
```

**Output:**

```text
1 + 1 = 2
1 + 1 = 2
... (32 lines total)
```

---

## Kernel walkthrough

The helper function `get_tensor` constructs a `Tensor` view from a raw device pointer, a length, and a stride. Because the compiler has no way to verify that the pointer is valid or that the memory it references outlives the kernel, the function — and the kernel entry point — must be marked `unsafe`.

Inside the kernel, the hand-built tensors are partitioned and loaded exactly like their safe counterparts. The only difference is how they were created.

On the host side, `device_pointer()` returns a typed `DevicePointer<T>` handle for an existing tensor. Kernel launchers convert that handle to the raw device pointer expected by the unsafe kernel. After the kernel completes, the original tensors still own the underlying memory, so we can read back the results with `to_host_vec()`.

---

(the-danger-of-async-raw-pointers)=
## Pointer lifetimes with async execution

`tokio::spawn` returns before the kernel finishes. Dropping or reallocating a
tensor during that time can leave the kernel accessing freed memory, causing
undefined behavior. Await the task handle before releasing the tensors.

---

## Key Takeaways

| Concept | What It Means |
|---------|---------------|
| **`device_pointer()`** | Returns a typed device pointer handle for a tensor. |
| **`make_tensor_view`** | Constructs a tensor from a pointer, shape, and strides inside a kernel. |
| **`unsafe` kernels** | Required whenever raw pointers bypass the type system's guarantees. |
| **Lifetime responsibility** | The programmer must ensure pointed-to memory outlives the kernel. |

---

### Exercise 1: Add Bounds Checking

What happens if `len` is not evenly divisible by the tile size? Add a guard in the kernel to handle the remainder.

### Exercise 2: In-Place Scaling

Modify the kernel to scale the output by a constant factor passed as an additional scalar argument.

### Exercise 3: Safe Wrapper

Write a safe Rust wrapper function around the unsafe kernel that validates the pointer lengths at the host level before launching.

---

## See also

- [Interoperability](../guide/interoperability.md) — a structured approach to pre-compiled CUDA C++ kernels using `AsyncKernelLaunch` instead of raw pointers
- [Tensors and Tiles](../guide/tensors-and-tiles.md) — tensor and tile basics before using pointer-based kernels
- [DSL API](../reference/dsl-api.md#low-level-memory-ops) — pointer tiles, tensor views, pointer loads/stores, and atomic operations
