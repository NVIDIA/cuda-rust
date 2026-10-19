/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke test: pointer-argument add kernel (`add_ptr`). Verifies that
//! `*mut T` kernel arguments wrapped via `make_tensor_view` produce the
//! same result as the safe `&Tensor` form.

use cuda_async::device_operation::DeviceOp;
use cutile::api::{arange, ones, zeros};
use cutile::tensor::{Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;

use crate::common;

#[cutile::module]
mod my_module {

    use cutile::core::*;

    unsafe fn get_tensor<T: ElementType>(ptr: *mut T, len: i32) -> Tensor<T, { [-1] }> {
        let shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[len] };
        let strides: Array<{ [-1] }> = Array::<{ [-1] }> { dims: &[1i32] };
        let ptr_tile: PointerTile<*mut T, { [] }> = pointer_to_tile(ptr);
        make_tensor_view(ptr_tile, shape, strides, new_token_unordered())
    }

    #[cutile::entry()]
    unsafe fn add_ptr<T: ElementType>(z_ptr: *mut T, x_ptr: *mut T, y_ptr: *mut T, len: i32) {
        let mut z_tensor: Tensor<T, { [-1] }> = get_tensor(z_ptr, len);
        let x_tensor: Tensor<T, { [-1] }> = get_tensor(x_ptr, len);
        let y_tensor: Tensor<T, { [-1] }> = get_tensor(y_ptr, len);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = shape![4i32];
        let tile_x = x_tensor.partition(tile_shape).load([pid.0]);
        let tile_y = y_tensor.partition(tile_shape).load([pid.0]);
        z_tensor
            .partition_mut(tile_shape)
            .store(tile_x + tile_y, [pid.0]);
    }
}

use my_module::add_ptr;

#[test]
fn smoke_add_ptr() {
    common::with_test_stack(|| {
        let len = 2usize.pow(5);
        let tile_size = 4usize;

        let z = zeros::<f32>(&[len]).sync().expect("zeros");
        let x: Tensor<f32> = arange(len).sync().expect("arange");
        let y: Tensor<f32> = ones(&[len]).sync().expect("ones");

        let z_ptr = z.device_pointer();
        let x_ptr = x.device_pointer();
        let y_ptr = y.device_pointer();

        unsafe { add_ptr(z_ptr, x_ptr, y_ptr, len as i32) }
            .grid(((len / tile_size) as u32, 1, 1))
            .sync()
            .expect("add_ptr kernel");

        let x_host = x.to_host_vec().sync().expect("x to_host");
        let y_host = y.to_host_vec().sync().expect("y to_host");
        let z_host = z.to_host_vec().sync().expect("z to_host");
        for i in 0..z_host.len() {
            assert_eq!(x_host[i] + y_host[i], z_host[i]);
        }
    });
}
