/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `*const T` in the device DSL, mirroring Rust's raw-pointer model:
//! obtaining and casting pointers is safe, dereferencing/viewing is the
//! gated act, and `*mut` works anywhere `*const` does. Covers
//! `Tensor::as_ptr`, `cast_const`/`cast_mut` (scalar and tile),
//! constness-preserving `pointer_to_tile`, `*const` entry params, and
//! gather loads through `PointerTile<*const E, S>`.

use cutile::prelude::*;

use crate::common;

#[cutile::module]
mod const_ptr_module {

    use cutile::core::*;

    /// Rebuild a read-only view from a `*const` base. The `cast_mut` is
    /// the explicit constness boundary (`make_tensor_view` takes `*mut`),
    /// mirroring Rust where the cast is safe and the view construction
    /// carries the safety contract.
    unsafe fn view_from_const<T: ElementType>(ptr: *const T, len: i32) -> Tensor<T, { [-1] }> {
        let shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[len] };
        let strides: Array<{ [-1] }> = Array::<{ [-1] }> { dims: &[1i32] };
        let ptr_tile: PointerTile<*mut T, { [] }> = pointer_to_tile(cast_mut(ptr));
        make_tensor_view(ptr_tile, shape, strides, new_token_unordered())
    }

    /// `x.as_ptr()` round-trips through the raw-pointer world and back to
    /// a view that loads the same data the safe path loads.
    #[cutile::entry()]
    fn as_ptr_roundtrip(z: &mut Tensor<f32, { [4] }>, x: &Tensor<f32, { [-1] }>, len: i32) {
        let x_base: *const f32 = x.as_ptr();
        // TODO (hme): document safety
        let x_view: Tensor<f32, { [-1] }> = unsafe { view_from_const(x_base, len) };
        let pid: (i32, i32, i32) = get_tile_block_id();
        let direct = x.load_tile(shape![4], [pid.0]);
        let via_ptr = x_view.partition(shape![4]).load([pid.0]);
        z.store(direct + via_ptr);
    }

    /// A `*const f32` entry param is accepted and readable; the scalar
    /// casts convert in both directions.
    #[cutile::entry()]
    unsafe fn read_through_const(z: &mut Tensor<f32, { [4] }>, x_ptr: *const f32, len: i32) {
        // *const -> *mut -> *const: both directions, like Rust's
        // cast_mut/cast_const.
        let x_mut: *mut f32 = cast_mut(x_ptr);
        let x_const: *const f32 = cast_const(x_mut);
        let x_view: Tensor<f32, { [-1] }> = view_from_const(x_const, len);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile = x_view.partition(shape![4]).load([pid.0]);
        z.store(tile);
    }

    /// Gather through a `PointerTile<*const f32, S>`: the read op is
    /// generic over the pointer's constness (`P: PointerTo<E>`), so the
    /// same `load_ptr_tko` accepts what a `*mut` tile satisfies too.
    #[cutile::entry()]
    unsafe fn gather_through_const(z: &mut Tensor<f32, { [4] }>, x_ptr: *const f32) {
        let base: PointerTile<*const f32, { [] }> = pointer_to_tile(x_ptr);
        let base: PointerTile<*const f32, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*const f32, { [4] }> = base.broadcast(shape![4]);
        // Gather indices 0..4 reversed: element i reads x[3 - i].
        let three = broadcast_scalar(3i32, shape![4]);
        let offsets: Tile<i32, { [4] }> = three - iota(shape![4]);
        let addrs: PointerTile<*const f32, { [4] }> = addptr_tile(base, offsets);
        // The tile-level casts convert both directions as well.
        let addrs: PointerTile<*mut f32, { [4] }> = cast_tile_mut(addrs);
        let addrs: PointerTile<*const f32, { [4] }> = cast_tile_const(addrs);
        let (tile, _token): (Tile<f32, { [4] }>, Token) = load_ptr_tko(
            addrs,
            ordering::Relaxed,
            Some(scope::Device),
            None,
            None,
            None,
            Latency::<0>,
        );
        z.store(tile);
    }
}

use const_ptr_module::{as_ptr_roundtrip, gather_through_const, read_through_const};

#[test]
fn as_ptr_matches_the_safe_path() {
    common::with_test_stack(|| {
        let len = 32usize;
        let x = api::arange::<f32>(len);
        let z_host = as_ptr_roundtrip(api::zeros(&[len]).partition([4]), x, len as i32)
            .grid(((len / 4) as u32, 1, 1))
            .first()
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("as_ptr_roundtrip kernel");
        for (i, v) in z_host.iter().enumerate() {
            assert_eq!(*v, 2.0 * i as f32, "index {i}");
        }
    });
}

#[test]
fn const_entry_param_loads() {
    common::with_test_stack(|| {
        let len = 32usize;
        let x = api::arange::<f32>(len).sync().expect("arange");
        let z_host = unsafe {
            read_through_const(
                api::zeros(&[len]).partition([4]),
                x.device_pointer(),
                len as i32,
            )
        }
        .grid(((len / 4) as u32, 1, 1))
        .first()
        .unpartition()
        .to_host_vec()
        .sync()
        .expect("read_through_const kernel");
        for (i, v) in z_host.iter().enumerate() {
            assert_eq!(*v, i as f32, "index {i}");
        }
    });
}

#[test]
fn gather_loads_through_a_const_pointer_tile() {
    common::with_test_stack(|| {
        let x = api::arange::<f32>(4).sync().expect("arange");
        let z_host =
            unsafe { gather_through_const(api::zeros(&[4]).partition([4]), x.device_pointer()) }
                .grid((1, 1, 1))
                .first()
                .unpartition()
                .to_host_vec()
                .sync()
                .expect("gather_through_const kernel");
        assert_eq!(z_host, vec![3.0, 2.0, 1.0, 0.0]);
    });
}
