/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Launcher-side validation between safe host code and a kernel launch: the
//! specialization a launch compiles must match the element types of the
//! buffers it is handed, however the generics were chosen. A user
//! `.generics(..)` list overrides the inferred element type, and a kernel
//! specialized wider than its buffers indexes past them.

use cutile::api;
use cutile::half::f16;
use cutile::prelude::*;

use crate::common;

#[cutile::module]
mod launcher_guards_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn copy_elements<T: ElementType>(z: &mut Tensor<T, { [128] }>, x: &Tensor<T, { [-1] }>) {
        let t: Tile<T, { [128] }> = load_tile_like(x, z);
        z.store(t);
    }

    /// Zeroes 128 elements through a raw pointer. The pointer is the only
    /// typed argument, so this isolates the pointer element-type check.
    #[cutile::entry()]
    unsafe fn zero_through_ptr<T: ElementType>(dst: *mut T) {
        let base: PointerTile<*mut T, { [] }> = pointer_to_tile(dst);
        let base: PointerTile<*mut T, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*mut T, { [128] }> = base.broadcast(shape![128]);
        let offsets: Tile<i32, { [128] }> = iota(shape![128]);
        let addrs: PointerTile<*mut T, { [128] }> = addptr_tile(base, offsets);
        let zeros: Tile<T, { [128] }> = constant(T::ZERO, shape![128]);
        store_ptr_tko(
            addrs,
            zeros,
            ordering::Relaxed,
            Some(scope::Device),
            None,
            None,
            Latency::<0>,
        );
    }

    /// Dumps IR into a directory that does not exist. The failed write must
    /// surface as a launch error, not a panic inside the single-flight compile.
    #[cutile::entry(dump_mlir_dir = "/nonexistent-cutile-ir-dump-dir/sub")]
    fn dump_to_missing_dir(z: &mut Tensor<f32, { [128] }>, x: &Tensor<f32, { [-1] }>) {
        let t: Tile<f32, { [128] }> = load_tile_like(x, z);
        z.store(t);
    }
}

use launcher_guards_module::{copy_elements, dump_to_missing_dir, zero_through_ptr};

#[test]
fn unwritable_dump_mlir_dir_is_an_error_not_a_panic() {
    common::with_test_stack(|| {
        let x = api::ones::<f32>(&[128]).sync().expect("alloc x");
        let mut z = api::zeros::<f32>(&[128]).sync().expect("alloc z");
        let err = dump_to_missing_dir((&mut z).partition([128]), &x)
            .sync()
            .err()
            .expect("an unwritable dump_mlir_dir must fail the launch");
        let msg = format!("{err}");
        assert!(msg.contains("IR dump"), "{msg}");
        assert!(msg.contains("nonexistent-cutile-ir-dump-dir"), "{msg}");
    });
}

/// `.compile()` / `.specialize()` execute their input ops. They must do so
/// under the execution lock like every other terminal — so nested use inside
/// a `then` closure is refused instead of racing the outer chain — and must
/// drain the stream before dropping inputs an allocating op materialized.
#[test]
fn compile_and_specialize_are_execution_terminals() {
    common::with_test_stack(|| {
        let meta_launch = || {
            copy_elements(
                api::meta::<f32>(&[128]).partition([128]),
                api::meta::<f32>(&[128]),
            )
        };
        let nested = value(())
            .then(move |_| {
                let compile = meta_launch().compile();
                let specialize = meta_launch().specialize();
                value((compile.is_err(), specialize.is_err()))
            })
            .sync()
            .expect("outer chain");
        assert_eq!(
            nested,
            (true, true),
            "nested compile()/specialize() must be refused while the outer chain holds the lock"
        );

        // Standalone, both work — including over allocating inputs, whose
        // tensors are released only after the stream has drained.
        meta_launch().compile().expect("compile over meta inputs");
        copy_elements(
            api::zeros::<f32>(&[128]).partition([128]),
            api::ones::<f32>(&[128]),
        )
        .compile()
        .expect("compile over allocating inputs");
        let _spec = copy_elements(
            api::zeros::<f32>(&[128]).partition([128]),
            api::ones::<f32>(&[128]),
        )
        .specialize()
        .expect("specialize over allocating inputs");
    });
}

#[test]
fn generics_must_match_tensor_element_types() {
    common::with_test_stack(|| {
        let x = api::arange::<f32>(128).sync().expect("alloc x");
        let mut z = api::zeros::<f32>(&[128]).sync().expect("alloc z");

        // An explicit f16 specialization over f32 tensors is refused before launch.
        let err = copy_elements((&mut z).partition([128]), &x)
            .generics(vec!["f16".to_string()])
            .sync()
            .err()
            .expect("f16 specialization over f32 tensors must be rejected");
        let msg = format!("{err}");
        assert!(msg.contains("element type mismatch"), "{msg}");
        assert!(msg.contains("f16") && msg.contains("f32"), "{msg}");

        // The dangerous direction: a specialization wider than its buffers
        // would read and write twice each allocation's length.
        let x16 = api::zeros::<f16>(&[128]).sync().expect("alloc x16");
        let mut z16 = api::zeros::<f16>(&[128]).sync().expect("alloc z16");
        let err = copy_elements((&mut z16).partition([128]), &x16)
            .generics(vec!["f32".to_string()])
            .sync()
            .err()
            .expect("f32 specialization over f16 tensors must be rejected");
        assert!(format!("{err}").contains("element type mismatch"), "{err}");

        // Matching explicit generics launch and copy the data.
        copy_elements((&mut z).partition([128]), &x)
            .generics(vec!["f32".to_string()])
            .sync()
            .expect("matching generics must launch");
        let host: Vec<f32> = z.to_host_vec().sync().expect("copy back");
        let expected: Vec<f32> = (0..128).map(|i| i as f32).collect();
        assert_eq!(host, expected);
    });
}

#[test]
fn generics_must_match_pointer_element_types() {
    common::with_test_stack(|| {
        let buf = api::ones::<f32>(&[128]).sync().expect("alloc");

        let err = unsafe { zero_through_ptr(buf.device_pointer()) }
            .generics(vec!["f16".to_string()])
            .grid((1, 1, 1))
            .sync()
            .expect_err("f16 specialization over a DevicePointer<f32> must be rejected");
        let msg = format!("{err}");
        assert!(msg.contains("element type mismatch"), "{msg}");
        assert!(msg.contains("DevicePointer<f32>"), "{msg}");

        unsafe { zero_through_ptr(buf.device_pointer()) }
            .generics(vec!["f32".to_string()])
            .grid((1, 1, 1))
            .sync()
            .expect("matching generics must launch");
        let host: Vec<f32> = buf.to_host_vec().sync().expect("copy back");
        assert!(host.iter().all(|&v| v == 0.0), "pointer store did not run");
    });
}
