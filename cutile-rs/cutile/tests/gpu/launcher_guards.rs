/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Launcher-side validation between safe host code and a kernel launch: the
//! specialization a launch compiles must match the element types of the
//! buffers it is handed, however the generics were chosen. A user
//! `.generics(..)` list overrides the inferred element type, and a kernel
//! specialized wider than its buffers indexes past them.
//! Stride constants must also agree with the runtime layout, including when
//! a custom input/output binding supplies stale specialization metadata.

use cuda_async::launch::AsyncKernelLaunch;
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

    #[cutile::entry()]
    fn copy_strided(z: &mut Tensor<f32, { [128, 1] }>, x: &Tensor<f32, { [-1, -1] }>) {
        let t: Tile<f32, { [128, 1] }> = load_tile_like(x, z);
        z.store(t);
    }

    #[cutile::entry()]
    fn mapped_stride_guard(_z: MappedPartitionMut<f32, { [128, 1] }, { [1, 2] }>) {}

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

use launcher_guards_module::{
    copy_elements, copy_strided, dump_to_missing_dir, mapped_stride_guard, zero_through_ptr,
};

/// Inject stale specialization metadata through the public binding traits.
/// Shapes, strides, retention and kernel arguments still come from the real
/// binding. This tests the generated guard without forging a device pointer.
struct OverrideSpec<I> {
    inner: I,
    spec: SpecializationBits,
}

fn with_first_stride_one(mut spec: SpecializationBits) -> SpecializationBits {
    spec.stride_one[0] = true;
    spec.stride_div[0].divisor = 1;
    spec
}

impl<I: KernelInputStored> KernelInputStored for OverrideSpec<I> {
    fn retain(&self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        self.inner.retain(ctx)
    }
    fn push_kernel_args(&self, launcher: &mut AsyncKernelLaunch) {
        self.inner.push_kernel_args(launcher);
    }
    fn shape(&self) -> &[i32] {
        self.inner.shape()
    }
    fn strides(&self) -> &[i32] {
        self.inner.strides()
    }
    fn spec(&self) -> &SpecializationBits {
        &self.spec
    }
    fn dtype_str(&self) -> &'static str {
        self.inner.dtype_str()
    }
}

impl<I: KernelInputStored> KernelInput<f32> for OverrideSpec<I> {
    type Stored = Self;
    type Returned = Self;
    fn prepare(self) -> Self {
        self
    }
    fn recover(stored: Self) -> Self {
        stored
    }
}

impl<I: KernelOutputStored<f32>> KernelOutputStored<f32> for OverrideSpec<I> {
    fn retain(&self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        self.inner.retain(ctx)
    }
    fn push_kernel_args(&self, launcher: &mut AsyncKernelLaunch) {
        self.inner.push_kernel_args(launcher);
    }
    fn grid(&self) -> Result<(u32, u32, u32), Error> {
        self.inner.grid()
    }
    fn map_shape_as_i32(&self) -> Option<Vec<i32>> {
        self.inner.map_shape_as_i32()
    }
    fn dtype_str(&self) -> &'static str {
        self.inner.dtype_str()
    }
    fn partition_shape_as_i32(&self) -> Vec<i32> {
        self.inner.partition_shape_as_i32()
    }
    fn partition_shape(&self) -> &[usize] {
        self.inner.partition_shape()
    }
    fn strides(&self) -> &[i32] {
        self.inner.strides()
    }
    fn strides_hint(&self) -> Vec<i32> {
        self.spec
            .stride_one
            .iter()
            .map(|&is_one| if is_one { 1 } else { -1 })
            .collect()
    }
    fn spec(&self) -> &SpecializationBits {
        &self.spec
    }
    fn shape_as_i32(&self) -> Vec<i32> {
        self.inner.shape_as_i32()
    }
}

impl<I: KernelOutputStored<f32>> KernelOutput<f32> for OverrideSpec<I> {
    type Stored = Self;
    type Returned = Self;
    fn prepare(self) -> Self {
        self
    }
    fn recover(stored: Self) -> Self {
        stored
    }
}

#[test]
fn stale_input_stride_constants_are_rejected_after_cache_fill() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let x = api::arange::<f32>(256)
            .reshape(&[128, 2])
            .sync()
            .expect("alloc x");
        let view = x.slice(&[0..128, 0..1]).expect("strided input view");
        assert_eq!(view.shape(), &[128, 1]);
        assert_eq!(view.strides(), &[2, 1]);
        let spec = with_first_stride_one(view.spec().clone());
        let mut z = api::zeros::<f32>(&[128, 1]).sync().expect("alloc z");

        // The JIT resolution is cached before argument validation. Repeating
        // the failed compile-only launch exercises the cached path too.
        for _ in 0..2 {
            let input = OverrideSpec {
                inner: &view,
                spec: spec.clone(),
            };
            let err = copy_strided((&mut z).partition([128, 1]), value(input))
                .compile()
                .expect_err("a unit stride specialization over stride 2 must be rejected");
            let msg = err.to_string();
            assert!(msg.contains("x strides mismatch"), "{msg}");
            assert!(msg.contains("[1, 1]") && msg.contains("[2, 1]"), "{msg}");
        }
    });
}

#[test]
fn stale_output_stride_constants_are_rejected_after_cache_fill() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let x = api::ones::<f32>(&[128, 2]).sync().expect("alloc x");
        let mut z = api::zeros::<f32>(&[128, 2]).sync().expect("alloc z");
        let spec = with_first_stride_one(z.spec().clone());
        for _ in 0..2 {
            let output = OverrideSpec {
                inner: (&mut z).partition([128, 1]),
                spec: spec.clone(),
            };
            let err = copy_strided(value(output), &x)
                .compile()
                .expect_err("a unit stride specialization over stride 2 must be rejected");
            let msg = err.to_string();
            assert!(msg.contains("z strides mismatch"), "{msg}");
            assert!(msg.contains("[1, 1]") && msg.contains("[2, 1]"), "{msg}");
        }

        let output = OverrideSpec {
            inner: (&mut z).partition([128, 1]).map([1, 2], 2),
            spec,
        };
        let err = mapped_stride_guard(value(output))
            .compile()
            .expect_err("mapped outputs must also validate their runtime strides");
        assert!(err.to_string().contains("_z strides mismatch"), "{err}");
    });
}

#[test]
fn different_dynamic_input_strides_reuse_the_same_specialization() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let mut previous_key = None;
        for row_stride in [32usize, 48, 32] {
            let x = api::arange::<f32>(128 * row_stride)
                .reshape(&[128, row_stride])
                .sync()
                .expect("alloc x");
            let view = x.slice(&[0..128, 0..1]).expect("strided input view");
            let mut z = api::zeros::<f32>(&[128, 1]).sync().expect("alloc z");
            // Both strides have the same divisibility hints and are dynamic.
            let key = copy_strided((&mut z).partition([128, 1]), &view)
                .l1_cache_key()
                .expect("specialization key");
            if let Some(previous) = &previous_key {
                assert_eq!(previous, &key);
            }
            previous_key = Some(key);
            copy_strided((&mut z).partition([128, 1]), &view)
                .sync()
                .expect("a dynamic input stride must be accepted");
            let host = z.to_host_vec().sync().expect("copy back");
            let expected: Vec<f32> = (0..128).map(|i| (i * row_stride) as f32).collect();
            assert_eq!(host, expected);
        }
    });
}

#[test]
fn dynamic_output_strides_are_accepted() {
    common::with_test_stack(|| {
        for columns in [2usize, 3] {
            let x = api::arange::<f32>(128 * columns)
                .reshape(&[128, columns])
                .sync()
                .expect("alloc x");
            let host = copy_strided(api::zeros::<f32>(&[128, columns]).partition([128, 1]), &x)
                .first()
                .unpartition()
                .to_host_vec()
                .sync()
                .expect("a dynamic output stride must be accepted");
            let expected: Vec<f32> = (0..128 * columns).map(|i| i as f32).collect();
            assert_eq!(host, expected);
        }
    });
}

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
