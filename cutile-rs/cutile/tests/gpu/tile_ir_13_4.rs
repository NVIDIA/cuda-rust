/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Execution coverage uses the selected assembler, not the build's CUDA headers.

use crate::common::supports_tile_ir as supports;
use cutile::prelude::*;
use cutile_compiler::cuda_tile_runtime_utils::ToolkitCapabilities;
use cutile_ir::bytecode::BytecodeVersion;
use cutile_ir::requirements::Feature;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn no_op() {}

    #[cutile::entry()]
    fn fast_accumulation(out: &mut Tensor<f32, { [16, 16] }>) {
        let a: Tile<f8e4m3fn, { [16, 32] }> = constant(f8e4m3fn::ZERO, shape![16, 32]);
        let b: Tile<f8e4m3fn, { [32, 16] }> = constant(f8e4m3fn::ZERO, shape![32, 16]);
        let c: Tile<f32, { [16, 16] }> = constant(7.0, shape![16, 16]);
        out.store(unsafe { mmaf_with_fast_acc(a, b, c, fast_acc::Enabled) });
    }

    #[cutile::entry()]
    fn sm107_scale(out: &mut Tensor<f32, { [16, 16] }>) {
        let a: Tile<f4e2m1fn, { [16, 16] }> = constant(f4e2m1fn::ZERO, shape![16, 16]);
        let c: Tile<f32, { [16, 16] }> = constant(0.0, shape![16, 16]);
        let sa: Tile<f8e5m3fnu, { [16, 1] }> = constant(f8e5m3fnu::ZERO, shape![16, 1]);
        let sb: Tile<f8e5m3fnu, { [1, 16] }> = constant(f8e5m3fnu::ZERO, shape![1, 16]);
        out.store(mmaf_scaled(a, a, c, sa, sb));
    }

    #[cutile::entry()]
    fn math_options(out: &mut Tensor<f32, { [4] }>) {
        let zero: Tile<f32, { [4] }> = constant(0.0, shape![4]);
        let one: Tile<f32, { [4] }> = constant(1.0, shape![4]);
        let two: Tile<f32, { [4] }> = constant(2.0, shape![4]);
        let three: Tile<f32, { [4] }> = constant(3.0, shape![4]);
        let result = unsafe {
            exp_with_rounding(zero, rounding::Approx)
                + tanh_with_rounding(zero, rounding::Approx)
                + fpowf(two, three)
        } + atan2(zero, one);
        out.store(result);
    }

    #[cutile::entry()]
    fn nearest_away(out: &mut Tensor<f32, { [4] }>) {
        let value: Tile<f16, { [4] }> = constant(f16::ONE, shape![4]);
        let widened: Tile<f32, { [4] }> = ftof(value, rounding::NearestAway);
        out.store(widened);
    }

    #[cutile::entry()]
    fn fp4_constant(out: &mut Tensor<u8, { [4] }>) {
        let value: Tile<f4e2m1fn, { [8] }> = constant(f4e2m1fn::ZERO, shape![8]);
        let packed: Tile<u8, { [4] }> = pack(value);
        out.store(packed);
    }

    #[cutile::entry()]
    fn loop_return(out: &mut Tensor<i32, { [4] }>, exit: i32) {
        loop {
            if exit != 0 {
                return;
            }
            break;
        }
        let value: Tile<i32, { [4] }> = constant(9, shape![4]);
        out.store(value);
    }

    #[cutile::entry()]
    fn insert_power(out: &mut Tensor<f32, { [8] }>) {
        let small: Tile<f32, { [4] }> = constant(2.0, shape![4]);
        let large: Tile<f32, { [8] }> = constant(1.0, shape![8]);
        let index: Tile<i32, { [] }> = constant(1, shape![]);
        let base: Tile<f32, { [8] }> = unsafe { insert(small, large, [index]) };
        let exponent: Tile<i32, { [8] }> = constant(3, shape![8]);
        out.store(unsafe { fpowi(base, exponent) });
    }

    #[cutile::entry()]
    unsafe fn producer(output: *mut i32) {
        let ptr: PointerTile<*mut i32, { [] }> = pointer_to_tile(output);
        let value: Tile<i32, { [] }> = constant(41, shape![]);
        let stored = unsafe {
            store_ptr_tko(
                ptr,
                value,
                ordering::Weak,
                None::<scope::Device>,
                None,
                None,
                Latency::<0>,
            )
        };
        let _signal = unsafe { gdc_launch_dependents_tko(Some(stored)) };
    }

    #[cutile::entry()]
    unsafe fn consumer(input: *mut i32, output: *mut i32) {
        // Scheduling does not establish visibility: every dependent memory
        // operation is downstream of this wait's token.
        let ready = unsafe { gdc_wait_tko(None) };
        let input_ptr: PointerTile<*mut i32, { [] }> = pointer_to_tile(input);
        let output_ptr: PointerTile<*mut i32, { [] }> = pointer_to_tile(output);
        let (value, loaded): (Tile<i32, { [] }>, Token) = unsafe {
            load_ptr_tko(
                input_ptr,
                ordering::Weak,
                None::<scope::Device>,
                None,
                None,
                Some(ready),
                Latency::<0>,
            )
        };
        let fenced = unsafe { memory_fence_alias_tko(loaded) };
        let one: Tile<i32, { [] }> = constant(1, shape![]);
        let _stored = unsafe {
            store_ptr_tko(
                output_ptr,
                value + one,
                ordering::Weak,
                None::<scope::Device>,
                None,
                Some(fenced),
                Latency::<0>,
            )
        };
    }
}

#[cutile::module(producer = "cutile-rs coverage tests")]
mod coverage {
    use cutile::core::*;
    use cutile::tileir::*;

    #[cuda_tile::global(constant = true, visibility = "private")]
    static BIAS: Global<AtomicI32, { [] }> = Global::new(7i32);

    #[cutile::entry()]
    unsafe fn gather_pipeline(
        input: &Tensor<f32, { [-1, -1, -1] }>,
        output: &Tensor<f32, { [-1, -1, -1] }>,
    ) {
        let src: GatherScatterView<f32, { [1, 4, 1] }, 1> =
            unsafe { make_gather_scatter_view(input, shape![1, 4, 1], padding::Zero) };
        let dst: GatherScatterView<f32, { [1, 4, 1] }, 1> =
            unsafe { make_gather_scatter_view(output, shape![1, 4, 1], padding::None) };
        let index: Tile<i32, { [4] }> = iota(shape![4]);
        let (value, done): (Tile<f32, { [1, 4, 1] }>, Token) = unsafe {
            load_view_raw(
                &src,
                (0i32, index, 0i32),
                None,
                ordering::Weak,
                scope::Device,
                tma::Disabled,
                None,
                [false, false, false],
                shape![1, 4, 1],
            )
        };
        let _done = unsafe {
            store_view_raw(
                &dst,
                value,
                (0i32, index, 0i32),
                Some(done),
                ordering::Weak,
                scope::Device,
                tma::Disabled,
                None,
                [false, false, false],
            )
        };
    }

    #[cutile::entry()]
    unsafe fn strided_pipeline(input: &Tensor<f32, { [-1] }>, output: &Tensor<f32, { [-1] }>) {
        let src: StridedView<f32, { [4] }, { [2] }, { [0] }> =
            unsafe { make_strided_view(input, shape![4], padding::Zero) };
        let dst: StridedView<f32, { [4] }, { [2] }, { [0] }> =
            unsafe { make_strided_view(output, shape![4], padding::None) };
        let (value, loaded): (Tile<f32, { [4] }>, Token) = unsafe {
            load_view_raw(
                &src,
                [0i32],
                None,
                ordering::Weak,
                scope::Device,
                tma::Disabled,
                None,
                [false],
                shape![4],
            )
        };
        let stored: Token = unsafe {
            store_view_raw(
                &dst,
                value,
                [0i32],
                Some(loaded),
                ordering::Weak,
                scope::Device,
                tma::Disabled,
                None,
                [false],
            )
        };
        let _reduced: Token = unsafe {
            atomic_red_view_raw(
                &dst,
                value,
                [0i32],
                atomic::AddF,
                ordering::Relaxed,
                scope::Device,
                Some(stored),
            )
        };
    }

    #[cutile::entry()]
    unsafe fn allocation_conversion(output: &mut Tensor<i32, { [4] }>) {
        let pointer: PointerTile<*mut f32, { [] }> =
            unsafe { alloca(1i64, 4i64, allocation::Local) };
        let x: Tile<f32, { [] }> = constant(5.9, shape![]);
        let stored = unsafe {
            store_ptr_tko(
                pointer,
                x,
                ordering::Weak,
                None::<scope::Device>,
                None,
                None,
                Latency::<0>,
            )
        };
        let (loaded, _token): (Tile<f32, { [] }>, Token) = unsafe {
            load_ptr_tko(
                pointer,
                ordering::Weak,
                None::<scope::Device>,
                None,
                None,
                Some(stored),
                Latency::<0>,
            )
        };
        let a: Tile<i32, { [] }> = ftoi(loaded, rounding::NearestIntToZero);
        let b: Tile<i32, { [] }> = unsafe { ftoi_saturating(loaded, saturating::Enabled) };
        let (bias, _bias_token): (Tile<i32, { [] }>, Token) =
            BIAS.load(ordering::Relaxed, scope::Device);
        let scalar = (a + b + bias).reshape(shape![1]);
        let sum: Tile<i32, { [4] }> = broadcast(scalar, shape![4]);
        output.store(sum);
    }

    #[cutile::entry()]
    unsafe fn classified_store(output: *mut f32) {
        let base: PointerTile<*mut f32, { [] }> = pointer_to_tile(output);
        let base: PointerTile<*mut f32, { [] }> = unsafe { ptr_with_attr_none(base) };
        let strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[1] };
        let tensor: Tensor<f32, { [8] }> =
            unsafe { make_tensor_view(base, shape![8], strides, new_token_unordered()) };
        let view: StridedView<f32, { [4] }, { [2] }, { [0] }> =
            unsafe { make_strided_view(&tensor, shape![4], padding::None) };
        let value: Tile<f32, { [4] }> = constant(2.0, shape![4]);
        let _stored = unsafe {
            store_view_raw(
                &view,
                value,
                [0i32],
                None,
                ordering::Weak,
                scope::Device,
                tma::Disabled,
                None,
                [true],
            )
        };
    }
}

#[test]
fn launch_opt_in_checks_capabilities_even_during_warmup() {
    crate::common::with_test_stack(|| {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();
        let caps = ToolkitCapabilities::for_device(0).unwrap();
        // The empty consumer has no predecessor-dependent accesses.
        let result = unsafe { kernels::no_op().programmatic_dependent_launch() }
            .grid((1, 1, 1))
            .compile_on(&stream);
        if caps.target.bytecode_version < BytecodeVersion::V13_4 {
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("programmatic dependent launch requires Tile IR 13.4"));
        } else if caps.target.sm().is_some_and(|sm| sm >= 90) {
            result.unwrap();
        } else {
            assert!(result.unwrap_err().to_string().contains("sm_90"));
        }
    });
}

#[test]
fn insert_and_integer_power_execute() {
    crate::common::with_test_stack(|| {
        if !supports(&[Feature::Insert, Feature::IntegerPower]) {
            return;
        }
        let output = api::zeros::<f32>(&[8]).sync().unwrap();
        let (output,) = kernels::insert_power(output.partition([8]))
            .grid((1, 1, 1))
            .sync()
            .unwrap();
        assert_eq!(
            output.unpartition().to_host_vec().sync().unwrap(),
            [1., 1., 1., 1., 8., 8., 8., 8.]
        );
    });
}

#[test]
fn dependent_launch_waits_for_producer_data() {
    crate::common::with_test_stack(|| {
        if !supports(&[Feature::ProgrammaticDependentLaunch]) {
            return;
        }
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();
        let input = api::zeros::<i32>(&[1]).sync_on(&stream).unwrap();
        let output = api::zeros::<i32>(&[1]).sync_on(&stream).unwrap();
        // Storage remains owned until both launches and readback complete.
        // Repeating also exercises the launch-site and compiled-kernel caches.
        for _ in 0..3 {
            unsafe {
                kernels::producer(input.device_pointer())
                    .grid((1, 1, 1))
                    .then(|_| {
                        kernels::consumer(input.device_pointer(), output.device_pointer())
                            .programmatic_dependent_launch()
                            .grid((1, 1, 1))
                    })
            }
            .sync_on(&stream)
            .unwrap();
        }
        assert_eq!(output.to_host_vec().sync_on(&stream).unwrap(), [42]);
    });
}

#[test]
fn strided_load_store_reduction_execute_on_13_3_and_13_4() {
    crate::common::with_test_stack(|| {
        if !supports(&[Feature::StridedView, Feature::AtomicRedView]) {
            return;
        }
        let input = api::ones::<f32>(&[8]).sync().unwrap();
        let output = api::zeros::<f32>(&[8]).sync().unwrap();
        unsafe { coverage::strided_pipeline(&input, &output) }
            .grid((1, 1, 1))
            .sync()
            .unwrap();
        assert_eq!(
            output.to_host_vec().sync().unwrap(),
            [2., 2., 2., 2., 0., 0., 0., 0.]
        );
    });
}

#[test]
fn allocation_saturation_and_private_constant_execute() {
    crate::common::with_test_stack(|| {
        if !supports(&[
            Feature::Allocation,
            Feature::SaturatingFToI,
            Feature::GlobalConstant,
            Feature::GlobalVisibility,
        ]) {
            return;
        }
        let output = api::zeros::<i32>(&[4]).sync().unwrap();
        let (output,) = unsafe { coverage::allocation_conversion(output.partition([4])) }
            .sync()
            .unwrap();
        assert_eq!(output.unpartition().to_host_vec().sync().unwrap(), [17; 4]);
    });
}

#[test]
fn classified_pointer_and_inbounds_store_execute() {
    crate::common::with_test_stack(|| {
        if !supports(&[Feature::PointerAttribute, Feature::Inbounds]) {
            return;
        }
        let output = api::zeros::<f32>(&[8]).sync().unwrap();
        unsafe { coverage::classified_store(output.device_pointer()) }
            .grid((1, 1, 1))
            .sync()
            .unwrap();
        assert_eq!(
            output.to_host_vec().sync().unwrap(),
            [2., 2., 2., 2., 0., 0., 0., 0.]
        );
    });
}

#[test]
fn loop_return_exits_kernel_only_on_taken_path() {
    crate::common::with_test_stack(|| {
        if !supports(&[Feature::LoopReturn]) {
            return;
        }
        for exit in [0, 1] {
            let output = api::zeros::<i32>(&[4]).sync().unwrap();
            let (output, _) = kernels::loop_return(output.partition([4]), exit)
                .grid((1, 1, 1))
                .sync()
                .unwrap();
            assert_eq!(
                output.unpartition().to_host_vec().sync().unwrap(),
                [if exit == 0 { 9 } else { 0 }; 4]
            );
        }
    });
}

#[test]
fn fp4_constant_pack_executes_on_sm_120() {
    crate::common::with_test_stack(|| {
        if !supports(&[Feature::Packing, Feature::Fp4]) {
            return;
        }
        let output = api::zeros::<u8>(&[4]).sync().unwrap();
        let (output,) = kernels::fp4_constant(output.partition([4])).sync().unwrap();
        assert_eq!(output.unpartition().to_host_vec().sync().unwrap(), [0; 4]);
    });
}

#[test]
fn arbitrary_rank_gather_scatter_and_math_options_execute() {
    crate::common::with_test_stack(|| {
        if !supports(&[Feature::GatherScatterView, Feature::ExpRounding]) {
            return;
        }
        let input = api::ones::<f32>(&[2, 4, 2]).sync().unwrap();
        let output = api::zeros::<f32>(&[2, 4, 2]).sync().unwrap();
        unsafe { coverage::gather_pipeline(&input, &output) }
            .grid((1, 1, 1))
            .sync()
            .unwrap();
        assert_eq!(
            output.to_host_vec().sync().unwrap(),
            [1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        );
        let output = api::zeros::<f32>(&[4]).sync().unwrap();
        let (output,) = kernels::math_options(output.partition([4])).sync().unwrap();
        for value in output.unpartition().to_host_vec().sync().unwrap() {
            assert!((value - 9.0).abs() < 1e-5);
        }
    });
}

#[test]
fn nearest_away_widening_executes() {
    crate::common::with_test_stack(|| {
        if !supports(&[Feature::NearestAway]) {
            return;
        }
        let output = api::zeros::<f32>(&[4]).sync().unwrap();
        let (output,) = kernels::nearest_away(output.partition([4])).sync().unwrap();
        assert_eq!(output.unpartition().to_host_vec().sync().unwrap(), [1.0; 4]);
    });
}

#[test]
fn fast_accumulation_executes_when_supported() {
    crate::common::with_test_stack(|| {
        if !supports(&[Feature::FastAccumulation, Feature::Fp8]) {
            return;
        }
        let output = api::zeros::<f32>(&[16, 16]).sync().unwrap();
        let (output,) = kernels::fast_accumulation(output.partition([16, 16]))
            .sync()
            .unwrap();
        assert_eq!(
            output.unpartition().to_host_vec().sync().unwrap(),
            vec![7.0; 256]
        );
    });
}

#[test]
fn new_scale_format_executes_only_on_sm107() {
    crate::common::with_test_stack(|| {
        if !supports(&[Feature::NewScaleMma]) {
            return;
        }
        let output = api::zeros::<f32>(&[16, 16]).sync().unwrap();
        let (output,) = kernels::sm107_scale(output.partition([16, 16]))
            .sync()
            .unwrap();
        assert_eq!(
            output.unpartition().to_host_vec().sync().unwrap(),
            vec![0.0; 256]
        );
    });
}
