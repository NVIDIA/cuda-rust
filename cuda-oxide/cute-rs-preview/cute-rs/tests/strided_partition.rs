/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Partition coverage, ordered reductions and disjoint stores on ordinary memory.

use cute_rs::partition::{StridedTensor, StridedTensorMut};

#[test]
fn nested_partitions_retain_their_types_and_compose_coordinates() {
    use cute_rs::partition::{Cyclic, Strided1D, Subrange};
    fn typed(_: &StridedTensor<'_, f32, Cyclic<Subrange<Cyclic<Strided1D, 4>>, 3>>) {}
    let storage: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let selected = StridedTensor::from_slice(&storage)
        .partition::<4>(1)
        .subview(2, 12)
        .partition::<3>(2);
    typed(&selected);
    assert_eq!(selected.load::<4>().into_array(), [17.0, 29.0, 41.0, 53.0]);
    let mut output = vec![-1.0; 100];
    let mut destination = StridedTensorMut::from_mut_slice(&mut output)
        .partition::<4>(1)
        .subview(2, 12)
        .partition::<3>(2);
    unsafe {
        selected.map_to(&mut destination, |value| value);
    }
    for (i, value) in output.iter().enumerate() {
        assert_eq!(
            *value,
            if [17, 29, 41, 53].contains(&i) {
                i as f32
            } else {
                -1.0
            }
        );
    }
}

#[test]
fn singleton_and_empty_nested_partitions_do_not_multiply_physical_stride() {
    let storage = [5.0_f32];
    let singleton = StridedTensor::from_slice(&storage)
        .partition::<{ usize::MAX }>(0)
        .partition::<{ usize::MAX }>(0);
    assert_eq!(singleton.get(0), 5.0);
    let empty = StridedTensor::from_slice(&storage)
        .partition::<{ usize::MAX }>(usize::MAX - 1)
        .subview(0, 0);
    assert_eq!(
        empty.fold(9.0, |_, _| panic!("empty layout must not access storage")),
        9.0
    );
}

#[test]
fn partition_layout_and_storage_identity_are_compile_time_contracts() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/compile_fail/cyclic_partition_mismatch.rs");
    tests.compile_fail("tests/compile_fail/strided_tensor_storage_mismatch.rs");
    tests.compile_fail("tests/compile_fail/strided_tensor_mut_reborrow.rs");
    tests.compile_fail("tests/compile_fail/strided_tensor_mut_move.rs");
    tests.compile_fail("tests/compile_fail/strided_layout_is_sealed.rs");
}

#[test]
fn encoded_storage_retains_its_element_contract_and_bits() {
    use cute_rs::partition::Strided1D;
    let bytes = [0x12_u8, 0x34, 0x56, 0x78];
    let encoded = StridedTensor::<cute_rs::E2M1, Strided1D, u8>::from_storage(&bytes);
    assert_eq!(
        encoded.partition::<2>(1).load::<2>().into_array(),
        [0x34, 0x78]
    );
}

#[test]
fn lane_partition_covers_tails_once_in_original_order() {
    let input: Vec<f32> = (0..70).map(|i| i as f32).collect();
    let mut output = vec![-1.0; input.len()];
    for lane in 0..32 {
        let source = StridedTensor::from_slice(&input).partition::<32>(lane);
        let mut destination = StridedTensorMut::from_mut_slice(&mut output).partition::<32>(lane);
        let expected: Vec<_> = (lane..70).step_by(32).map(|i| i as f32).collect();
        let actual = source.fold(Vec::new(), |mut values, value| {
            values.push(value);
            values
        });
        assert_eq!(actual, expected);
        // SAFETY: each lane writes its own positions; iterations are sequential.
        unsafe { source.map_to(&mut destination, |value| value + 1.0) };
    }
    assert_eq!(output, input.iter().map(|v| v + 1.0).collect::<Vec<_>>());
}

#[test]
fn register_gather_and_scatter_keep_stride_and_guards() {
    let input: Vec<f32> = (0..128).map(|i| i as f32).collect();
    let mut output = [-9.0; 130];
    let gathered = StridedTensor::from_slice(&input)
        .partition::<32>(7)
        .load::<4>();
    assert_eq!(gathered.into_array(), [7.0, 39.0, 71.0, 103.0]);
    let mut destination = StridedTensorMut::from_mut_slice(&mut output)
        .subview(1, 128)
        .partition::<32>(7);
    // SAFETY: this is the sole writable view and its four coordinates are distinct.
    unsafe { destination.store(gathered) };
    for (i, value) in output.iter().enumerate() {
        assert_eq!(
            *value,
            if [8, 40, 72, 104].contains(&i) {
                (i - 1) as f32
            } else {
                -9.0
            }
        );
    }
}

#[test]
fn fold_does_not_reassociate_floating_point_values() {
    let input = [1.0e20_f32, 1.0, -1.0e20, 3.0];
    assert_eq!(
        StridedTensor::from_slice(&input).fold(0.0, |a, b| a + b),
        3.0
    );
}

#[test]
fn empty_and_outside_lanes_never_access_storage() {
    let input = [7.0_f32];
    for lane in 1..32 {
        let view = StridedTensor::from_slice(&input).partition::<32>(lane);
        assert!(view.is_empty());
        assert_eq!(view.fold(5.0, |a, b| a + b), 5.0);
    }
    let empty = StridedTensor::from_slice(&input).subview(1, 0);
    assert!(empty.partition::<32>(31).is_empty());
}

#[test]
#[should_panic]
fn subview_rejects_overflowing_extent() {
    let input = [1.0_f32];
    let _ = StridedTensor::from_slice(&input).subview(usize::MAX, 2);
}

#[test]
#[should_panic]
fn fixed_load_rejects_short_partition() {
    let input = [1.0_f32; 63];
    let _ = StridedTensor::from_slice(&input)
        .partition::<32>(31)
        .load::<2>();
}

#[test]
fn exclusive_update_changes_only_the_selected_coordinate() {
    let mut data = [3_u32, 8, 10];
    let mut view = StridedTensorMut::from_mut_slice(&mut data);
    unsafe { view.update_at(1, |value| value | 4) };
    assert_eq!(data, [3, 12, 10]);
}

#[test]
fn register_gather_expresses_permuted_logical_coordinates() {
    let tile = cute_rs::RegisterTile::from_array([3.0_f32, 7.0]);
    assert_eq!(tile.gather([1, 0]).into_array(), [7.0, 3.0]);
}

#[test]
fn shared_tile_loads_follow_its_declared_layout() {
    type Layout = cute_rs::Composed<cute_rs::Swizzle<2, 0, 2>, 0, cute_rs::RowMajor<4, 4>>;
    let mut storage = [0.0_f32; 16];
    for row in 0..4 {
        for col in 0..4 {
            let offset = cute_rs::static_layout::offset_2d::<Layout>(row, col);
            storage[offset] = (10 * row + col) as f32;
        }
    }
    let tile = cute_rs::SmemTile::<f32, Layout> {
        base: storage.as_mut_ptr(),
        capacity: storage.len(),
        layout: core::marker::PhantomData,
    };
    // SAFETY: host-backed storage has the same layout, is initialized and has no writer.
    assert_eq!(
        unsafe { tile.load_row::<4>(2, 0) }.into_array(),
        [20.0, 21.0, 22.0, 23.0]
    );
}
