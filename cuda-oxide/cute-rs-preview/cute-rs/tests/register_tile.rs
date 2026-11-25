/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![doc = "Behavioral contracts for typed register-tile composition."]
#![feature(f16)]

use cute_rs::tensor::RegisterTile;

#[test]
fn construction_preserves_lane_order_and_supports_constants() {
    const TILE: RegisterTile<i32, 4> = RegisterTile::from_array([3, -2, 7, 1]);
    assert_eq!(TILE.as_array(), &[3, -2, 7, 1]);
    assert_eq!(TILE.into_array(), [3, -2, 7, 1]);
}

#[test]
fn producer_constructs_each_register_once_in_order() {
    let mut coordinates = Vec::new();
    let tile = RegisterTile::<i32, 4>::from_fn(|i| {
        coordinates.push(i);
        7 + i as i32 * 3
    });
    assert_eq!(coordinates, [0, 1, 2, 3]);
    assert_eq!(tile.into_array(), [7, 10, 13, 16]);
    let singleton = RegisterTile::<usize, 1>::from_fn(|i| i + 9);
    assert_eq!(singleton.into_array(), [9]);
}

#[test]
fn empty_producer_never_calls_its_callback() {
    let empty = RegisterTile::<i32, 0>::from_fn(|_| panic!("empty producer invoked"));
    assert_eq!(empty.into_array(), []);
}

#[test]
fn map_visits_each_lane_once_in_order() {
    let mut visited = Vec::new();
    let tile = RegisterTile::from_array([3, -2, 7, 1]);
    let output: RegisterTile<i64, 4> = tile.map(|value| {
        visited.push(value);
        i64::from(value) * visited.len() as i64
    });
    assert_eq!(visited, [3, -2, 7, 1]);
    assert_eq!(output.into_array(), [3, -4, 21, 4]);
}

#[test]
fn zip_map_pairs_lanes_once_and_allows_explicit_type_conversion() {
    let lhs = RegisterTile::from_array([1.5_f16, -2.0, 0.25, 8.0]);
    let rhs = RegisterTile::from_array([2_i32, 3, 4, -1]);
    let mut visited = Vec::new();
    let output: RegisterTile<f32, 4> = lhs.zip_map(rhs, |left, right| {
        visited.push((left as f32, right));
        (left as f32) * (right as f32)
    });
    assert_eq!(visited, [(1.5, 2), (-2.0, 3), (0.25, 4), (8.0, -1)]);
    assert_eq!(output.into_array(), [3.0, -6.0, 1.0, -8.0]);
}

#[test]
fn empty_tiles_do_not_invoke_transforms() {
    let tile = RegisterTile::<f16, 0>::from_array([]);
    let mapped: RegisterTile<f32, 0> = tile.map(|_| panic!("no lanes to map"));
    let zipped: RegisterTile<u32, 0> = mapped
        .zip_map(RegisterTile::<i32, 0>::from_array([]), |_, _| {
            panic!("no lanes to zip")
        });
    assert_eq!(zipped.into_array(), []);
}

#[test]
fn fold_preserves_order_with_an_explicit_accumulator_type() {
    let tile = RegisterTile::from_array([1_i16, 2, 3]);
    let mut visited = Vec::new();
    let result = tile.fold(5_i64, |acc, value| {
        visited.push(value);
        acc * 10 + i64::from(value)
    });
    assert_eq!(result, 5123);
    assert_eq!(visited, [1, 2, 3]);
    assert_eq!(
        RegisterTile::<f32, 0>::from_array([]).fold(7, |_, _| panic!("empty")),
        7
    );
}

#[test]
fn arithmetic_operators_apply_per_lane_in_order() {
    let lhs = RegisterTile::from_array([8.0_f32, -6.0, 1.5, 0.0]);
    let rhs = RegisterTile::from_array([2.0_f32, 3.0, -0.5, 4.0]);
    assert_eq!((lhs + rhs).into_array(), [10.0, -3.0, 1.0, 4.0]);
    assert_eq!((lhs - rhs).into_array(), [6.0, -9.0, 2.0, -4.0]);
    assert_eq!((lhs * rhs).into_array(), [16.0, -18.0, -0.75, 0.0]);
    assert_eq!((lhs / rhs).into_array(), [4.0, -2.0, -3.0, 0.0]);
    let negated = (-lhs).into_array();
    assert_eq!(negated, [-8.0, 6.0, -1.5, 0.0]);
    assert!(negated[3].is_sign_negative());
}

#[test]
fn integer_arithmetic_keeps_the_element_type_and_truncates_quotients() {
    let lhs = RegisterTile::from_array([7_i32, -4, 9]);
    let rhs = RegisterTile::from_array([2_i32, 3, -3]);
    let difference: RegisterTile<i32, 3> = lhs - rhs;
    assert_eq!(difference.into_array(), [5, -7, 12]);
    assert_eq!((lhs * rhs).into_array(), [14, -12, -27]);
    assert_eq!((lhs / rhs).into_array(), [3, -1, -3]);
    assert_eq!((-rhs).into_array(), [-2, -3, 3]);
}

#[test]
fn empty_tiles_support_every_operator() {
    let empty = RegisterTile::<f32, 0>::from_array([]);
    assert_eq!((empty + empty).into_array(), []);
    assert_eq!((empty - empty).into_array(), []);
    assert_eq!((empty * empty).into_array(), []);
    assert_eq!((empty / empty).into_array(), []);
    assert_eq!((-empty).into_array(), []);
}

#[test]
fn warp_reduction_sum_exposes_an_unsafe_collective_boundary() {
    let _: unsafe fn(f32) -> f32 = cute_rs::arch::warp_reduction_sum_f32;
}
