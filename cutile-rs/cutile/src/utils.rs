/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Utility functions and traits for cuTile Rust.
//!
//! This module provides helper utilities for testing, debugging, and working with
//! floating-point values and tensors.
//!
//! ## Floating-Point Comparisons
//!
//! The [`Float`] trait provides approximate equality comparisons for floating-point
//! types (`f16`, `f32`, `f64`). This is essential for testing GPU computations where
//! exact floating-point equality is often inappropriate due to rounding errors.
//!

use half::f16;

/// Trait for approximate floating-point comparisons.
///
/// Provides methods to compare floating-point values with tolerance for rounding errors.
/// This is crucial for testing GPU computations where exact equality is often inappropriate.
///
/// Implemented for `f16`, `f32`, and `f64`.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::utils::Float;
///
/// let a = 0.1f32 + 0.2f32;
/// let b = 0.3f32;
///
/// // Exact equality might fail due to rounding
/// assert!(a.close(b, 1e-6));
///
/// // Or use machine epsilon
/// assert!(a.epsilon_close(b));
/// ```
pub trait Float {
    /// Compares two values with a specified tolerance.
    ///
    /// Returns `true` if the absolute difference between the values is less than `tolerance`.
    fn close(&self, other: Self, tolerance: Self) -> bool;

    /// Compares two values using machine epsilon as the tolerance.
    ///
    /// Returns `true` if the values are equal within the type's epsilon (smallest
    /// representable difference).
    fn epsilon_close(&self, other: Self) -> bool;
}
impl Float for f16 {
    fn close(&self, other: Self, tolerance: f16) -> bool {
        f16::from_f32((self - other).to_f32().abs()) < tolerance
    }
    fn epsilon_close(&self, other: Self) -> bool {
        self.close(other, f16::EPSILON)
    }
}
impl Float for f32 {
    fn close(&self, other: Self, tolerance: f32) -> bool {
        (self - other).abs() < tolerance
    }
    fn epsilon_close(&self, other: Self) -> bool {
        self.close(other, f32::EPSILON)
    }
}
impl Float for f64 {
    fn close(&self, other: Self, tolerance: f64) -> bool {
        (self - other).abs() < tolerance
    }
    fn epsilon_close(&self, other: Self) -> bool {
        self.close(other, f64::EPSILON)
    }
}
