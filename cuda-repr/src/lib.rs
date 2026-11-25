// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Representation contracts shared by host buffers and device element types.
//!
//! This crate has no dependencies and builds without a CUDA toolkit or `std`.

#![no_std]

/// An element representation that can replace `Inner` in an owned device buffer.
///
/// This marker permits a buffer owner to change its element type without moving
/// or copying the allocation. It does **not** permit converting references:
/// for example, an atomic wrapper allows mutation through a shared reference,
/// whereas a shared reference to its ordinary integer representation does not.
/// Neither type is required to implement `Copy`.
///
/// # Safety
///
/// - `Self` and `Inner` must have identical size and alignment on both host and
///   device, with matching representations across those targets.
/// - Every valid initialized representation of either type must be a valid
///   initialized representation of the other, in both directions. Padding in
///   one type must not become an uninitialized value byte in the other.
/// - Neither representation may require element destruction. Changing between
///   them must not introduce ownership obligations, such as new ownership of
///   allocations or reference counts.
///
/// Consumers must preserve allocation ownership and lifetimes, and must uphold
/// borrowing and synchronization requirements when switching access modes.
/// This trait provides no reference conversions and no synchronization.
pub unsafe trait DeviceRepr<Inner>: Sized {}

// SAFETY: Wrapping is transparent over T with the same representations and
// ownership obligations. Copy excludes destruction in either representation.
unsafe impl<T: Copy> DeviceRepr<T> for core::num::Wrapping<T> {}
