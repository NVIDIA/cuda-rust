/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Common operand roles and major-mode markers (`cutlass.cute.nvgpu.common`).

pub use crate::utils::blockscaled::{Mkl, Nkl};

/// Left matrix input (`A` operand); cannot be substituted for [`OperandB`].
pub enum OperandA {}

/// Right matrix input (`B` operand); cannot be substituted for [`OperandA`].
pub enum OperandB {}
