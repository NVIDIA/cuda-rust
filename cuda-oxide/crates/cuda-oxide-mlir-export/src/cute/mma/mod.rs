/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Matrix-load address plans and architecture-specific tensor-core instructions.

pub(in crate::cute) mod ldmatrix;
pub(in crate::cute) mod sm100;
pub(in crate::cute) mod sm100_attention;
pub(in crate::cute) mod sm120;
