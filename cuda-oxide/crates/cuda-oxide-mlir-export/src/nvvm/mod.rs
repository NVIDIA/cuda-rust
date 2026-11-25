/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Private NVVM primitive mappings grouped by instruction capability.

pub(crate) mod attention;
pub(crate) mod cluster;
pub(crate) mod numeric;
pub(crate) mod sm120;
pub(crate) mod sregs;
pub(crate) mod sync_restrict;
pub(crate) mod tcgen05;
