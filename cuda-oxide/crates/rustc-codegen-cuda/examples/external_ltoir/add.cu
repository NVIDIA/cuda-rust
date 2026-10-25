/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef EXTERNAL_ADDEND
#define EXTERNAL_ADDEND 3
#endif

extern "C" __device__ int external_add(int value) {
    return value + EXTERNAL_ADDEND;
}
