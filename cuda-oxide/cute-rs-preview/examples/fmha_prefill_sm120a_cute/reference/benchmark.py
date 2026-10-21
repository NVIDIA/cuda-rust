#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated CuTeDSL timing; shares all options and checks with launch.py."""

from launch import parse_args, run


if __name__ == "__main__":
    run(parse_args())
