#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query local NVML fields used for nominal memory-bandwidth rooflines.

The nominal device-memory bandwidth calculation is:

    max_memory_clock_mhz * transfers_per_clock * bus_width_bits / 8 / 1000

This intentionally records the local driver-reported clock and bus-width
inputs so paper rooflines can be reproduced when moving between GPUs.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import byref, c_char_p, c_uint, c_void_p


NVML_CLOCK_SM = 1
NVML_CLOCK_MEM = 2


def _load_nvml():
    return ctypes.CDLL("libnvidia-ml.so.1")


def _call(rc: int, name: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{name} failed with NVML status {rc}")


def query(device: int) -> dict[str, object]:
    lib = _load_nvml()

    lib.nvmlInit_v2.restype = ctypes.c_int
    lib.nvmlShutdown.restype = ctypes.c_int
    lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [c_uint, ctypes.POINTER(c_void_p)]
    lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
    lib.nvmlDeviceGetName.argtypes = [c_void_p, c_char_p, c_uint]
    lib.nvmlDeviceGetName.restype = ctypes.c_int
    lib.nvmlDeviceGetMaxClockInfo.argtypes = [c_void_p, c_uint, ctypes.POINTER(c_uint)]
    lib.nvmlDeviceGetMaxClockInfo.restype = ctypes.c_int
    lib.nvmlDeviceGetMemoryBusWidth.argtypes = [c_void_p, ctypes.POINTER(c_uint)]
    lib.nvmlDeviceGetMemoryBusWidth.restype = ctypes.c_int

    _call(lib.nvmlInit_v2(), "nvmlInit_v2")
    try:
        handle = c_void_p()
        _call(
            lib.nvmlDeviceGetHandleByIndex_v2(c_uint(device), byref(handle)),
            "nvmlDeviceGetHandleByIndex_v2",
        )

        name_buf = ctypes.create_string_buffer(128)
        _call(lib.nvmlDeviceGetName(handle, name_buf, len(name_buf)), "nvmlDeviceGetName")

        mem_clock = c_uint()
        sm_clock = c_uint()
        bus_width = c_uint()
        _call(
            lib.nvmlDeviceGetMaxClockInfo(handle, NVML_CLOCK_MEM, byref(mem_clock)),
            "nvmlDeviceGetMaxClockInfo(memory)",
        )
        _call(
            lib.nvmlDeviceGetMaxClockInfo(handle, NVML_CLOCK_SM, byref(sm_clock)),
            "nvmlDeviceGetMaxClockInfo(sm)",
        )
        _call(
            lib.nvmlDeviceGetMemoryBusWidth(handle, byref(bus_width)),
            "nvmlDeviceGetMemoryBusWidth",
        )

        return {
            "device": device,
            "name": name_buf.value.decode(),
            "max_memory_clock_mhz": mem_clock.value,
            "max_sm_clock_mhz": sm_clock.value,
            "memory_bus_width_bits": bus_width.value,
        }
    finally:
        lib.nvmlShutdown()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument(
        "--transfers-per-clock",
        type=float,
        default=2.0,
        help="Memory transfers per reported memory-clock tick; record this per GPU.",
    )
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--value-only", action="store_true")
    args = ap.parse_args()

    info = query(args.device)
    bw_gb_s = (
        float(info["max_memory_clock_mhz"])
        * args.transfers_per_clock
        * (float(info["memory_bus_width_bits"]) / 8.0)
        / 1000.0
    )

    if args.value_only:
        print(f"{bw_gb_s:.6f}")
    elif args.markdown:
        print(
            "| {name} | {device} | {clock:.0f} MHz | {bus} bit | {xfer:g} | "
            "`{clock:.0f} * {xfer:g} * {bus} / 8 / 1000` | {bw:.3f} GB/s |".format(
                name=info["name"],
                device=info["device"],
                clock=float(info["max_memory_clock_mhz"]),
                bus=info["memory_bus_width_bits"],
                xfer=args.transfers_per_clock,
                bw=bw_gb_s,
            )
        )
    else:
        for key in (
            "device",
            "name",
            "max_memory_clock_mhz",
            "memory_bus_width_bits",
            "max_sm_clock_mhz",
        ):
            print(f"{key}: {info[key]}")
        print(f"transfers_per_clock: {args.transfers_per_clock:g}")
        print(
            "nominal_memory_bandwidth_gb_s: "
            f"{bw_gb_s:.3f}"
        )
        print(
            "formula: max_memory_clock_mhz * transfers_per_clock * "
            "memory_bus_width_bits / 8 / 1000"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
