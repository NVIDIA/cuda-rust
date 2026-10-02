# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuda-gdb exception handling and GPU-coredump coverage."""


def test_gpu_assert_exception(gdb):
    """gpu_assert! produces a Warp Assert with message and source backtrace."""
    gdb("set args assert_fail")
    with gdb("run") as check:
        check.matches(r"CUDA_EXCEPTION_12|Warp Assert", "CUDA Warp Assert is caught")
        check.matches(r"intentional assertion failure", "assertion message is printed")
    with gdb("backtrace") as check:
        check.matches(r"src/main\.rs:[0-9]+", "assert source line is shown")
        check.matches(r"#0.*debuginfo_assert_fail", "backtrace is at assert site")
    gdb("info line")
    gdb("kill")


def test_debug_trap_exception(gdb):
    """debug::trap() produces a CUDA exception at its source location."""
    gdb("set args trap_fail")
    with gdb("run") as check:
        check.matches(
            r"CUDA_EXCEPTION_[0-9]+|CUDA Exception|received signal CUDA",
            "CUDA exception is caught after trap",
        )
    with gdb("backtrace") as check:
        check.matches(r"src/main\.rs:[0-9]+", "trap source location is shown")
        check.matches(r"debuginfo_trap_fail", "trap kernel is in backtrace")
    gdb("kill")


def test_debug_breakpoint_sigtrap(gdb, lines):
    """debug::breakpoint() produces SIGTRAP at the expected source line."""
    gdb("set args breakpoint")
    with gdb("run") as check:
        check.matches(
            r"SIGTRAP|received signal CUDA_EXCEPTION|brkpt|Breakpoint",
            "software breakpoint signal fires",
        )
    with gdb("backtrace") as check:
        check.matches(
            f"src/main\\.rs:{lines.brkpt}",
            f"backtrace shows line {lines.brkpt}",
        )
        check.matches(r"debuginfo_breakpoint", "breakpoint kernel is in backtrace")
    gdb("info line")
    gdb("kill")


def test_null_deref_mmu_fault(gdb):
    """A null dereference produces an MMU/illegal-address fault with source."""
    gdb("set args null_deref")
    with gdb("run") as check:
        check.matches(
            r"CUDA_EXCEPTION_30|Warp MMU Fault|CUDA_EXCEPTION_[0-9]+.*Illegal",
            "MMU or illegal-address fault is caught",
        )
        check.matches(
            r"The exception was triggered at PC.*main\.rs|src/main\.rs",
            "errorPC maps to source",
        )
    with gdb("backtrace") as check:
        check.matches(r"src/main\.rs:[0-9]+|null_ptr", "null-deref source line is shown")
        check.matches(r"debuginfo_null_deref", "null-deref kernel is in backtrace")
    gdb("info line")
    gdb("kill")


def test_oob_bounds_check_trap(gdb, lines):
    """A Rust bounds trap maps to the indexing line and identifies its thread."""
    gdb("set args oob_index")
    with gdb("run") as check:
        check.matches(r"SIGTRAP|CUDA_EXCEPTION", "bounds trap fires")
        check.matches(r"thread \(4,0,0\)", "first OOB thread (4,0,0) is identified")
    with gdb("backtrace") as check:
        check.matches(
            f"src/main\\.rs:{lines.oob_idx}",
            f"backtrace points to indexing line {lines.oob_idx}",
        )
        check.not_matches(
            f"src/main\\.rs:{lines.oob_fn}",
            f"backtrace points to function signature line {lines.oob_fn}",
        )
    gdb("info registers pc errorpc")
    gdb("kill")


def test_coredump_generate_and_inspect(core_file, gdb):
    """A generated GPU coredump supports offline kernel/frame/register inspection."""
    assert core_file.exists() and core_file.stat().st_size > 0, \
        "coredump file not generated (CUDA_ENABLE_COREDUMP_ON_EXCEPTION may not be supported)"
    with gdb(f"target cudacore {core_file}") as check:
        check.matches(r"Opening GPU coredump", "coredump opens")
    gdb("info cuda devices")
    with gdb("info cuda kernels") as check:
        check.matches(r"debuginfo_null_deref", "active kernel is listed")
    with gdb("backtrace") as check:
        check.matches(
            r"null_deref.*src/main\.rs|src/main\.rs.*null_deref",
            "fault location is in backtrace",
        )
    gdb("info line")
    with gdb("info registers") as check:
        check.matches(r"^pc |^R[0-9]+ ", "registers are accessible")
    gdb("quit")
