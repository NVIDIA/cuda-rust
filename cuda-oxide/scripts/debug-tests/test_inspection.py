# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuda-gdb value, type, source, and memory-inspection coverage."""

from conftest import breakpoint_has_location


def test_values_source_inspection_and_exit(gdb, lines):
    """One values stop covers primitive locals, source commands, CUDA focus, and exit."""
    gdb("set args values")
    with gdb(f"break src/main.rs:{lines.values}") as check:
        check.matches(
            f"src/main\\.rs(?::|, line ){lines.values}",
            f"source breakpoint resolves at line {lines.values}",
        )
    with gdb("run") as check:
        check.matches(r"CUDA thread hit Breakpoint", "source breakpoint hits")
        check.matches(
            f"src/main\\.rs:{lines.values}",
            f"stop is shown at line {lines.values}",
        )

    gdb("disable 1")
    # On the tested cuda-gdb versions this multi-location source breakpoint
    # stops in the inlined `wrapping_add` frame. `frame function
    # debuginfo_values` cannot distinguish its caller because the inline frame
    # shares the kernel's PC and leaves `wrapping_add` selected. Move to the
    # immediate caller instead, then verify it before inspecting the locals.
    with gdb("up") as check:
        check.matches(
            r"^#[0-9]+ .*debuginfo_values",
            "the values kernel frame is selected",
        )
    with gdb("info locals") as check:
        check.matches(r"^x =", "scalar local x is visible")
        check.matches(r"^y =", "scalar local y is visible")
        check.matches(r"^flag =", "scalar local flag is visible")
        check.matches(r"^big =", "scalar local big is visible")
        check.matches(r"^tid =", "scalar local tid is visible")
        check.matches(r"^buf =", "array local buf is visible")
        check.matches(r"^ptr = 0x[0-9a-f]{6,}", "ptr has a valid GPU address")
        check.matches(r"^fptr = 0x[0-9a-f]{6,}", "fptr has a valid GPU address")
        check.matches(r"^scaled =", "inline-helper result is visible")

    with gdb("print x") as check:
        check.value_matches(r"1", "x = 1")
    with gdb("print y") as check:
        check.value_matches(r"1\.5", "y = 1.5")
    with gdb("print flag") as check:
        check.value_matches(r"(?:true|1)", "flag = true")
    with gdb("print big") as check:
        check.value_matches(r"0", "big = 0")

    pointer_value = r"(?:\([^)]+\) )?0x[0-9a-f]{6,}"
    with gdb("print slot") as check:
        check.value_matches(pointer_value, "slot is a valid GPU address")
    with gdb("print buf") as check:
        check.value_matches(r"[\[{]0, 10, 20, 30[\]}]", "buf shows all elements")
    with gdb("print buf[0]") as check:
        check.value_matches(r"0", "buf[0] = 0")
    with gdb("print buf[2]") as check:
        check.value_matches(r"20", "buf[2] = 20")
    with gdb("print ptr") as check:
        check.value_matches(pointer_value, "ptr is printable")
    with gdb("print fptr") as check:
        check.value_matches(pointer_value, "fptr is printable")
    with gdb("print scaled") as check:
        check.value_matches(r"4", "scaled = 4")

    with gdb("whatis x") as check:
        check.matches(r"type = i32", "x has type i32")
    with gdb("whatis y") as check:
        check.matches(r"type = f32", "y has type f32")
    with gdb("whatis flag") as check:
        check.matches(r"type = bool", "flag has type bool")
    with gdb("whatis big") as check:
        check.matches(r"type = u64", "big has type u64")
    with gdb("whatis slot") as check:
        check.matches(r"type = \*mut i32", "slot has type &mut i32")
    with gdb("whatis buf") as check:
        check.matches(r"type = \[i32; 4\]", "buf has type [i32; 4]")
    with gdb("whatis ptr") as check:
        check.matches(r"type = \*const i32|\*mut i32", "ptr has raw i32 pointer type")

    with gdb("info args") as check:
        check.matches(r"input = 0x[0-9a-f]+", "input argument is a pointer")
        check.matches(r"fdata = 0x[0-9a-f]+", "fdata argument is a pointer")
        check.matches(r"DisjointSlice \{ptr:", "out argument is DisjointSlice")
        check.matches(r"len: [0-9]+", "DisjointSlice shows len")
    with gdb("backtrace") as check:
        check.matches(r"#[0-9]+ +debuginfo_values", "backtrace shows values kernel")
        check.matches(r"src/main\.rs:[0-9]+", "backtrace has file:line")
    with gdb("list") as check:
        check.matches(
            r"let scaled: i32|gpu_assert!|wrapping_add",
            "list shows nearby Rust source",
        )
    with gdb(f"info line src/main.rs:{lines.values}") as check:
        check.matches(r"starts at address 0x[0-9a-f]+", "info line shows start address")
        check.matches(r"ends at 0x[0-9a-f]+", "info line shows end address")
        check.matches(r"src/main\.rs", "info line references src/main.rs")
    with gdb("info breakpoints") as check:
        check.validate(
            "breakpoint table lists source breakpoint",
            lambda text, _output: breakpoint_has_location(
                text, 1, "src/main.rs", lines.values,
            ),
        )
    with gdb("info frame") as check:
        check.matches(r"source language rust", "frame reports Rust")
    with gdb("show language") as check:
        check.matches(r"currently rust", "current language is Rust")
    with gdb("cuda thread") as check:
        check.matches(r"thread \([0-9,]+\)", "CUDA thread coordinates")
    with gdb("cuda block") as check:
        check.matches(r"block \([0-9,]+\)", "CUDA block coordinates")
    with gdb("cuda warp") as check:
        check.matches(r"^warp [0-9]+", "CUDA warp number")
    with gdb("cuda lane") as check:
        check.matches(r"^lane [0-9]+", "CUDA lane number")
    with gdb("info cuda threads") as check:
        check.matches(r"Kernel [0-9]+", "CUDA thread table shows kernel")

    gdb("delete breakpoints")
    with gdb("continue") as check:
        check.matches(r"exited normally", "inferior exits normally")
        check.matches(r"PASS", "combined values kernel passes host correctness check")


def test_aggregates_inspection(gdb, lines):
    """Runtime/user structs, tuple, and enum are inspectable at one stop."""
    gdb("set args aggregates")
    gdb(f"break src/main.rs:{lines.aggregates}")
    with gdb("run") as check:
        check.matches(r"CUDA thread hit Breakpoint", "aggregate breakpoint fires")
    gdb("disable 1")
    with gdb("info locals") as check:
        check.matches(r"^pair =", "tuple pair is visible")
        check.matches(r"^p =", "Point local p is visible")
        check.matches(r"^r =", "Rect local r is visible")
        check.matches(r"^state =", "NavState local state is visible")
        check.matches(r"^dir_a =", "Direction local dir_a is visible")
    with gdb("print idx") as check:
        check.matches(r"ThreadIndex \{raw: 0\}", "idx is ThreadIndex{raw:0}")
    with gdb("print out") as check:
        check.matches(
            r"DisjointSlice \{ptr: 0x[0-9a-f]+, len: [0-9]+\}",
            "out is DisjointSlice{ptr,len}",
        )
    with gdb("print pair") as check:
        check.value_matches(r"\(0, 1\)", "tuple pair has the expected value")
    with gdb("print dir_a") as check:
        check.value_matches(
            r"(?:debug_tests::)?Direction::East",
            "enum dir_a has the expected value",
        )
    with gdb("ptype idx") as check:
        check.matches(r"struct ThreadIndex", "ptype shows ThreadIndex")
        check.matches(r"raw:", "ThreadIndex shows raw field")
    with gdb("ptype Point") as check:
        check.matches(r"struct Point", "ptype Point shows its definition")
    with gdb("ptype Rect") as check:
        check.matches(r"struct Rect", "ptype Rect shows its definition")
    gdb("ptype NavState")
    gdb("kill")


def test_memory_spaces_inspection(gdb, lines):
    """Shared, global, and constant memory are inspectable at one stop."""
    gdb("set args memory_spaces")
    gdb(f"break src/main.rs:{lines.memory_spaces}")
    with gdb("run") as check:
        check.matches(r"CUDA thread hit", "GPU breakpoint fires")
    gdb("disable 1")
    # Raw-pointer arithmetic stops in an inline `add` frame at this source
    # location. Select and verify its kernel caller before reading the locals.
    with gdb("up") as check:
        check.matches(
            r"^#[0-9]+ .*debuginfo_memory_spaces",
            "the memory-spaces kernel frame is selected",
        )
    with gdb("info locals") as check:
        check.matches(
            r"^TILE = \[0, 1, 2, 3, 4, 5, 6, 7, 0 <repeats 24 times>\]$",
            "TILE has the expected shared-memory contents",
        )
        check.matches(r"^i =", "scalar local i is visible")
    with gdb("print TILE") as check:
        check.value_matches(
            r"\[0, 1, 2, 3, 4, 5, 6, 7, 0 <repeats 24 times>\]",
            "TILE has the expected shared-memory contents",
        )
    # This matches host Rust debugging: GDB resolves a static in the current
    # module unqualified, but a crate-root static imported with `use super::*`
    # still requires its DWARF-qualified name.
    with gdb("print debug_tests::GLOBAL_COUNTER") as check:
        check.value_matches(r"7", "GLOBAL_COUNTER has the expected value")
    with gdb("ptype debug_tests::GLOBAL_COUNTER") as check:
        check.matches(r"type = @global u64", "GLOBAL_COUNTER is in global memory")
    with gdb("print coeff_val") as check:
        check.value_matches(r"2\.5", "coeff_val has the expected value")
    with gdb("print COEFF") as check:
        check.value_matches(
            r"ConstantMemory \{0: UnsafeCell \{value: 2\.5\}\}",
            "COEFF has the expected constant-memory value",
        )
    gdb("kill")
