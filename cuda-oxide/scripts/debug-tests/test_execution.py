# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuda-gdb breakpoints, stepping, and frame-navigation coverage."""

import re

import pytest

from conftest import KnownDebugInfoFailure


def test_values_named_and_inlined_frames(gdb):
    """PTX, named breakpoints, inline attribution, and frame switching work."""
    gdb("set args values")
    gdb("break debuginfo_values")
    with gdb("run") as check:
        check.matches(r"CUDA thread hit Breakpoint", "break by kernel name hits")
        check.matches(r"debuginfo_values", "stop is in debuginfo_values")
        check.transcript_not_matches(
            r"INVALID_PTX|JIT compilation failed|No device code|CUDA_ERROR_INVALID_PTX",
            "PTX JIT error present",
        )

    gdb("disable 1")
    with gdb("break cuda_device::DisjointSlice::<'_, i32>::get_mut") as check:
        check.matches(
            r"Breakpoint 2 at 0x[0-9a-f]+: file .*disjoint\.rs, line [0-9]+",
            "the emitted get_mut name resolves",
        )
    with gdb("continue") as check:
        check.matches(
            r"CUDA thread hit Breakpoint 2",
            "fully-qualified get_mut breakpoint fires, including at an inline callee",
        )

    gdb("disable 2")
    with gdb("backtrace") as check:
        check.matches(
            r"^#[0-9]+ .*DisjointSlice.*get_mut",
            "inlined get_mut is visible",
        )
        check.matches(
            r"^#[0-9]+ .*debuginfo_values",
            "kernel caller frame is visible",
        )
        check.matches(r"get_mut.*disjoint\.rs", "get_mut is attributed to disjoint.rs")
        check.not_matches(
            r"get_mut.*src/main\.rs",
            "get_mut is incorrectly attributed to main.rs",
        )

    with gdb("frame function debuginfo_values") as check:
        check.matches(r"^#[0-9]+ .*debuginfo_values", "kernel frame is selected")
    with gdb("info args") as check:
        check.matches(
            r"input = 0x[0-9a-f]+",
            "kernel frame shows input pointer argument",
        )
    with gdb("info locals") as check:
        check.matches(r"^idx =", "kernel frame shows idx local")
    with gdb("down") as check:
        check.matches(
            r"^#[0-9]+ .*DisjointSlice.*get_mut",
            "get_mut frame is selected relative to its kernel caller",
        )
    with gdb("info args") as check:
        check.matches(r"^self =", "get_mut frame shows its self argument")
        check.matches(r"^idx =", "get_mut frame shows its index argument")
    gdb("kill")


def test_values_break_on_launch(gdb):
    """set cuda break_on_launch application stops in the values launch."""
    gdb("set args values")
    gdb("set cuda break_on_launch application")
    with gdb("run") as launch:
        launch.matches(
            r"kernel entry function breakpoint|CUDA thread hit",
            "break_on_launch fires",
        )
    gdb("info line")
    with gdb("backtrace") as check:
        check.validate(
            "values kernel frame is accessible",
            lambda text, output: re.search(
                r"debuginfo_values|#0.*src/main",
                output.section(launch) + text,
                re.MULTILINE,
            ) is not None,
        )
    gdb("kill")


def test_loop_inspection_and_step(gdb, lines):
    """First-iteration loop locals are visible and next advances a source line."""
    gdb("set args loop")
    gdb(f"break src/main.rs:{lines.loop}")
    gdb("run")
    gdb("disable 1")
    with gdb("frame") as before:
        before.matches(
            f"src/main\\.rs:{lines.loop}",
            f"current source line is {lines.loop}",
        )
    with gdb("info locals") as check:
        check.matches(r"^i =", "info locals shows i")
        check.matches(r"^acc =", "info locals shows acc")
    with gdb("print i") as check:
        check.matches(r"^\$[0-9]+ = 0$", "i = 0")
    with gdb("print acc") as check:
        check.matches(r"^\$[0-9]+ = 0$", "acc = 0")
    with gdb("print n") as check:
        check.matches(r"^\$[0-9]+ = 5$", "n = 5")
    gdb("next")
    with gdb("frame") as check:
        check.matches(r"src/main\.rs:[0-9]+", "source line after next is reported")
        check.matches(r"debuginfo_loop", "stepping remains in loop kernel")
        check.validate(
            "next advances to a different source line",
            lambda text, output: re.findall(r"src/main\.rs:([0-9]+)", text)[-1]
            != re.findall(r"src/main\.rs:([0-9]+)", output.section(before))[-1],
        )
    gdb("kill")


@pytest.mark.xfail(
    strict=True,
    raises=KnownDebugInfoFailure,
    reason="generated debug info omits scaled from the inlined deep_middle frame",
)
def test_mixed_physical_and_inlined_callstack(gdb, lines):
    """A forced inline frame remains visible between physical call frames."""
    gdb("set args deep_stack")
    gdb(f"break src/main.rs:{lines.deep_leaf_bp}")
    with gdb("run") as check:
        check.matches(r"CUDA thread hit Breakpoint", "deep_leaf breakpoint fires")
    gdb("disable 1")
    with gdb("backtrace") as check:
        check.matches(
            r"^#0 .*debug_tests::deep_leaf.*\n"
            r"#1 .*?(?:debug_tests::)?deep_middle.*\n"
            r"#2 .*debug_tests::deep_outer.*\n"
            r"#3 .*debuginfo_deep_stack",
            "backtrace has leaf, inline, outer, and kernel frames in order",
        )
    with gdb("frame 0") as check:
        check.matches(r"^#0 .*debug_tests::deep_leaf", "leaf frame is selected")
    with gdb("info args") as check:
        check.matches(r"^v = 3$", "frame 0 argument v = 3")
    gdb("info locals")
    with gdb("frame 1") as check:
        check.matches(
            r"^#1 .*?(?:debug_tests::)?deep_middle",
            "inline middle frame is selected",
        )
    gdb("info args")
    with gdb("info locals") as check:
        check.matches(
            r"^scaled = 3$",
            "inline frame local scaled = 3",
            known_failure=r"\s*No locals\.\s*",
        )
    with gdb("frame 2") as check:
        check.matches(r"^#2 .*debug_tests::deep_outer", "outer frame is selected")
    gdb("info args")
    with gdb("info locals") as check:
        check.matches(r"^offset = 4$", "frame 2 local offset = 4")
    gdb("kill")
