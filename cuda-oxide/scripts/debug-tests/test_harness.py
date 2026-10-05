# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit regressions for the cuda-gdb transcript harness."""

import subprocess

import pytest

from conftest import (
    EXAMPLE_DIR, GdbSession, KnownDebugInfoFailure, breakpoint_has_location,
    build_example, gdb_value_matches,
)


def command_section(number, body=""):
    label = f"command_{number:03d}"
    return (
        f"@@CUDA_OXIDE_GDB_BEGIN:{label}@@\n"
        f"{body}\n"
        f"@@CUDA_OXIDE_GDB_END:{label}@@\n"
    )


def mocked_session(tmp_path, monkeypatch, output, returncode=0):
    completed = subprocess.CompletedProcess(
        args=["cuda-gdb"], returncode=returncode, stdout=output,
    )
    monkeypatch.setattr("conftest.subprocess.run", lambda *args, **kwargs: completed)
    return GdbSession("cuda-gdb", tmp_path / "binary", tmp_path / "session.gdb")


def test_nonzero_cuda_gdb_exit_fails(tmp_path, monkeypatch):
    session = mocked_session(
        tmp_path, monkeypatch, command_section(1), returncode=17,
    )
    session("show version")

    with pytest.raises(pytest.fail.Exception, match="exited with status 17"):
        session.execute_and_check()


def test_unchecked_command_must_reach_its_end_marker(tmp_path, monkeypatch):
    session = mocked_session(tmp_path, monkeypatch, command_section(1))
    session("set args values")
    session("run")

    with pytest.raises(pytest.fail.Exception, match="command 2.*markers are missing"):
        session.execute_and_check()


def test_final_quit_may_omit_its_end_marker(tmp_path, monkeypatch):
    quit_begin = "@@CUDA_OXIDE_GDB_BEGIN:command_002@@\n"
    session = mocked_session(
        tmp_path, monkeypatch, command_section(1) + quit_begin,
    )
    session("target cudacore fixture.core")
    session("quit")

    session.execute_and_check()


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("$1 = 1.59999", r"1\.5"),
        ("$2 = 200", r"20"),
        ("$3 = <optimized out>", r"\(0, 1\)"),
        ("$4 = <optimized out>", r"Direction::East"),
        ("$5 = <optimized out>", r"\[0, 1, 2, 3\]"),
        ("Cannot access memory at address 0x0", r"ConstantMemory .*2\.5"),
    ],
)
def test_exact_value_matching_rejects_incorrect_output(output, expected):
    assert not gdb_value_matches(output, expected)


def test_exact_value_matching_accepts_complete_expected_value():
    assert gdb_value_matches("$12 = (0, 1)\n", r"\(0, 1\)")


@pytest.mark.parametrize(
    "output",
    [
        "1 breakpoint keep y 0x1234 in debuginfo_values at src/main.rs:183",
        "\n".join([
            "1 breakpoint keep y <MULTIPLE>",
            "1.1 y 0x1234 in wrapping_add at src/main.rs:183",
            "1.2 y 0x5678 in debuginfo_values at src/main.rs:183",
        ]),
    ],
)
def test_breakpoint_location_accepts_single_and_multiple_rows(output):
    assert breakpoint_has_location(output, 1, "src/main.rs", 183)


def test_breakpoint_location_is_scoped_to_requested_breakpoint():
    output = "\n".join([
        "1 breakpoint keep y <MULTIPLE>",
        "1.1 y 0x1234 in wrapping_add at src/main.rs:184",
        "2 breakpoint keep y 0x5678 in debuginfo_values at src/main.rs:183",
    ])
    assert not breakpoint_has_location(output, 1, "src/main.rs", 183)


def known_local_session(tmp_path, monkeypatch, local="No locals.",
                        frame="deep_middle", returncode=0, include_end=True):
    output = command_section(1, frame)
    if include_end:
        output += command_section(2, local)
    else:
        output += "@@CUDA_OXIDE_GDB_BEGIN:command_002@@\n" + local
    session = mocked_session(tmp_path, monkeypatch, output, returncode)
    with session("frame 1") as check:
        check.matches(r"^deep_middle$", "inline frame is selected")
    with session("info locals") as check:
        check.matches(r"^scaled = 3$", "scaled = 3",
                      known_failure=r"\s*No locals\.\s*")
    return session


def test_only_known_missing_local_raises_expected_failure(tmp_path, monkeypatch):
    session = known_local_session(tmp_path, monkeypatch)
    with pytest.raises(KnownDebugInfoFailure, match="scaled = 3"):
        session.execute_and_check()


@pytest.mark.parametrize("overrides", [
    {"local": "scaled = 30"},
    {"local": "Cannot access memory at address 0x0"},
    {"frame": "wrong_frame"},
    {"returncode": 17},
    {"returncode": -9},
    {"include_end": False},
])
def test_known_local_does_not_hide_other_failures(tmp_path, monkeypatch, overrides):
    session = known_local_session(tmp_path, monkeypatch, **overrides)
    with pytest.raises(pytest.fail.Exception):
        session.execute_and_check()


def test_recovered_local_passes_so_strict_xfail_can_report_xpass(tmp_path, monkeypatch):
    session = known_local_session(tmp_path, monkeypatch, local="scaled = 3")
    session.execute_and_check()


def test_timeout_is_not_a_known_local_failure(tmp_path, monkeypatch):
    session = known_local_session(tmp_path, monkeypatch)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("cuda-gdb", 180, output=(
            command_section(1, "deep_middle") + command_section(2, "No locals.")
        ).encode())

    monkeypatch.setattr("conftest.subprocess.run", timeout)
    with pytest.raises(pytest.fail.Exception, match="timed out"):
        session.execute_and_check()


def test_build_uses_the_binary_fixtures_target_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("CARGO_TARGET_DIR", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("CARGO_BUILD_TARGET", "aarch64-unknown-linux-gnu")
    calls = []
    host = "x86_64-unknown-linux-gnu"
    monkeypatch.setattr("conftest.subprocess.check_output", lambda *a, **k: host + "\n")

    def build(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, returncode=0)

    monkeypatch.setattr("conftest.subprocess.run", build)
    binary = build_example("sm_120")
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ["cargo", "oxide", "build", "debug-tests"]
    assert kwargs["env"]["CARGO_TARGET_DIR"] == str(EXAMPLE_DIR / "target")
    assert kwargs["env"]["CARGO_BUILD_TARGET"] == host
    assert kwargs["env"]["CUDA_OXIDE_DEBUG"] == "full"
    assert kwargs["env"]["CUDA_OXIDE_TARGET"] == "sm_120"
    assert binary == EXAMPLE_DIR / "target" / host / "release" / "debug-tests"


def test_debugger_executes_one_fail_fast_command_file(tmp_path, monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 1, stdout=(
            "@@CUDA_OXIDE_GDB_BEGIN:command_001@@\n"
            "Undefined command: this-is-an-invalid-command\n"
        ))

    monkeypatch.setattr("conftest.subprocess.run", run)
    script = tmp_path / "session.gdb"
    session = GdbSession("cuda-gdb", tmp_path / "binary", script)
    session("this-is-an-invalid-command")
    session("show version")
    with pytest.raises(pytest.fail.Exception, match="exited with status 1"):
        session.execute_and_check()
    assert calls == [["cuda-gdb", "--batch", "-x", str(script), str(tmp_path / "binary")]]
    assert "this-is-an-invalid-command" in script.read_text()
