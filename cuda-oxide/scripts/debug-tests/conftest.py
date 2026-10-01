# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""pytest fixtures for the cuda-gdb debug-info assessment (CUDA_OXIDE_DEBUG=full).

Tests that are currently expected to fail are marked with ``pytest.mark.xfail``.
Search for that marker to find the affected tests; README.md records the full
list and the exact missing behavior. Other failures remain errors.

Environment:
  CUDA_OXIDE_CUDA_GDB   override path to cuda-gdb
  CUDA_OXIDE_TARGET     override device arch (e.g. sm_90)

cuda-gdb's raw transcript is only printed when pytest is run with -s
(--capture=no); by default it is captured and hidden like any other
test stdout.
"""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent
EXAMPLE = "debug-tests"
EXAMPLE_DIR = REPO_ROOT / "crates" / "rustc-codegen-cuda" / "examples" / EXAMPLE
DISJOINT_RS = REPO_ROOT / "crates" / "cuda-device" / "src" / "disjoint.rs"
MAIN_RS = EXAMPLE_DIR / "src" / "main.rs"
ANSI_CYAN = "\033[36m"
ANSI_RED = "\033[31m"
ANSI_RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Source-file helpers
# ---------------------------------------------------------------------------

def line_after_marker(pattern):
    """Return 1-indexed line number of the line AFTER the first match in main.rs."""
    for i, line in enumerate(MAIN_RS.read_text().splitlines(), 1):
        if re.search(pattern, line):
            return i + 1
    raise ValueError(f"Marker not found in main.rs: {pattern!r}")


def grep_line(pattern, path=None, last=False):
    """Return the 1-indexed line number of the first (or last) match."""
    path = Path(path or MAIN_RS)
    result = None
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if re.search(pattern, line):
            if not last:
                return i
            result = i
    if result is None:
        raise ValueError(f"Pattern not found in {path}: {pattern!r}")
    return result


def find_cuda_gdb():
    env_val = os.environ.get("CUDA_OXIDE_CUDA_GDB")
    if env_val:
        return env_val
    toolkit = os.environ.get("CUDA_TOOLKIT_PATH")
    if toolkit:
        candidate = Path(toolkit).expanduser() / "bin" / "cuda-gdb"
        if candidate.exists():
            return str(candidate)
    found = shutil.which("cuda-gdb")
    if found:
        return found
    return "/usr/local/cuda/bin/cuda-gdb"


def detect_arch():
    env_val = os.environ.get("CUDA_OXIDE_TARGET")
    if env_val:
        return env_val
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        )
        cc = out.strip().splitlines()[0].replace(".", "").strip()
        return f"sm_{cc}"
    except (subprocess.CalledProcessError, IndexError):
        return None


def build_example(arch):
    env = os.environ.copy()
    env["CUDA_OXIDE_DEBUG"] = "full"
    env["CUDA_OXIDE_TARGET"] = arch
    host = subprocess.check_output(
        ["rustc", "--print", "host-tuple"], text=True, cwd=str(REPO_ROOT),
    ).strip()
    # cuda-gdb runs this binary locally. An inherited cross target (including
    # Cargo's build.target configuration) must not select a different artifact.
    env["CARGO_BUILD_TARGET"] = host
    # Keep the build and binary fixture on the same artifact path, including
    # when the caller uses a shared Cargo target directory elsewhere.
    env["CARGO_TARGET_DIR"] = str(EXAMPLE_DIR / "target")
    result = subprocess.run(
        ["cargo", "oxide", "build", EXAMPLE],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), check=False,
    )
    if result.returncode != 0:
        tail = (result.stdout + result.stderr)[-3000:]
        pytest.fail(f"failed to build {EXAMPLE} with CUDA_OXIDE_DEBUG=full:\n{tail}")
    return EXAMPLE_DIR / "target" / host / "release" / EXAMPLE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cuda_gdb():
    path = find_cuda_gdb()
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        pytest.skip(f"cuda-gdb not found: {path} (set CUDA_OXIDE_CUDA_GDB)")
    return path


@pytest.fixture(scope="session")
def arch():
    try:
        subprocess.check_call(["nvidia-smi", "-L"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("no usable NVIDIA GPU / driver (nvidia-smi -L failed)")

    detected = detect_arch()
    if not detected:
        pytest.skip("could not read GPU compute capability")
    return detected


@pytest.fixture(scope="session")
def binary(arch):
    if not EXAMPLE_DIR.exists():
        pytest.fail(f"example not found: {EXAMPLE_DIR}")

    path = build_example(arch)
    if not path.exists():
        pytest.fail(f"binary not found: {path}")
    return path


@dataclass(frozen=True)
class Lines:
    values: int
    aggregates: int
    loop: int
    memory_spaces: int
    brkpt: int
    getmut_body: int
    deep_leaf_bp: int
    oob_fn: int
    oob_idx: int


@dataclass(frozen=True)
class GdbOutput:
    """Raw cuda-gdb output plus output isolated by command."""

    raw: str
    sections: dict
    returncode: Optional[int]
    timed_out: bool

    def section(self, command):
        label = command.label
        try:
            return self.sections[label]
        except KeyError:
            raise AssertionError(
                f"{command.display}: output markers are missing"
            ) from None


@dataclass(frozen=True)
class GdbExpectation:
    description: str
    pattern: Optional[str] = None
    should_match: bool = True
    transcript: bool = False
    validator: object = None
    known_failure: Optional[str] = None


class KnownDebugInfoFailure(AssertionError):
    """Only a specifically recognized missing-debug-info result failed."""


@dataclass
class GdbCommand:
    """One deferred command and the checks attached to its output."""

    number: int
    text: str
    captured: bool = False
    expectations: list = field(default_factory=list)

    @property
    def label(self):
        return f"command_{self.number:03d}"

    @property
    def display(self):
        return f"command {self.number} ({self.text!r})"

    def __enter__(self):
        self.captured = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def matches(self, pattern, description, *, known_failure=None):
        self.expectations.append(GdbExpectation(
            description, pattern, known_failure=known_failure,
        ))

    def not_matches(self, pattern, description):
        self.expectations.append(GdbExpectation(
            description, pattern, should_match=False,
        ))

    def transcript_matches(self, pattern, description):
        self.expectations.append(GdbExpectation(
            description, pattern, transcript=True,
        ))

    def transcript_not_matches(self, pattern, description):
        self.expectations.append(GdbExpectation(
            description, pattern, should_match=False, transcript=True,
        ))

    def validate(self, description, predicate):
        """Defer an arbitrary check of (this command's output, full output)."""
        self.expectations.append(GdbExpectation(
            description, validator=predicate,
        ))

    def value_matches(self, expected, description):
        """Require a ``print`` result whose complete value matches ``expected``."""
        self.validate(
            description,
            lambda text, _output: gdb_value_matches(text, expected),
        )


_SECTION_RE = re.compile(
    r"@@CUDA_OXIDE_GDB_BEGIN:(command_[0-9]+)@@\r?\n"
    r"(.*?)"
    r"@@CUDA_OXIDE_GDB_END:\1@@",
    re.DOTALL,
)


def parse_gdb_sections(output):
    """Extract output bracketed by the markers emitted for labeled commands."""
    sections = {}
    for match in _SECTION_RE.finditer(output):
        label, body = match.groups()
        if label in sections:
            raise ValueError(f"duplicate cuda-gdb output section: {label!r}")
        sections[label] = body
    return sections


def gdb_value_matches(text, expected):
    """Return whether a ``print`` command produced exactly the expected value."""
    return re.search(
        rf"^\$[0-9]+ = (?:{expected})$", text, re.MULTILINE,
    ) is not None


def breakpoint_has_location(text, number, source, line):
    """Check a breakpoint's parent/child rows for one source location."""
    row = re.compile(rf"^\s*{number}(?:\.[0-9]+)?\s")
    location = re.compile(rf"{re.escape(source)}:{line}(?:\s|$)")
    return any(
        row.search(output_line) and location.search(output_line)
        for output_line in text.splitlines()
    )


def annotate_gdb_output(output, commands, inline_failures, trailing_failures):
    """Insert commands and failed checks into the pytest-visible transcript."""
    annotated = output
    trailing = list(trailing_failures)
    unplaced_commands = []
    for command in commands:
        begin_marker = f"@@CUDA_OXIDE_GDB_BEGIN:{command.label}@@"
        command_line = (
            f"{ANSI_CYAN}[cuda-gdb command {command.number}] "
            f"{command.text}{ANSI_RESET}"
        )
        if begin_marker in annotated:
            annotated = annotated.replace(
                begin_marker, command_line + "\n" + begin_marker, 1,
            )
        else:
            unplaced_commands.append(command_line + " (not reached in transcript)")

        messages = inline_failures.get(command.label, [])
        if not messages:
            continue
        marker = f"@@CUDA_OXIDE_GDB_END:{command.label}@@"
        annotations = "".join(
            f"\n{ANSI_RED}[cuda-gdb assertion FAILED] {message}{ANSI_RESET}"
            for message in messages
        )
        if marker in annotated:
            annotated = annotated.replace(marker, marker + annotations, 1)
        else:
            trailing.extend(messages)

    if unplaced_commands or trailing:
        annotated = annotated.rstrip("\n") + "\n"
    if unplaced_commands:
        annotated += "\n".join(unplaced_commands) + "\n"
    if trailing:
        annotated += "\n".join(
            f"{ANSI_RED}[cuda-gdb assertion FAILED] {message}{ANSI_RESET}"
            for message in trailing
        )
        annotated += "\n"
    return annotated


class GdbSession:
    """Collect commands during a test, then run and check one cuda-gdb session."""

    def __init__(self, cuda_gdb, binary, script):
        self.cuda_gdb = cuda_gdb
        self.binary = binary
        self.script = script
        self.commands = []
        self.executed = False

    def __call__(self, text):
        if self.executed:
            raise RuntimeError("cannot add a command after cuda-gdb has run")
        command = GdbCommand(len(self.commands) + 1, text)
        self.commands.append(command)
        return command

    def execute_and_check(self):
        if self.executed:
            raise RuntimeError("cuda-gdb session has already run")
        self.executed = True
        if not self.commands:
            return

        script_lines = ["set pagination off", "set breakpoint pending on"]
        for command in self.commands:
            script_lines.extend([
                f"echo \\n@@CUDA_OXIDE_GDB_BEGIN:{command.label}@@\\n",
                command.text,
                f"echo \\n@@CUDA_OXIDE_GDB_END:{command.label}@@\\n",
            ])
        self.script.write_text("\n".join(script_lines) + "\n")

        env = os.environ.copy()
        toolkit = env.get("CUDA_TOOLKIT_PATH", "/usr/local/cuda")
        cuda_lib = str(Path(toolkit).expanduser() / "lib64")
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{cuda_lib}:/usr/lib/x86_64-linux-gnu:{existing}"

        try:
            # Sourcing the script makes the first command error fatal. Separate
            # --ex arguments can hide an earlier error behind a successful final
            # command, even when both output markers have been printed.
            gdb_args = [
                self.cuda_gdb, "--batch", "-x", str(self.script), str(self.binary),
            ]
            result = subprocess.run(
                gdb_args,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=180,
                cwd=str(EXAMPLE_DIR), env=env, check=False,
            )
            output = result.stdout
            returncode = result.returncode
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            def as_text(value):
                if value is None:
                    return ""
                if isinstance(value, bytes):
                    return value.decode(errors="replace")
                return value

            output = as_text(exc.stdout) + "\nTIMEOUT\n"
            returncode = None
            timed_out = True

        result = GdbOutput(
            raw=output,
            sections=parse_gdb_sections(output),
            returncode=returncode,
            timed_out=timed_out,
        )
        failures = []
        known_failures = []
        inline_failures = {}
        trailing_failures = []

        def record_failure(message, command=None, *, known=False):
            failures.append(message)
            if known:
                known_failures.append(message)
            if command is None:
                trailing_failures.append(message)
            else:
                inline_failures.setdefault(command.label, []).append(message)

        if timed_out:
            record_failure("cuda-gdb timed out")
        elif returncode != 0:
            record_failure(f"cuda-gdb exited with status {returncode}")

        for index, command in enumerate(self.commands):
            section = result.sections.get(command.label)
            final_quit = (
                index == len(self.commands) - 1
                and command.text.strip() == "quit"
            )
            begin_marker = f"@@CUDA_OXIDE_GDB_BEGIN:{command.label}@@"
            if section is None and not (final_quit and begin_marker in result.raw):
                record_failure(
                    f"{command.display}: output markers are missing", command,
                )
            if not command.captured or section is None:
                continue
            for expectation in command.expectations:
                if expectation.transcript and expectation.validator is None:
                    matched = re.search(
                        expectation.pattern, result.raw, re.MULTILINE,
                    ) is not None
                    if matched != expectation.should_match:
                        record_failure(
                            f"{command.display}: {expectation.description}",
                            command,
                        )
                    continue
                if section is None:
                    continue
                if expectation.validator is not None:
                    try:
                        passed = expectation.validator(section, result)
                    except Exception as exc:  # Report validator bugs with context.
                        record_failure(
                            f"{command.display}: {expectation.description}: {exc}",
                            command,
                        )
                    else:
                        if not passed:
                            record_failure(
                                f"{command.display}: {expectation.description}",
                                command,
                            )
                    continue

                matched = re.search(
                    expectation.pattern, section, re.MULTILINE,
                ) is not None
                if matched != expectation.should_match:
                    record_failure(
                        f"{command.display}: {expectation.description}",
                        command,
                        known=(
                            expectation.known_failure is not None
                            and re.fullmatch(expectation.known_failure, section) is not None
                        ),
                    )

        print(annotate_gdb_output(
            output, self.commands, inline_failures, trailing_failures,
        ))
        if failures:
            message = "cuda-gdb checks failed:\n- " + "\n- ".join(failures)
            # A known missing local must not hide broken frames, incorrect
            # values, missing commands, timeouts, or abnormal debugger exits.
            if len(known_failures) == len(failures):
                raise KnownDebugInfoFailure(message)
            pytest.fail(message)


@pytest.fixture(scope="session")
def lines():
    return Lines(
        values=line_after_marker(r"GDB BREAK VALUES"),
        aggregates=line_after_marker(r"GDB BREAK AGGREGATES"),
        loop=line_after_marker(r"GDB BREAK \(first iteration\)"),
        memory_spaces=line_after_marker(r"GDB BREAK MEMORY SPACES"),
        brkpt=line_after_marker(r"GDB BREAK BRKPT"),
        getmut_body=grep_line(r"size_of::<T>\(\) != 0 && idx\.is_valid", DISJOINT_RS),
        deep_leaf_bp=grep_line(r"let result: i32 = doubled\.wrapping_add"),
        oob_fn=grep_line(r"pub fn debuginfo_oob_index"),
        oob_idx=grep_line(r"let v: i32 = input\[tid\];\s*// ← OOB"),
    )


@pytest.fixture(scope="module")
def core_file(binary, tmp_path_factory):
    """Generates a GPU coredump from a null_deref crash, once per module."""
    path = tmp_path_factory.mktemp("coredump") / "debuginfo.core"
    env = os.environ.copy()
    env["CUDA_ENABLE_COREDUMP_ON_EXCEPTION"] = "1"
    env["CUDA_COREDUMP_FILE"] = str(path)
    try:
        subprocess.run(
            [str(binary), "null_deref"],
            capture_output=True, text=True, timeout=60,
            env=env, cwd=str(EXAMPLE_DIR), check=False,
        )
    except subprocess.TimeoutExpired:
        pass

    return path


@pytest.fixture
def gdb(cuda_gdb, binary, tmp_path, request):
    """Record one cuda-gdb session; pytest runs it after the test body."""
    session = GdbSession(
        cuda_gdb,
        binary,
        tmp_path / f"{request.node.name}.gdb",
    )
    request.node._cuda_oxide_gdb_session = session
    return session


@pytest.hookimpl(wrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    """Run deferred cuda-gdb commands as part of pytest's normal call phase."""
    result = yield
    session = getattr(pyfuncitem, "_cuda_oxide_gdb_session", None)
    if session is not None:
        session.execute_and_check()
    return result
