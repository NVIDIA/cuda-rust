#!/usr/bin/env python3
"""Validate generated Rustlantis programs with Miri Tree Borrows."""

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

RUSTLANTIS = Path(__file__).resolve().parent.parent / "rustlantis"

CONFIG = """
bb_max_len = 12
max_switch_targets = 3
max_bb_count = 6
max_bb_count_hard = 12
max_fn_count = 8
max_args_count = 6
var_dump_chance = 0.0
static_count = 8
tuple_max_len = 3
array_max_len = 3
struct_max_fields = 3
adt_max_variants = 2
composite_count = 8
adt_count = 0

[backends]
"""


def execute(command, cwd, env):
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            command, 124, "", "Process timed out after 180 seconds."
        )


def has_reference_call(source):
    signatures = re.findall(
        r"(?m)^\s*(?:pub\s+)?fn\s+(fn\d+)\((.*?)\)\s*->",
        source,
    )
    reference_functions = {
        name for name, parameters in signatures
        if "&'static " in parameters
    }
    called_functions = set(
        re.findall(r"\bCall\([^\n]*\b(fn\d+)\(", source)
    )
    return bool(reference_functions & called_functions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--toolchain", default="nightly-2026-10-11")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("/tmp/rustlantis-miri-regressions"),
    )
    args = parser.parse_args()

    if args.start < 0 or args.count < 1:
        parser.error("--start must be nonnegative and --count must be positive")

    env = os.environ.copy()
    env.pop("RUSTFLAGS", None)
    env.pop("RUSTC_WRAPPER", None)

    miri_env = env.copy()
    miri_env["MIRIFLAGS"] = "-Zmiri-tree-borrows"

    passed = 0
    static_programs = 0
    reference_call_programs = 0
    overlapping_programs = 0

    with tempfile.TemporaryDirectory(prefix="rustlantis-miri-") as directory:
        work = Path(directory)
        (work / "src").mkdir()
        (work / "config.toml").write_text(CONFIG)
        (work / "Cargo.toml").write_text(
            '[package]\n'
            'name = "rustlantis-miri-sweep"\n'
            'version = "0.1.0"\n'
            'edition = "2024"\n'
        )

        negative_fixture = (
            Path(__file__).resolve().parent
            / "tests"
            / "miri_static_provenance_invalid.rs"
        )

        (work / "src/main.rs").write_text(negative_fixture.read_text())

        negative_result = execute(
            [
                "cargo", f"+{args.toolchain}", "miri",
                "run", "--quiet",
                "--manifest-path", str(work / "Cargo.toml"),
            ],
            work,
            miri_env,
        )

        if (
            negative_result.returncode == 0
            or "Undefined Behavior" not in negative_result.stderr
        ):
            print("FAIL: Miri did not detect the expected provenance violation")
            print(negative_result.stderr[-3000:])
            return 1

        print("Miri negative control: PASS (UB detected)", flush=True)

        fixture_path = (
            Path(__file__).resolve().parent
            / "tests"
            / "miri_static_provenance.rs"
        )

        (work / "src/main.rs").write_text(fixture_path.read_text())

        fixture_result = execute(
            [
                "cargo", f"+{args.toolchain}", "miri",
                "run", "--quiet",
                "--manifest-path", str(work / "Cargo.toml"),
            ],
            work,
            miri_env,
        )

        if fixture_result.returncode:
            args.artifacts.mkdir(parents=True, exist_ok=True)
            log = args.artifacts / "miri_static_provenance.log"
            log.write_text(fixture_result.stdout + fixture_result.stderr)
            print("FAIL: deterministic provenance fixture")
            print(f"Log: {log}")
            print(fixture_result.stderr[-4000:])
            return 1

        print("Deterministic provenance fixture: PASS", flush=True)

        # Regression: seed 5 previously generated an invalid local
        # reference reborrow even with static generation disabled.
        assert CONFIG.count("static_count = 8\n") == 1
        zero_static_config = CONFIG.replace("static_count = 8\n", "")
        (work / "config.toml").write_text(zero_static_config)

        zero_generated = execute(
            [
                "cargo", f"+{args.toolchain}", "run", "--quiet",
                "--manifest-path", str(RUSTLANTIS / "Cargo.toml"),
                "-p", "generate", "--", "--debug", "5",
            ],
            work,
            env,
        )

        zero_source = zero_generated.stdout

        unexpected_static = bool(
            re.search(r"\bStatic(?:Mut)?\(static\d+\)", zero_source)
            or re.search(
                r"(?m)^\s*static(?:\s+mut)?\s+static\d+\s*:",
                zero_source,
            )
        )

        if zero_generated.returncode:
            zero_result = zero_generated
            zero_stage = "generator"
        elif unexpected_static:
            zero_result = subprocess.CompletedProcess(
                [],
                1,
                "",
                "Unexpected static declaration or operand with static_count=0",
            )
            zero_stage = "configuration"
        else:
            (work / "src/main.rs").write_text(zero_source)

            zero_result = execute(
                [
                    "cargo", f"+{args.toolchain}", "miri",
                    "run", "--quiet",
                    "--manifest-path", str(work / "Cargo.toml"),
                ],
                work,
                miri_env,
            )
            zero_stage = "miri"

        if zero_result.returncode:
            args.artifacts.mkdir(parents=True, exist_ok=True)

            (args.artifacts / "seed-5-zero-static.rs").write_text(
                zero_source
            )

            (args.artifacts / "seed-5-zero-static.log").write_text(
                zero_result.stdout + zero_result.stderr
            )

            print(
                f"FAIL: zero-static seed 5 regression "
                f"(stage={zero_stage})"
            )
            print(zero_result.stderr[-3000:])
            return 1

        print("Zero-static seed 5 regression: PASS", flush=True)

        # Restore the configured static workload for the requested sweep.
        (work / "config.toml").write_text(CONFIG)

        for seed in range(args.start, args.start + args.count):
            generated = execute(
                [
                    "cargo", f"+{args.toolchain}", "run", "--quiet",
                    "--manifest-path", str(RUSTLANTIS / "Cargo.toml"),
                    "-p", "generate", "--", "--debug", str(seed),
                ],
                work,
                env,
            )

            source = generated.stdout
            if generated.returncode:
                failure = generated
                stage = "generator"
            else:
                uses_static = bool(
                    re.search(r"\bStatic(?:Mut)?\(static\d+\)", source)
                )
                calls_with_refs = has_reference_call(source)

                static_programs += uses_static
                reference_call_programs += calls_with_refs
                overlapping_programs += uses_static and calls_with_refs

                (work / "src/main.rs").write_text(source)

                failure = execute(
                    [
                        "cargo", f"+{args.toolchain}", "miri",
                        "run", "--quiet",
                        "--manifest-path", str(work / "Cargo.toml"),
                    ],
                    work,
                    miri_env,
                )
                stage = "miri"

            if failure.returncode:
                args.artifacts.mkdir(parents=True, exist_ok=True)
                prefix = args.artifacts / f"seed-{seed}"

                if source:
                    prefix.with_suffix(".rs").write_text(source)

                prefix.with_suffix(".log").write_text(
                    failure.stdout + failure.stderr
                )

                print(f"FAIL: seed={seed}, stage={stage}")
                print(f"Artifacts: {prefix}")
                print(failure.stderr[-2500:])
                break

            passed += 1

            if passed % 25 == 0:
                print(f"Miri PASS: {passed}/{args.count}", flush=True)

    print("\n=== MIRI SWEEP SUMMARY ===")
    print(f"Passed: {passed}/{args.count}")
    print(f"Programs using static operands: {static_programs}")
    print(f"Programs calling functions with reference parameters: {reference_call_programs}")
    print(f"Programs containing both features: {overlapping_programs}")

    return 0 if passed == args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
