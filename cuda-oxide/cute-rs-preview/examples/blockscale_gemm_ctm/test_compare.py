#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only acceptance tests for Nsight Systems comparison evidence."""

import contextlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest import mock

try:
    import compare
except ImportError:
    compare = None


def gpu_report(**changes):
    report = {
        "verification": "bit-exact-pass", "checked_outputs": 4194304,
        "mnk": [2048, 2048, 1024], "compute_capability": [12, 0],
        "ctas": 170, "warmup": 2, "samples": 3, "replay_verified": True,
        "median_ms": 999, "p10_ms": 998, "p90_ms": 1000, "tflops": 1,
    }
    report.update(changes)
    return report


def trace(path, launches):
    """Insert in reverse order so readers must order by device timestamp."""
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
                   "(start INTEGER, end INTEGER, shortName INTEGER)")
        names = {name: i for i, name in enumerate(dict.fromkeys(x[0] for x in launches))}
        db.executemany("INSERT INTO StringIds VALUES (?, ?)",
                       [(i, name) for name, i in names.items()])
        db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?)",
                       [(i * 1000000, i * 1000000 + duration, names[name])
                        for i, (name, duration) in reversed(list(enumerate(launches)))])


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(compare, "comparison runner has not been implemented")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def write_report(self, report):
        path = self.directory / "result.json"
        path.write_text(json.dumps(report))
        return path

    def test_accepts_full_initial_and_replay_verification(self):
        started = time.time_ns()
        path = self.write_report(gpu_report())
        result = compare.validate_report(path, 170, 2, 3, started)
        self.assertEqual(result["checked_outputs"], 4194304)

    def test_rejects_failed_incomplete_or_mismatched_verification(self):
        for key, bad_value in [
            ("verification", "failed"), ("checked_outputs", 1),
            ("mnk", [2048, 2048, 2048]), ("compute_capability", [12, 1]),
            ("ctas", 169), ("warmup", 3), ("samples", 2),
            ("replay_verified", False), ("replay_verified", 1),
        ]:
            with self.subTest(key=key, bad_value=bad_value):
                path = self.write_report(gpu_report(**{key: bad_value}))
                with self.assertRaises(ValueError):
                    compare.validate_report(path, 170, 2, 3, 0)
        for key in gpu_report():
            if key in {"median_ms", "p10_ms", "p90_ms", "tflops"}:
                continue
            with self.subTest(missing=key):
                report = gpu_report()
                del report[key]
                path = self.write_report(report)
                with self.assertRaises(ValueError):
                    compare.validate_report(path, 170, 2, 3, 0)

    def test_rejects_stale_missing_and_malformed_reports(self):
        path = self.directory / "result.json"
        with self.assertRaises(ValueError):
            compare.validate_report(path, 170, 2, 3, 0)
        self.write_report(gpu_report())
        os.utime(path, ns=(1, 1))
        with self.assertRaisesRegex(ValueError, "stale"):
            compare.validate_report(path, 170, 2, 3, time.time_ns())
        for contents in ["{", "[]", "null"]:
            path.write_text(contents)
            with self.assertRaises(ValueError):
                compare.validate_report(path, 170, 2, 3, 0)

    def test_discards_correctness_and_warmup_launches_in_timestamp_order(self):
        path = self.directory / "trace.sqlite"
        trace(path, [("blockscale_gemm", n) for n in [900000, 800000, 700000, 31000, 11000, 21000]])
        self.assertEqual(compare.kernel_samples(path, 2, 3), [31.0, 11.0, 21.0])

    def test_accepts_explicit_mangled_name_for_all_132_launches(self):
        name = "kernel_cutlass_blockscale_gemm_1..."
        path = self.directory / "mangled.sqlite"
        trace(path, [(name, 900000)] * 101 + [(name, n * 1000) for n in range(1, 32)])
        self.assertEqual(compare.kernel_samples(path, 100, 31, expected_name=name),
                         list(range(1, 32)))

    def test_explicit_kernel_name_rejects_partial_matches_and_wrong_counts(self):
        expected = "kernel_cutlass_blockscale_gemm_1..."
        for index, wrong in enumerate(["blockscale_gemm", expected + "_other",
                                       "prefix_" + expected, expected[:-1]]):
            with self.subTest(wrong=wrong):
                path = self.directory / f"wrong-mangled-{index}.sqlite"
                trace(path, [(wrong, 1000)] + [(expected, 1000)] * 131)
                with self.assertRaisesRegex(ValueError, "unexpected kernel shortName"):
                    compare.kernel_samples(path, 100, 31, expected_name=expected)
        for count in (131, 133):
            with self.subTest(count=count):
                path = self.directory / f"wrong-count-{count}.sqlite"
                trace(path, [(expected, 1000)] * count)
                with self.assertRaisesRegex(ValueError, "expected 132 kernel launches"):
                    compare.kernel_samples(path, 100, 31, expected_name=expected)

    def test_rejects_wrong_kernel_or_wrong_launch_count(self):
        for name, launches in [
            ("wrong", [("other_kernel", 1000)] * 6),
            ("missing", [("blockscale_gemm", 1000)] * 5),
            ("extra", [("blockscale_gemm", 1000)] * 7),
            ("mixed", [("blockscale_gemm", 1000)] * 5 + [("other_kernel", 1000)]),
            ("invalid-duration", [("blockscale_gemm", 1000)] * 5 + [("blockscale_gemm", -1)]),
        ]:
            with self.subTest(name=name):
                path = self.directory / (name + ".sqlite")
                trace(path, launches)
                with self.assertRaises(ValueError):
                    compare.kernel_samples(path, 2, 3)

    def test_rejects_unresolved_names_and_missing_trace_without_creating_it(self):
        path = self.directory / "missing.sqlite"
        with self.assertRaises(ValueError):
            compare.kernel_samples(path, 2, 3)
        self.assertFalse(path.exists())
        trace(path, [("blockscale_gemm", 1000)] * 6)
        with sqlite3.connect(path) as db:
            db.execute("DELETE FROM StringIds")
        with self.assertRaises(ValueError):
            compare.kernel_samples(path, 2, 3)

    def test_percentiles_use_rust_rounded_rank_including_even_median(self):
        result = compare.summarize_samples([10, 1, 9, 2, 8, 3, 7, 4, 6, 5])
        self.assertEqual(result, {"median_us": 6, "p10_us": 2, "p90_us": 9})
        self.assertEqual(compare.summarize_samples([7]),
                         {"median_us": 7, "p10_us": 7, "p90_us": 7})

    def test_aggregate_uses_round_medians_and_keeps_every_sample(self):
        entries = [
            {"label": "rust", "samples_us": [1, 100, 1], "median_us": 1},
            {"label": "cpp", "samples_us": [4, 5, 6], "median_us": 5},
            {"label": "rust", "samples_us": [9, 10, 11], "median_us": 10},
        ]
        result = compare.aggregate_rounds(entries)
        self.assertEqual(result["rust"]["round_medians_us"], [1, 10])
        self.assertEqual(result["rust"]["median_of_round_medians_us"], 10)
        self.assertEqual(result["rust"]["samples_us"], [1, 100, 1, 9, 10, 11])
        self.assertEqual(result["cpp"]["median_of_round_medians_us"], 5)

    def test_rotation_balances_starting_order(self):
        self.assertEqual(compare.execution_order(["a", "b", "c"], 4),
                         [["a", "b", "c"], ["b", "c", "a"],
                          ["c", "a", "b"], ["a", "b", "c"]])

    def arguments(self):
        for name in ("rust", "cpp"):
            path = self.directory / name
            path.write_text(name)
            path.chmod(0o755)
        return ["--binary", f"rust={self.directory / 'rust'}",
                "--binary", f"cpp={self.directory / 'cpp'}",
                "--output-dir", str(self.directory / "evidence"),
                "--rounds", "2", "--warmup", "2", "--samples", "3"]

    def fake_command(self, command, timeout, environment, log_path):
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "1")
        log_path.write_text("test tool output\n")
        if command[1:2] == ["profile"]:
            result = Path(command[command.index("--json") + 1])
            result.write_text(json.dumps(gpu_report()))
            prefix = Path(command[command.index("--output") + 1])
            prefix.with_suffix(".nsys-rep").write_bytes(b"synthetic trace")
        elif command[1:2] == ["export"]:
            path = Path(command[command.index("--output") + 1])
            trace(path, [("blockscale_gemm", n) for n in [900000, 800000, 700000, 31000, 11000, 21000]])
        return 0

    def test_runner_persists_serial_order_active_timings_and_fresh_evidence(self):
        arguments = self.arguments()
        with mock.patch.object(compare, "run_command", side_effect=self.fake_command), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(compare.main(arguments), 0)
            self.assertEqual(compare.main(arguments), 0)
        runs = sorted((self.directory / "evidence").iterdir())
        self.assertEqual(len(runs), 2)
        for directory in runs:
            result = json.loads((directory / "report.json").read_text())
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["kernel_names"],
                             {"rust": "blockscale_gemm", "cpp": "blockscale_gemm"})
            self.assertEqual(result["execution_order"], [["rust", "cpp"], ["cpp", "rust"]])
            self.assertEqual([r["label"] for r in result["runs"]], ["rust", "cpp", "cpp", "rust"])
            self.assertEqual(result["summary"]["rust"]["median_of_round_medians_us"], 21)
            self.assertEqual(result["summary"]["rust"]["samples_us"], [31, 11, 21, 31, 11, 21])
            self.assertEqual(len(result["binaries"]["rust"]["sha256"]), 64)
            self.assertIn("nsys_version", result)
            self.assertIn("gpu_snapshot", result)
            self.assertTrue(all(r["status"] == "passed" for r in result["runs"]))

    def test_runner_records_and_applies_exact_kernel_name_per_label(self):
        name = "kernel_cutlass_blockscale_gemm_1..."

        def mangled_command(command, timeout, environment, log_path):
            status = self.fake_command(command, timeout, environment, log_path)
            if command[1:2] == ["export"]:
                path = Path(command[command.index("--output") + 1])
                if path.stem.endswith("-cpp"):
                    with sqlite3.connect(path) as db:
                        db.execute("UPDATE StringIds SET value = ?", (name,))
            return status

        with mock.patch.object(compare, "run_command", side_effect=mangled_command), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(compare.main(self.arguments() + [
                "--kernel-name", "rust=blockscale_gemm", "--kernel-name", f"cpp={name}",
            ]), 0)
        directory, = (self.directory / "evidence").iterdir()
        result = json.loads((directory / "report.json").read_text())
        self.assertEqual(result["kernel_names"], {"rust": "blockscale_gemm", "cpp": name})
        self.assertEqual(result["kernel_name_field"], "shortName")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["summary"]["cpp"]["samples_us"], [31, 11, 21, 31, 11, 21])

    def test_kernel_name_options_reject_unknown_empty_and_duplicate_labels(self):
        arguments = self.arguments()
        for specifications in [["unknown=kernel"], ["rust="], ["rust=   "], ["rust"],
                               ["=kernel"], ["rust=first", "rust=second"]]:
            extra = [value for specification in specifications
                     for value in ("--kernel-name", specification)]
            with self.subTest(specifications=specifications), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    mock.patch.object(compare, "run_command") as run:
                with self.assertRaises(SystemExit) as caught:
                    compare.main(arguments + extra)
                self.assertEqual(caught.exception.code, 2)
                run.assert_not_called()

    def test_missing_correctness_report_stops_before_accepting_timings(self):
        def missing_report(command, timeout, environment, log_path):
            status = self.fake_command(command, timeout, environment, log_path)
            if command[1:2] == ["profile"]:
                Path(command[command.index("--json") + 1]).unlink()
            return status

        with mock.patch.object(compare, "run_command", side_effect=missing_report), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(compare.main(self.arguments()), 1)
        directory, = (self.directory / "evidence").iterdir()
        result = json.loads((directory / "report.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["runs"]), 1)
        self.assertEqual(result["runs"][0]["status"], "failed")
        self.assertNotIn("summary", result)
        self.assertNotIn("samples_us", result["runs"][0])

    def test_invalid_cli_is_rejected_before_running_tools(self):
        arguments = self.arguments()
        for extra in [["--rounds", "0"], ["--warmup", "-1"],
                      ["--samples", "0"], ["--ctas", "257"],
                      ["--timeout", "nan"], ["--timeout", "0"],
                      ["--binary", f"rust={self.directory / 'rust'}"]]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    compare.main(arguments + extra)
                self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
