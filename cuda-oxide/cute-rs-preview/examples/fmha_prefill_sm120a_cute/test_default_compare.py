#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the default FMHA comparison and its runner protocol."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixture = load("default_compare_fixture", HERE / "reference/launch.py")
comparison = load("default_compare", HERE / "compare.py") if (HERE / "compare.py").exists() else None


class DefaultComparisonTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(comparison, "default FMHA comparison is not implemented")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.binary = self.directory / "rust"
        self.binary.write_text("test runner")
        self.binary.chmod(0o755)
        self.mutate = lambda report: None

    def run_runner(self, command, **kwargs):
        rust = command[0] == str(self.binary)
        if rust:
            self.assertEqual(command[1], "--rust-only", "Rust subprocess must not recurse")
            cli = command[2:]
        else:
            self.assertNotIn("--rust-only", command)
            cli = command[2:]
        launch = (HERE if self.workload == "prefill" else HERE.parent / "fmha_decode_sm120a_cute") / "reference/launch.py"
        args = load("test_reference_runner", launch).parse_args(cli)
        self.assertEqual((args.seq_q, args.seq_k, args.heads, args.input_scale, args.zero_query),
                         (1, 64 if self.workload == "prefill" else 65, 1, 4, True))
        inputs = fixture.make_inputs(1, args.seq_k, 1, 4, True)
        expected = fixture.reference_cpu(*inputs, 1, args.seq_k, 1, False)
        output = fixture.decode_f16(fixture.encode_f16(expected))
        samples = ([3.0, 1.0, 2.0] if rust else [6.0, 2.0, 4.0]) if args.iters else []
        report = {
            "implementation": "cute-rs" if rust else "cutlass-cute-dsl",
            "workload": args.workload, "shape": [1, 1, args.seq_k, 1, 64],
            "causal": False, "input_scale": 4, "zero_query": True,
            "input_hashes": dict(zip("qkv", map(fixture.bits_hash, inputs))),
            "output_hash": fixture.bits_hash(output), "verification": "cpu-pass",
            "replay_bitwise_equal": True, "validation": fixture.verify(output, expected),
            "metric_value": (2.0 if rust else 4.0) if samples else None,
            "metric_unit": "us", "samples_us": samples,
            "provenance": {
                "gpu": "test GPU", "compute_capability": [12, 0], "device": args.device,
                "warmup": args.warmup if args.iters else 0, "iters": args.iters,
                "dtype": "float16", "timing_mode": "direct",
                "tool": "cuda-events-single-kernel" if self.workload == "prefill" else "cuda-events-kernel-sequence",
                "shape_specialization": "runtime", "causal_specialization": "static",
                "algorithm": "single-query-split-kv", "kv_chunk": 64,
                "split_count": 2, "kernel_count": 2,
                "kernel_sequence": ["fmha_decode_partial", "fmha_decode_merge"],
            },
        }
        if rust:
            self.mutate(report)
        args.json.write_text(json.dumps(report))
        args.output.write_bytes(fixture.encode_f16(output))
        return subprocess.CompletedProcess(command, 0, stdout="runner diagnostics\n", stderr="")

    def invoke(self, workload, iters=3):
        self.workload = workload
        result_path = self.directory / "combined.json"
        argv = ["--workload", workload, "--rust-binary", str(self.binary),
                "--seq-q", "1", "--seq-k", "64" if workload == "prefill" else "65",
                "--heads", "1", "--noncausal", "--input-scale", "4", "--zero-query",
                "--device", "2", "--warmup", "2", "--iters", str(iters),
                "--json", str(result_path)]
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(comparison.subprocess, "run", side_effect=self.run_runner), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = comparison.main(argv)
        return status, stdout.getvalue(), stderr.getvalue(), result_path

    def test_prints_both_medians_and_saves_checked_reports_for_each_workload(self):
        for workload in ("prefill", "decode"):
            with self.subTest(workload=workload):
                status, stdout, _, path = self.invoke(workload)
                self.assertEqual(status, 0)
                self.assertIn("CUDA-event median", stdout)
                self.assertIn("runtime / noncausal", stdout)
                self.assertIn("CuTeDSL: 4.000 us\ncute-rs: 2.000 us", stdout)
                report = json.loads(path.read_text())
                self.assertEqual(report["verification"], "both-pass")
                self.assertEqual(report["cuda_oxide"]["metric_value"], 2)
                self.assertEqual(report["python_reference"]["provenance"]["device"], 2)
                self.assertEqual(report["output_comparison"]["elements"], 64)

    def test_zero_iterations_verifies_both_without_inventing_timings(self):
        for workload in ("prefill", "decode"):
            status, stdout, _, path = self.invoke(workload, iters=0)
            self.assertEqual(status, 0)
            self.assertIn("CuTeDSL: verified (timing disabled)\n"
                          "cute-rs: verified (timing disabled)", stdout)
            self.assertIsNone(json.loads(path.read_text())["cuda_oxide"]["metric_value"])

    def test_rejects_invalid_timing_or_mismatched_device_before_printing_summary(self):
        for field, value in (("metric_value", 999), ("samples_us", [0, 1, 2]),
                             ("metric_unit", "ms"), ("input_hashes", {})):
            with self.subTest(field=field):
                self.mutate = lambda report: report.update({field: value})
                status, stdout, stderr, _ = self.invoke("prefill")
                self.assertEqual(status, 1)
                self.assertEqual(stdout, "")
                self.assertIn("FAIL", stderr)
        self.mutate = lambda report: report["provenance"].update(gpu="different GPU")
        status, stdout, _, _ = self.invoke("decode")
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")

    def test_child_failure_surfaces_diagnostics_without_success_summary(self):
        self.run_runner = lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", "missing GPU dependency")
        status, stdout, stderr, _ = self.invoke("prefill")
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertIn("missing GPU dependency", stderr)


class AllCasesTests(unittest.TestCase):
    workload = "prefill"
    dimensions = (1024, 1024, 8)
    selection = ["--shape-mode", "static", "--noncausal"]
    expected_cases = [("runtime", True), ("runtime", False), ("static", True), ("static", False)]
    unsupported_shapes = [("--shape-mode", "runtime", "--seq-q", "64", "--seq-k", "64"),
                          ("--shape-mode", "runtime", "--heads", "3")]

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "all.json"
        self.seen = []

    def compare_case(self, options, cli):
        _, launch, _ = comparison.load_reference(self.workload)
        args = launch.parse_args(cli)
        mode = args.shape_mode if self.workload == "prefill" else args.specialization
        self.seen.append((mode, args.causal))
        # The selected mode/mask must be replaced without losing other settings.
        self.assertEqual((args.seq_q, args.seq_k, args.heads), self.dimensions)
        self.assertEqual((args.device, args.warmup, args.input_scale, args.zero_query), (2, 7, 4, True))
        return {
            "workload": args.workload, "verification": "both-pass",
            "cuda_oxide": {"metric_value": 2.0 if args.iters else None,
                           "shape": [1, *self.dimensions, 64], "causal": args.causal,
                           "provenance": {"warmup": args.warmup, "iters": args.iters,
                                          "shape_specialization": mode}},
            "python_reference": {"metric_value": 4.0 if args.iters else None},
            "timing": {"mode": "direct", "statistic": "median", "unit": "us"},
        }

    def invoke(self, *extra, failure=None):
        argv = ["--workload", self.workload, "--rust-binary", "/unused-rust",
                "--all-cases", *self.selection,
                "--device", "2", "--warmup", "7", "--iters", "3",
                "--input-scale", "4", "--zero-query", "--json", str(self.path), *extra]
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(comparison, "compare", side_effect=failure or self.compare_case) as runner, \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = comparison.main(argv)
        return status, stdout.getvalue(), stderr.getvalue(), runner

    def test_all_modes_are_measured_and_saved_despite_single_case_flags(self):
        status, stdout, _, runner = self.invoke()
        self.assertEqual(status, 0)
        self.assertEqual(runner.call_count, len(self.expected_cases))
        expected = self.expected_cases
        self.assertEqual(self.seen, expected)
        report = json.loads(self.path.read_text())
        self.assertEqual(report["verification"], "all-pass")
        self.assertEqual([(case["shape_mode"], case["causal"]) for case in report["cases"]], expected)
        self.assertIn("CUDA-event median", stdout)
        self.assertIn("Nsight kernel-active", stdout)
        for mode, causal in expected:
            self.assertIn(f"| {mode.title()} | {'Causal' if causal else 'Noncausal'} | 2.000 | 4.000 | -50.0% |", stdout)

    def test_verification_only_table_has_no_fabricated_times_or_relative_deltas(self):
        status, stdout, _, _ = self.invoke("--iters", "0")
        self.assertEqual(status, 0)
        self.assertEqual(stdout.count("| verified | verified | n/a |"), len(self.expected_cases))
        self.assertNotIn("-50.0%", stdout)

    def test_unsupported_shape_fails_before_running_any_case(self):
        for extra in self.unsupported_shapes:
            with self.subTest(extra=extra):
                status, stdout, stderr, runner = self.invoke(*extra)
                self.assertEqual(status, 1)
                runner.assert_not_called()
                self.assertEqual(stdout, "")
                self.assertIn("--all-cases", stderr)
                self.assertFalse(self.path.exists())

    def test_failed_case_cannot_produce_success_table_or_json(self):
        def fail_second(options, cli):
            if self.seen:
                raise RuntimeError("second case failed validation")
            return self.compare_case(options, cli)

        status, stdout, stderr, runner = self.invoke(failure=fail_second)
        self.assertEqual(status, 1)
        self.assertEqual(runner.call_count, 2)
        self.assertEqual(stdout, "")
        self.assertIn("failed validation", stderr)
        self.assertFalse(self.path.exists())


class DecodeAllCasesTests(AllCasesTests):
    workload = "decode"
    dimensions = (1, 1024, 8)
    selection = ["--specialization", "auto", "--noncausal"]
    expected_cases = [("runtime", False), ("static", False)]
    unsupported_shapes = [("--specialization", "runtime", "--seq-k", "66"),
                          ("--specialization", "runtime", "--heads", "3")]


if __name__ == "__main__":
    unittest.main()
