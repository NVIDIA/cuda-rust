#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only acceptance tests for matched attention profiling evidence."""

import contextlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from launch import ATOL, RTOL, bits_hash, encode_f16, make_inputs

try:
    import compare
except ImportError:
    compare = None


REFERENCE_NAME = "kernel_FlashAttentionForwardAmpere_1_64_64"


def trace(path, launches, *, full_names=None):
    """Insert out of order; durations are nanoseconds, including auxiliary work."""
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
                   "(start INTEGER, end INTEGER, shortName INTEGER, deviceId INTEGER)")
        names = {name: i for i, name in enumerate(dict.fromkeys(name for name, _ in launches))}
        db.executemany("INSERT INTO StringIds VALUES (?, ?)", [(i, name) for name, i in names.items()])
        db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, 0)",
                       [(i * 1000000, i * 1000000 + duration, names[name])
                        for i, (name, duration) in reversed(list(enumerate(launches)))])
        if full_names is not None:
            db.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN demangledName INTEGER")
            full_ids = {name: len(names) + i for i, name in enumerate(dict.fromkeys(full_names))}
            db.executemany("INSERT INTO StringIds VALUES (?, ?)", [(i, name) for name, i in full_ids.items()])
            db.executemany("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET demangledName=? WHERE start=?",
                           [(full_ids[name], i * 1000000) for i, name in enumerate(full_names)])


def options():
    return SimpleNamespace(seq_q=64, seq_k=64, heads=1, causal=True, input_scale=1,
                           zero_query=False, warmup=2, samples=3, device=0, shape_mode="runtime")


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(compare, "comparison runner has not been implemented")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.args = options()
        self.hashes = compare.expected_input_hashes(self.args)
        self.output = [0.0] * (64 * 64)

    def report(self):
        return {
            "implementation": "cute-rs", "workload": "prefill-fmha",
            "shape": [1, self.args.seq_q, self.args.seq_k, self.args.heads, 64],
            "causal": True, "input_scale": 1,
            "zero_query": False, "input_hashes": self.hashes,
            "output_hash": bits_hash(self.output), "verification": "cpu-pass",
            "validation": {"elements": len(self.output), "max_abs_error": 0.0,
                           "max_tolerance_ratio": 0.0, "atol": ATOL, "rtol": RTOL,
                           "oracle": "cpu-fp64-stable-softmax"},
            "replay_bitwise_equal": True, "metric_value": 999.0, "metric_unit": "us",
            "samples_us": [999.0] * 3,
            "provenance": {"gpu": "test SM120 device", "compute_capability": [12, 0],
                           "device": 0, "warmup": 2, "iters": 3, "dtype": "float16",
                           "causal_specialization": "static",
                           "shape_specialization": self.args.shape_mode,
                           "tool": "cuda-events-single-kernel", "timing_mode": "direct"},
        }

    def write_evidence(self, report=None, values=None):
        result, output = self.directory / "runner.json", self.directory / "output.f16"
        result.write_text(json.dumps(self.report() if report is None else report))
        output.write_bytes(encode_f16(self.output if values is None else values))
        return result, output

    def validate(self, paths, started=0):
        return compare.validate_report(*paths, self.args, started, self.hashes)

    def test_checks_every_output_and_hash_with_full_cpu_oracle_evidence(self):
        started = time.time_ns()
        report, values = self.validate(self.write_evidence(), started)
        self.assertEqual(report["validation"]["elements"], 4096)
        self.assertEqual(values, self.output)

    def test_streamed_input_hashes_match_shared_fixture_for_prefill_and_decode(self):
        for query, scale, zero in ((64, 1, False), (1, 16, False), (64, 4, True)):
            with self.subTest(query=query, scale=scale, zero=zero):
                args = options()
                args.seq_q, args.input_scale, args.zero_query = query, scale, zero
                inputs = make_inputs(args.seq_q, args.seq_k, args.heads, scale, zero)
                self.assertEqual(compare.expected_input_hashes(args),
                                 {name: bits_hash(values) for name, values in zip("qkv", inputs)})

    def test_rejects_wrong_shape_inputs_or_incomplete_cpu_verification(self):
        mutations = [
            ("verification", "failed"), ("replay_bitwise_equal", 1),
            ("shape", [True, 64, 64, 1, 64]), ("input_hashes", {"q": "0" * 16}),
            ("validation.elements", 1), ("validation.max_tolerance_ratio", 1.01),
            ("validation.max_abs_error", float("nan")), ("validation.atol", 1.0),
            ("validation.oracle", "gpu-library"), ("provenance.warmup", 1),
            ("provenance.iters", 31), ("provenance.device", 1),
        ]
        for field, value in mutations:
            with self.subTest(field=field):
                report = self.report()
                owner = report
                parts = field.split(".")
                for part in parts[:-1]:
                    owner = owner[part]
                owner[parts[-1]] = value
                with self.assertRaises(ValueError):
                    self.validate(self.write_evidence(report))

    def test_rejects_stale_corrupt_truncated_or_nonfinite_output(self):
        paths = self.write_evidence()
        os.utime(paths[0], ns=(1, 1))
        with self.assertRaisesRegex(ValueError, "stale"):
            self.validate(paths, time.time_ns())
        for values in (self.output[:-1], [1.0] + self.output[1:],
                       [float("nan")] + self.output[1:]):
            with self.subTest(length=len(values), first=values[0]):
                with self.assertRaises(ValueError):
                    self.validate(self.write_evidence(values=values))

    def test_exact_kernel_samples_strip_two_checks_and_all_warmups(self):
        path = self.directory / "trace.sqlite"
        trace(path, [("torch_fill", 777000), ("fmha_prefill", 900000),
                     ("torch_fill", 555000), ("fmha_prefill", 800000)]
              + [("fmha_prefill", n) for n in [700000, 600000, 31000, 11000, 21000]])
        result = compare.kernel_samples(path, 2, 3, expected_name="fmha_prefill")
        self.assertEqual(result["samples_us"], [31.0, 11.0, 21.0])
        self.assertEqual(result["correctness_samples_us"], [900.0, 800.0])
        self.assertEqual(result["warmup_samples_us"], [700.0, 600.0])
        self.assertEqual(result["auxiliary_kernel_counts"], {"torch_fill": 2})

    def test_resolves_one_reference_name_then_requires_all_133_launches(self):
        path = self.directory / "reference.sqlite"
        trace(path, [(REFERENCE_NAME, 900000)] * 102
              + [(REFERENCE_NAME, n * 1000) for n in range(1, 32)])
        result = compare.kernel_samples(path, 100, 31,
                                        name_contains="FlashAttentionForwardAmpere")
        self.assertEqual(result["kernel_name"], REFERENCE_NAME)
        self.assertEqual(result["samples_us"], list(range(1, 32)))

    def test_resolves_truncated_short_name_using_full_demangled_identity(self):
        path = self.directory / "truncated.sqlite"
        short = "kernel_cutlass_kernel_flash_at_1..."
        full = "kernel_cutlass_kernel_flash_attention_v2FlashAttentionForwardAmpere_object_at__tensor_0"
        launches = [(short, 900000)] * 102 + [(short, n * 1000) for n in range(1, 32)]
        trace(path, launches + [("vectorized_elementwise_kernel", 1000)] * 2,
              full_names=[full] * 133 + ["void at::native::vectorized_elementwise_kernel<FillFunctor>"] * 2)
        for selector in ({"name_contains": "FlashAttentionForwardAmpere"},
                         {"expected_name": short}, {"expected_name": full}):
            with self.subTest(selector=selector):
                result = compare.kernel_samples(path, 100, 31, **selector)
                self.assertEqual(result["kernel_name"], full)
                self.assertEqual(result["kernel_name_field"], "demangledName")
                self.assertEqual(result["kernel_short_names"], [short])
                self.assertEqual(result["samples_us"], list(range(1, 32)))
                self.assertEqual(sum(result["auxiliary_kernel_counts"].values()), 2)

    def test_rejects_ambiguous_truncated_name_and_keeps_distinct_full_kernels(self):
        path = self.directory / "colliding.sqlite"
        short = "kernel_cutlass_kernel_flash_at_1..."
        first, second = REFERENCE_NAME + "_first", REFERENCE_NAME + "_second"
        trace(path, [(short, 1000)] * 7 + [(short, 2000)] * 7,
              full_names=[first] * 7 + [second] * 7)
        for selector in ({"name_contains": "FlashAttentionForwardAmpere"}, {"expected_name": short}):
            with self.subTest(selector=selector):
                with self.assertRaisesRegex(ValueError, "one.*kernel name"):
                    compare.kernel_samples(path, 2, 3, **selector)
        result = compare.kernel_samples(path, 2, 3, expected_name=first)
        self.assertEqual(result["kernel_name"], first)
        self.assertEqual(result["samples_us"], [1.0] * 3)
        self.assertEqual(result["auxiliary_kernel_counts"], {second: 7})

    def test_rejects_wrong_counts_ambiguous_names_and_invalid_durations(self):
        cases = {
            "missing": [("fmha_prefill", 1000)] * 6,
            "extra": [("fmha_prefill", 1000)] * 8,
            "partial-name": [("fmha_prefill_other", 1000)] * 7,
            "invalid-duration": [("fmha_prefill", 1000)] * 6 + [("fmha_prefill", -1)],
        }
        for name, launches in cases.items():
            with self.subTest(name=name):
                path = self.directory / f"{name}.sqlite"
                trace(path, launches)
                with self.assertRaises(ValueError):
                    compare.kernel_samples(path, 2, 3, expected_name="fmha_prefill")
        path = self.directory / "ambiguous.sqlite"
        trace(path, [(REFERENCE_NAME, 1000)] * 7 + [(REFERENCE_NAME + "_other", 1000)])
        with self.assertRaisesRegex(ValueError, "one.*kernel name"):
            compare.kernel_samples(path, 2, 3, name_contains="FlashAttentionForwardAmpere")

    def test_rejects_missing_trace_unresolved_names_or_multiple_devices(self):
        path = self.directory / "missing.sqlite"
        with self.assertRaises(ValueError):
            compare.kernel_samples(path, 2, 3, expected_name="fmha_prefill")
        self.assertFalse(path.exists())
        trace(path, [("fmha_prefill", 1000)] * 7)
        with sqlite3.connect(path) as db:
            db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET deviceId=1 WHERE start=0")
        with self.assertRaisesRegex(ValueError, "device"):
            compare.kernel_samples(path, 2, 3, expected_name="fmha_prefill")
        with sqlite3.connect(path) as db:
            db.execute("DELETE FROM StringIds")
        with self.assertRaises(ValueError):
            compare.kernel_samples(path, 2, 3, expected_name="fmha_prefill")

    def test_aggregate_uses_round_medians_and_preserves_every_raw_sample(self):
        entries = [{"label": "rust", "samples_us": values, "median_us": median}
                   for values, median in [([1, 100, 1], 1), ([9, 10, 11], 10), ([3, 2, 4], 3)]]
        result = compare.aggregate_rounds(entries)["rust"]
        self.assertEqual(result["round_medians_us"], [1, 10, 3])
        self.assertEqual(result["median_of_round_medians_us"], 3)
        self.assertEqual(result["samples_us"], [1, 100, 1, 9, 10, 11, 3, 2, 4])

    def arguments(self):
        binary = self.directory / "rust"
        binary.write_text("synthetic executable")
        binary.chmod(0o755)
        return ["--rust-binary", str(binary), "--output-dir", str(self.directory / "evidence"),
                "--seq-q", "64", "--seq-k", "64", "--heads", "1",
                "--warmup", "2", "--samples", "3", "--rounds", "3"]

    def fake_command(self, command, timeout, environment, log_path):
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "GPU-externally-selected")
        log_path.write_text("synthetic tool version\n")
        if command[1:2] == ["profile"]:
            result = Path(command[command.index("--json") + 1])
            prefix = Path(command[command.index("--output") + 1])
            output = Path(command[-1])
            report = self.report()
            report["causal"] = "--causal" in command
            if "--shape-mode" in command:
                report["provenance"]["shape_specialization"] = command[command.index("--shape-mode") + 1]
            label = "cutlass" if "cutlass" in prefix.name else "rust"
            if label == "cutlass":
                report["implementation"] = "cutlass-cute-dsl"
            if label == getattr(self, "invalid_specialization_label", None):
                if self.invalid_specialization is None:
                    del report["provenance"]["causal_specialization"]
                else:
                    report["provenance"]["causal_specialization"] = self.invalid_specialization
            if label == getattr(self, "invalid_shape_label", None):
                if self.invalid_shape_specialization is None:
                    del report["provenance"]["shape_specialization"]
                else:
                    report["provenance"]["shape_specialization"] = self.invalid_shape_specialization
            result.write_text(json.dumps(report))
            output.write_bytes(encode_f16(self.output))
            prefix.with_suffix(".nsys-rep").write_bytes(b"synthetic trace")
        elif command[1:2] == ["export"]:
            path = Path(command[command.index("--output") + 1])
            name = (REFERENCE_NAME if "cutlass" in path.name
                    else getattr(self, "rust_kernel_name", "fmha_prefill"))
            durations = [900000, 800000, 700000, 600000, 31000, 11000, 21000]
            if getattr(self, "drop_last_launch", False):
                durations.pop()
            trace(path, [(name, n) for n in durations])
        return 0

    def test_runner_keeps_external_device_and_separates_active_and_event_metrics(self):
        arguments = self.arguments()
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-externally-selected"}), \
                mock.patch.object(compare, "run_command", side_effect=self.fake_command), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(compare.main(arguments), 0)
        directory, = (self.directory / "evidence").iterdir()
        result = json.loads((directory / "report.json").read_text())
        self.assertEqual(result["status"], "passed")
        self.assertEqual([entry["label"] for entry in result["runs"]],
                         ["rust", "cutlass", "cutlass", "rust", "rust", "cutlass"])
        self.assertEqual(result["summary"]["rust"]["median_of_round_medians_us"], 21)
        self.assertEqual(result["summary"]["rust"]["samples_us"], [31, 11, 21] * 3)
        self.assertEqual(result["runs"][0]["runner_report"]["metric_value"], 999)
        self.assertIn("source_snapshot", result["provenance"])
        self.assertIn("nsys_version", result["provenance"])
        self.assertEqual(result["provenance"]["gpu"]["gpu"], "test SM120 device")
        self.assertEqual(result["parameters"]["shape_mode"], "runtime")
        for command in result["commands"]:
            if command["command"][1:2] == ["profile"]:
                argv = command["command"]
                self.assertEqual(argv[argv.index("--shape-mode") + 1], "runtime")

    def test_static_mode_is_forwarded_to_both_runners_and_recorded(self):
        self.args.seq_q, self.args.seq_k, self.args.heads = 1024, 1024, 8
        self.args.shape_mode = "static"
        self.hashes = compare.expected_input_hashes(self.args)
        self.output = [0.0] * (1024 * 8 * 64)
        arguments = self.arguments() + ["--seq-q", "1024", "--seq-k", "1024", "--heads", "8",
                                        "--shape-mode", "static", "--rounds", "1"]
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-externally-selected"}), \
                mock.patch.object(compare, "run_command", side_effect=self.fake_command), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(compare.main(arguments), 0, errors.getvalue())
        directory, = (self.directory / "evidence").iterdir()
        result = json.loads((directory / "report.json").read_text())
        self.assertEqual(result["parameters"]["shape_mode"], "static")
        self.assertEqual([run["label"] for run in result["runs"]], ["rust", "cutlass"])
        for run in result["runs"]:
            self.assertEqual(run["runner_report"]["provenance"]["shape_specialization"], "static")
        profiles = [entry["command"] for entry in result["commands"] if entry["command"][1:2] == ["profile"]]
        self.assertEqual(len(profiles), 2)
        for command in profiles:
            self.assertEqual(command[command.index("--shape-mode") + 1], "static")

    def test_static_comparison_rejects_unlisted_shapes_before_running_tools(self):
        for seq_q, seq_k, heads in ((1, 1024, 8), (512, 1024, 8), (1024, 512, 8), (1024, 1024, 4)):
            arguments = self.arguments() + ["--seq-q", str(seq_q), "--seq-k", str(seq_k),
                                            "--heads", str(heads), "--noncausal"]
            with self.subTest(seq_q=seq_q, seq_k=seq_k, heads=heads):
                self.assertEqual(compare.parse_arguments(arguments).shape_mode, "runtime")
                with mock.patch.object(compare, "run_command") as command, \
                        contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit):
                    compare.main(arguments + ["--shape-mode", "static"])
                command.assert_not_called()
                self.assertIn("static shape mode requires", errors.getvalue())

    def test_requested_static_mode_rejects_runtime_report(self):
        report = self.report()
        self.args.shape_mode = "static"
        with self.assertRaisesRegex(ValueError, "shape_specialization"):
            self.validate(self.write_evidence(report))

    def test_runner_rejects_missing_wrong_or_invalid_shape_mode_on_either_side(self):
        for label in ("rust", "cutlass"):
            for mode in (None, "static", "auto"):
                with self.subTest(label=label, mode=mode):
                    self.invalid_shape_label = label
                    self.invalid_shape_specialization = mode
                    evidence = self.directory / f"invalid-shape-{label}-{mode}"
                    arguments = self.arguments() + ["--output-dir", str(evidence), "--rounds", "1"]
                    with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-externally-selected"}), \
                            mock.patch.object(compare, "run_command", side_effect=self.fake_command), \
                            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(compare.main(arguments), 1)
                    directory, = evidence.iterdir()
                    result = json.loads((directory / "report.json").read_text())
                    self.assertEqual(result["status"], "failed")
                    self.assertIn("shape_specialization", result["error"])
                    self.assertEqual(len(result["runs"]), 1 if label == "rust" else 2)
                    self.assertEqual(result["runs"][-1]["label"], label)
                    self.assertNotIn("summary", result)

    def test_runner_resolves_generic_names_for_both_causal_specializations(self):
        for causal, name in ((True, "fmha_prefill__generic_causal_hash"),
                             (False, "fmha_prefill__generic_noncausal_hash")):
            with self.subTest(causal=causal):
                self.rust_kernel_name = name
                evidence = self.directory / f"generic-{causal}"
                arguments = self.arguments() + ["--output-dir", str(evidence), "--rounds", "1",
                                                "--causal" if causal else "--noncausal"]
                with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-externally-selected"}), \
                        mock.patch.object(compare, "run_command", side_effect=self.fake_command), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
                    self.assertEqual(compare.main(arguments), 0, errors.getvalue())
                directory, = evidence.iterdir()
                result = json.loads((directory / "report.json").read_text())
                self.assertEqual(result["runs"][0]["kernel_name"], name)
                self.assertEqual(result["runs"][0]["samples_us"], [31, 11, 21])
                self.assertEqual(result["parameters"]["causal"], causal)
                for run in result["runs"]:
                    self.assertEqual(run["runner_report"]["causal"], causal)
                    self.assertEqual(run["runner_report"]["provenance"]["causal_specialization"], "static")

    def test_runner_rejects_missing_or_runtime_causal_specialization_on_either_side(self):
        for label in ("rust", "cutlass"):
            for specialization in (None, "runtime"):
                with self.subTest(label=label, specialization=specialization):
                    self.invalid_specialization_label = label
                    self.invalid_specialization = specialization
                    evidence = self.directory / f"invalid-{label}-{specialization}"
                    arguments = self.arguments() + ["--output-dir", str(evidence), "--rounds", "1"]
                    with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-externally-selected"}), \
                            mock.patch.object(compare, "run_command", side_effect=self.fake_command), \
                            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(compare.main(arguments), 1)
                    directory, = evidence.iterdir()
                    result = json.loads((directory / "report.json").read_text())
                    self.assertEqual(result["status"], "failed")
                    self.assertIn("causal_specialization", result["error"])
                    self.assertEqual(len(result["runs"]), 1 if label == "rust" else 2)
                    self.assertEqual(result["runs"][-1]["label"], label)
                    self.assertEqual(result["runs"][-1]["status"], "failed")
                    self.assertNotIn("summary", result)

    def test_failed_trace_stops_serial_run_and_never_publishes_a_summary(self):
        self.drop_last_launch = True
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-externally-selected"}), \
                mock.patch.object(compare, "run_command", side_effect=self.fake_command), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(compare.main(self.arguments()), 1)
        directory, = (self.directory / "evidence").iterdir()
        result = json.loads((directory / "report.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["runs"]), 1)
        self.assertEqual(result["runs"][0]["status"], "failed")
        self.assertNotIn("summary", result)

    def test_build_manifest_must_match_binary_before_any_tool_runs(self):
        manifest = self.directory / "build.json"
        manifest.write_text(json.dumps({"binary_sha256": "0" * 64}))
        arguments = self.arguments() + ["--rust-build-manifest", str(manifest)]
        with mock.patch.object(compare, "run_command") as command, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(compare.main(arguments), 1)
        command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
