#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU acceptance tests for ordered split-KV decode profiling evidence."""

import contextlib
import importlib.util
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

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixture = load("decode_comparison_fixture", HERE.parent.parent / "fmha_prefill_sm120a_cute/reference/launch.py")
compare = load("decode_comparison", HERE / "compare.py") if (HERE / "compare.py").exists() else None
PARTIAL, MERGE = "fmha_decode_partial", "fmha_decode_merge"


def trace(path, rows, *, full_names=None):
    """Rows are (name, start, end, device, stream), deliberately inserted backwards."""
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
                   "(start INTEGER, end INTEGER, shortName INTEGER, deviceId INTEGER, streamId INTEGER)")
        names = {name: i for i, name in enumerate(dict.fromkeys(row[0] for row in rows))}
        db.executemany("INSERT INTO StringIds VALUES (?, ?)", [(i, name) for name, i in names.items()])
        db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)",
                       [(start, end, names[name], device, stream)
                        for name, start, end, device, stream in reversed(rows)])
        if full_names:
            db.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN demangledName INTEGER")
            full = {name: len(names) + i for i, name in enumerate(dict.fromkeys(full_names))}
            db.executemany("INSERT INTO StringIds VALUES (?, ?)", [(i, name) for name, i in full.items()])
            db.executemany("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET demangledName=? WHERE start=?",
                           [(full[name], row[1]) for name, row in zip(full_names, rows)])


def sequences():
    return [row for index, duration in enumerate((900, 800, 700, 600, 31, 11, 21))
            for row in ((PARTIAL, index * 10000000, index * 10000000 + duration * 1000, 0, 7),
                        (MERGE, index * 10000000 + duration * 1000 + 4000,
                         index * 10000000 + duration * 1000 + 9000, 0, 7))]


def options():
    return SimpleNamespace(seq_q=1, seq_k=65, heads=1, causal=False, input_scale=1,
                           zero_query=False, warmup=2, samples=3, device=0, specialization="auto")


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(compare, "dedicated decode comparison is not implemented")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.args = options()
        self.hashes = compare.expected_input_hashes(self.args)
        self.output = fixture.decode_f16(fixture.encode_f16(fixture.reference_cpu(
            *fixture.make_inputs(1, 65, 1), 1, 65, 1, False)))

    def report(self):
        return {
            "implementation": "cute-rs", "workload": "fmha-decode",
            "shape": [1, 1, 65, 1, 64], "causal": False, "input_scale": 1,
            "zero_query": False, "input_hashes": self.hashes,
            "output_hash": fixture.bits_hash(self.output), "verification": "cpu-pass",
            "validation": {"elements": 64, "max_abs_error": 0.0,
                           "max_tolerance_ratio": 0.0, "atol": fixture.ATOL, "rtol": fixture.RTOL,
                           "oracle": "cpu-fp64-stable-softmax"},
            "replay_bitwise_equal": True, "metric_value": 999.0, "metric_unit": "us",
            "samples_us": [999.0] * 3,
            "provenance": {"gpu": "test SM120 device", "compute_capability": [12, 0],
                           "device": 0, "warmup": 2, "iters": 3, "dtype": "float16",
                           "tool": "cuda-events-kernel-sequence", "timing_mode": "direct",
                           "algorithm": "single-query-split-kv", "kv_chunk": 64,
                           "split_count": 2, "kernel_count": 2,
                           "shape_specialization": "runtime",
                           "kernel_sequence": [PARTIAL, MERGE]},
        }

    def write_evidence(self, report=None, values=None):
        result, output = self.directory / "runner.json", self.directory / "output.f16"
        result.write_text(json.dumps(self.report() if report is None else report))
        output.write_bytes(fixture.encode_f16(self.output if values is None else values))
        return result, output

    def validate(self, paths, started=0):
        return compare.validate_report(*paths, self.args, started, self.hashes)

    def sample(self, path, **selectors):
        return compare.sequence_samples(path, 2, 3, **(selectors or {"expected_names": [PARTIAL, MERGE]}))

    def test_strips_whole_correctness_and_warmup_sequences_and_separates_active_from_wall(self):
        path = self.directory / "trace.sqlite"
        trace(path, sequences() + [("fill", 950000, 951000, 0, 7)])
        result = self.sample(path)
        self.assertEqual(result["active_samples_us"], [36, 16, 26])
        self.assertEqual(result["wall_samples_us"], [40, 20, 30])
        self.assertEqual(result["correctness_active_samples_us"], [905, 805])
        self.assertEqual(result["warmup_active_samples_us"], [705, 605])
        self.assertEqual(result["kernel_launches"], {PARTIAL: 7, MERGE: 7})
        self.assertEqual(result["auxiliary_kernel_counts"], {"fill": 1})
        self.assertEqual(result["trace_stream_id"], 7)

    def test_rejects_missing_extra_reordered_overlapping_or_interleaved_kernels(self):
        rows = sequences()
        cases = {
            "missing": rows[:-1],
            "extra": rows + [(MERGE, 90000000, 90001000, 0, 7)],
            "reordered": [(MERGE, *rows[0][1:]), (PARTIAL, *rows[1][1:])] + rows[2:],
            "overlap": [rows[0], (MERGE, 100, 1000, 0, 7)] + rows[2:],
            "inside-pair": rows + [("other", 31000 + 40000000, 32000 + 40000000, 0, 7)],
            "between-samples": rows + [("other", 49000000, 49001000, 0, 7)],
        }
        for name, launches in cases.items():
            with self.subTest(name=name):
                path = self.directory / f"{name}.sqlite"
                trace(path, launches)
                with self.assertRaises(ValueError):
                    self.sample(path)

    def test_rejects_mismatched_or_missing_device_stream_and_context(self):
        for column in ("deviceId", "streamId", "contextId"):
            for value in (1, None):
                with self.subTest(column=column, value=value):
                    path = self.directory / f"{column}-{value}.sqlite"
                    trace(path, sequences())
                    with sqlite3.connect(path) as db:
                        if column == "contextId":
                            db.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN contextId INTEGER DEFAULT 0")
                        db.execute(f"UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET {column}=? WHERE start=0", (value,))
                    with self.assertRaises(ValueError):
                        self.sample(path)

    def test_resolves_full_demangled_names_and_rejects_ambiguous_selectors(self):
        rows = sequences()
        path = self.directory / "full.sqlite"
        full = [f"authored_{row[0]}_specialization" for row in rows]
        trace(path, [("truncated", *row[1:]) for row in rows], full_names=full)
        result = self.sample(path, name_contains=[PARTIAL, MERGE])
        self.assertEqual(result["kernel_names"], full[:2])
        self.assertEqual(result["kernel_name_field"], "demangledName")
        with self.assertRaises(ValueError):
            self.sample(path, expected_names=["truncated", MERGE])

    def test_rejects_trace_without_stream_metadata_or_missing_database(self):
        missing = self.directory / "missing.sqlite"
        with self.assertRaises(ValueError):
            self.sample(missing)
        self.assertFalse(missing.exists())
        path = self.directory / "no-stream.sqlite"
        trace(path, sequences())
        with sqlite3.connect(path) as db:
            db.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL DROP COLUMN streamId")
        with self.assertRaises(ValueError):
            self.sample(path)

    def test_checks_every_output_hash_and_decode_sequence_metadata(self):
        report, values = self.validate(self.write_evidence(), time.time_ns() - 1000000000)
        self.assertEqual(report["validation"]["elements"], 64)
        self.assertEqual(values, self.output)
        changes = {"workload": "single-query-mha", "replay_bitwise_equal": 1,
                   "validation.elements": 1, "validation.max_tolerance_ratio": 1.01,
                   "validation.max_abs_error": float("nan"), "validation.atol": 1.0,
                   "validation.oracle": "gpu", "provenance.tool": "cuda-events-single-kernel",
                   "provenance.kernel_sequence": [MERGE, PARTIAL], "provenance.split_count": 1,
                   "provenance.kernel_count": 1, "provenance.iters": 2,
                   "input_hashes": {"q": "0" * 16}, "shape": [True, 1, 65, 1, 64]}
        for field, value in changes.items():
            with self.subTest(field=field):
                changed = self.report()
                keys = field.split(".")
                owner = changed if len(keys) == 1 else changed[keys[0]]
                owner[keys[-1]] = value
                with self.assertRaises(ValueError):
                    self.validate(self.write_evidence(changed))

    def test_rejects_stale_truncated_tampered_nonfinite_outputs(self):
        paths = self.write_evidence()
        os.utime(paths[0], ns=(1, 1))
        with self.assertRaises(ValueError):
            self.validate(paths, time.time_ns())
        for values in (self.output[:-1], [1.0] + self.output[1:], [float("nan")] + self.output[1:]):
            with self.subTest(values=values[:1]):
                with self.assertRaises(ValueError):
                    self.validate(self.write_evidence(values=values))

    def test_recomputes_full_cpu_oracle_even_when_false_runner_evidence_is_self_consistent(self):
        wrong = [1.0] * 64
        report = self.report()
        report["output_hash"] = fixture.bits_hash(wrong)
        with self.assertRaisesRegex(ValueError, "output mismatch"):
            self.validate(self.write_evidence(report, wrong))

    def test_rejects_missing_invalid_and_inconsistent_event_sequence_samples(self):
        for value in ([], [1, 2], [1, 2, -1], [1, True, 2], [1, 2, 3]):
            with self.subTest(samples=value):
                report = self.report()
                report["samples_us"] = value
                with self.assertRaises(ValueError):
                    self.validate(self.write_evidence(report))
        paths = self.write_evidence()
        paths[0].write_text(paths[0].read_text().replace('"metric_value": 999.0', '"metric_value": 1e999'))
        with self.assertRaises(ValueError):
            self.validate(paths)

    def test_specialization_selection_and_report_mismatch_are_rejected(self):
        cases = ((64, 8, "auto", "static"), (65, 3, "auto", "static"),
                 (1024, 8, "auto", "static"), (4097, 8, "auto", "static"),
                 (65, 1, "auto", "runtime"), (1024, 8, "runtime", "runtime"),
                 (65, 3, "static", "static"))
        self.assertTrue(hasattr(compare, "effective_specialization"), "specialization selection is not implemented")
        for length, heads, requested, expected in cases:
            with self.subTest(length=length, heads=heads, requested=requested):
                self.assertEqual(compare.effective_specialization(length, heads, requested), expected)
        with self.assertRaises(ValueError):
            compare.effective_specialization(65, 1, "static")
        report = self.report()
        report["provenance"]["shape_specialization"] = "static"
        with self.assertRaisesRegex(ValueError, "shape_specialization"):
            self.validate(self.write_evidence(report))

    def test_cli_propagates_modes_and_rejects_unsupported_forced_static(self):
        arguments = self.arguments()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                compare.parse_arguments(arguments + ["--specialization", "static"])
        args = compare.parse_arguments(arguments + ["--specialization", "runtime"])
        self.assertEqual(args.specialization, "runtime")

    def test_input_hashes_match_fixture_for_tails_large_logits_and_zero_query(self):
        for length, heads, scale, zero in ((1, 1, 1, False), (65, 3, 16, False), (257, 8, 1, True)):
            args = options()
            args.seq_k, args.heads, args.input_scale, args.zero_query = length, heads, scale, zero
            expected = {name: fixture.bits_hash(values) for name, values in
                        zip("qkv", fixture.make_inputs(1, length, heads, scale, zero))}
            self.assertEqual(compare.expected_input_hashes(args), expected)

    def test_aggregate_preserves_distinct_raw_metrics_and_round_medians(self):
        entries = [{"label": "rust", "active_samples_us": active, "wall_samples_us": wall}
                   for active, wall in (([1, 100, 1], [3, 102, 3]), ([9, 10, 11], [13, 14, 15]),
                                        ([3, 2, 4], [4, 3, 5]))]
        result = compare.aggregate_rounds(entries)["rust"]
        self.assertEqual(result["active"]["round_medians_us"], [1, 10, 3])
        self.assertEqual(result["active"]["median_of_round_medians_us"], 3)
        self.assertEqual(result["wall"]["median_of_round_medians_us"], 4)
        self.assertEqual(result["active"]["samples_us"], [1, 100, 1, 9, 10, 11, 3, 2, 4])

    def arguments(self):
        binary = self.directory / "rust"
        binary.write_text("synthetic executable")
        binary.chmod(0o755)
        return ["--rust-binary", str(binary), "--output-dir", str(self.directory / "evidence"),
                "--seq-k", "65", "--heads", "1", "--warmup", "2", "--samples", "3", "--rounds", "3"]

    def fake_command(self, command, timeout, environment, log_path):
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "GPU-external")
        log_path.write_text("synthetic version\n")
        if command[1:2] == ["profile"]:
            result = Path(command[command.index("--json") + 1])
            prefix = Path(command[command.index("--output") + 1])
            output = Path(command[-1])
            report = self.report()
            if "authored-cute-dsl" in prefix.name:
                report["implementation"] = "cutlass-cute-dsl"
            result.write_text(json.dumps(report))
            output.write_bytes(fixture.encode_f16(self.output))
            prefix.with_suffix(".nsys-rep").write_bytes(b"synthetic raw trace")
        elif command[1:2] == ["export"]:
            path = Path(command[command.index("--output") + 1])
            rows = sequences()
            if "authored-cute-dsl" in path.name:
                rows = [(f"authored_{row[0]}_specialized", *row[1:]) for row in rows]
            else:
                rows = [(f"{row[0]}__generic_hash", *row[1:]) for row in rows]
            trace(path, rows[:-1] if getattr(self, "drop_last_launch", False) else rows)
        return 0

    def test_command_flow_alternates_complete_serial_runs_and_preserves_all_three_metrics(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-external"}), \
                mock.patch.object(compare, "run_command", side_effect=self.fake_command), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(compare.main(self.arguments()), 0)
        directory, = (self.directory / "evidence").iterdir()
        report = json.loads((directory / "report.json").read_text())
        self.assertEqual(report["status"], "passed")
        self.assertEqual([entry["label"] for entry in report["runs"]],
                         ["rust", "authored-cute-dsl", "authored-cute-dsl", "rust", "rust", "authored-cute-dsl"])
        self.assertEqual(report["summary"]["rust"]["active"]["median_of_round_medians_us"], 26)
        self.assertEqual(report["summary"]["rust"]["wall"]["median_of_round_medians_us"], 30)
        self.assertEqual(report["summary"]["rust"]["active"]["samples_us"], [36, 16, 26] * 3)
        self.assertEqual(report["runs"][0]["runner_report"]["metric_value"], 999)
        self.assertEqual(report["environment"]["CUDA_VISIBLE_DEVICES"], "GPU-external")
        self.assertEqual(report["parameters"]["specialization"], "auto")
        for command in (entry["command"] for entry in report["commands"] if "profile" in entry["command"]):
            self.assertEqual(command[command.index("--specialization") + 1], "auto")
        for previous, following in zip(report["runs"], report["runs"][1:]):
            self.assertLessEqual(previous["finished_ns"], following["started_ns"])

    def test_failed_sequence_stops_before_another_runner_and_never_publishes_summary(self):
        self.drop_last_launch = True
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-external"}), \
                mock.patch.object(compare, "run_command", side_effect=self.fake_command), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(compare.main(self.arguments()), 1)
        directory, = (self.directory / "evidence").iterdir()
        report = json.loads((directory / "report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["runs"]), 1)
        self.assertNotIn("summary", report)

    def test_build_manifest_rejects_wrong_binary_before_any_command(self):
        manifest = self.directory / "build.json"
        manifest.write_text(json.dumps({"binary_sha256": "0" * 64}))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(compare.main(self.arguments() + ["--rust-build-manifest", str(manifest)]), 1)
        directory, = (self.directory / "evidence").iterdir()
        report = json.loads((directory / "report.json").read_text())
        self.assertEqual(report["commands"], [])
        self.assertEqual(report["status"], "failed")

    def test_validation_sweep_covers_tails_heads_and_unequal_split_uniform_weights(self):
        self.assertTrue((HERE / "validate.py").exists(), "dedicated decode validator is not implemented")
        validation = load("decode_validation_test", HERE / "validate.py")
        regular = {(sk, heads) for _, sk, heads, scale, zero in validation.CASES if scale == 1 and not zero}
        self.assertEqual(regular, {(sk, heads) for sk in (1, 3, 31, 32, 33, 63, 64, 65, 127, 257, 1024, 4097)
                                  for heads in (1, 3, 8)})
        self.assertTrue(any(sk > 64 and sk % 64 and zero for _, sk, _, _, zero in validation.CASES))
        self.assertTrue(any(sk > 64 and scale == 16 for _, sk, _, scale, _ in validation.CASES))

    def test_validation_plan_pairs_every_portfolio_shape_in_both_modes(self):
        validation = load("decode_validation_specialization_test", HERE / "validate.py")
        self.assertTrue(hasattr(validation, "validation_plan"), "runtime/static sweep is not implemented")
        plan = validation.validation_plan(validation.CASES, "auto")
        self.assertEqual(len(plan), 52)
        self.assertEqual([item[5] for item in plan[:40]], ["auto"] * 40)
        self.assertEqual({(sk, heads, scale, mode) for _, sk, heads, scale, _, mode in plan[40:]},
                         {(sk, heads, scale, mode) for sk, heads, scale in
                          ((64, 8, 1), (65, 3, 1), (1024, 8, 1), (4097, 8, 1), (65, 3, 16), (4097, 8, 16))
                          for mode in ("runtime", "static")})

    def test_validator_records_fresh_checked_outputs_and_stops_on_nonzero_exit(self):
        validation = load("decode_validation_flow_test", HERE / "validate.py")
        binary = self.directory / "rust"
        binary.write_text("synthetic executable")
        binary.chmod(0o755)
        calls = []

        def execute(command, timeout, environment, log_path):
            calls.append(command)
            log_path.write_text("synthetic run\n")
            if len(calls) == 3:
                return 9
            report = self.report()
            report["provenance"].update(warmup=0, iters=0)
            report.update(samples_us=[], metric_value=None)
            if command[0] != str(binary):
                report["implementation"] = "cutlass-cute-dsl"
            Path(command[command.index("--json") + 1]).write_text(json.dumps(report))
            Path(command[command.index("--output") + 1]).write_bytes(fixture.encode_f16(self.output))
            return 0

        parent = self.directory / "validation-evidence"
        with mock.patch.object(validation.compare, "run_command", side_effect=execute), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = validation.main(["--binary", str(binary), "--reference", "--output-dir", str(parent)],
                                     cases=(("first", 65, 1, 1, False), ("second", 65, 1, 1, False)))
        self.assertEqual(result, 1)
        directory, = parent.iterdir()
        report = json.loads((directory / "report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual([entry["status"] for entry in report["checks"]], ["passed", "passed", "failed"])
        self.assertEqual(report["checks"][-1]["exit_code"], 9)
        self.assertEqual(len(report["cross_checks"]), 1)
        self.assertEqual(report["checks"][0]["independent_cpu_validation"]["elements"], 64)

    def test_validator_checks_runtime_static_outputs_directly_and_records_parity_failure(self):
        validation = load("decode_validation_pair_flow", HERE / "validate.py")
        binary = self.directory / "rust"
        binary.write_text("synthetic executable")
        binary.chmod(0o755)
        cache = {}

        def execute(command, timeout, environment, log_path):
            seq_k = int(command[command.index("--seq-k") + 1])
            heads = int(command[command.index("--heads") + 1])
            scale = int(command[command.index("--input-scale") + 1])
            mode = command[command.index("--specialization") + 1]
            key = (seq_k, heads, scale)
            if key not in cache:
                inputs = fixture.make_inputs(1, seq_k, heads, scale)
                expected = fixture.reference_cpu(*inputs, 1, seq_k, heads, False)
                cache[key] = expected, {name: fixture.bits_hash(values) for name, values in zip("qkv", inputs)}
            expected, hashes = cache[key]
            values = fixture.decode_f16(fixture.encode_f16(expected))
            if fail_cross_check:
                # Both runners remain within CPU tolerance, but opposite signed
                # errors exceed the direct runtime/static comparison tolerance.
                values[0] += 0.00019 * (1 if mode == "runtime" else -1)
            elif mode == "static":
                values[0] += 2 ** -13
            values = fixture.decode_f16(fixture.encode_f16(values))
            report = self.report()
            report.update(shape=[1, 1, seq_k, heads, 64], input_scale=scale, input_hashes=hashes,
                          output_hash=fixture.bits_hash(values), validation=fixture.verify(values, expected),
                          samples_us=[], metric_value=None)
            report["provenance"].update(warmup=0, iters=0, shape_specialization=mode,
                                        split_count=(seq_k + 63) // 64)
            if command[0] != str(binary):
                report["implementation"] = "cutlass-cute-dsl"
            log_path.write_text("synthetic specialization execution\n")
            Path(command[command.index("--json") + 1]).write_text(json.dumps(report))
            Path(command[command.index("--output") + 1]).write_bytes(fixture.encode_f16(values))
            return 0

        for fail_cross_check in (False, True):
            with self.subTest(fail_cross_check=fail_cross_check):
                parent = self.directory / f"pairs-{fail_cross_check}"
                with mock.patch.object(validation.compare, "run_command", side_effect=execute), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    result = validation.main(["--binary", str(binary), "--reference", "--output-dir", str(parent)],
                                             cases=())
                directory, = parent.iterdir()
                report = json.loads((directory / "report.json").read_text())
                if fail_cross_check:
                    self.assertEqual(result, 1)
                    self.assertEqual(report["checks"][-1]["status"], "failed")
                    self.assertIn("output mismatch", report["error"])
                    self.assertEqual(len(report["checks"]), 3)
                else:
                    self.assertEqual(result, 0)
                    self.assertEqual(len(report["checks"]), 24)
                    self.assertEqual(len(report["specialization_cross_checks"]), 12)
                    self.assertTrue(all(not entry["bitwise_equal"] and entry["max_abs_error"] > 0
                                        for entry in report["specialization_cross_checks"]))


if __name__ == "__main__":
    unittest.main()
