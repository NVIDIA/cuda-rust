# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU orchestration tests; GPU execution and telemetry are external boundaries."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import compare
from contract import Config


class ComparisonTests(unittest.TestCase):
    def run_fake_case(self, *, corrupt_timed_output=False, corrupt_reference=None, **kwargs):
        cfg = Config(kv_length=1, splits=kwargs.pop("splits", 1))
        # All-zero Q/K/V and one valid key imply exact zero output and log2(1)=0 LSE.
        data = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in cfg.arrays().items()}
        output = np.zeros((1, 128, 512), dtype="<f2")
        lse = np.zeros((1, 128), dtype="<f4")
        events, budgets = [], []
        snapshots = {"state": 0, "processes": 0}
        rust_report = {"cutlass_optimization_level": "2", "includes_split_reduction": True}

        class Reference:
            clusters = 1
            torch = SimpleNamespace(__version__="test")

            def validate(self, data, atol):
                events.append("cutedsl-validate")
                return {"output": {"max_abs": 0.0}, "lse": {"max_abs": 0.0}}

            timed = False

            def read(self):
                result, result_lse = output.copy(), lse.copy()
                if self.timed and corrupt_reference is not None:
                    target = result_lse if corrupt_reference == "lse" else result
                    target.flat[0] = -0.0 if corrupt_reference == "signed-zero" else 1.0
                return result, result_lse

            def benchmark(self, warmup, iterations, rounds):
                events.append("cutedsl-time")
                self.timed = True
                budgets.append(("cutedsl", warmup, iterations, rounds))
                return [1.0] * rounds

        def subprocess_run(command, **unused):
            if command[0] == "nvidia-smi":
                kind = "processes" if command[1].startswith("--query-compute-apps") else "state"
                snapshots[kind] += 1
                label = f"{kind}-{snapshots[kind]}"
                events.append(label)
                return SimpleNamespace(stdout=label)
            directory = Path(command[command.index("--output-dir") + 1])
            timed = "--validation-only" not in command
            events.append("rust-time" if timed else "rust-validate")
            if timed:
                budgets.append(("rust", *(int(command[command.index(flag) + 1])
                                           for flag in ("--warmup", "--iterations", "--rounds"))))
            written = output.copy()
            if timed and corrupt_timed_output:
                written.flat[0] = 1
            written.tofile(directory / "output.f16")
            lse.tofile(directory / "lse.f32")
            return SimpleNamespace(stdout=json.dumps(rust_report))

        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "kernel"
            binary.write_bytes(b"test executable placeholder")
            with patch.object(compare, "generate", return_value=data), \
                 patch.object(compare, "sm100_mla_decode_fp16_cutedsl", return_value=Reference()), \
                 patch.object(compare.subprocess, "run", side_effect=subprocess_run), \
                 patch.object(compare.importlib.metadata, "version", return_value="4.7.0"):
                report = compare.run_case(binary, cfg, warmup=2, iterations=7, rounds=3, **kwargs)
        return report, events, budgets

    def test_default_order_reports_actual_timed_work_and_before_after_state(self):
        report, events, budgets = self.run_fake_case()
        self.assertFalse(report["timing"]["includes_split_reduction"])
        self.assertEqual(report["timing"]["order"],
                         ["sm100_mla_decode_fp16_cutedsl", "sm100_mla_decode_fp16_cute_rs"])
        self.assertEqual(events, ["cutedsl-validate", "rust-validate", "state-1", "processes-1",
                                  "cutedsl-time", "rust-time", "state-2", "processes-2"])
        self.assertEqual(budgets, [("cutedsl", 2, 7, 3), ("rust", 2, 7, 3)])
        self.assertEqual(report["environment"]["gpu_state"], "state-1")
        self.assertEqual(report["environment"]["gpu_processes"], "processes-1")
        self.assertEqual(report["environment"]["gpu_state_after_timing"], "state-2")
        self.assertEqual(report["environment"]["gpu_processes_after_timing"], "processes-2")
        self.assertTrue(report["rust"]["includes_split_reduction"], "preserve native report verbatim")

    def test_rust_first_preserves_validation_and_equal_timing_budgets(self):
        report, events, budgets = self.run_fake_case(timing_order="rust-first", splits=3)
        self.assertEqual(events, ["cutedsl-validate", "rust-validate", "state-1", "processes-1",
                                  "rust-time", "cutedsl-time", "state-2", "processes-2"])
        self.assertEqual(budgets, [("rust", 2, 7, 3), ("cutedsl", 2, 7, 3)])
        self.assertTrue(report["timing"]["includes_split_reduction"])
        self.assertEqual(report["timing"]["order"],
                         ["sm100_mla_decode_fp16_cute_rs", "sm100_mla_decode_fp16_cutedsl"])

    def test_validation_only_has_no_timing_or_post_timing_snapshot(self):
        report, events, budgets = self.run_fake_case(validation_only=True)
        self.assertEqual(events, ["cutedsl-validate", "rust-validate", "state-1", "processes-1"])
        self.assertEqual(budgets, [])
        self.assertNotIn("timing", report)
        self.assertNotIn("gpu_state_after_timing", report["environment"])

    def test_timed_output_corruption_is_rejected_in_both_orders(self):
        for order in ("cutedsl-first", "rust-first"):
            with self.subTest(order=order), self.assertRaisesRegex(AssertionError, "output changed"):
                self.run_fake_case(timing_order=order, corrupt_timed_output=True)

    def test_reference_repeatability_in_both_orders_including_signed_zero(self):
        for order in ("cutedsl-first", "rust-first"):
            for field in ("output", "lse", "signed-zero"):
                with self.subTest(order=order, field=field), self.assertRaisesRegex(
                        AssertionError, "CuTeDSL output changed"):
                    self.run_fake_case(timing_order=order, corrupt_reference=field)

    def test_cli_exposes_timing_order_without_changing_default(self):
        parser = compare.parser()
        default = parser.parse_args(["--rust-bin", "/tmp/kernel"])
        self.assertEqual(getattr(default, "timing_order", None), "cutedsl-first")
        explicit = parser.parse_args(["--rust-bin", "/tmp/kernel", "--timing-order", "rust-first"])
        self.assertEqual(explicit.timing_order, "rust-first")

    def test_cli_supports_cargo_run_aliases(self):
        args = compare.parser().parse_args([
            "--rust-bin", "/tmp/kernel", "--iters", "7", "--windows", "4", "--check-only"])
        self.assertEqual((args.iterations, args.rounds, args.validation_only), (7, 4, True))

    def test_invalid_timing_counts_fail_before_input_allocation(self):
        for invalid in ({"warmup": -1}, {"iterations": 0}, {"iterations": -1},
                        {"rounds": 0}, {"rounds": -1}, {"timeout": 0}, {"clusters": 0}):
            with self.subTest(invalid=invalid), \
                 patch.object(compare, "generate", side_effect=AssertionError("allocated inputs")), \
                 self.assertRaises(ValueError):
                compare.run_case("/tmp/kernel", Config(), **invalid)


if __name__ == "__main__":
    unittest.main()
