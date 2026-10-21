# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the input ABI and independent decode oracle."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class ContractTests(unittest.TestCase):
    def contract(self):
        path = Path(__file__).with_name("contract.py")
        self.assertTrue(path.is_file(), "the validated decode input contract is missing")
        spec = importlib.util.spec_from_file_location("mla_contract", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_rejects_unsupported_modes_and_sizes(self):
        c = self.contract()
        for kwargs in ({"heads": 64}, {"query_length": 2}, {"page_size": 64},
                       {"kv_length": 0}, {"splits": 0}, {"splits": 257},
                       {"softmax_scale": float("nan")}, {"output_scale": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                c.Config(**kwargs).validate()

    def test_zero_query_excludes_tail_and_uses_page_table(self):
        c = self.contract()
        cfg = c.Config(kv_length=129, splits=3, output_scale=2.0)
        data = c.generate(cfg, seed=7)
        data["q_latent"].fill(0)
        data["q_rope"].fill(0)
        data["page_table"][:] = [[1, 0]]
        data["c_latent"][0].fill(3)
        data["c_latent"][1].fill(1)
        out, lse = c.oracle(cfg, data)
        np.testing.assert_allclose(out, 2 * (128 + 3) / 129, atol=1e-6)
        np.testing.assert_allclose(lse, math.log2(129), atol=1e-6)

    def test_input_roundtrip_preserves_exact_bytes(self):
        c = self.contract()
        cfg = c.Config(batch=2, kv_length=257, splits=2)
        data = c.generate(cfg, seed=91)
        with tempfile.TemporaryDirectory() as directory:
            c.write_inputs(directory, cfg, data)
            restored = c.read_inputs(directory, cfg)
            for key in data:
                self.assertEqual(data[key].tobytes(), restored[key].tobytes(), key)
            path = Path(directory) / "q_latent.f16"
            path.write_bytes(path.read_bytes()[:-2])
            with self.assertRaises(ValueError):
                c.read_inputs(directory, cfg)

    def test_split_ranges_cover_each_key_once(self):
        c = self.contract()
        for length, splits in ((129, 3), (2049, 7), (128, 32), (4096, 1)):
            ranges = c.split_ranges(length, splits)
            covered = [i for start, end in ranges for i in range(start, end)]
            self.assertEqual(covered, list(range(length)))

    def test_reference_validation_honors_requested_tolerance(self):
        import cutedsl
        adapter = object.__new__(cutedsl.sm100_mla_decode_fp16_cutedsl)
        adapter.cfg = None
        adapter.checked = False
        adapter.run = lambda: None
        adapter.read = lambda: (np.array([0.002]), np.array([0.0]))
        with patch.object(cutedsl, "oracle", return_value=(np.array([0.0]), np.array([0.0]))):
            with self.assertRaises(AssertionError):
                adapter.validate({}, atol=0.001)
            self.assertFalse(adapter.checked)
            adapter.validate({}, atol=0.01)
            self.assertTrue(adapter.checked)

    def test_comparison_rejects_nonfinite_and_reports_errors(self):
        c = self.contract()
        expected = np.array([0.0, 1.0], dtype=np.float32)
        with self.assertRaises(AssertionError):
            c.check_close(np.array([0.0, float("nan")]), expected)
        with self.assertRaises(AssertionError):
            c.check_close(np.array([0.0, 1.2]), expected)
        with self.assertRaises(AssertionError):
            c.check_close(expected, np.array([0.0, float("nan")]))
        with self.assertRaises(AssertionError):
            c.check_close(expected, np.array([0.0, float("inf")]))
        metrics = c.check_close(np.array([0.0, 1.001]), expected)
        self.assertAlmostEqual(metrics["max_abs"], 0.001)


if __name__ == "__main__":
    unittest.main()
