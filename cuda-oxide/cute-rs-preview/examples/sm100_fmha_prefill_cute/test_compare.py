# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the shared fixture, masks, and measurement gates."""
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from compare import Config, checked_errors, oracle_row, read_f16, write_f16, verify_sources, ROOT, require_matched_optimization


class FixtureTests(unittest.TestCase):
    def test_timing_rejects_compiler_optimization_mismatch(self):
        require_matched_optimization({'compiler_optimization_level': 2})
        for result in [{}, {'compiler_optimization_level': 3}]:
            with self.assertRaisesRegex(ValueError, 'optimization level 2'):
                require_matched_optimization(result)

    def test_reference_helper_mutation_is_rejected(self):
        verify_sources()
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / 'reference'
            shutil.copytree(ROOT / 'reference', reference)
            helper = reference / 'helpers/fmha_helpers.py'
            helper.write_bytes(helper.read_bytes() + b'\n')
            with self.assertRaisesRegex(RuntimeError, 'fmha_helpers.py'):
                verify_sources(reference)

    def test_gqa_and_causal_use_correct_kv_head(self):
        config = Config(batch=1, query=2, keys=3, heads=4, kv_heads=2, causal=True)
        config.validate()
        self.assertEqual([config.kv_head(h) for h in range(4)], [0, 0, 1, 1])
        self.assertEqual(config.valid_keys(0), 1)
        self.assertEqual(config.valid_keys(1), 2)

    def test_reject_invalid_profiles_before_allocation(self):
        for overrides in [dict(head_dim=64), dict(heads=3, kv_heads=2), dict(query=0),
                          dict(keys=0), dict(batch=0), dict(kv_heads=0), dict(query=2**31)]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                Config(**overrides).validate()

    def test_mask_excludes_keys_from_denominator(self):
        q = [0.0, 0.0]
        k = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        v = [[2.0, -2.0], [8.0, 4.0], [100.0, 100.0]]
        self.assertEqual(oracle_row(q, k, v, 1, 1.0), [2.0, -2.0])
        self.assertEqual(oracle_row(q, k, v, 2, 1.0), [5.0, 1.0])

    def test_oracle_stable_for_large_logits(self):
        actual = oracle_row([1000.0], [[1000.0], [-1000.0]], [[3.0], [9.0]], 2, 1.0)
        self.assertEqual(actual, [3.0])

    def test_exact_half_interchange_and_size_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'q.f16'
            values = [0.0, 1.0, -2.0, 0.5, 65504.0]
            write_f16(path, values)
            self.assertEqual(read_f16(path, len(values)), values)
            with self.assertRaises(ValueError):
                read_f16(path, len(values) + 1)

    def test_nonfinite_and_large_errors_reject_timing(self):
        for actual in [[math.nan], [math.inf], [2.0]]:
            with self.assertRaises(ValueError):
                checked_errors(actual, [1.0], atol=0.1)
        self.assertAlmostEqual(checked_errors([1.01], [1.0], atol=0.1)['max_abs'], 0.01)


if __name__ == '__main__':
    unittest.main()
