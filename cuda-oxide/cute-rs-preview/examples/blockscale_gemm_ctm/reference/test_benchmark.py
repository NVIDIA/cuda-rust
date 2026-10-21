# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail closed on invalid benchmark requests and incomplete GPU output."""
import contextlib
import io
import unittest

import numpy as np

import benchmark


class BenchmarkTests(unittest.TestCase):
    def test_invalid_launch_counts_are_rejected_before_gpu_work(self):
        for args in (["--ctas", "0"], ["--ctas", "257"],
                     ["--warmup", "-1"], ["--samples", "0"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    benchmark.parse_arguments(args)

    def test_bitwise_verification_rejects_poison_and_signed_zero_difference(self):
        expected = np.array([[0x0000, 0x3c00]], dtype=np.uint16)
        benchmark.verify_bits(expected.copy(), expected)
        for got in (np.array([[0x8000, 0x3c00]], dtype=np.uint16),
                    np.array([[0x0000, 0x7e00]], dtype=np.uint16)):
            with self.assertRaisesRegex(ValueError, "mismatch"):
                benchmark.verify_bits(got, expected)

    def test_verification_rejects_missing_outputs_and_wrong_storage_type(self):
        expected = np.array([[0, 1]], dtype=np.uint16)
        for got in (expected[:, :1], expected.astype(np.float16)):
            with self.assertRaises(ValueError):
                benchmark.verify_bits(got, expected)


if __name__ == "__main__":
    unittest.main()
