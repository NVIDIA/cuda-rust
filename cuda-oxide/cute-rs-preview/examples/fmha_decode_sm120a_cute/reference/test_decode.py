# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the single-query example's shape and attention contract."""

from contextlib import redirect_stderr
from io import StringIO
import unittest

import launch


class DecodeTests(unittest.TestCase):
    def test_defaults_select_one_query_noncausal_mha(self):
        args = launch.parse_args([])
        self.assertEqual((args.seq_q, args.seq_k, args.heads, args.causal), (1, 1024, 8, False))
        self.assertEqual((args.specialization, args.shape_specialization), ("auto", "static"))

    def test_supported_shapes_allow_static_and_runtime_modes(self):
        for seq_k, heads in ((64, 8), (65, 3), (1024, 8), (4097, 8)):
            for request, effective in (("auto", "static"), ("static", "static"), ("runtime", "runtime")):
                with self.subTest(seq_k=seq_k, heads=heads, request=request):
                    args = launch.parse_args(["--seq-k", str(seq_k), "--heads", str(heads),
                                              "--specialization", request])
                    self.assertEqual(args.shape_specialization, effective)

    def test_auto_falls_back_and_explicit_static_rejects_unlisted_shapes(self):
        for seq_k, heads in ((1, 8), (65, 8), (1024, 3), (4096, 8)):
            argv = ["--seq-k", str(seq_k), "--heads", str(heads)]
            with self.subTest(seq_k=seq_k, heads=heads):
                self.assertEqual(launch.parse_args(argv).shape_specialization, "runtime")
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    launch.parse_args(argv + ["--specialization", "static"])

    def test_rejects_causal_and_prefill_invocations(self):
        for argv in (["--causal"], ["--seq-q", "64"], ["--seq-k", "0"], ["--heads", "0"]):
            with self.subTest(argv=argv), redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                launch.parse_args(argv)
            self.assertNotEqual(error.exception.code, 0)

    def test_zero_query_averages_all_cache_rows_for_each_head(self):
        q = [0.0] * 128
        k = [0.0] * 256
        v = [2.0] * 64 + [10.0] * 64 + [4.0] * 64 + [14.0] * 64
        got = launch.reference_cpu(q, k, v, 1, 2, 2, False)
        self.assertEqual(got, [3.0] * 64 + [12.0] * 64)

    def test_ragged_cache_lengths_have_nonempty_last_split(self):
        for length, splits in ((1, 1), (3, 1), (63, 1), (64, 1), (65, 2), (257, 5), (1024, 16)):
            with self.subTest(length=length):
                args = launch.parse_args(["--seq-k", str(length)])
                self.assertEqual(args.seq_k, length)
                self.assertEqual(launch.split_count(length), splits)
                rows = [list(range(s * 64, min((s + 1) * 64, length))) for s in range(splits)]
                self.assertTrue(all(rows))
                self.assertEqual([row for split in rows for row in split], list(range(length)))

    def test_rejects_index_overflow_and_negative_measurement_arguments(self):
        for argv in (["--seq-k", str(1 << 32)], ["--warmup", "-1"], ["--iters", "-1"], ["--device", "-1"]):
            with self.subTest(argv=argv), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                launch.parse_args(argv)

    def test_ragged_zero_query_averages_every_split(self):
        # The final one-row split must contribute to the denominator and numerator.
        length = 65
        q = [0.0] * 64
        k = [0.0] * (length * 64)
        v = [float(row) for row in range(length) for _ in range(64)]
        got = launch.reference_cpu(q, k, v, 1, length, 1, False)
        self.assertEqual(got, [32.0] * 64)


if __name__ == "__main__":
    unittest.main()
