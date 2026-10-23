# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks for the shared fixture, oracle, and exhaustive verifier."""

import contextlib
import io
import math
import importlib.util
import unittest
from unittest import mock

import launch


class FixtureTests(unittest.TestCase):
    def test_shape_mode_defaults_to_runtime_and_static_accepts_both_masks(self):
        self.assertEqual(launch.parse_args([]).shape_mode, "runtime")
        for mask in ("--causal", "--noncausal"):
            with self.subTest(mask=mask):
                args = launch.parse_args(["--shape-mode", "static", mask])
                self.assertEqual((args.seq_q, args.seq_k, args.heads), (1024, 1024, 8))
                self.assertEqual(args.shape_mode, "static")

    def test_static_mode_rejects_every_unsupported_dimension(self):
        for dimensions in ((1, 1024, 8), (512, 1024, 8), (1024, 512, 8), (1024, 1024, 4)):
            seq_q, seq_k, heads = dimensions
            argv = ["--seq-q", str(seq_q), "--seq-k", str(seq_k), "--heads", str(heads), "--noncausal"]
            with self.subTest(dimensions=dimensions):
                self.assertEqual(launch.parse_args(argv).shape_mode, "runtime")
                with contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit):
                    launch.parse_args(argv + ["--shape-mode", "static"])
                self.assertIn("static shape mode requires", errors.getvalue())

    def test_static_views_keep_shapes_and_strides_while_runtime_marks_both_dynamic(self):
        for mode in ("runtime", "static"):
            with self.subTest(mode=mode):
                tensor, view = mock.Mock(), mock.Mock()
                tensor.dim_order.return_value = (0, 1, 2, 3)
                view.mark_layout_dynamic.return_value = view
                view.mark_compact_shape_dynamic.return_value = view
                from_dlpack = mock.Mock(return_value=view)
                self.assertIs(launch.make_tensor_view(tensor, from_dlpack, mode), view)
                from_dlpack.assert_called_once_with(tensor, assumed_align=16)
                if mode == "runtime":
                    view.mark_layout_dynamic.assert_called_once_with(leading_dim=3)
                    view.mark_compact_shape_dynamic.assert_called_once_with(
                        mode=3, stride_order=(0, 1, 2, 3), divisibility=8,
                    )
                else:
                    view.mark_layout_dynamic.assert_not_called()
                    view.mark_compact_shape_dynamic.assert_not_called()
                    tensor.dim_order.assert_not_called()

    def test_dyadic_inputs_are_exact_fp16_and_distinguish_heads_and_operands(self):
        q, k, v = launch.make_inputs(2, 2, 2)
        for values in (q, k, v):
            self.assertEqual(len(values), 256)
            self.assertTrue(all(-0.25 <= x <= 0.2421875 for x in values))
            self.assertEqual(launch.decode_f16(launch.encode_f16(values)), values)
            self.assertGreater(len(set(values)), 40)
        self.assertNotEqual(q, k)
        self.assertNotEqual(k, v)
        self.assertNotEqual(q[:64], q[64:128])

    def test_input_byte_hashes_match_rust_fixture_abi(self):
        q, k, v = launch.make_inputs(2, 2, 2)
        self.assertEqual([launch.bits_hash(x) for x in (q, k, v)],
                         ["ab1887afc727f6b1", "c758962db6dc1ae0", "e543f46a99a93138"])
        scaled_q, scaled_k, scaled_v = launch.make_inputs(2, 2, 2, 16)
        self.assertEqual(scaled_q, [x * 16 for x in q])
        self.assertEqual(scaled_k, [x * 16 for x in k])
        self.assertEqual(scaled_v, v)
        self.assertEqual(launch.make_inputs(2, 2, 2, 16, True)[0], [0.0] * 256)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional CPU Torch oracle")
    def test_batched_cpu_oracle_matches_scalar_for_scaled_logits_and_both_masks(self):
        for scale in (1, 4, 16):
            for causal in (False, True):
                q, k, v = launch.make_inputs(2, 2, 2, scale)
                scalar = launch.reference_cpu(q, k, v, 2, 2, 2, causal)
                batched = launch.reference_cpu(q, k, v, 2, 2, 2, causal, use_torch=True)
                self.assertLess(max(abs(x - y) for x, y in zip(scalar, batched)), 1e-12)

    def test_oracle_masks_future_keys_and_keeps_bshd_heads_separate(self):
        zeros = [0.0] * 256
        values = [2.0] * 64 + [10.0] * 64 + [4.0] * 64 + [14.0] * 64
        full = launch.reference_cpu(zeros, zeros, values, 2, 2, 2, False)
        causal = launch.reference_cpu(zeros, zeros, values, 2, 2, 2, True)
        self.assertEqual(full, ([3.0] * 64 + [12.0] * 64) * 2)
        self.assertEqual(causal, [2.0] * 64 + [10.0] * 64 + [3.0] * 64 + [12.0] * 64)

    def test_oracle_subtracts_row_max_before_exponentiating(self):
        result = launch.reference_cpu(
            [100.0] * 64, [100.0] * 64 + [-100.0] * 64,
            [3.0] * 64 + [-7.0] * 64, 1, 2, 1, False,
        )
        self.assertEqual(result, [3.0] * 64)

    def test_verifier_rejects_nonfinite_wrong_length_and_last_element_error(self):
        expected = [0.0, 0.01, -0.02, 0.03]
        self.assertEqual(launch.verify(expected, expected)["elements"], 4)
        for replacement in (math.nan, math.inf, -math.inf, -0.03):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                launch.verify(expected[:-1] + [replacement], expected)
        with self.assertRaises(ValueError):
            launch.verify(expected[:-1], expected)
        with self.assertRaises(ValueError):
            launch.verify(expected, [0.0, 0.01, -0.02, math.nan])


if __name__ == "__main__":
    unittest.main()
