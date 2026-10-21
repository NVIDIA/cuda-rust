# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the public FP16 correctness oracle."""

import shutil
import subprocess
import unittest

import torch

from public_reference import oracle
import compare


class ReferenceTests(unittest.TestCase):
    """Exercise layouts and numerics independently of the CuTe kernel."""

    def test_seed_preserves_rng_and_produces_shared_integer_inputs(self):
        state = torch.random.get_rng_state().clone()
        first, outputs = oracle.prepare_tensors("fp16", (3, 5, 4), True, seed=17, device="cpu")
        second, _ = oracle.prepare_tensors("fp16", (3, 5, 4), True, seed=17, device="cpu")
        other, _ = oracle.prepare_tensors("fp16", (3, 5, 4), True, seed=18, device="cpu")
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        self.assertFalse(torch.equal(first[0], other[0]))
        for tensor in first:
            self.assertEqual(tensor.dtype, torch.float16)
            self.assertTrue(bool(((tensor >= -2) & (tensor < 2)).all()))
            self.assertTrue(torch.equal(tensor, tensor.round()))
        a, b, bias = first
        self.assertEqual(a.shape, (1, 3, 4))
        self.assertTrue(a.is_contiguous())
        self.assertEqual(b.shape, (1, 4, 5))
        self.assertEqual(b.stride(), (20, 1, 4))
        self.assertTrue(b.transpose(1, 2).is_contiguous())
        self.assertEqual(bias.shape, (3,))
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].shape, (1, 3, 5))
        self.assertEqual(outputs[0].dtype, torch.float16)

    def test_hand_computed_product_and_row_bias(self):
        a = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float16)
        b = torch.tensor([[[5, 7, 9], [6, 8, 10]]], dtype=torch.float16)
        c = torch.tensor([[[17, 23, 29], [39, 53, 67]]], dtype=torch.float16)
        oracle.verify("fp16", a, b, c)
        bias = torch.tensor([10, -10], dtype=torch.float16)
        biased = torch.tensor([[[27, 33, 39], [29, 43, 57]]], dtype=torch.float16)
        oracle.verify("fp16", a, b, bias, biased)
        with self.assertRaises(AssertionError):
            oracle.verify("fp16", a, b, bias, c)

    def test_bias_is_added_before_half_rounding(self):
        a = torch.tensor([[[1, 1]]], dtype=torch.float16)
        b = torch.tensor([[[2048], [1]]], dtype=torch.float16)
        bias = torch.tensor([1], dtype=torch.float16)
        oracle.verify("fp16", a, b, bias, torch.tensor([[[2050]]], dtype=torch.float16))
        with self.assertRaises(AssertionError):
            oracle.verify("fp16", a, b, bias, torch.tensor([[[2048]]], dtype=torch.float16))

    def test_nonfinite_and_invalid_results_fail(self):
        a = torch.ones((1, 2, 3), dtype=torch.float16)
        b = torch.ones((1, 3, 4), dtype=torch.float16)
        for bad in (torch.full((1, 2, 4), float("nan"), dtype=torch.float16),
                    torch.full((1, 2, 4), float("inf"), dtype=torch.float16),
                    torch.ones((1, 2, 4), dtype=torch.float16),
                    torch.full((2, 4), 3, dtype=torch.float16),
                    torch.full((1, 2, 4), 3, dtype=torch.float32)):
            with self.subTest(shape=bad.shape, dtype=bad.dtype):
                with self.assertRaises(AssertionError):
                    oracle.verify("fp16", a, b, bad)
        a[0, 0, 0] = float("nan")
        with self.assertRaises(AssertionError):
            oracle.verify("fp16", a, b, torch.full((1, 2, 4), float("nan"), dtype=torch.float16))

    def test_only_fp16_positive_shapes_are_supported(self):
        for family, dims in (("fp32", (2, 3, 4)), ("fp16", (0, 3, 4)),
                             ("fp16", (2, 3)), ("fp16", (2, 3, 1.5))):
            with self.subTest(family=family, dims=dims):
                with self.assertRaises(ValueError):
                    oracle.prepare_tensors(family, dims, device="cpu")
        with self.assertRaises(ValueError):
            oracle.verify("fp32")

    def test_incompatible_inputs_and_bias_fail(self):
        a = torch.ones((1, 2, 3), dtype=torch.float16)
        b = torch.ones((1, 3, 4), dtype=torch.float16)
        c = torch.full((1, 2, 4), 3, dtype=torch.float16)
        for inputs in ((a.float(), b, c), (a[0], b, c),
                       (a, b[:, :2], c), (a, b, torch.ones(4, dtype=torch.float16), c)):
            with self.subTest(shapes=[x.shape for x in inputs]):
                with self.assertRaises(AssertionError):
                    oracle.verify("fp16", *inputs)

    def test_outer_autocast_does_not_change_reference_rounding(self):
        a = torch.tensor([[[1, 1]]], dtype=torch.float16)
        b = torch.tensor([[[2048], [1]]], dtype=torch.float16)
        expected = torch.tensor([[[2050]]], dtype=torch.float16)
        bias = torch.tensor([1], dtype=torch.float16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            oracle.verify("fp16", a, b, bias, expected)

    def test_non_direct_measurement_is_rejected_before_launch(self):
        with self.assertRaises(ValueError):
            compare.measure(torch, oracle, None, [], None, 0, 1, 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_measured_noop_cannot_reuse_a_valid_output(self):
        inputs, outputs = oracle.prepare_tensors("fp16", (2, 3, 4), device="cuda")
        outputs[0].copy_(torch.bmm(*inputs))

        def noop(*args, stream):
            """Deliberately leave a poisoned result unwritten."""

        with self.assertRaises(AssertionError):
            compare.measure(torch, oracle, noop, inputs, outputs[0], 0, 1, 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_direct_measurement_uses_explicit_stream_and_verifies_result(self):
        inputs, outputs = oracle.prepare_tensors("fp16", (2, 3, 4), device="cuda")
        observed_streams = []

        def product(a, b, c, stream):
            """Use a real GPU product to exercise timing and stream ordering."""
            observed_streams.append((stream.cuda_stream, torch.cuda.current_stream().cuda_stream))
            torch.bmm(a, b, out=c)

        latency = compare.measure(torch, oracle, product, inputs, outputs[0], 2, 3, 1)
        self.assertGreater(latency, 0)
        self.assertEqual(len(observed_streams), 5)
        self.assertTrue(all(explicit == current for explicit, current in observed_streams))
        oracle.verify("fp16", *inputs, outputs[0])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_seeded_inputs_are_identical_across_devices(self):
        cpu, _ = oracle.prepare_tensors("fp16", (3, 5, 4), True, seed=42, device="cpu")
        gpu, _ = oracle.prepare_tensors("fp16", (3, 5, 4), True, seed=42, device="cuda")
        self.assertTrue(all(torch.equal(a, b.cpu()) for a, b in zip(cpu, gpu)))

    def test_clock_snapshot_is_explicit_and_best_effort(self):
        snapshot = oracle.clock_state("cpu")["clock_snapshot"]
        self.assertEqual(snapshot["status"], "unavailable")
        self.assertEqual(snapshot["phase"], "post-timing")
        self.assertIn("timestamp_utc", snapshot)

    @unittest.skipUnless(torch.cuda.is_available() and shutil.which("nvidia-smi"), "GPU telemetry unavailable")
    def test_clock_snapshot_accepts_torch_gpu_uuid_format(self):
        uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
        if uuid is None:
            self.skipTest("PyTorch device UUID unavailable")
        query_uuid = str(uuid)
        if not query_uuid.startswith(("GPU-", "MIG-")):
            query_uuid = "GPU-" + query_uuid
        try:
            result = subprocess.run(
                ["nvidia-smi", "--id=" + query_uuid,
                 "--query-gpu=uuid,clocks.current.sm,clocks.current.memory,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            )
            fields = result.stdout.strip().split(",")
            for field in fields[1:]:
                float(field)
        except (subprocess.SubprocessError, ValueError):
            self.skipTest("nvidia-smi clock telemetry unavailable")
        snapshot = oracle.clock_state()["clock_snapshot"]
        self.assertEqual(snapshot["status"], "available", snapshot)
        self.assertEqual(snapshot["gpu_uuid"], fields[0].strip())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_reference_restores_precision_after_success_and_failure(self):
        old = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("high")
            a = torch.tensor([[[1.0009765625, 1.0009765625]]], device="cuda", dtype=torch.float16)
            b = torch.tensor([[[1024], [1024]]], device="cuda", dtype=torch.float16)
            expected = torch.tensor([[[2050]]], device="cuda", dtype=torch.float16)
            oracle.verify("fp16", a, b, expected)
            self.assertEqual(torch.get_float32_matmul_precision(), "high")
            with self.assertRaises(AssertionError):
                oracle.verify("fp16", a, b, torch.zeros_like(expected))
            self.assertEqual(torch.get_float32_matmul_precision(), "high")
        finally:
            torch.set_float32_matmul_precision(old)


if __name__ == "__main__":
    unittest.main()
