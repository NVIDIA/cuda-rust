# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the public kernel's validation and prepared-call ownership."""

import contextlib
import unittest
from unittest import mock

try:
    import torch
    try:
        from . import gemm
    except ImportError:
        import gemm
except ImportError:
    gemm = None


@unittest.skipIf(gemm is None, 'install the optional pinned public-reference dependencies')
class GemmContractTests(unittest.TestCase):
    """Exercise actual tensor metadata and cache behavior without an SM100 device."""

    def test_dimension_and_launch_bounds_reject_before_compilation(self):
        """Malformed dimensions and padded coordinate overflow never reach the compiler."""
        for dims in [(0, 8, 8), (128, 8, 8), (256, 7, 8), (256, 8, 7),
                     (256, 8, 2**31), (True, 8, 8), (256, 2**31 - 8, 8)]:
            with self.subTest(dims=dims), self.assertRaises((TypeError, ValueError)):
                gemm.compile_kernel(*dims)
        for clusters in (0, -1, 2**30):
            with self.subTest(clusters=clusters), self.assertRaises(ValueError):
                gemm.compile_kernel(256, 352, 64, clusters=clusters)

    def test_exact_strides_and_output_alias_are_rejected(self):
        """The host contract catches wrong B storage and destructive output overlap."""
        a = torch.empty((1, 256, 8), dtype=torch.float16)
        b = torch.empty((1, 8, 8), dtype=torch.float16)
        c = torch.empty_like(a)
        with mock.patch.object(gemm, '_sm100'):
            with self.assertRaisesRegex(ValueError, 'transpose'):
                gemm._arguments((a, b, c), False)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                gemm._arguments((a, b.transpose(1, 2), a), False)

    def test_preparation_launches_nothing_and_calls_reuse_owned_views(self):
        """Eager JIT/view preparation queues no GPU work; matching calls reuse the views."""
        self._exercise_prepared_cache(mutate=False)

    def test_prepared_cache_rejects_storage_replacement(self):
        """Tensor identity must not hide a replacement backing allocation."""
        self._exercise_prepared_cache(mutate=True)

    def _exercise_prepared_cache(self, mutate):
        """Use real CPU tensors; replace only CUDA-specific target and execution services."""
        a = torch.empty((1, 256, 8), dtype=torch.float16)
        b = torch.empty((1, 8, 8), dtype=torch.float16).transpose(1, 2)
        c = torch.empty_like(a)
        launches, conversions = [], []
        stream = gemm.driver.CUstream(0)

        def enqueue(*args):
            launches.append(args)

        def view(tensor, **kwargs):
            conversions.append(tensor)
            return tensor

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(gemm, '_sm100'))
            stack.enter_context(mock.patch.object(torch.cuda, 'device', side_effect=lambda _: contextlib.nullcontext()))
            stack.enter_context(mock.patch.object(gemm.utils, 'HardwareInfo'))
            gemm.utils.HardwareInfo.return_value.get_max_active_clusters.return_value = 1
            compiler = stack.enter_context(mock.patch.object(gemm, 'compile_kernel', return_value=enqueue))
            stack.enter_context(mock.patch.object(gemm, 'from_dlpack', side_effect=view))
            prepared = gemm.prepare_callable(inputs=(a, b), output=c)
            self.assertEqual(launches, [])
            self.assertEqual(len(conversions), 4)
            self.assertIs(prepared(a, b, c, stream=stream), c)
            self.assertIs(prepared(a, b, c, stream=stream), c)
            self.assertEqual(len(launches), 2)
            self.assertEqual(len(conversions), 4)
            compiler.assert_called_once()
            if mutate:
                a.set_(torch.empty_like(a))
                with self.assertRaisesRegex(ValueError, 'changed its storage'):
                    prepared(a, b, c, stream=stream)
                self.assertEqual(len(launches), 2)
                self.assertEqual(len(conversions), 4)
                compiler.assert_called_once()


if __name__ == '__main__':
    unittest.main()
