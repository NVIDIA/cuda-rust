# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only byte-layout and independent exact-result checks for the CTM fixture."""

import importlib
import importlib.util
import struct
import unittest

import numpy as np


def scalar_mix(value):
    """Rust's wrapping u32 arithmetic, evaluated with Python integers."""
    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = value * 0x7FEB352D % (1 << 32)
    value ^= value >> 15
    value = value * 0x846CA68B % (1 << 32)
    return value ^ (value >> 16)


def scalar_nibble(row, k, operand):
    multiplier, seed = {
        "a": (0x9E3779B9, 0xA511E9B3),
        "b": (0x85EBCA6B, 0x63D83595),
    }[operand]
    return scalar_mix((row * multiplier & 0xFFFFFFFF) ^ k ^ seed) % 16


def scalar_scale(row, group, operand):
    multiplier, seed = {
        "a": (0xC2B2AE35, 0),
        "b": (0x27D4EB2F, 0x165667B1),
    }[operand]
    return 126 + scalar_mix((row * multiplier & 0xFFFFFFFF) ^ group ^ seed) % 3


def scalar_result_bits(row, column):
    """Accumulate integer sixteenths, independently of NumPy or BLAS."""
    twice_magnitude = (0, 1, 2, 3, 4, 6, 8, 12)
    total = 0
    for group in range(32):
        block = 0
        for k in range(group * 32, (group + 1) * 32):
            a = scalar_nibble(row, k, "a")
            b = scalar_nibble(column, k, "b")
            product = twice_magnitude[a % 8] * twice_magnitude[b % 8]
            block += -product if (a ^ b) & 8 else product
        # Each value and each scale was doubled: products are in sixteenths.
        total += block * (1 << (scalar_scale(row, group, "a") - 126)) * (
            1 << (scalar_scale(column, group, "b") - 126)
        )
    return struct.unpack("<H", struct.pack("<e", total / 16))[0]


class FixtureTests(unittest.TestCase):
    data = None

    def fixture_module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("fixture"), "fixture.py must implement build_fixture()"
        )
        return importlib.import_module("fixture")

    def fixture(self):
        if FixtureTests.data is None:
            FixtureTests.data = self.fixture_module().build_fixture()
        return FixtureTests.data

    def test_mix32_wraps_both_multiplications(self):
        module = self.fixture_module()
        np.testing.assert_array_equal(
            module._mix32(np.array([0, 1, 0xFFFFFFFF, 0xA511E9B3, 0x63D83595], np.uint32)),
            np.array([0, 0x688990C0, 0x6768824A, 0x86A24FA0, 0x7CC4ACAF], np.uint32),
        )

    def test_buffers_have_exact_transport_shapes_and_dtypes(self):
        fixture = self.fixture()
        self.assertEqual(set(fixture), {"a", "b", "sa", "sb", "expected"})
        for name, shape, dtype in (
            ("a", (2048, 512), np.uint8),
            ("b", (2048, 512), np.uint8),
            ("sa", (128, 256), np.uint16),
            ("sb", (128, 256), np.uint16),
            ("expected", (2048, 2048), np.uint16),
        ):
            with self.subTest(buffer=name):
                self.assertEqual(fixture[name].shape, shape)
                self.assertEqual(fixture[name].dtype, np.dtype(dtype))
                self.assertTrue(fixture[name].flags.c_contiguous)

    def test_low_k_nibble_occupies_low_bits_and_rows_do_not_repeat(self):
        fixture = self.fixture()
        # A nibbles [0,3,5,11,0,13,6,12]; B [15,2,3,13,5,4,14,2].
        self.assertEqual(fixture["a"][0, :4].tolist(), [0x30, 0xB5, 0xD0, 0xC6])
        self.assertEqual(fixture["b"][0, :4].tolist(), [0x2F, 0xD3, 0x45, 0x2E])
        for operand in ("a", "b"):
            packed = fixture[operand]
            observed = set(np.unique(packed & 15)) | set(np.unique(packed >> 4))
            self.assertEqual(observed, set(range(16)))
            for row in (0, 1, 31, 32, 127, 128, 1023, 2047):
                want = [
                    scalar_nibble(row, k, operand)
                    | (scalar_nibble(row, k + 1, operand) << 4)
                    for k in range(0, 1024, 2)
                ]
                with self.subTest(operand=operand, row=row):
                    np.testing.assert_array_equal(packed[row], want)

    def test_scales_use_little_endian_words_and_canonical_atoms(self):
        fixture = self.fixture()
        self.assertEqual(fixture["sa"][0, :2].tolist(), [0x7F7E, 0x7F7E])
        self.assertEqual(fixture["sb"][0, :2].tolist(), [0x7E7E, 0x807F])
        self.assertEqual(int(fixture["sa"][1, 0]), 0x7F7F)
        self.assertEqual(int(fixture["sb"][1, 0]), 0x8080)
        for operand, name in (("a", "sa"), ("b", "sb")):
            raw = fixture[name].astype("<u2", copy=False).view(np.uint8)
            self.assertEqual(set(np.unique(raw)), {126, 127, 128})
            # Physical order is [row/128][group/4][row%32][row/32%4][group%4].
            atoms = raw.reshape(16, 8, 32, 4, 4)
            for row in (0, 1, 31, 32, 63, 64, 95, 96, 127, 128, 2047):
                for group in range(32):
                    with self.subTest(operand=operand, row=row, group=group):
                        self.assertEqual(
                            int(atoms[row // 128, group // 4, row % 32, row // 32 % 4, group % 4]),
                            scalar_scale(row, group, operand),
                        )

    def test_expected_fp16_bits_match_independent_integer_submatrix(self):
        actual = self.fixture()["expected"]
        for row in (0, 1, 31, 32, 127, 128, 1023, 2047):
            for column in (0, 1, 32, 127, 128, 2047):
                with self.subTest(row=row, column=column):
                    self.assertEqual(int(actual[row, column]), scalar_result_bits(row, column))
        self.assertTrue(np.isfinite(actual.view(np.float16)).all())
        self.assertTrue(np.any(actual.view(np.float16) != 0))


if __name__ == "__main__":
    unittest.main()
