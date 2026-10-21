# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject traces that could silently include warmups or omit decode work."""
import copy
import unittest

from extract_nsight import extract_stream


class TimedInvocationTests(unittest.TestCase):
    """Exercise event selection independently of the SQLite storage format."""

    def setUp(self):
        """Create two validation pairs, one warmup pair and one timed pair."""
        self.kernels = [
            {'name': 'fmha_decode_' + component, 'start': i * 10000 + start,
             'end': i * 10000 + end}
            for i in range(4)
            for component, start, end in [('partial', 1000, 3000), ('merge', 4000, 5000)]
        ]
        self.events = [{'timestamp': 30500}, {'timestamp': 36000}]
        self.runner = {'provenance': {'warmup': 1, 'iters': 1}, 'samples_us': [5.5]}

    def extract(self, kernels=None, events=None):
        """Run selection with this fixture's expected two-stage invocation."""
        return extract_stream(self.kernels if kernels is None else kernels,
                              self.events if events is None else events,
                              self.runner, True, 'cuda_oxide')

    def test_only_timed_pair_contributes(self):
        result = self.extract()
        self.assertEqual(result['active_samples_us'], [3.0])
        self.assertEqual(result['span_samples_us'], [4.0])
        self.assertEqual(result['excluded_workload_kernel_count'], 6)
        self.assertEqual(result['selected_kernel_count'], 2)

    def test_missing_or_extra_event_fails(self):
        for events in [self.events[:1], self.events + [{'timestamp': 37000}]]:
            with self.subTest(events=events), self.assertRaises(ValueError):
                self.extract(events=events)

    def test_kernel_crossing_event_boundary_fails(self):
        with self.assertRaises(ValueError):
            self.extract(events=[{'timestamp': 32000}, self.events[1]])

    def test_overlapping_kernels_fail(self):
        kernels = copy.deepcopy(self.kernels)
        kernels[-2]['end'] = kernels[-1]['start'] + 100
        with self.assertRaises(ValueError):
            self.extract(kernels=kernels)

    def test_reversed_decode_stages_fail(self):
        kernels = copy.deepcopy(self.kernels)
        kernels[-2]['name'], kernels[-1]['name'] = kernels[-1]['name'], kernels[-2]['name']
        with self.assertRaises(ValueError):
            self.extract(kernels=kernels)

    def test_missing_warmup_or_timed_kernel_fails(self):
        for index in [0, 4, 7]:
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.extract(kernels=self.kernels[:index] + self.kernels[index + 1:])

    def test_extra_kernel_in_timed_interval_fails(self):
        extra = {'name': 'auxiliary', 'start': 33300, 'end': 33500}
        kernels = sorted(self.kernels + [extra], key=lambda row: row['start'])
        with self.assertRaises(ValueError):
            self.extract(kernels=kernels)

    def test_event_clock_disagreement_is_retained(self):
        self.runner['samples_us'] = [7.0]
        result = self.extract()
        self.assertEqual(result['trace_minus_reported_event_samples_us'], [-1.5])
        self.assertEqual(result['active_samples_us'], [3.0])


if __name__ == '__main__':
    unittest.main()
