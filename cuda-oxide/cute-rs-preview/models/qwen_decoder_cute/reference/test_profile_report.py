# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for measured intervals and allocation identity accounting."""

import importlib.util
from pathlib import Path
import json
import sqlite3
import unittest


class ProfileReportTests(unittest.TestCase):
    """Regression fixtures for overlapping activity and honest allocation accounting."""
    def setUp(self):
        """Import the actual report module before each isolated SQLite fixture."""
        path = Path(__file__).with_name("profile_report.py")
        self.assertTrue(path.exists(), "profiling report implementation is missing")
        spec = importlib.util.spec_from_file_location("profile_report", path)
        self.report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.report)

    def test_union_does_not_double_count_overlapping_gpu_work(self):
        """Count overlapping GPU intervals once and handle an empty trace."""
        self.assertEqual(self.report.interval_union([(10, 20), (12, 15), (18, 30), (40, 45)]), 25)
        self.assertEqual(self.report.interval_union([]), 0)

    def test_peak_matches_lifetimes_and_address_reuse(self):
        # timestamp, process, device, context, address, bytes, operation
        """Track peak live bytes across frees and reuse of the same address."""
        rows = [
            (1, 1, 0, 1, 100, 12, 0),
            (2, 1, 0, 1, 200, 8, 0),
            (3, 1, 0, 1, 100, 12, 1),
            (4, 1, 0, 1, 100, 16, 0),
            (5, 1, 0, 1, 200, 8, 1),
            (6, 1, 0, 1, 100, 16, 1),
        ]
        result = self.report.allocation_peak(rows, 0, 1)
        self.assertTrue(result["exact_for_traced_device_allocations"])
        self.assertEqual(result["peak_bytes"], 24)
        self.assertEqual(result["peak_timestamp_ns"], 4)
        self.assertEqual(result["final_live_bytes"], 0)
        self.assertEqual(result["allocation_count"], 3)

    def test_same_address_on_different_devices_is_not_an_alias(self):
        """Keep device identities distinct when allocation addresses coincide."""
        rows = [(1, 1, 0, 1, 100, 12, 0), (2, 1, 1, 1, 100, 8, 0)]
        result = self.report.allocation_peak(rows, 0, 1)
        self.assertEqual(result["peak_bytes"], 20)
        self.assertEqual(result["per_device"]["0"]["peak_bytes"], 12)
        self.assertEqual(result["per_device"]["1"]["peak_bytes"], 8)

    def test_incomplete_or_inconsistent_trace_cannot_claim_exact_peak(self):
        """Withhold an exact peak for missing, duplicate or unordered allocation events."""
        cases = [
            [(1, 1, 0, 1, 100, 12, 1)],  # allocation was outside capture
            [(1, 1, 0, 1, 100, 12, 0), (2, 1, 0, 1, 100, 8, 1)],
            [(1, 1, 0, 1, 100, 12, 0), (2, 1, 0, 1, 100, 12, 0)],
            [(1, 1, 0, 1, 100, 12, 9)],  # unknown operation
            [(1, 1, 0, 1, 100, 12, 0), (1, 1, 0, 1, 200, 8, 1)],
        ]
        for rows in cases:
            with self.subTest(rows=rows):
                result = self.report.allocation_peak(rows, 0, 1)
                self.assertFalse(result["exact_for_traced_device_allocations"])
                self.assertIsNone(result["peak_bytes"])
                self.assertTrue(result["anomalies"])

    def test_sqlite_summary_excludes_private_metadata_and_maps_cuda_device(self):
        # Minimal populated export schema; queries execute against real SQLite.
        """Exclude process environment while preserving physical-to-CUDA device mapping."""
        with sqlite3.connect(":memory:") as connection:
            connection.executescript("""
                CREATE TABLE StringIds(id INTEGER,value TEXT);
                INSERT INTO StringIds VALUES(1,'linear'),(2,'cuLaunchKernel');
                CREATE TABLE META_DATA_EXPORT(name TEXT,value TEXT);
                INSERT INTO META_DATA_EXPORT VALUES('EXPORT_SCHEMA_VERSION','3.28.0'),('PRIVATE_EXTRA','private sentinel');
                CREATE TABLE META_DATA_CAPTURE(name TEXT,value TEXT);
                INSERT INTO META_DATA_CAPTURE VALUES('PROCESS_0:ENVIRONMENT_VARIABLE','private sentinel');
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,demangledName INTEGER,gridX INTEGER,gridY INTEGER,gridZ INTEGER,blockX INTEGER,blockY INTEGER,blockZ INTEGER);
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(10,30,1,1,1,1,128,1,1);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER);
                INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(1,15,2);
                CREATE TABLE TARGET_INFO_CUDA_DEVICE(gpuId INTEGER,cudaId INTEGER,pid INTEGER,uuid TEXT);
                INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES(1,0,123,'physical-gpu-one');
            """)
            result = self.report.summarize(connection)
        self.assertNotIn("private sentinel", json.dumps(result))
        self.assertEqual(result["kernel_count"], 1)
        self.assertEqual(result["kernel_span"]["api"][0]["total_ns"], 5)
        self.assertEqual(result["cuda_device_mapping"][0]["gpuId"], 1)
        self.assertEqual(result["cuda_device_mapping"][0]["cudaId"], 0)
        self.assertFalse(result["device_allocations"]["available"])

    def activity_database(self):
        """Create a small in-memory kernel/graph trace and register connection cleanup."""
        connection = sqlite3.connect(":memory:")
        self.addCleanup(connection.close)
        connection.executescript("""
            CREATE TABLE StringIds(id INTEGER,value TEXT);
            INSERT INTO StringIds VALUES(1,'linear'),(2,'cuLaunchKernel');
            CREATE TABLE META_DATA_EXPORT(name TEXT,value TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,demangledName INTEGER,gridX INTEGER,gridY INTEGER,gridZ INTEGER,blockX INTEGER,blockY INTEGER,blockZ INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(10,20,1,1,1,1,128,1,1),(40,50,1,1,1,1,128,1,1);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE(start INTEGER,end INTEGER);
        """)
        return connection

    def test_graph_envelopes_are_activity_without_double_counting_kernels(self):
        """Union graph envelopes with kernels without treating envelope time as kernel time."""
        connection = self.activity_database()
        connection.executescript("""
            INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES(15,35),(25,45);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(start INTEGER,end INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES(30,42);
        """)
        result = self.report.summarize(connection)
        span = result["kernel_span"]
        self.assertEqual(span["gpu_activity_union_ns"], 40)
        self.assertEqual(span["no_recorded_gpu_activity_ns"], 0)
        self.assertEqual(span["kernel_union_ns"], 20)
        self.assertEqual(result["summed_kernel_ns"], 20)
        self.assertEqual(result["kernel_count"], 2)
        self.assertEqual(result["graph_trace"]["count"], 2)
        self.assertEqual(result["graph_trace"]["summed_envelope_ns"], 40)
        self.assertEqual(result["graph_trace"]["envelope_union_ns"], 30)
        self.assertIsNone(span["kernel_active_fraction"])
        self.assertEqual(span["recorded_kernel_fraction"], 0.5)
        self.assertEqual(result["kernel_timing_coverage"], "incomplete_or_unverified_for_graphs")

    def test_graph_envelopes_extend_observed_span_beyond_kernel_bounds(self):
        """Include graph-only edges in the span and clip APIs/copies to those bounds."""
        connection = self.activity_database()
        connection.executescript("""
            INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES(0,8),(55,70);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(60,80,2);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(start INTEGER,end INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES(-10,12),(65,80);
        """)
        span = self.report.summarize(connection)["kernel_span"]
        self.assertEqual((span["start_ns"], span["end_ns"]), (0, 70))
        self.assertEqual(span["elapsed_ns"], 70)
        self.assertEqual(span["gpu_activity_union_ns"], 45)
        self.assertEqual(span["no_recorded_gpu_activity_ns"], 25)
        self.assertEqual(span["api"][0]["total_ns"], 10)

    def test_graph_only_capture_does_not_require_individual_kernel_events(self):
        """Summarize graph activity even when no per-kernel events were captured."""
        connection = self.activity_database()
        connection.executescript("""
            DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL;
            INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES(10,30),(40,50);
        """)
        result = self.report.summarize(connection)
        span = result["kernel_span"]
        self.assertEqual(span["gpu_activity_union_ns"], 30)
        self.assertEqual(span["no_recorded_gpu_activity_ns"], 10)
        self.assertEqual(span["kernel_union_ns"], 0)
        self.assertEqual(result["kernel_count"], 0)
        self.assertIsNone(span["kernel_active_fraction"])

    def test_graph_only_export_can_omit_the_empty_kernel_table(self):
        """Accept graph-only SQLite exports that omit the kernel table altogether."""
        connection = self.activity_database()
        connection.executescript("""
            DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL;
            INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES(10,30);
        """)
        result = self.report.summarize(connection)
        self.assertEqual(result["kernel_span"]["gpu_activity_union_ns"], 20)
        self.assertEqual(result["kernel_count"], 0)
        self.assertEqual(result["kernels"], [])
        self.assertEqual(result["kernel_grids"], [])

    def test_empty_graph_table_preserves_nongraph_activity_accounting(self):
        """Keep ordinary kernel-only accounting when the graph table is empty."""
        result = self.report.summarize(self.activity_database())
        span = result["kernel_span"]
        self.assertEqual((span["start_ns"], span["end_ns"]), (10, 50))
        self.assertEqual(span["gpu_activity_union_ns"], 20)
        self.assertEqual(span["no_recorded_gpu_activity_ns"], 20)
        self.assertEqual(span["kernel_active_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
