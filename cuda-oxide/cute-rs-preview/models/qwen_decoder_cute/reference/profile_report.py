#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize an Nsight Systems SQLite export without exporting its environment.

Validated against the installed Nsight Systems 2026.3.2 / SQLite schema 3.28.0.
The source report contains inherited process environment and stays untracked.
Only explicitly selected metadata is emitted. No CUDA or third-party imports.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3


def interval_union(intervals):
    """Nanoseconds covered by intervals; overlapping streams count only once."""
    total = 0
    left = right = None
    for start, end in sorted(intervals):
        if left is None:
            left, right = start, end
        elif start > right:
            total += right - left
            left, right = start, end
        else:
            right = max(right, end)
    return total + (0 if left is None else right - left)


def allocation_peak(rows, allocation, deallocation):
    """Replay allocation identities, declining exactness on incomplete records.

    Address reuse is normal after a free. Identity includes process/device/context.
    This measures live requested device bytes, not driver/module overhead or the
    physical reservation of CUDA pools. Equal timestamps with multiple events do
    not establish their order, so an exact peak is not claimed in that case.
    """
    live = {}
    devices = defaultdict(lambda: {"live_bytes": 0, "peak_bytes": 0})
    total = peak = allocations = frees = allocated_bytes = 0
    peak_time = previous_time = None
    anomalies = []
    for time, process, device, context, address, size, operation in rows:
        key = (process, device, context, address)
        if previous_time is not None and time <= previous_time:
            anomalies.append("unordered or simultaneous memory events")
        previous_time = time
        delta = 0
        if size < 0:
            anomalies.append("negative allocation size")
            continue
        if operation == allocation:
            allocations += 1
            allocated_bytes += size
            if key in live:
                anomalies.append("allocation reuses a still-live address")
                continue
            live[key] = size
            delta = size
        elif operation == deallocation:
            frees += 1
            original = live.pop(key, None)
            if original is None:
                anomalies.append("free has no allocation in this capture")
                continue
            if original != size:
                anomalies.append("free size differs from matching allocation")
            delta = -original
        else:
            anomalies.append("unknown memory operation")
        total += delta
        state = devices[str(device)]
        state["live_bytes"] += delta
        state["peak_bytes"] = max(state["peak_bytes"], state["live_bytes"])
        if total > peak:
            peak, peak_time = total, time
    exact = not anomalies
    if not exact:
        for state in devices.values():
            state["peak_bytes"] = None
    return {
        "exact_for_traced_device_allocations": exact,
        "peak_bytes": peak if exact else None,
        "peak_timestamp_ns": peak_time if exact else None,
        "final_live_bytes": total,
        "allocation_count": allocations,
        "free_count": frees,
        "total_allocated_bytes": allocated_bytes,
        "per_device": dict(devices),
        "anomalies": sorted(set(anomalies)),
        "scope": "Requested live bytes in traced device allocations; excludes driver/module overhead and pool reservations.",
    }


def require_columns(connection, table, names):
    """Reject a SQLite table that lacks columns required by this schema reader."""
    present = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
    missing = set(names) - present
    if missing:
        raise ValueError(f"unsupported schema: {table} lacks {sorted(missing)}")


def rows_as_dicts(connection, query, parameters=()):
    """Execute a parameterized query and map each result row by column name."""
    cursor = connection.execute(query, parameters)
    names = [entry[0] for entry in cursor.description]
    return [dict(zip(names, row)) for row in cursor]


def summarize(connection):
    """Summarize observed CUDA activity, API calls and allocation lifetimes.

    Times are nanoseconds. Overlapping activity is counted once for union spans;
    graph envelopes may include kernels whose individual events were omitted.
    The result marks incomplete coverage and emits only selected metadata, never
    the captured process environment. Profiling spans are not benchmark latency."""
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    require_columns(connection, "StringIds", ("id", "value"))
    has_kernels = "CUPTI_ACTIVITY_KIND_KERNEL" in tables
    if has_kernels:
        require_columns(connection, "CUPTI_ACTIVITY_KIND_KERNEL", ("start", "end", "demangledName", "gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ"))
    require_columns(connection, "CUPTI_ACTIVITY_KIND_RUNTIME", ("start", "end", "nameId"))
    metadata = dict(connection.execute(
        "SELECT name,value FROM META_DATA_EXPORT WHERE name IN ('EXPORT_PRODUCT_NAME','EXPORT_PRODUCT_VERSION','EXPORT_SCHEMA_VERSION')"
    ))
    kernel_groups = rows_as_dicts(connection, """
        SELECT s.value AS name, COUNT(*) AS count, SUM(k.end-k.start) AS total_ns,
               AVG(k.end-k.start) AS mean_ns, MIN(k.end-k.start) AS min_ns,
               MAX(k.end-k.start) AS max_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
        GROUP BY k.demangledName ORDER BY total_ns DESC
    """) if has_kernels else []
    kernel_grids = rows_as_dicts(connection, """
        SELECT s.value AS name, gridX,gridY,gridZ,blockX,blockY,blockZ,
               COUNT(*) AS count,SUM(k.end-k.start) AS total_ns,
               AVG(k.end-k.start) AS mean_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
        GROUP BY k.demangledName,gridX,gridY,gridZ,blockX,blockY,blockZ
        ORDER BY total_ns DESC
    """) if has_kernels else []
    api_groups = rows_as_dicts(connection, """
        SELECT s.value AS name,COUNT(*) AS count,SUM(r.end-r.start) AS total_ns,
               AVG(r.end-r.start) AS mean_ns,MAX(r.end-r.start) AS max_ns
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id
        GROUP BY r.nameId ORDER BY total_ns DESC
    """)
    copies = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        require_columns(connection, "CUPTI_ACTIVITY_KIND_MEMCPY", ("start", "end", "bytes", "copyKind"))
        copies = rows_as_dicts(connection, """
            SELECT e.label AS direction,COUNT(*) AS records,SUM(m.bytes) AS bytes,
                   SUM(m.end-m.start) AS total_ns
            FROM CUPTI_ACTIVITY_KIND_MEMCPY m JOIN ENUM_CUDA_MEMCPY_OPER e ON m.copyKind=e.id
            GROUP BY m.copyKind ORDER BY total_ns DESC
        """)
    intervals = list(connection.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL")) if has_kernels else []
    graphs = []
    if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables:
        require_columns(connection, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE", ("start", "end"))
        graphs = list(connection.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE"))
    activity = intervals + graphs
    if not activity:
        raise ValueError("capture contains no CUDA kernel or graph events")
    start, end = min(a for a, _ in activity), max(b for _, b in activity)
    # Graph-level tracing can omit all graph node kernels. Its envelopes are
    # recorded activity, but may include gaps between nodes: they are not kernel
    # busy time. Union them with kernels/copies so mixed traces cannot double count.
    # Clip copies/memsets to the observed kernel/graph span; initial uploads stay
    # in full-capture copy totals without inflating the activity ratio.
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if table in tables:
            activity.extend((max(a, start), min(b, end)) for a, b in connection.execute(
                f"SELECT start,end FROM {table} WHERE end>? AND start<?", (start, end)
            ))
    kernel_union = interval_union(intervals)
    activity_union = interval_union(activity)
    api_window = rows_as_dicts(connection, """
        SELECT s.value AS name,COUNT(*) AS count,
               SUM(MIN(r.end,?)-MAX(r.start,?)) AS total_ns
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id
        WHERE r.end>? AND r.start<? GROUP BY r.nameId ORDER BY total_ns DESC
    """, (end, start, start, end))
    memory = {"available": False, "reason": "capture lacks CUDA_GPU_MEMORY_USAGE_EVENTS"}
    if "CUDA_GPU_MEMORY_USAGE_EVENTS" in tables:
        require_columns(connection, "CUDA_GPU_MEMORY_USAGE_EVENTS", ("start", "globalPid", "deviceId", "contextId", "address", "bytes", "memoryOperationType", "memKind"))
        kinds = dict(connection.execute("SELECT name,id FROM ENUM_CUDA_MEM_KIND"))
        operations = dict(connection.execute("SELECT name,id FROM ENUM_CUDA_DEV_MEM_EVENT_OPER"))
        device_kind = kinds["CUDA_MEMOPR_MEMORY_KIND_DEVICE"]
        memory = allocation_peak(connection.execute("""
            SELECT start,globalPid,deviceId,contextId,address,bytes,memoryOperationType
            FROM CUDA_GPU_MEMORY_USAGE_EVENTS WHERE memKind=? ORDER BY start,rowid
        """, (device_kind,)), operations["CUDA_DEV_MEM_EVENT_OPR_ALLOCATION"], operations["CUDA_DEV_MEM_EVENT_OPR_DEALLOCATION"])
        memory["available"] = True
    gpu = []
    if "TARGET_INFO_GPU" in tables:
        gpu = rows_as_dicts(connection, "SELECT id,name,uuid,totalMemory,computeMajor,computeMinor FROM TARGET_INFO_GPU")
    cuda_devices = []
    if "TARGET_INFO_CUDA_DEVICE" in tables:
        cuda_devices = rows_as_dicts(connection, "SELECT gpuId,cudaId,pid,uuid FROM TARGET_INFO_CUDA_DEVICE")
    return {
        "schema_version": 1,
        "source_metadata": metadata,
        "gpu": gpu,
        "cuda_device_mapping": cuda_devices,
        "scope": (
            "All events present in the capture. "
            + ("The kernel_span is the observed first-to-last kernel/graph interval, not necessarily an application/NVTX synthesis boundary." if graphs else "Kernel span is the observed first-to-last kernel interval, not necessarily an application/NVTX synthesis boundary.")
        ),
        "timing_caveat": (
            "Profiled timing includes instrumentation. CPU API totals include waits and may overlap GPU execution; do not add them to GPU totals."
            + (" Graph envelopes are included in GPU activity coverage, not kernel busy time. Individual graph node timing is incomplete or unverified; kernel counts/totals describe recorded kernels only. Use node-level tracing for kernel attribution." if graphs else "")
        ),
        "kernel_timing_coverage": "incomplete_or_unverified_for_graphs" if graphs else "individual_kernel_records",
        "graph_trace": {
            "count": len(graphs),
            "summed_envelope_ns": sum(b - a for a, b in graphs),
            "envelope_union_ns": interval_union(graphs),
        },
        "kernel_span": {
            "start_ns": start, "end_ns": end, "elapsed_ns": end - start,
            "kernel_union_ns": kernel_union,
            "gpu_activity_union_ns": activity_union,
            "no_recorded_gpu_activity_ns": end - start - activity_union,
            "kernel_active_fraction": None if graphs else kernel_union / (end - start),
            "recorded_kernel_fraction": kernel_union / (end - start),
            "gpu_activity_includes_graph_envelopes": bool(graphs),
            "api": api_window,
        },
        "kernel_count": sum(row["count"] for row in kernel_groups),
        "summed_kernel_ns": sum(row["total_ns"] for row in kernel_groups),
        "kernels": kernel_groups,
        "kernel_grids": kernel_grids,
        "api": api_groups,
        "copies": copies,
        "device_allocations": memory,
    }


def main():
    """Read an Nsight SQLite export and write a JSON summary with its file hash."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    database = args.sqlite.resolve(strict=True)
    if args.output.resolve() == database:
        parser.error("output must not overwrite the SQLite input")
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        result = summarize(connection)
    with database.open("rb") as source:
        result["source_sqlite_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}: {result['kernel_count']} kernels; traced device peak {result['device_allocations'].get('peak_bytes')} bytes")


if __name__ == "__main__":
    main()
