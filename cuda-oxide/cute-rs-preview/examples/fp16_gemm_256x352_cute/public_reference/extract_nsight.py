# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract GPU kernel durations from a profiled FP16 comparison.

Requires Nsight Systems 2026.3 SQLite CUDA software tracing with GPU event
timestamps and the JSON report from the same invocation of compare.py.
Consecutive CUDA event records delimit single-kernel samples independently
of the report's median. Process, device, context, green context and stream
must all match. Missing records, extra timed kernels and dropped data fail.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics


# Context and stream IDs repeat between processes; all five fields must match.
IDENTITY = ('globalPid', 'deviceId', 'contextId', 'greenContextId', 'streamId')


def require(condition, message):
    """Reject incomplete or inconsistent input evidence."""
    if not condition:
        raise ValueError(message)


def sha256(path):
    """Hash a source artifact without reading the complete file into memory."""
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def identity(record):
    """Return the full CUDA stream identity, including the process."""
    return tuple(record[name] for name in IDENTITY)


def classify(name, rust_label):
    """Recognize the Rust entry point and the checked-in Python GEMM only."""
    if name == 'fp16_gemm_256x352':
        return rust_label
    if name.startswith('kernel_cutlass_kernel_public_referencegemm_Gemm_object_at_'):
        return 'python_ctm'
    return None


def stats(values):
    """Summarize nonempty microsecond durations without assuming normality."""
    return {'count': len(values), 'median_us': statistics.median(values),
            'mean_us': statistics.mean(values), 'min_us': min(values), 'max_us': max(values)}


def extract(database, comparison):
    """Return validated per-implementation timing summaries and sample rows."""
    source = json.loads(comparison.read_text())
    require(source['verification'] == 'both-pass', 'comparison correctness did not pass')
    require(source['cluster_counts_match'], 'comparison launch cluster counts differ')
    protocol = source['provenance']
    require(protocol['timing_mode'] == 'direct' and protocol['graph_launches'] == 1,
            'only direct single-kernel event samples are supported')
    iterations, warmup = protocol['iters'], protocol['warmup']
    require(iterations > 0 and warmup >= 0, 'invalid timing counts')
    rust_label = source['cuda_oxide']['implementation']
    require(rust_label in ('cute-rs', 'ctm-rs'), f'unsupported Rust implementation: {rust_label}')
    event_reports = {rust_label: source['cuda_oxide'], 'python_ctm': source['python_reference']}
    with sqlite3.connect(f'file:{database.resolve()}?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        kernels = [dict(row) for row in connection.execute('''
            SELECT k.*, s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName = s.id ORDER BY k.start
        ''')]
        events = [dict(row) for row in connection.execute('''
            SELECT * FROM CUPTI_ACTIVITY_KIND_CUDA_EVENT ORDER BY timestamp
        ''')]
        diagnostics = [dict(row) for row in connection.execute('SELECT * FROM DIAGNOSTIC_EVENT')]
        session = dict(connection.execute('SELECT * FROM TARGET_INFO_SESSION_START_TIME').fetchone())
    require(not any('dropped' in record['text'].lower() for record in diagnostics),
            'Nsight reported dropped records; trace is incomplete')
    grouped_kernels, grouped_events = defaultdict(list), defaultdict(list)
    workload_streams = defaultdict(set)
    for kernel in kernels:
        key = identity(kernel)
        grouped_kernels[key].append(kernel)
        label = classify(kernel['name'], rust_label)
        if label:
            workload_streams[key].add(label)
    for event in events:
        key = identity(event)
        if key in workload_streams:
            require(event['timestamp'] > 0, 'GPU event completion timestamp is unavailable')
            grouped_events[key].append(event)
    report = {
        'case': database.stem, 'nsight_sqlite': database.name,
        'nsight_sqlite_sha256': sha256(database),
        'comparison_report': comparison.name, 'comparison_report_sha256': sha256(comparison),
        'session': session, 'mnk': source['mnk'], 'has_bias': source['has_bias'],
        'seed': source['seed'], 'input_sha256': source['input_sha256'],
        'warmup': warmup, 'iterations': iterations, 'diagnostics': diagnostics,
        'method': 'Consecutive GPU CUDA-event pairs matched by process/device/context/green-context/stream; one complete GEMM per pair; median kernel end-start duration.',
        'event_cross_check': 'Only the median is recorded by compare.py; compare the independently captured GPU-event median with that report using max(0.1 us, 0.2%) tolerance.',
        'implementations': {},
    }
    rows = []
    for key, labels in workload_streams.items():
        require(len(labels) == 1, f'ambiguous workload stream {key}: {labels}')
        label = next(iter(labels))
        require(label not in report['implementations'], f'multiple workload streams for {label}')
        stream_events = grouped_events[key]
        require(len(stream_events) == 2 * iterations,
                f'{label}: expected {2 * iterations} GPU events, got {len(stream_events)}')
        stream_kernels = grouped_kernels[key]
        workload_kernels = [kernel for kernel in stream_kernels if classify(kernel['name'], rust_label) == label]
        # Rust verifies an extra untimed launch except in the single-launch mode.
        preliminary = 0 if label == 'python_ctm' or source['cuda_oxide']['provenance']['single_launch'] else 1
        expected_kernels = warmup + iterations + preliminary
        require(len(workload_kernels) == expected_kernels,
                f'{label}: expected {expected_kernels} total GEMMs, got {len(workload_kernels)}')
        names = {kernel['name'] for kernel in workload_kernels}
        require(len(names) == 1, f'{label}: ambiguous GEMM kernel names: {names}')
        durations, event_durations, local_rows = [], [], []
        for index in range(iterations):
            start_event, end_event = stream_events[index * 2:index * 2 + 2]
            begin, finish = start_event['timestamp'], end_event['timestamp']
            require(begin < finish, f'{label}: inverted CUDA event pair {index}')
            overlapping = [kernel for kernel in stream_kernels
                           if kernel['start'] < finish and begin < kernel['end']]
            require(len(overlapping) == 1,
                    f'{label}: sample {index} contains {len(overlapping)} overlapping kernels; expected one')
            kernel = overlapping[0]
            require(classify(kernel['name'], rust_label) == label,
                    f'{label}: unexpected timed kernel: {kernel["name"]}')
            require(begin <= kernel['start'] < kernel['end'] <= finish,
                    f'{label}: kernel crosses event boundary in sample {index}')
            require((kernel['gridX'], kernel['gridY'], kernel['gridZ']) ==
                    (2 * source['python_reference']['clusters'], 1, 1),
                    f'{label}: timed grid disagrees with the comparison report')
            require((kernel['blockX'], kernel['blockY'], kernel['blockZ']) == (192, 1, 1),
                    f'{label}: unexpected block dimensions')
            duration = (kernel['end'] - kernel['start']) / 1000
            event_duration = (finish - begin) / 1000
            durations.append(duration)
            event_durations.append(event_duration)
            local_rows.append({
                'implementation': label, 'sample': index, **dict(zip(IDENTITY, key)),
                'start_event_id': start_event['eventId'], 'end_event_id': end_event['eventId'],
                'start_event_ns': begin, 'end_event_ns': finish,
                'kernel_start_ns': kernel['start'], 'kernel_end_ns': kernel['end'],
                'kernel_duration_us': duration, 'event_duration_us': event_duration,
                'correlation_id': kernel['correlationId'], 'grid_id': kernel['gridId'],
            })
        delta = statistics.median(event_durations) - event_reports[label]['metric_value']
        require(abs(delta) <= max(0.1, statistics.median(event_durations) * 0.002),
                f'{label}: GPU event median disagrees with reported median by {delta} us')
        report['implementations'][label] = {
            'kernel_name': next(iter(names)), 'stream_identity': dict(zip(IDENTITY, key)),
            'selected_kernel_count': len(local_rows), 'cuda_event_count': len(stream_events),
            'total_workload_kernel_count': len(workload_kernels),
            'excluded_warmup_kernel_count': warmup,
            'excluded_preliminary_kernel_count': preliminary,
            'other_stream_kernel_count': len(stream_kernels) - len(workload_kernels),
            'kernel_duration': stats(durations), 'trace_event_duration': stats(event_durations),
            'reported_profiled_event_median_us': event_reports[label]['metric_value'],
            'event_median_cross_check_delta_us': delta,
        }
        rows.extend(local_rows)
    require(set(report['implementations']) == {rust_label, 'python_ctm'},
            f'missing implementation: {set(report["implementations"])}')
    return report, rows


def main():
    """Write a compact JSON summary and independently inspectable CSV timestamps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument('--comparison', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--samples', type=Path)
    args = parser.parse_args()
    comparison = args.comparison or args.database.with_suffix('.json')
    output = args.output or args.database.with_name(args.database.stem + '-kernels.json')
    samples = args.samples or args.database.with_name(args.database.stem + '-samples.csv')
    report, rows = extract(args.database, comparison)
    with samples.open('w', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    report['samples_csv'] = samples.name
    report['samples_csv_sha256'] = sha256(samples)
    output.write_text(json.dumps(report, indent=2) + '\n')
    for label, result in report['implementations'].items():
        print(label, json.dumps(result['kernel_duration']))


if __name__ == '__main__':
    main()
