# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract only complete timed invocations enclosed by GPU CUDA events.

The workload's JSON supplies expected windows/iterations, never timestamps.
CUDA event records independently delimit the windows. Kernel records are
matched by process, device, context, green context and stream before selection.
Missing records, unexpected components and overlapping kernels fail closed.

Requires the Nsight Systems 2026.3 SQLite schema: CUPTI kernel records,
CUDA-event completion timestamps, StringIds, session metadata and diagnostics.
The matching comparison JSON must come from the same profiled invocation.
Both workloads record consecutive start/end event pairs on one stream per
implementation; split decode submits alternating main and reduction kernels.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics

# CUDA stream identity includes the process because context and stream IDs repeat.
IDENTITY = ('globalPid', 'deviceId', 'contextId', 'greenContextId', 'streamId')


def require(condition, message):
    """Reject an incomplete or inconsistent trace with a descriptive error."""
    if not condition:
        raise ValueError(message)


def sha256(path):
    """Hash an evidence file without loading the complete trace into memory."""
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def identity(record):
    """Return the full CUDA stream identity for an event or kernel record."""
    return tuple(record[name] for name in IDENTITY)


def classify(name, decode):
    """Identify the pinned attention implementation and its kernel component."""
    if name.startswith('sm100_') and name.endswith('_cute_rs'):
        return 'cute_rs', 'reduction' if '_reduce_' in name else 'main'
    if decode and 'BlackwellMultiHeadLatentAttentionForwardFP16' in name:
        return 'cutedsl', 'reduction' if '_reduction_kernel_' in name else 'main'
    if not decode and 'BlackwellFusedMultiHeadAttentionForward' in name:
        return 'cutedsl', 'main'
    return None


def stats(values):
    """Summarize a nonempty sequence of microsecond execution durations."""
    return {'count': len(values), 'mean_us': statistics.mean(values),
            'median_us': statistics.median(values), 'min_us': min(values), 'max_us': max(values)}


def extract(database, comparison):
    """Validate a profiled comparison and return selected GPU timing evidence."""
    source = json.loads(comparison.read_text())
    decode = 'configuration' in source
    config = source['configuration'] if decode else source['config']
    timing = source['timing']
    iterations = timing['iterations']
    windows = timing['rounds' if decode else 'windows']
    expected_components = ['main', 'reduction'] if decode and config['splits'] > 1 else ['main']
    if decode:
        event_reports = {'cute_rs': source['rust'], 'cutedsl': source['cutedsl']}
    else:
        event_reports = {name: timing[f'sm100_fmha_prefill_{suffix}']
                         for name, suffix in [('cute_rs', 'cute_rs'), ('cutedsl', 'cutedsl')]}
    connection = sqlite3.connect(f'file:{database.resolve()}?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    kernel_records = [dict(row) for row in connection.execute('''
        SELECT k.*, s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id ORDER BY k.start
    ''')]
    event_records = [dict(row) for row in connection.execute('''
        SELECT * FROM CUPTI_ACTIVITY_KIND_CUDA_EVENT ORDER BY timestamp
    ''')]
    grouped_kernels = defaultdict(list)
    workload_identities = defaultdict(set)
    for kernel in kernel_records:
        grouped_kernels[identity(kernel)].append(kernel)
        component = classify(kernel['name'], decode)
        if component:
            workload_identities[identity(kernel)].add(component[0])
    grouped_events = defaultdict(list)
    for event in event_records:
        if identity(event) in workload_identities:
            require(event['timestamp'] > 0, 'GPU event timestamp was not captured')
            grouped_events[identity(event)].append(event)
    diagnostics = [dict(row) for row in connection.execute('SELECT * FROM DIAGNOSTIC_EVENT')]
    report = {
        'case': database.stem,
        'nsight_sqlite': str(database.resolve()), 'nsight_sqlite_sha256': sha256(database),
        'comparison_report': str(comparison.resolve()), 'comparison_report_sha256': sha256(comparison),
        'session': dict(connection.execute('SELECT * FROM TARGET_INFO_SESSION_START_TIME').fetchone()),
        'config': config, 'iterations_per_window': iterations, 'windows': windows,
        'warmup_per_window': timing['warmup'], 'kernel_components_per_invocation': expected_components,
        'method': 'GPU CUDA-event pairs, matched process/device/context/green-context/stream; sum kernel end-start durations within each complete invocation; median of window means',
        'diagnostics': diagnostics, 'implementations': {},
    }
    for key, events in grouped_events.items():
        labels = workload_identities[key]
        require(len(labels) == 1, f'ambiguous implementation on stream {key}: {labels}')
        label = next(iter(labels))
        require(label not in report['implementations'], f'multiple timed streams for {label}')
        require(len(events) == windows * 2, f'{label}: expected {windows * 2} GPU events, got {len(events)}')
        reference_event_samples = event_reports[label]['samples_us']
        require(len(reference_event_samples) == windows, f'{label}: invalid comparison window count')
        results = []
        total_selected = 0
        all_invocations = []
        for index in range(windows):
            start, end = events[index * 2:index * 2 + 2]
            begin, finish = start['timestamp'], end['timestamp']
            require(begin < finish, f'{label}: inverted event window {index}')
            selected = [kernel for kernel in grouped_kernels[key]
                        if begin <= kernel['start'] and kernel['end'] <= finish]
            overlapping = [kernel for kernel in grouped_kernels[key]
                           if kernel['start'] < finish and begin < kernel['end']]
            require(len(selected) == len(overlapping), f'{label}: kernel crosses event boundary in window {index}')
            expected_count = iterations * len(expected_components)
            require(len(selected) == expected_count,
                    f'{label}: window {index} expects {expected_count} kernels, got {len(selected)}; trace may have dropped events')
            require(all(a['end'] <= b['start'] for a, b in zip(selected, selected[1:])),
                    f'{label}: overlapping kernels cannot be summed on window {index}')
            invocations = []
            component_durations = defaultdict(list)
            raw = []
            for offset in range(0, len(selected), len(expected_components)):
                invocation = selected[offset:offset + len(expected_components)]
                actual = [classify(kernel['name'], decode) for kernel in invocation]
                require(actual == [(label, component) for component in expected_components],
                        f'{label}: unexpected kernel components in window {index}: {actual}')
                invocations.append(sum(kernel['end'] - kernel['start'] for kernel in invocation) / 1000)
                for component, kernel in zip(expected_components, invocation):
                    component_durations[component].append((kernel['end'] - kernel['start']) / 1000)
                    raw.append({'invocation': offset // len(expected_components), 'component': component,
                                'start_ns': kernel['start'], 'end_ns': kernel['end'],
                                'correlation_id': kernel['correlationId'], 'grid_id': kernel['gridId'],
                                'name': kernel['name']})
            event_interval_us = (finish - begin) / 1000 / iterations
            difference_us = event_interval_us - reference_event_samples[index]
            # Event timers and CUPTI clocks have different timestamp resolution.
            require(abs(difference_us) <= max(0.1, event_interval_us * 0.002),
                    f'{label}: GPU event pair disagrees with recorded window {index}: {difference_us} us/invocation')
            total_selected += len(selected)
            all_invocations.extend(invocations)
            results.append({'index': index, 'start_event': start, 'end_event': end,
                            'trace_event_interval_per_invocation_us': event_interval_us,
                            'reported_profiled_event_per_invocation_us': reference_event_samples[index],
                            'event_cross_check_delta_us': difference_us,
                            'invocation_kernel_duration': stats(invocations),
                            'components': {name: stats(values) for name, values in component_durations.items()},
                            'selected_kernels': raw})
        means = [window['invocation_kernel_duration']['mean_us'] for window in results]
        component_means = {component: [window['components'][component]['mean_us'] for window in results]
                           for component in expected_components}
        report['implementations'][label] = {
            'stream_identity': dict(zip(IDENTITY, key)),
            'selected_kernel_count': total_selected,
            'excluded_stream_kernel_count': len(grouped_kernels[key]) - total_selected,
            'kernel_duration_us': {'window_means': means, 'median_window_mean': statistics.median(means),
                                   'min_window_mean': min(means), 'max_window_mean': max(means),
                                   'all_invocations': stats(all_invocations)},
            'component_duration_us': {component: {'window_means': values, 'median_window_mean': statistics.median(values)}
                                      for component, values in component_means.items()},
            'windows': results,
        }
    require(set(report['implementations']) == {'cute_rs', 'cutedsl'},
            f'missing implementation event windows: {set(report["implementations"])}')
    require(not any('dropped' in item['text'].lower() for item in diagnostics),
            'Nsight reports dropped events; reject trace as incomplete')
    connection.close()
    return report


def main():
    """Write JSON timing evidence from a SQLite export and matched comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument('--comparison', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    comparison = args.comparison or args.database.with_name(args.database.stem + '-profiled.json')
    output = args.output or args.database.with_name(args.database.stem + '-kernels.json')
    report = extract(args.database, comparison)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(output)
    for implementation, result in report['implementations'].items():
        print(implementation, json.dumps(result['kernel_duration_us']))


if __name__ == '__main__':
    main()
