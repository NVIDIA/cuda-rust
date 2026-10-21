# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select timed SM120 prefill/decode invocations from an all-cases capture.

Requires Nsight Systems 2026.3 SQLite GPU event timestamps and the comparison
JSON from that same profiled run. Each case runs CuTeDSL then Rust in separate
processes. Each stream has two correctness invocations, warmups, then one
invocation per start/end event pair. Unexpected counts or ordering fail closed.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import statistics


IDENTITY = ('globalPid', 'deviceId', 'contextId', 'greenContextId', 'streamId')


def require(condition, message):
    """Reject incomplete evidence instead of publishing a partial timing."""
    if not condition:
        raise ValueError(message)


def classify(name, decode):
    """Resolve implementation and stage using the pinned workload names."""
    if decode:
        for component in ('partial', 'merge'):
            if 'fmha_decode_' + component in name:
                label = 'cuda_oxide' if name.startswith('fmha_decode_') else 'python_reference'
                return label, component
    elif name.startswith('fmha_prefill_'):
        return 'cuda_oxide', 'main'
    elif 'FlashAttentionForwardAmpere' in name:
        return 'python_reference', 'main'
    return None


def extract_stream(kernels, events, runner, decode, label):
    """Check a single CUDA stream and select only event-enclosed invocations."""
    warmup, iterations = (runner['provenance'][key] for key in ('warmup', 'iters'))
    require(iterations > 0 and warmup >= 0, 'invalid sample configuration')
    components = ['partial', 'merge'] if decode else ['main']
    expected = [(label, component) for component in components]
    workload = [kernel for kernel in kernels if classify(kernel['name'], decode)]
    require([classify(kernel['name'], decode) for kernel in workload]
            == expected * (2 + warmup + iterations), 'incomplete or misordered workload launches')
    require(all(kernel['end'] > kernel['start'] for kernel in kernels), 'invalid kernel duration')
    require(all(a['end'] <= b['start'] for a, b in zip(kernels, kernels[1:])),
            'overlapping kernels on the selected stream')
    require(len(events) == 2 * iterations, 'missing or extra GPU events')
    require(all(0 < a['timestamp'] < b['timestamp'] for a, b in zip(events, events[1:])),
            'invalid or overlapping GPU event intervals')
    require(len(runner['samples_us']) == iterations, 'invalid event report sample count')
    selected_all, active, spans, event_intervals = [], [], [], []
    component_samples = {component: [] for component in components}
    for index in range(iterations):
        begin, finish = [event['timestamp'] for event in events[2 * index:2 * index + 2]]
        selected = [kernel for kernel in kernels if kernel['start'] < finish and begin < kernel['end']]
        require(all(begin <= kernel['start'] and kernel['end'] <= finish for kernel in selected),
                'kernel crosses a GPU event boundary')
        require([classify(kernel['name'], decode) for kernel in selected] == expected,
                'missing, extra or misordered timed kernels')
        interval = (finish - begin) / 1000
        # Keep both observations. GPU completion-marker intervals can differ
        # from CUDA elapsed-time reports; ordinal selection is checked below.
        event_intervals.append(interval)
        durations = [(kernel['end'] - kernel['start']) / 1000 for kernel in selected]
        active.append(sum(durations))
        spans.append((selected[-1]['end'] - selected[0]['start']) / 1000)
        for component, duration in zip(components, durations):
            component_samples[component].append(duration)
        selected_all.extend(selected)
    excluded = (2 + warmup) * len(components)
    require(selected_all == workload[excluded:], 'events did not select exactly the post-warmup launches')
    return {
        'kernel_names': list(dict.fromkeys(kernel['name'] for kernel in workload)),
        'selected_kernel_count': len(selected_all), 'excluded_workload_kernel_count': excluded,
        'auxiliary_kernel_count': len(kernels) - len(workload),
        'active_samples_us': active, 'active_median_us': statistics.median(active),
        'span_samples_us': spans, 'span_median_us': statistics.median(spans),
        'component_samples_us': component_samples,
        'profiled_event_samples_us': runner['samples_us'], 'trace_event_intervals_us': event_intervals,
        'trace_minus_reported_event_samples_us': [trace - reported for trace, reported
                                                  in zip(event_intervals, runner['samples_us'])],
        'selected_timestamp_bounds_ns': [selected_all[0]['start'], selected_all[-1]['end']],
        'event_timestamp_bounds_ns': [events[0]['timestamp'], events[-1]['timestamp']],
    }


def sha256(path):
    """Bind the result to the full local trace or report without copying it."""
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def extract(database, comparison):
    """Match ordered all-cases child processes by full CUDA stream identity."""
    report = json.loads(comparison.read_text())
    require(report['verification'] == 'all-pass', 'comparison did not pass all cases')
    decode = report['workload'] == 'fmha-decode'
    cases = report['cases']
    streams, events = defaultdict(list), defaultdict(list)
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        for row in db.execute('SELECT k.*, s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k '
                              'JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start'):
            streams[tuple(row[key] for key in IDENTITY)].append(dict(row))
        for row in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_CUDA_EVENT ORDER BY timestamp'):
            events[tuple(row[key] for key in IDENTITY)].append(dict(row))
        diagnostics = [dict(row) for row in db.execute('SELECT * FROM DIAGNOSTIC_EVENT')]
        session = dict(db.execute('SELECT * FROM TARGET_INFO_SESSION_START_TIME').fetchone())
    require(not any('dropped' in row['text'].lower() or 'incomplete' in row['text'].lower()
                    for row in diagnostics), 'Nsight reports an incomplete capture')
    workload_streams = [(key, kernels) for key, kernels in streams.items()
                        if any(classify(kernel['name'], decode) for kernel in kernels)]
    workload_streams.sort(key=lambda item: item[1][0]['start'])
    require(len(workload_streams) == len(cases) * 2, 'unexpected workload process/stream count')
    results = []
    for index, case in enumerate(cases):
        require(case['verification'] == 'both-pass', 'case validation failed')
        require(case['timing']['order'] == ['python_reference', 'cuda_oxide'], 'unexpected runner order')
        result = {'shape_mode': case['shape_mode'], 'causal': case['causal'],
                  'shape': case['cuda_oxide']['shape'], 'implementations': {}}
        for offset, label in enumerate(case['timing']['order']):
            key, kernels = workload_streams[index * 2 + offset]
            runner = case[label]
            require(runner['verification'] == 'cpu-pass' and runner['replay_bitwise_equal'],
                    'runner correctness or replay check failed')
            timings = extract_stream(kernels, events[key], runner, decode, label)
            timings['stream_identity'] = dict(zip(IDENTITY, key))
            result['implementations'][label] = timings
        results.append(result)
    diagnostic_counts = Counter((row['severity'], re.sub(r'/tmp/\S+', '<local-path>', row['text']))
                                for row in diagnostics)
    return {'sqlite_file': database.name, 'sqlite_sha256': sha256(database),
            'comparison_file': comparison.name, 'comparison_sha256': sha256(comparison),
            'session': session, 'workload': report['workload'],
            'diagnostics': [{'severity': severity, 'text': text, 'count': count}
                            for (severity, text), count in diagnostic_counts.items()],
            'cases': results}


def main():
    """Write portable JSON evidence from a capture and its comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = extract(args.database, args.comparison)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for case in report['cases']:
        print(case['shape_mode'], 'causal' if case['causal'] else 'noncausal',
              {label: values['active_median_us'] for label, values in case['implementations'].items()})


if __name__ == '__main__':
    main()
