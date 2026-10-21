#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded exhaustive oracle/reference checks, including ring and scheduler reuse."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CASES = [
    ('single-key', '1,1,1,1,1', False, 'uniform'),
    ('short-causal', '1,17,31,4,2', True, 'uniform'),
    ('query-tail', '1,129,128,2,2', False, 'gaussian'),
    ('both-tails-gqa', '1,257,385,4,2', True, 'gaussian'),
    ('kv-ring-wrap', '1,512,1153,8,2', False, 'gaussian'),
    ('unequal-causal', '1,769,257,4,1', True, 'gaussian'),
    ('reference-smoke', '1,1024,1024,8,8', False, 'gaussian'),
    ('persistent-reuse', '4,4096,4096,8,8', False, 'gaussian'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rust-bin', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--sanitizer', choices=['memcheck', 'synccheck'])
    parser.add_argument('--case', action='append', choices=[x[0] for x in CASES])
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, shape, causal, pattern in CASES:
        if args.case and name not in args.case:
            continue
        command = [sys.executable, str(ROOT / 'compare.py'), '--rust-bin', str(args.rust_bin),
                   '--shape', shape, '--pattern', pattern, '--check-only', '--timeout', str(args.timeout)]
        if causal:
            command += ['--causal']
        if args.sanitizer:
            command += ['--sanitizer', args.sanitizer]
        if args.output_dir:
            command += ['--json', str(args.output_dir / f'{name}.json')]
        print(f'Checking {name}: {shape}', flush=True)
        subprocess.run(command, check=True, timeout=args.timeout + 60)
    print('All selected prefill validation cases passed.')


if __name__ == '__main__':
    main()
