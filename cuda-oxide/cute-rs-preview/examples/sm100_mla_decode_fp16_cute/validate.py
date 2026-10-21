# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded GPU correctness matrix; compare.py times only after these checks pass."""
import argparse
import json
from pathlib import Path

from compare import run_case
from contract import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-bin", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    cases = [(1, 128, 1, 1, "zero-query", 1.0), (1, 129, 3, 1, "zero-query", 1.0),
             (1, 1024, 1, 1, "random", 1.0), (2, 2049, 3, 1, "random", 1.0),
             (4, 4096, 8, 2, "random", 1.0), (1, 257, 2, 1, "random", 1.73),
             (1, 32768, 256, 1, "random", 1.0), (1, 129, 256, 2, "random", 1.0),
             (4, 8193, 1, 1, "random", 1.0)]
    reports = []
    for batch, length, splits, clusters, pattern, output_scale in cases:
        reports.append(run_case(args.rust_bin, Config(batch=batch, kv_length=length, splits=splits, output_scale=output_scale),
                                clusters=clusters, pattern=pattern, validation_only=True, timeout=180))
    rendered = json.dumps({"passed": len(reports), "cases": reports}, indent=2)
    if args.json:
        args.json.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
