# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dump Tile IR MLIR for clean vs ambiguous-alias kernels.
Look for scope annotations on token-ordered operations.

Usage:
    python3 dump_mlir.py
"""

import cuda.tile as ct
from cuda.tile._compile import (
    _bind_kernel_arguments, _get_final_ir, get_sm_arch,
    _get_max_supported_bytecode_version, dev_features_enabled,
)
from cuda.tile._const_utils import get_constant_annotations
from cuda.tile._context import TileContextConfig
from cuda.tile._compiler_options import CompilerOptions
from cuda.tile._ir2bytecode import generate_bytecode_for_kernel
from cuda.tile._bytecode import writer as bc
import inspect, os, torch

ConstInt = ct.Constant[int]


@ct.kernel
def clean_separation(x, y, TILE: ConstInt):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.store(y, index=(bid,), tile=tx + tx)


@ct.kernel
def multi_read_ambiguous(x, y, cond: int, TILE: ConstInt):
    bid = ct.bid(0)
    t1 = ct.load(x, index=(bid,), shape=(TILE,))
    t2 = ct.load(x, index=(bid,), shape=(TILE,))
    t3 = ct.load(x, index=(bid,), shape=(TILE,))
    if cond:
        target = x
    else:
        target = y
    ct.store(target, index=(bid,), tile=t1 + t2 + t3)


def get_mlir(kernel_obj, args, label):
    pyfunc = kernel_obj._pyfunc
    param_names = tuple(inspect.signature(pyfunc).parameters.keys())
    const_args = get_constant_annotations(pyfunc)
    ir_args = _bind_kernel_arguments(param_names, args, const_args)

    config = TileContextConfig(
        temp_dir=os.environ.get('TMPDIR', '/tmp'),
        log_keys=frozenset(),
        compiler_timeout_sec=None,
        enable_crash_dump=False,
        cache_dir=None,
        cache_size_limit=100_000_000,
    )

    bytecode_version = _get_max_supported_bytecode_version(
        os.environ.get('TMPDIR', '/tmp'), allow_dev=dev_features_enabled())
    func_ir = _get_final_ir(pyfunc, ir_args, config, bytecode_version)
    compiler_options = CompilerOptions()
    sm_arch = get_sm_arch()

    bytecode_buf = bytearray()
    with bc.write_bytecode(num_functions=1, buf=bytecode_buf,
                           version=bytecode_version) as writer:
        generate_bytecode_for_kernel(func_ir, compiler_options, sm_arch,
                                     writer, anonymize_debug_attr=False)

    # Write bytecode to file and use tileiras to convert to MLIR text
    import subprocess, tempfile
    tmpdir = os.environ.get('TMPDIR', '/tmp')
    bc_path = os.path.join(tmpdir, f'{label.replace(" ", "_")}.tileirbc')
    with open(bc_path, 'wb') as f:
        f.write(bytecode_buf)

    tileiras = os.path.join(tmpdir, 'pylibs/nvidia/cu13/bin/tileiras')
    if not os.path.exists(tileiras):
        # Try finding it
        import glob
        candidates = glob.glob(os.path.join(tmpdir, 'pylibs/**/tileiras'), recursive=True)
        tileiras = candidates[0] if candidates else 'tileiras'

    try:
        result = subprocess.run(
            [tileiras, '--emit=mlir', bc_path],
            capture_output=True, text=True, timeout=30)
        mlir = result.stdout if result.returncode == 0 else f"tileiras error: {result.stderr}"
    except Exception as e:
        mlir = f"(Could not run tileiras: {e})"

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(mlir)

    # Highlight scope-related lines
    for line in mlir.split('\n'):
        if 'scope' in line.lower() or 'tko' in line.lower() or 'token' in line.lower():
            print(f"  >>> {line.strip()}")


N = 1024
TILE = 128
x = torch.randn(N, dtype=torch.float32, device='cuda')
y = torch.zeros(N, dtype=torch.float32, device='cuda')

get_mlir(clean_separation, (x, y, TILE), "CLEAN (separate alias sets)")
get_mlir(multi_read_ambiguous, (x, y, 1, TILE), "MULTI-READ AMBIGUOUS (merged alias set)")
