# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Call tilelib_relu's `relu` from a cuTile kernel through `foreign_call`.

    python call_from_cutile.py --dry-run           # bytecode only; runs on CUDA 13.4
    python call_from_cutile.py [path/to.tilelib]   # full run; needs the CUDA 13.5 tileiras

The kernel loads one 128-float tile, hands it to the library's `relu`
declaration together with a zero tile for the result slot, and stores what
comes back. `--dry-run` compiles that kernel to Tile IR bytecode without
tileiras (`return_cubin=False`) and checks that the program really calls
`relu`: the caller half of the contract, produced on this machine. The full
run also needs the library half, `tilelib_relu.tilelib`, which only a CUDA
13.5 tileiras can build (`cargo oxide tilelib tilelib_relu --arch sm_120`),
and a cuTile Python whose supported bytecode versions include 13.5.

Checked against cuTile Python 1.7 (checkout 770b493) with the
`cuda-tile-preview` package from that checkout's experimental/tile_preview
folder; `cuda.tile_preview` is not on PyPI. The dry run reaches into private
`cuda.tile._*` modules (`_bytecode`, `_cext`, `_compile`), which can move
between releases.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import cuda.tile as ct
from cuda import tile_preview as tp

TILE = 128
LIBRARY_VERSION = "0.1.0"
HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_TILELIB = HERE.parent / "tilelib_relu.tilelib"
# preview$call in the bytecode opcode table; upstream's test_foreign_call.py
# looks for the same varint-encoded value.
PREVIEW_CALL_OPCODE = 0x4002

# Set by main() before the kernel is traced. foreign_call reads the library
# as a compile-time constant, so it has to be a module global, not an argument.
PROVIDER: tp.Tilelib | None = None


def relu_tile(x, y):
    """One block runs `relu` over one 128-float tile of `x` into `y`."""
    pid = ct.bid(0)
    tile = ct.load(x, index=(pid,), shape=(TILE,))
    # Every declaration argument needs an input tile, the `out` slot included.
    result_slot = ct.zeros((TILE,), dtype=ct.float32)
    (out,) = tp.foreign_call(
        tilelibs=(PROVIDER,),
        symbol_name="relu",
        constant_params=(),
        inputs=(tile, result_slot),
        output_types=((ct.float32, (TILE,)),),
    )
    ct.store(y, index=(pid,), tile=out)


def dry_run() -> None:
    """Compile the kernel to 13.5 bytecode and show the preview$call to `relu` is in it."""
    from cuda.tile._bytecode.basic import encode_varint
    from cuda.tile._bytecode.version import BytecodeVersion
    from cuda.tile._cext import CallingConvention
    from cuda.tile._compile import compile_tile
    from cuda.tile.compilation import ArrayConstraint, KernelSignature

    constraint = ArrayConstraint(
        dtype=ct.float32,
        ndim=1,
        index_dtype=ct.int32,
        stride_lower_bound_incl=0,
        alias_groups=(),
        may_alias_internally=False,
    )
    signature = KernelSignature(
        [constraint, constraint], CallingConvention.cutile_python_v2(), symbol="relu_tile"
    )
    result = compile_tile(
        relu_tile,
        [signature],
        bytecode_version=BytecodeVersion.V_13_5,
        return_final_ir=True,
        return_bytecode=True,
        return_cubin=False,
    )
    bytecode = bytes(result.bytecode)
    major, minor = bytecode[8], bytecode[9]
    preview = int.from_bytes(bytecode[10:12], "little") & 1
    opcode = bytearray()
    encode_varint(PREVIEW_CALL_OPCODE, opcode)
    has_call_op = bytes(opcode) in bytecode
    # The callee's name is read from the final IR, not from the bytecode's
    # string table: the kernel symbol `relu_tile` contains `relu`, so a bare
    # search of the bytes would pass without any call.
    ir_text = "\n".join(str(block) for block in result.final_ir)
    calls_relu = "preview_foreign_call" in ir_text and 'symbol_name="relu"' in ir_text
    print(f"bytecode: {len(bytecode)} bytes, header {bytecode[8:12].hex()} "
          f"(version {major}.{minor}, preview flag {'set' if preview else 'clear'}), "
          f"preview$call opcode {'present' if has_call_op else 'absent'}; "
          f"final IR {'calls' if calls_relu else 'does not call'} `relu`")
    if (major, minor) != (13, 5) or not preview or not has_call_op or not calls_relu:
        print("FAIL: the program does not carry a 13.5 preview call to `relu`")
        sys.exit(1)
    print("PASS: the caller side compiles; the program calls `relu` through "
          "preview$call (dry run, no cubin)")


def full_run() -> None:
    """Link the library with tileiras, run on the GPU, compare with numpy."""
    import cupy as cp
    from cuda.tile import _compile as compile_module
    from cuda.tile._bytecode.version import BytecodeVersion

    # A launched kernel is compiled at the newest bytecode version this cuTile
    # Python lists and the installed tileiras accepts. The 1.7 checkout lists
    # versions above 13.4 only when its extension is built with development
    # features, so every version its encoder knows is listed here; the probe
    # against tileiras still decides which one is used. foreign_call needs
    # 13.5, and an assembler reads preview content only at the version it
    # calls current, so this cuTile Python must encode that version too. A
    # release that lists these versions itself needs no such lines.
    for version in BytecodeVersion:
        if (version > BytecodeVersion.V_13_4
                and version not in compile_module._SUPPORTED_VERSIONS):
            compile_module._SUPPORTED_VERSIONS.append(version)

    relu_kernel = ct.kernel(relu_tile)
    n = 32 * TILE
    x = cp.linspace(-4.0, 4.0, n, dtype=cp.float32)
    y = cp.zeros_like(x)
    try:
        ct.launch(cp.cuda.get_current_stream(), (n // TILE, 1, 1), relu_kernel, (x, y))
    except Exception as error:  # noqa: BLE001  (every failure here is reported, not hidden)
        explain(error)
        sys.exit(1)
    expected = np.maximum(cp.asnumpy(x), 0.0)
    got = cp.asnumpy(y)
    np.testing.assert_array_equal(got, expected)
    print(f"PASS: relu from the tile library matches numpy on {n} elements")


def explain(error: Exception) -> None:
    """Report the failure and, when it is the toolkit, say so plainly."""
    print(f"FAIL: {type(error).__name__}: {error}")
    versions = tileiras_versions()
    if versions and "13.5" not in versions:
        print(f"The installed tileiras lists bytecode versions {', '.join(versions)}. "
              "foreign_call needs 13.5, which ships with CUDA 13.5 or newer; until then "
              "only --dry-run can run on this machine.")
    else:
        print("A 13.5 tileiras is also only used when this cuTile Python lists 13.5 "
              "among its supported bytecode versions.")


def tileiras_versions() -> list[str]:
    """The bytecode versions the tileiras on this machine accepts, or []."""
    candidates = [shutil.which("tileiras")]
    for root in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"):
        if root:
            candidates.append(os.path.join(root, "bin", "tileiras"))
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            output = subprocess.run([path, "--list-versions"], capture_output=True,
                                    text=True, check=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        return [line.strip() for line in output.splitlines() if line.strip()]
    return []


def main() -> None:
    global PROVIDER
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tilelib", nargs="?", default=DEFAULT_TILELIB, type=pathlib.Path,
                        help="the tile library (default: ../tilelib_relu.tilelib)")
    parser.add_argument("--dry-run", action="store_true",
                        help="compile to bytecode only; no tileiras, no GPU")
    args = parser.parse_args()

    if args.dry_run and not args.tilelib.is_file():
        # Tilelib() insists that the file exists; the bytecode only records
        # its path, so a placeholder stands in for the real library.
        with tempfile.TemporaryDirectory() as scratch:
            placeholder = pathlib.Path(scratch) / args.tilelib.name
            placeholder.write_bytes(b"placeholder")
            print(f"{args.tilelib} is not built yet; using a placeholder for the dry run")
            PROVIDER = tp.Tilelib(path=placeholder, version=LIBRARY_VERSION)
            dry_run()
        return

    if not args.tilelib.is_file():
        print(f"FAIL: {args.tilelib} does not exist; build it with "
              "`cargo oxide tilelib tilelib_relu --arch sm_120` (CUDA 13.5 tileiras)")
        sys.exit(1)
    PROVIDER = tp.Tilelib(path=args.tilelib, version=LIBRARY_VERSION)
    if args.dry_run:
        dry_run()
    else:
        full_run()


if __name__ == "__main__":
    main()
