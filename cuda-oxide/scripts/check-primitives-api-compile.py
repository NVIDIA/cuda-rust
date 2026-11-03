#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build in-tree tools and validate compile-only Rust primitives against CUTLASS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import tomllib
from functools import cache
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "cutlass-primitives-mlir22"
ARCH = "sm_100a"
KERNELS = ["primitives_grid_constant", "primitives_lifecycle", "primitives_tma", "primitives_sync", "primitives_mma", "primitives_attention"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def probe_functions(text):
    """Read the exporter's generic syntax; follow CFG edges, not block order."""
    functions = {}
    for chunk in text.split('        "func.func"()')[1:]:
        name = re.search(r'sym_name = "([^"]+)"', chunk).group(1)
        functions[name] = chunk.split('\n        })', 1)[0]
    return functions


def instruction_path(function):
    # Block labels are whatever the printer assigns (Pliron names unnamed
    # blocks block<N>v<M>); only the CFG edges matter.
    # The probes are acyclic. Checked failure paths return early; the longest
    # path exercises their successful issue sequence. Never use textual order.
    pieces = re.split(r"^          \^([A-Za-z0-9_]+)(?:\([^\n]*\))?:\n", function, flags=re.M)
    blocks = dict(zip(pieces[1::2], pieces[2::2]))
    require(bool(blocks), "probe has no basic blocks")
    visiting = set()

    @cache
    def visit(block):
        require(block not in visiting, "unexpected loop in instruction probe")
        visiting.add(block)
        body = blocks[block]
        successors = re.findall(r"\^([A-Za-z0-9_]+)", body)
        suffix = max((visit(s) for s in successors), key=len, default=[])
        visiting.remove(block)
        return body.splitlines() + suffix

    return visit(next(iter(blocks)))


def unwrap_single_results(text):
    """The published printer writes a single result type as `-> (T)`; the checks
    below were written against `-> T`. Multi-result lists keep their parentheses."""
    return re.sub(r"\) -> \(([^(),]+)\)", r") -> \1", text)


def verify_attention_ir(lines):
    """Check the CTM facade's instruction contracts at the consumer boundary."""
    text = "\n".join(lines)
    definitions = {line.strip().split(" = ", 1)[0]: line for line in lines if " = " in line}
    constants = {
        name: (struct.unpack("!f", int(value, 16).to_bytes(4, "big"))[0]
               if kind == "f32" and value.startswith("0x") else float(value))
        for name, value, kind in re.findall(
            r'(%v\d+) = "arith.constant"\(\) <\{value = ([^ :]+) : (i32|i64|f32)', text)
    }

    def operands(line):
        args = re.search(r'"[^\"]+"\(([^)]*)\)', line).group(1)
        return args.split(", ") if args else []

    def constant_value(arg):
        if arg in constants:
            return constants[arg]
        # Carrier constructors retain arrays in the exported, unoptimized IR.
        # Follow each extract back to the constant stored at that array index.
        extract = definitions[arg]
        require('"llvm.extractvalue"' in extract, "expected a constant or carrier register")
        index = int(re.search(r"position = array<i64: (\d+)>", extract).group(1))
        load = definitions[operands(extract)[0]]
        require('"llvm.load"' in load, "expected carrier array load")
        pointer = operands(load)[0]
        initializers = []
        for line in lines:
            if '"llvm.store"' not in line:
                continue
            value, destination = operands(line)
            gep = definitions.get(destination, "")
            if '"llvm.getelementptr"' in gep:
                base, offset = operands(gep)
                if base == pointer and constants[offset] == index:
                    initializers.append(value)
        require(len(initializers) == 1, "carrier register must have one constant initializer")
        return constants[initializers[0]]

    def values(line):
        return [constant_value(arg) for arg in operands(line)]

    def one(fragment):
        matches = [line for line in lines if fragment in line]
        require(len(matches) == 1, f"expected exactly one attention instruction {fragment}")
        return matches[0]

    one('"nvvm.cp.async.commit.group"')
    waits = [line for line in lines if '"nvvm.cp.async.wait.group"' in line]
    require([int(re.search(r"n = (\d+) : i32", line).group(1)) for line in waits] == [0, 7],
            "cp.async wait boundary immediates changed")
    for name, value in [("llvm.nvvm.ex2.approx.f32", 1.25),
                        ("llvm.nvvm.ex2.approx.ftz.f32", 2.5),
                        ("llvm.nvvm.rcp.approx.ftz.f", 4.0)]:
        line = one(f'intrin = "{name}"')
        require(values(line) == [value] and re.search(r"\(f32\) -> \(?f32\)?", line),
                f"attention math input or exact modifier changed: {name}")
    shuffle = one('intrin = "llvm.nvvm.shfl.sync.bfly.f32"')
    args = operands(shuffle)
    require(len(args) == 4 and int(constants[args[0]]) & 0xffffffff == 0xffffffff
            and values(shuffle)[1:] == [8.0, 2.0, 31.0],
            "shuffle must retain full-warp mask, XOR lane selector and clamp 31")
    pack = one('intrin = "llvm.nvvm.ff2f16x2.rn"')
    require(values(pack) == [2.5, 1.25] and '(f32, f32) -> vector<2xf16>' in pack,
            "FP16 packing must preserve round-nearest and LLVM high/low operand order")
    packed_result = pack.strip().split(" = ", 1)[0]
    require(any('"llvm.bitcast"' in line and operands(line) == [packed_result]
                and '(vector<2xf16>) -> i32' in line for line in lines),
            "packed half pair lost its raw u32 carrier")
    mma = one('"nvvm.mma.sync"')
    for attr in ["#nvvm.shape<m = 16, n = 8, k = 16>", "layoutA = #nvvm.mma_layout<row>",
                 "layoutB = #nvvm.mma_layout<col>", "multiplicandAPtxType = #nvvm.mma_type<f16>",
                 "multiplicandBPtxType = #nvvm.mma_type<f16>", "operandSegmentSizes = array<i32: 4, 2, 4>"]:
        require(attr in mma, f"FP16 MMA selector changed: {attr}")
    args = operands(mma)
    require(len(args) == 10, "FP16 MMA operand count changed")
    registers = []
    for arg in args[:6]:
        cast = definitions[arg]
        require('"llvm.bitcast"' in cast and '(i32) -> vector<2xf16>' in cast,
                "FP16 MMA packed register must be bitcast, never numerically converted")
        registers.append(int(constant_value(operands(cast)[0])))
    require(registers == [0x3c003c00, 0x40004000, 0x42004200, 0x44004400, 0x46004600, 0x48004800]
            and [constant_value(arg) for arg in args[6:]] == [1.0, 2.0, 3.0, 4.0],
            "FP16 MMA must preserve A[4], B[2], C[4] register order")
    aggregate = mma.strip().split(" = ", 1)[0]
    extracts = [line for line in lines if '"llvm.extractvalue"' in line and operands(line) == [aggregate]]
    require([int(re.search(r"position = array<i64: (\d+)>", line).group(1)) for line in extracts] == [0, 1, 2, 3],
            "FP16 MMA result register order changed")
    for index, extract in enumerate(extracts):
        result = extract.strip().split(" = ", 1)[0]
        inserts = [line for line in lines if '"llvm.insertvalue"' in line
                   and operands(line)[-1] == result]
        require(len(inserts) == 1 and f"position = array<i64: {index}>" in inserts[0],
                "FP16 MMA result adapter reordered the accumulator carrier")
    instructions = re.findall(r'"((?:nvvm|llvm\.nvvm)\.[^"]+)"', text)
    require(instructions == ["nvvm.cp.async.commit.group", "nvvm.cp.async.wait.group",
                             "nvvm.cp.async.wait.group", "llvm.nvvm.ex2.approx.f32",
                             "llvm.nvvm.ex2.approx.ftz.f32", "llvm.nvvm.rcp.approx.ftz.f",
                             "llvm.nvvm.shfl.sync.bfly.f32", "llvm.nvvm.ff2f16x2.rn", "nvvm.mma.sync"],
            "attention probe acquired hidden instructions or changed issue order")
    return instructions


def verify_ir(text, log, evidence):
    text = unwrap_single_results(text)
    require(not re.search(r'"cute\.|!cute\.', text), "primitives input contains CuTe operations/types")
    functions = probe_functions(text)
    require(set(KERNELS) <= functions.keys(), "missing primitive probe function")
    paths = {name: instruction_path(functions[name]) for name in KERNELS}
    attention = verify_attention_ir(paths["primitives_attention"])
    events = lambda lines: re.findall(r'"(nvvm\.[^"]+)"', "\n".join(lines))
    lifecycle = events(paths["primitives_lifecycle"])
    require(lifecycle == ["nvvm.tcgen05.alloc", "nvvm.tcgen05.relinquish_alloc_permit", "nvvm.tcgen05.dealloc", "nvvm.mbarrier.init", "nvvm.mbarrier.inval"], f"unexpected lifecycle instructions: {lifecycle}")
    mma = paths["primitives_mma"]
    issues = [line for line in mma if '"nvvm.tcgen05.mma"' in line]
    require(len(issues) == 8, "expected exactly eight MMA issues")
    require([re.search(r"collectorop<([^>]+)>", line).group(1) for line in issues] == ["fill", "use", "lastuse", "discard"] * 2, "collector sequence changed")
    require(events(mma) == ["nvvm.tcgen05.mma"] * 8 + ["nvvm.tcgen05.ld", "nvvm.tcgen05.wait"], "MMA probe acquired hidden instructions or lost explicit wait")
    constants = dict(re.findall(r'(%v\d+) = "arith.constant"\(\) <\{value = (-?\d+) : i\d+', text))
    operands = lambda line: re.search(r'"[^\"]+"\(([^)]*)\)', line).group(1).split(", ")
    require([int(constants[operands(line)[3]]) for line in issues] == [0x10300010] * 4 + [0x10280010] * 4, "MMA shape descriptors changed")
    extracts = [line for line in mma if '"vector.extract"' in line]
    require([int(re.search(r"static_position = array<i64: (\d+)>", line).group(1)) for line in extracts] == list(range(32)), "TMEM register result order changed")
    wait = next(i for i, line in enumerate(mma) if '"nvvm.tcgen05.wait"' in line)
    decode_name = "ctm_rs__tcgen05__mma__TmemRaw32__decode_f32"
    decode_calls = [i for i, line in enumerate(mma) if f"callee = @{decode_name}" in line]
    require(len(decode_calls) == 1 and decode_calls[0] > wait, "TMEM decode must follow the explicit wait")
    decode = functions[decode_name]
    require(decode.count('(i32) -> f32') == 32, "expected 32 raw-register bit interpretations")
    require(not any('"llvm.alloca"' in line and '32 x i32' in line for line in decode.splitlines()), "raw TMEM carrier escaped into a local allocation")
    tma = paths["primitives_tma"]
    loads = [line for line in tma if '"nvvm.cp.async.bulk.tensor.shared.cluster.global"' in line]
    require(len(loads) == 3, "expected three typed TMA loads")
    for line in loads:
        args = operands(line)
        require([int(constants[args[i]]) for i in [2, 3, 5]] == [-32, 4096, 3], "TMA coordinate/mask order changed")
    remote_arrivals = [line for line in tma if '"nvvm.mbarrier.arrive"' in line and 'cluster' in line]
    require(len(remote_arrivals) == 1, "expected one remote arrival")
    definitions = {line.strip().split(" = ", 1)[0]: line for line in tma if " = " in line}
    remote_pointer = definitions[operands(remote_arrivals[0])[0]]
    remote_offset = definitions[operands(remote_pointer)[0]]
    require('"llvm.ptrtoint"' in remote_offset and '(!llvm.ptr<7>)' in remote_offset,
            "remote barrier offset must come from the cluster window, never the CTA window")
    origin = definitions[operands(remote_offset)[0]]
    while '"llvm.addrspacecast"' in origin or '"builtin.unrealized_conversion_cast"' in origin:
        require('!llvm.ptr<3>' not in origin, "remote barrier address passed through the CTA window")
        origin = definitions[operands(origin)[0]]
    require('"nvvm.mapa"' in origin and int(constants[operands(origin)[1]]) == 1,
            "remote arrival must preserve the mapped rank-1 address")
    cluster_expect = [line for line in tma if '"nvvm.mbarrier.arrive.expect_tx"' in line and '#nvvm.mem_scope<cluster>' in line]
    cluster_wait = [line for line in tma if '"nvvm.mbarrier.wait.parity"' in line and '#nvvm.mbar_scope<cluster>' in line]
    require(len(cluster_expect) == 1 and 'relaxed = true' in cluster_expect[0], "missing relaxed cluster expect-tx")
    require(len(cluster_wait) == 1 and '#nvvm.mem_order<acquire>' in cluster_wait[0], "missing acquire cluster parity probe")
    sync = paths["primitives_sync"]
    helper_call = next(line for line in sync if 'callee = @primitives_api_compile__kernel__generic_shared_identity' in line)
    sync_definitions = {line.strip().split(" = ", 1)[0]: line for line in sync if " = " in line}
    argument = sync_definitions[operands(helper_call)[0]]
    require('"llvm.addrspacecast"' in argument and '(!llvm.ptr<3>) -> !llvm.ptr' in argument,
            "ordinary Rust helper call lost shared-to-generic address conversion")
    require(any('sym_name = "__cuda_oxide_dynamic_shared"' in line and 'alignment = 1024 : i64' in line
                for line in text.splitlines()), "dynamic shared allocation lost requested alignment")
    tma_events = events(tma)
    require(tma_events.count("nvvm.cp.async.bulk.commit.group") == 1 and tma_events.count("nvvm.cp.async.bulk.wait_group") == 2, "TMA commit/wait count changed")
    for name, count in [("primitives_grid_constant", 1), ("primitives_tma", 4)]:
        signature = functions[name].splitlines()[0]
        for attr in ["llvm.byval = !llvm.array<128 x i8>", "llvm.align = 64 : i64", "nvvm.grid_constant"]:
            require(signature.count(attr) == count, f"incorrect tensor-map ABI in {name}: {attr}")
    (evidence / "rust-mir.log").write_text(log)
    before = log.split("=== dialect-mir module (pre-verify) ===", 1)[1].split("=== Verifying dialect-mir module ===", 1)[0]
    after = log.split("=== dialect-mir module (after preparation) ===", 1)[1].split("=== Running official CUTLASS compiler", 1)[0]
    pairs = lambda section: Counter(re.findall(r'(nvvm\.\w+) [^\n]*cuda_oxide_intrinsic_marker: builtin.string "(v1:i\d+)"', section))
    require(pairs(before) == pairs(after), "preparation changed imported primitive identities or issue counts")
    counts = Counter()
    for (_, marker), count in pairs(after).items():
        counts[marker] += count
    for marker, count in {"v1:i0359": 1, "v1:i0360": 1, "v1:i0763": 8, "v1:i0664": 1, "v1:i0357": 1, "v1:i0331": 3, "v1:i0336": 1, "v1:i0308": 1, "v1:i0100": 1, "v1:i0342": 2, "v1:i0367": 1, "v1:i0887": 1,
                          "v1:i0094": 1, "v1:i0096": 2, "v1:i0819": 1, "v1:i0820": 1,
                          "v1:i0788": 1, "v1:i0055": 1, "v1:i0081": 1, "v1:i0106": 1}.items():
        require(counts[marker] == count, f"incorrect actual Rust importer count for {marker}: {counts[marker]}")
    for section in [before, after]:
        line = next(line for line in section.splitlines() if '"v1:i0664"' in line)
        results = line.split(" = ", 1)[0].strip()
        require(len(results.split(", ")) == 32 and line.split(" -> ", 1)[1].count("builtin.integer ui32") == 32, "raw TMEM result widths/count changed")
        require(f"mir.construct_array ({results})" in section, "Rust TMEM result adapter reordered registers")
    require("cuda_oxide_intrinsic_marker" not in text, "source markers leaked into consumer IR")
    checks = {name: events(lines) for name, lines in paths.items()}
    checks["primitives_attention"] = attention
    return checks


def verify_gemm_ir(text, log, evidence, name):
    text = unwrap_single_results(text)
    require(not re.search(r'"cute(?:_nvgpu)?\.|!cute(?:_nvgpu)?\.', text),
            "primitives GEMM contains CuTe operations/types")
    require('"nvvm.trap"' not in text and '"llvm.intr.trap"' in text,
            "checked construction must lower to the registered trap intrinsic")
    globals_ = [line for line in text.splitlines() if '"llvm.mlir.global"' in line]
    require(len(globals_) == 1 and 'alignment = 1024 : i64' in globals_[0]
            and 'addr_space = 3 : i32' in globals_[0] and '!llvm.array<0 x i8>' in globals_[0],
            "GEMM must use one 1024-byte-aligned dynamic shared allocation")
    functions = probe_functions(text)
    kernel = "blockscale_gemm" if name == "blockscale_gemm_ctm" else "fp16_gemm_256x352"
    require(kernel in functions, "missing GEMM kernel")
    maps = 5 if kernel == "blockscale_gemm" else 4
    signature = functions[kernel].splitlines()[0]
    for attr in ["llvm.byval = !llvm.array<128 x i8>", "llvm.align = 64 : i64", "nvvm.grid_constant"]:
        require(signature.count(attr) == maps, f"incorrect GEMM tensor-map ABI: {attr}")
    instructions = Counter(re.findall(r'"((?:nvvm|llvm\.intr)\.[^"]+)"', text))
    if kernel == "blockscale_gemm":
        for instruction in ["nvvm.mma.block_scale", "nvvm.ldmatrix", "nvvm.stmatrix",
                            "nvvm.cp.async.bulk.tensor.shared.cluster.global",
                            "nvvm.cp.async.bulk.tensor.global.shared.cta"]:
            require(instructions[instruction] > 0, f"missing SM120 operation {instruction}")
        require("#nvvm.block_scale_kind<mxf4>" in text and "#nvvm.scale_vec_size<x2>" in text,
                "expected dense K64 MXFP4 with two UE8M0 scales")
        require('"nvvm.tcgen05.' not in text, "SM120 GEMM must use register MMA")
    else:
        require(instructions["nvvm.tcgen05.mma"] == 8, "expected four fill/last-use MMA pairs")
        issues = [line for line in text.splitlines() if '"nvvm.tcgen05.mma"' in line]
        require([re.search(r"collectorop<([^>]+)>", line).group(1) for line in issues]
                == ["fill", "lastuse"]*4, "collector pairing changed")
        require(instructions["nvvm.tcgen05.ld"] == 1 and instructions["nvvm.tcgen05.wait"] == 1,
                "expected one explicit TMEM load and wait site")
    require("cuda_oxide_intrinsic_marker" not in text, "source markers leaked into consumer IR")
    (evidence / "rust-mir.log").write_text(log)
    return dict(instructions)


def sha256(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def run(command, env, log):
    print(f"Running {' '.join(map(str, command))}\n  log: {log}", flush=True)
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"command exited {result.returncode}: {log}\n{result.stderr[-6000:]}")
    return result.stdout


def artifact(output, name, suffix=None):
    matches = []
    for line in output.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("reason") != "compiler-artifact" or item["target"]["name"] != name:
            continue
        if suffix:
            matches.extend(Path(f) for f in item["filenames"] if f.endswith(suffix))
        elif item.get("executable"):
            matches.append(Path(item["executable"]))
    if len(matches) != 1 or not matches[0].is_file():
        raise RuntimeError(f"expected one source-built artifact for {name}, found {matches}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tools-dir", type=Path, help="isolated cache for source-built tools")
    parser.add_argument("--example", choices=["api", "fp16", "blockscale"], default="api",
                        help="compile probes or a complete primitives GEMM")
    args = parser.parse_args()
    fixture, target, manifest_path = {
        "api": ("primitives_api_compile", "sm_100a",
                "crates/rustc-codegen-cuda/tests/fixtures/primitives_api_compile/Cargo.toml"),
        "fp16": ("fp16_gemm_256x352_ctm", "sm_100a",
                 "cute-rs-preview/examples/fp16_gemm_256x352_ctm/Cargo.toml"),
        "blockscale": ("blockscale_gemm_ctm", "sm_120a",
                       "cute-rs-preview/examples/blockscale_gemm_ctm/Cargo.toml"),
    }[args.example]
    evidence = args.output_dir.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    tools = (args.tools_dir or evidence / "tools").resolve()
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=evidence))
    env = os.environ.copy()
    channel = tomllib.loads((ROOT / "rust-toolchain.toml").read_text())["toolchain"]["channel"]
    env.update(RUSTUP_TOOLCHAIN=channel, RUSTC_BOOTSTRAP="1", CARGO_ENCODED_RUSTFLAGS="", RUSTFLAGS="")
    sysroot = subprocess.check_output(["rustc", "--print", "sysroot"], cwd=ROOT, env=env, text=True).strip()
    rustc_version = subprocess.check_output(["rustc", "-vV"], cwd=ROOT, env=env, text=True)
    host = re.search(r"^host: (.+)$", rustc_version, re.M).group(1)
    for variable in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
        env[variable] = str(Path(sysroot) / "lib") + (":" + env[variable] if env.get(variable) else "")
    common = ["cargo", "build", "--offline", "--target", host, "--message-format=json-render-diagnostics"]
    cli = artifact(run(common + ["-p", "cargo-oxide", "--target-dir", str(tools / "cli")], env, run_dir / "cli.log"), "cargo-oxide")
    backend = artifact(run(common + ["--manifest-path", "crates/rustc-codegen-cuda/Cargo.toml", "--lib", "--target-dir", str(tools / "backend")], env, run_dir / "backend.log"), "rustc_codegen_cuda", ".so")
    compiler = Path(env.get("CUDA_OXIDE_CUTLASS_COMPILER", str(Path.home() / ".cargo/cuda-oxide/toolchains/cutlass/4.7.0/x86_64-cu13/lib/libCutlassCompiler.so")))
    if not compiler.is_file():
        raise RuntimeError(f"pinned CUTLASS compiler not installed: {compiler}")
    mlir = evidence / "primitives.mlir"
    cubin = evidence / f"{fixture}.cubin"
    for path in (mlir, cubin, evidence / "manifest.json"):
        path.unlink(missing_ok=True)
    env.update(CUDA_OXIDE_BACKEND=str(backend), CUDA_OXIDE_CUTLASS_COMPILER=str(compiler),
               CUDA_OXIDE_DEVICE_BACKEND="cutlass-mlir", CUDA_OXIDE_MLIR_PROFILE=PROFILE,
               CUDA_OXIDE_DUMP_MIR="1", CUDA_OXIDE_VERBOSE="1",
               CUDA_OXIDE_PTX_DIR=str(evidence), CUDA_OXIDE_MLIR_OUTPUT=str(mlir))
    command = [str(cli), "build", "--arch", target, "--cargo-target-dir", str(run_dir / "target"),
               "--device-codegen-crate", fixture, "--", "--manifest-path",
               manifest_path, "--release", "--offline"]
    run(command, env, run_dir / "fixture.log")
    if not mlir.is_file() or not cubin.is_file() or cubin.stat().st_size == 0:
        raise RuntimeError(f"missing fresh MLIR/cubin evidence in {evidence}")
    checks = (verify_ir(mlir.read_text(), (run_dir / "fixture.log").read_text(), evidence)
              if args.example == "api" else
              verify_gemm_ir(mlir.read_text(), (run_dir / "fixture.log").read_text(), evidence, fixture))
    manifest = {
        "profile": PROFILE, "target": target, "fixture": fixture, "command": command, "rustc": rustc_version,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "backend": str(backend), "backend_sha256": sha256(backend),
        "compiler": str(compiler), "compiler_sha256": sha256(compiler),
        "mlir_sha256": sha256(mlir), "cubin_sha256": sha256(cubin), "run_directory": str(run_dir),
        "validation": "production compile_to_cubin: nonempty kernel code, exact target and parameter ABI",
        "gpu_execution": False,
        "instruction_paths": checks,
    }
    (evidence / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {PROFILE} / {target}\nMLIR: {mlir}\nCUDA image: {cubin}\nManifest: {evidence / 'manifest.json'}")


if __name__ == "__main__":
    main()
