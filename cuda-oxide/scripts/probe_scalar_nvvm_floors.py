# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 ssoj13
# SPDX-License-Identifier: Apache-2.0
# Adapted from the licensed scalar catalog and libnvvm-sys fixtures; retains their attribution.
from pathlib import Path
import ctypes as C
import os
import json
import re
import hashlib
import subprocess
import struct
import argparse

parser = argparse.ArgumentParser(
    description="Reproduce scalar arithmetic CUDA 13.3 libNVVM floor evidence"
)
parser.add_argument("--libnvvm", type=Path, required=True)
parser.add_argument("--nvjitlink", type=Path, required=True)
parser.add_argument("--ptxas", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
out = args.out
out.mkdir(parents=True, exist_ok=True)
nv_path = args.libnvvm
link_path = args.nvjitlink
directories = []
if hasattr(os, "add_dll_directory"):
    for directory in {nv_path.parent, link_path.parent}:
        directories.append(os.add_dll_directory(str(directory)))
nv = C.CDLL(str(nv_path))
link = C.CDLL(str(link_path))


def function(lib, name, args):
    f = getattr(lib, name)
    f.argtypes = args
    f.restype = C.c_int
    return f


ptr = C.c_void_p
size = C.c_size_t
uint = C.c_uint
chars = C.POINTER(C.c_char_p)
new = function(nv, "nvvmCreateProgram", [C.POINTER(ptr)])
destroy = function(nv, "nvvmDestroyProgram", [C.POINTER(ptr)])
add = function(nv, "nvvmAddModuleToProgram", [ptr, ptr, size, C.c_char_p])
compile_program = function(nv, "nvvmCompileProgram", [ptr, C.c_int, chars])
get_size = function(nv, "nvvmGetCompiledResultSize", [ptr, C.POINTER(size)])
get_result = function(nv, "nvvmGetCompiledResult", [ptr, ptr])
log_size = function(nv, "nvvmGetProgramLogSize", [ptr, C.POINTER(size)])
get_log = function(nv, "nvvmGetProgramLog", [ptr, ptr])
version = function(nv, "nvvmLLVMVersion", [C.c_char_p, C.POINTER(C.c_int)])
architectures = []
for arch in [70, 72, 75, 80]:
    v = C.c_int()
    result = version(f"compute_{arch}".encode(), C.byref(v))
    print("architecture", arch, "status", result, "LLVM", v.value, flush=True)
    architectures.append({"compute": arch, "status": result, "llvm": v.value})
    assert (result == 0) == (arch >= 75)
ir_version = function(nv, "nvvmIRVersion", [C.POINTER(C.c_int)] * 4)
ir_values = [C.c_int() for _ in range(4)]
assert ir_version(*map(C.byref, ir_values)) == 0
root = Path(__file__).resolve().parents[1]
records = [
    r
    for r in json.loads((root / "intrinsics/catalog.json").read_text(encoding="utf-8"))[
        "intrinsics"
    ]
    if r["family"] == "scalar_arithmetic"
]
assert len(records) == 64
parts = [
    "; SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n",
    "; SPDX-FileCopyrightText: Copyright (c) 2026 ssoj13\n",
    "; SPDX-License-Identifier: Apache-2.0\n",
    'target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"\n',
    'target triple = "nvptx64-nvidia-cuda"\n',
]
annotations = []
used = []
for i, r in enumerate(records):
    typ = {"f32": "float", "f64": "double"}[r["scalar_arithmetic"]["format"]]
    count = len(r["llvm"]["arguments"])
    register = "f" if typ == "float" else "d"
    params = ", ".join(
        [f"{typ} addrspace(1)* %out"] + [f"{typ} %a{j}" for j in range(count)]
    )
    opcode = (
        r["expected_ptx"]["mnemonic"] + "." + ".".join(r["expected_ptx"]["modifiers"])
    )
    asm = opcode + " " + ", ".join(f"${j}" for j in range(count + 1)) + ";"
    constraint = ",".join(["=" + register] + [register] * count)
    inputs = ", ".join(f"{typ} %a{j}" for j in range(count))
    parts.append(
        f'define void @probe_{r["id"]}({params}) {{\n %value = call {typ} asm "{asm}", "{constraint}"({inputs})\n store {typ} %value, {typ} addrspace(1)* %out\n ret void\n}}\n'
    )
    signature = ", ".join([f"{typ} addrspace(1)*"] + [typ] * count)
    annotations.append(
        f'!{i} = !{{void ({signature})* @probe_{r["id"]}, !"kernel", i32 1}}\n'
    )
    used.append(f"i8* bitcast (void ({signature})* @probe_{r['id']} to i8*)")
parts.append(
    "@llvm.used = appending global [64 x i8*] ["
    + ", ".join(used)
    + '], section "llvm.metadata"\n'
)
parts.append("!nvvm.annotations = !{" + ", ".join(f"!{i}" for i in range(64)) + "}\n")
parts += annotations
parts.append("!nvvmir.version = !{!64}\n!64 = !{i32 2, i32 0, i32 3, i32 2}\n")
source = "".join(parts).encode()
(out / "scalar.ll").write_bytes(source)


def nvvm_compile(options):
    p = ptr()
    assert new(C.byref(p)) == 0
    try:
        data = C.create_string_buffer(source)
        assert add(p, data, len(source), b"scalar.ll") == 0
        opts = (C.c_char_p * len(options))(*[o.encode() for o in options])
        status = compile_program(p, len(options), opts)
        n = size()
        log_size(p, C.byref(n))
        log = C.create_string_buffer(n.value or 1)
        get_log(p, log)
        if status:
            raise RuntimeError(
                f"nvvm status {status}: {log.value.decode(errors='replace')}"
            )
        assert get_size(p, C.byref(n)) == 0
        result = C.create_string_buffer(n.value)
        assert get_result(p, result) == 0
        return result.raw
    finally:
        destroy(C.byref(p))


ptx = nvvm_compile(["-arch=compute_75"])
(out / "scalar.ptx").write_bytes(ptx.rstrip(b"\0"))
text = ptx.decode(errors="replace")
for r in records:
    opcode = (
        r["expected_ptx"]["mnemonic"] + "." + ".".join(r["expected_ptx"]["modifiers"])
    )
    assert len(re.findall(r"\b" + re.escape(opcode) + r"\s", text)) == 1, r["id"]
ptxas = args.ptxas
subprocess.run(
    [
        str(ptxas),
        "-arch=sm_75",
        str(out / "scalar.ptx"),
        "-o",
        str(out / "scalar.cubin"),
    ],
    check=True,
)
lto = nvvm_compile(["-arch=compute_75", "-gen-lto"])
(out / "scalar.ltoir").write_bytes(lto)


def linker(name, args):
    return function(link, "__" + name + "_13_3", args)


link_new = linker("nvJitLinkCreate", [C.POINTER(ptr), uint, chars])
link_add = linker("nvJitLinkAddData", [ptr, C.c_int, ptr, size, C.c_char_p])
link_complete = linker("nvJitLinkComplete", [ptr])
linked_size = linker("nvJitLinkGetLinkedCubinSize", [ptr, C.POINTER(size)])
linked_result = linker("nvJitLinkGetLinkedCubin", [ptr, ptr])
link_destroy = linker("nvJitLinkDestroy", [C.POINTER(ptr)])
h = ptr()
options = (C.c_char_p * 2)(b"-arch=sm_75", b"-lto")
assert link_new(C.byref(h), 2, options) == 0
try:
    lto_buffer = C.create_string_buffer(lto)
    assert link_add(h, 3, lto_buffer, len(lto), b"scalar.ltoir") == 0
    status = link_complete(h)
    if status:
        ls = linker("nvJitLinkGetErrorLogSize", [ptr, C.POINTER(size)])
        lr = linker("nvJitLinkGetErrorLog", [ptr, ptr])
        n = size()
        ls(h, C.byref(n))
        buf = C.create_string_buffer(n.value or 1)
        lr(h, buf)
        raise RuntimeError(
            f"link status {status}: {buf.value.decode(errors='replace')}"
        )
    n = size()
    assert linked_size(h, C.byref(n)) == 0
    buf = C.create_string_buffer(n.value)
    assert linked_result(h, buf) == 0
    (out / "linked.cubin").write_bytes(buf.raw)
finally:
    link_destroy(C.byref(h))


def entries(file):
    data = file.read_bytes()
    assert data[:6] == b"\x7fELF\x02\x01"
    offset = struct.unpack_from("<Q", data, 40)[0]
    stride, count, string_index = struct.unpack_from("<HHH", data, 58)
    headers = [
        struct.unpack_from("<IIQQQQIIQQ", data, offset + i * stride)
        for i in range(count)
    ]
    string_header = headers[string_index]
    strings = data[string_header[4] : string_header[4] + string_header[5]]
    names = [strings[h[0] : strings.index(b"\0", h[0])].decode() for h in headers]
    return sorted(n[len(".text.") :] for n in names if n.startswith(".text.probe_"))


expected = sorted("probe_" + r["id"] for r in records)
assert entries(out / "scalar.cubin") == expected
assert entries(out / "linked.cubin") == expected
nv_version = function(nv, "nvvmVersion", [C.POINTER(C.c_int)] * 2)
nv_values = [C.c_int(), C.c_int()]
assert nv_version(*map(C.byref, nv_values)) == 0
link_version = function(link, "nvJitLinkVersion", [C.POINTER(uint)] * 2)
link_values = [uint(), uint()]
assert link_version(*map(C.byref, link_values)) == 0
proof = {
    "compute": "compute_75",
    "terminal": "sm_75",
    "count": 64,
    "runtime": "unexecuted",
    "architectures": architectures,
    "ir_version": [v.value for v in ir_values],
    "nvvm_api_version": [v.value for v in nv_values],
    "link_version": [v.value for v in link_values],
    "cubin_entries": expected,
    "artifacts": {
        p.name: {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
        for p in out.iterdir()
        if p.is_file() and p.name != "proof.json"
    },
    "tools": {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [nv_path, link_path, ptxas]
    },
}
(out / "proof.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
print("All 64 forms compiled, assembled, and LTO-linked for sm_75", flush=True)
