# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invoke an external reference checkout; no reference implementation is vendored."""

import importlib
import pathlib
import sys
import types

source, operation, version, option = sys.argv[1:]
root = pathlib.Path(source) / "src" / "cuda" / "tile" / "_bytecode"
for name, path in [("cuda", root.parent.parent), ("cuda.tile", root.parent),
                   ("cuda.tile._bytecode", root)]:
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package

encoding = importlib.import_module("cuda.tile._bytecode.encodings")
builder = importlib.import_module("cuda.tile._bytecode.code_builder")
versions = importlib.import_module("cuda.tile._bytecode.version")
version = getattr(versions.BytecodeVersion, "V_" + version.replace(".", "_"))
cb = builder.CodeBuilder(bytearray(), version, None, None, [])
value = builder.Value
type_id = encoding.TypeId
if operation in ("Pointer", "TensorView", "F8E5M3FNU"):
    type_module = importlib.import_module("cuda.tile._bytecode.type")
    table = type_module.TypeTable(version)
    attr = type_module.PtrAttr.Default if option == "true" else type_module.PtrAttr.Missing
    if operation == "Pointer":
        result = table.pointer(type_id(0), attr)
    elif operation == "TensorView":
        result = table.tensor_view(type_id(0), [4], [1], attr)
    else:
        result = table.simple(type_module.SimpleType.FNV8E5M3FNU)
    print(next(data for data, ident in table.items() if ident == result).hex())
    sys.exit(0)
function = getattr(encoding, "encode_" + operation + "Op")
if operation in ("GdcLaunchDependentsTko", "GdcWaitTko"):
    function(cb, type_id(0), value(0) if option == "true" else None)
elif operation == "Insert":
    function(cb, type_id(1), value(0), value(1), [value(2)])
elif operation == "FPowI":
    function(cb, type_id(1), value(0), value(1))
elif operation == "MemoryFenceAliasTko":
    function(cb, type_id(0), value(0))
elif operation == "FToI":
    rounding = next(r for r in encoding.RoundingMode if r.value == b"\x06")
    function(cb, type_id(1), value(0), encoding.Signedness.Unsigned,
             rounding, option == "true")
elif operation == "LoadViewTko":
    weak = next(r for r in encoding.MemoryOrderingSemantics if r.value == b"\x00")
    function(cb, type_id(1), type_id(2), value(0), [value(1)], value(2),
             weak, None, None, [option == "true"])
elif operation == "StoreViewTko":
    weak = next(r for r in encoding.MemoryOrderingSemantics if r.value == b"\x00")
    function(cb, type_id(0), value(0), value(1), [value(2)], value(3),
             weak, None, None, [option == "true"])
elif operation == "Atan2":
    function(cb, type_id(1), value(0), value(1))
else:
    raise ValueError(operation)
# Rust's helper writes the body; compare the opcode separately in its test.
print(cb.buf[1:].hex())
