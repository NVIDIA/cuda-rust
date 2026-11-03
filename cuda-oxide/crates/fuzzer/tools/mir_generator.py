#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Generate one small rustlantis custom-MIR case for cuda-oxide.

This is a deliberately small Stage 2 adapter. It does not try to support the
full rustlantis output space. It generates one program built from scalars and
composites, extracts the first generated custom-MIR function, and rewrites
rustlantis' `dump_var(...)` terminators into calls to the cuda-oxide harness'
generic `dump_var`.

Composites reach the device, and the trace boundary now takes them: a dump site
or return position holding a tuple or an array folds as its leaves. What is
still refused there is a shape with no leaf reading, such as a reference, a
slice, or a tuple wider than the arity the trace API implements. An aggregate
in an argument position is refused separately, by `literal_for_type`, which has
no literal to construct for one.

By default it emits a complete `generated_case.rs` module for the
`rustlantis-smoke` example: imports, adapted MIR function, deterministic call
arguments, and a `compute_rustlantis_trace()` wrapper.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_RUSTLANTIS_DIR = (Path(__file__).resolve().parent.parent / "rustlantis").resolve()

TINY_CONFIG = """\
bb_max_len = 8
max_switch_targets = 2
max_bb_count = 3
max_bb_count_hard = 6
max_fn_count = 1
max_args_count = 3
var_dump_chance = 1.0
tuple_max_len = 2
array_max_len = 2
struct_max_fields = 2
adt_max_variants = 2
composite_count = 3
adt_count = 0

[backends.llvm]
type = "llvm"
toolchain = "nightly"
flags = ["-Zmir-opt-level=0"]
"""


def run(cmd: list[str], *, cwd: Path) -> str:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    return proc.stdout


def generate_source(rustlantis_dir: Path, seed: int, *, build: bool) -> str:
    if build:
        run(["cargo", "build", "-q", "-p", "generate"], cwd=rustlantis_dir)

    generator = rustlantis_dir / "target" / "debug" / "generate"
    if not generator.exists():
        raise SystemExit(f"generator not found: {generator}")

    with tempfile.TemporaryDirectory(prefix="rustlantis-cuda-oxide-") as tmp:
        tmpdir = Path(tmp)
        (tmpdir / "config.toml").write_text(TINY_CONFIG)
        return run([str(generator), str(seed)], cwd=tmpdir)


def extract_first_custom_mir_fn(source: str) -> str:
    start = source.find("#[custom_mir")
    if start < 0:
        raise SystemExit("no #[custom_mir] function found")

    fn_pos = source.find("pub fn ", start)
    if fn_pos < 0:
        raise SystemExit("custom MIR function header not found")

    body_start = source.find("{", fn_pos)
    if body_start < 0:
        raise SystemExit("custom MIR function body not found")

    depth = 0
    for idx in range(body_start, len(source)):
        ch = source[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start : idx + 1]

    raise SystemExit("unterminated custom MIR function")


def split_args(args: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(args):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(args[start:idx].strip())
            start = idx + 1
    last = args[start:].strip()
    if last:
        parts.append(last)
    return parts


def split_type_at_semicolon(text: str) -> tuple[str, int] | None:
    """Split a Rust type off the front of `text`, up to its terminating `;`.

    Returns the type and the offset just past that semicolon, or `None` when
    `text` holds no terminator.

    An array type carries a semicolon inside its brackets (`[u128; 1]`), so
    the terminator is the first `;` at bracket depth zero rather than the
    first `;` at all. Depth is tracked over `(` and `[` exactly as
    `split_args` does, and for the same reason.
    """
    depth = 0
    for idx, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == ";" and depth == 0:
            return text[:idx].strip(), idx + 1
    return None


def normalize_dump_arg(arg: str) -> str:
    arg = arg.strip()
    for wrapper in ("Move", "Copy"):
        prefix = f"{wrapper}("
        if arg.startswith(prefix) and arg.endswith(")"):
            return arg[len(prefix) : -1].strip()
    return arg


def collect_types(fn_src: str) -> dict[str, str]:
    header = re.search(r"pub fn\s+\w+\((?P<args>.*?)\)\s*->", fn_src, re.S)
    if not header:
        raise SystemExit("function header parse failed")

    types: dict[str, str] = {}
    for arg in split_args(header.group("args")):
        match = re.match(r"(?:mut\s+)?(?P<name>_\d+)\s*:\s*(?P<ty>[^,]+)$", arg.strip())
        if match:
            types[match.group("name")] = match.group("ty").strip()

    for match in re.finditer(r"let\s+(?P<name>_\d+)\s*:\s*", fn_src):
        parsed = split_type_at_semicolon(fn_src[match.end() :])
        if parsed is not None:
            types[match.group("name")] = parsed[0]

    return types


def function_args(fn_src: str) -> list[tuple[str, str]]:
    header = re.search(r"pub fn\s+\w+\((?P<args>.*?)\)\s*->", fn_src, re.S)
    if not header:
        raise SystemExit("function header parse failed")

    args: list[tuple[str, str]] = []
    for arg in split_args(header.group("args")):
        match = re.match(r"(?:mut\s+)?(?P<name>_\d+)\s*:\s*(?P<ty>[^,]+)$", arg.strip())
        if not match:
            raise SystemExit(f"unsupported function argument syntax: {arg}")
        args.append((match.group("name"), match.group("ty").strip()))
    return args


def return_type(fn_src: str) -> str:
    match = re.search(r"pub fn\s+\w+\(.*?\)\s*->\s*(?P<ret>[^{]+){", fn_src, re.S)
    if not match:
        raise SystemExit("function return type parse failed")
    return match.group("ret").strip()


def dump_tuple(values: list[str]) -> str:
    moved = [f"Move({value})" for value in values]
    if len(moved) == 1:
        return f"({moved[0]},)"
    return f"({', '.join(moved)})"


def tuple_type(types: list[str]) -> str:
    if len(types) == 1:
        return f"({types[0]},)"
    return f"({', '.join(types)})"


def format_rust_block(src: str) -> str:
    lines: list[str] = []
    indent = 0
    for raw in src.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("}"):
            indent = max(indent - 1, 0)
        lines.append(f"{'    ' * indent}{line}")
        if line.endswith("{"):
            indent += 1
    return "\n".join(lines)


def literal_for_type(ty: str, idx: int) -> str:
    literals = {
        "bool": ["false", "true"],
        "i8": ["98_i8", "(-17_i8)", "42_i8"],
        "i16": ["1234_i16", "(-567_i16)", "42_i16"],
        "i32": ["10_i32", "(-20_i32)", "42_i32"],
        "i64": ["10_i64", "(-20_i64)", "42_i64"],
        "i128": ["10_i128", "(-20_i128)", "42_i128"],
        "isize": ["10_isize", "(-20_isize)", "42_isize"],
        "u8": ["98_u8", "17_u8", "42_u8"],
        "u16": ["1234_u16", "567_u16", "42_u16"],
        "u32": ["10_u32", "20_u32", "42_u32"],
        "u64": ["10_u64", "20_u64", "42_u64"],
        "u128": ["10_u128", "20_u128", "42_u128"],
        "usize": ["10_usize", "20_usize", "42_usize"],
        "char": ["'a'", "'\\u{3a9}'", "'\\u{1f980}'"],
        # Exactly representable in binary floating point, so the literal a
        # seed is given is the value both backends start from.
        "f32": ["1.5_f32", "(-0.25_f32)", "42.0_f32"],
        "f64": ["1.5_f64", "(-0.25_f64)", "42.0_f64"],
    }
    if ty not in literals:
        raise SystemExit(f"unsupported function argument type for Stage 2 adapter: {ty}")
    values = literals[ty]
    return values[idx % len(values)]


SCALAR_TRACE_TYPES = frozenset(
    {
        "bool",
        "i8",
        "i16",
        "i32",
        "i64",
        "i128",
        "isize",
        "u8",
        "u16",
        "u32",
        "u64",
        "u128",
        "usize",
        "char",
        "f32",
        "f64",
    }
)

# `TraceValue` is implemented for tuples up to arity 5, matching `TraceDump`.
MAX_TRACE_TUPLE_ARITY = 5


def supported_trace_type(ty: str) -> bool:
    """Report whether the harness can fold a value of this type into the trace.

    A scalar folds directly. An aggregate folds as its leaves, so it is
    supported exactly when every leaf is, which makes this a recursive
    question rather than a set membership one.
    """
    ty = ty.strip()
    if ty in SCALAR_TRACE_TYPES:
        return True

    if ty.startswith("[") and ty.endswith("]"):
        parsed = split_type_at_semicolon(ty[1:-1])
        if parsed is None:
            return False
        return supported_trace_type(parsed[0])

    if ty.startswith("(") and ty.endswith(")"):
        fields = split_args(ty[1:-1])
        if len(fields) > MAX_TRACE_TUPLE_ARITY:
            return False
        return all(supported_trace_type(field) for field in fields)

    return False


def adapt_function(fn_src: str, fn_name: str) -> str:
    types = collect_types(fn_src)
    dump_pattern = re.compile(
        r"Call\((?P<dest>[^=]+)=\s*dump_var\((?P<args>.*?)\),\s*ReturnTo\((?P<target>bb\d+)\),\s*UnwindUnreachable\(\)\)",
        re.S,
    )
    dump_matches = list(dump_pattern.finditer(fn_src))

    dump_locals: list[tuple[str, list[str]]] = []
    dump_idx = 0

    def rewrite_dump(match: re.Match[str]) -> str:
        nonlocal dump_idx
        raw_args = split_args(match.group("args"))
        values = [normalize_dump_arg(arg) for arg in raw_args]
        kept_values = [value for value in values if types.get(value) != "()"]
        kept_types = [types[value] for value in kept_values]
        for ty in kept_types:
            if not supported_trace_type(ty):
                raise SystemExit(f"unsupported dumped type for Stage 2 adapter: {ty}")

        if not kept_values:
            return f"Goto({match.group('target')})"

        local = f"__rl_dump{dump_idx}"
        dump_idx += 1
        dump_locals.append((local, kept_types))
        tuple_expr = dump_tuple(kept_values)
        return (
            f"{local} = {tuple_expr};\n"
            f"Call({match.group('dest').strip()} = dump_var(Move({local})), "
            f"ReturnTo({match.group('target')}), UnwindUnreachable())"
        )

    adapted = re.sub(
        r"#\[custom_mir\(dialect = \"runtime\", phase = \"initial\"\)\]",
        '#[custom_mir(dialect = "runtime", phase = "initial")]',
        fn_src,
        count=1,
    )
    adapted = re.sub(
        r"pub fn\s+\w+\((?P<args>.*?)\)\s*->\s*[^{]+{",
        lambda match: (
            f"fn {fn_name}({', '.join(split_args(match.group('args')))}) "
            f"-> {return_type(fn_src)} {{"
        ),
        adapted,
        count=1,
        flags=re.S,
    )
    adapted = dump_pattern.sub(rewrite_dump, adapted)
    if dump_locals:
        local_decls = "\n".join(
            f"        let {name}: {tuple_type(local_types)};" for name, local_types in dump_locals
        )
        # Insert after the whole `type RET = ..;` alias. Matching its type
        # with `[^;]+` would stop inside an array return type and leave the
        # remainder of that type stranded after the inserted declarations,
        # which rustc then reads as a statement.
        #
        # A missing anchor is an adapter failure, not something to skip:
        # silently dropping the declarations leaves the rewritten dump calls
        # referencing undeclared locals, and the resulting rustc error would
        # be misreported as a backend COMPILE_FAIL instead of
        # UNSUPPORTED [adapter].
        anchor = re.search(r"type RET\s*=\s*", adapted)
        if anchor is None:
            raise SystemExit("expected a `type RET = ..;` alias to anchor the dump-local declarations")
        parsed = split_type_at_semicolon(adapted[anchor.end() :])
        if parsed is None:
            raise SystemExit("expected a depth-0 `;` terminating the `type RET` alias")
        end = anchor.end() + parsed[1]
        adapted = f"{adapted[:end]}\n{local_decls}{adapted[end:]}"

    return format_rust_block(adapted)


def generated_module(fn_src: str, fn_name: str, seed: int) -> str:
    adapted = adapt_function(fn_src, fn_name)
    args = [literal_for_type(ty, idx) for idx, (_, ty) in enumerate(function_args(fn_src))]
    call_args = ", ".join(args)
    ret_ty = return_type(fn_src)
    has_dump_site = "dump_var(Move(__rl_dump" in adapted

    if has_dump_site or ret_ty == "()":
        trace_lines = [
            f"    let _ = {fn_name}({call_args});",
            "    trace_finish()",
        ]
    elif supported_trace_type(ret_ty):
        trace_lines = [
            f"    let result = {fn_name}({call_args});",
            "    dump_var((result,));",
            "    trace_finish()",
        ]
    else:
        raise SystemExit(f"unsupported return type for return-value tracing: {ret_ty}")

    return "\n".join(
        [
            "/*",
            " * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
            " * SPDX-License-Identifier: Apache-2.0",
            " */",
            "",
            "// AUTO-GENERATED by crates/fuzzer/tools/mir_generator.py.",
            f"// rustlantis seed: {seed}",
            "// Adapted dump calls update the fuzzer crate's global trace state.",
            "",
            "// Machine-generated MIR-shaped code is lint-hostile by design",
            "// (explicit casts, redundant temps); clippy findings carry no",
            "// signal here, and checked-in cases sit inside the example's",
            "// `cargo clippy -- -D warnings` CI gate.",
            "#![allow(unused_assignments, unused_parens, overflowing_literals)]",
            "#![allow(clippy::all)]",
            "",
            "use core::intrinsics::mir::*;",
            "use fuzzer::{dump_var, trace_finish, trace_reset};",
            "",
            adapted,
            "",
            "#[inline(never)]",
            "pub fn compute_rustlantis_trace() -> u64 {",
            "    trace_reset();",
            *trace_lines,
            "}",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--fn-name", default="fn1")
    parser.add_argument("--rustlantis-dir", type=Path, default=DEFAULT_RUSTLANTIS_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument(
        "--function-only",
        action="store_true",
        help="emit only the adapted custom-MIR function instead of a generated_case.rs module",
    )
    args = parser.parse_args()

    source = generate_source(args.rustlantis_dir, args.seed, build=not args.no_build)
    fn_src = extract_first_custom_mir_fn(source)
    adapted = (
        adapt_function(fn_src, args.fn_name)
        if args.function_only
        else generated_module(fn_src, args.fn_name, args.seed)
    )

    if args.output:
        args.output.write_text(adapted + "\n")
    else:
        print(adapted)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
