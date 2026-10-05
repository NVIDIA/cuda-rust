#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify the example classification assumptions and the full-debug route in
# scripts/smoketest.sh. Most classification failures are otherwise only
# discoverable by running the suite on a GPU:
#
#   1. Every name in a *_EXAMPLES array is a real example directory.  These
#      arrays drive classify(), so a typo or a renamed example does not fail
#      loudly -- the entry simply matches nothing and the example silently
#      falls through to the `standard` category and the wrong verdict rules.
#
#   2. No example is claimed by two category arrays.  classify() returns the
#      first category that matches, so a second claim is not a conflict it can
#      report -- it is silently ignored, taking the example's arch gate and
#      verdict rules with it.  Listing `wgmma` in TCGEN05_EXAMPLES, say, gates
#      a Hopper example on Blackwell and both existing guards still pass.
#
#   3. Every `standard` example can actually report success.  That category's
#      verdict requires a SUCCESS/PASS/Complete marker in the output, and an
#      example that verifies its results but never prints one is reported as
#      `FAIL (no success marker)`.  That is a false failure, and it has landed
#      before: 981b9eb0 had to add a marker to disjoint_from_raw_parts after
#      #670 (the new example) and #665 (stricter verdicts) merged in one batch.
#
#   4. The full-debug census stays on the direct LLVM build route, explicitly
#      requests device debug, runs permanent debug-info contracts, and cannot
#      accidentally inherit the optimized shape/libNVVM gates it was created
#      to separate from.
#
# None of these need a GPU, and none is reachable from the compile-only CI
# lane, which collapses every category into verdict_compile.
#
# Run this after adding an example or editing a *_EXAMPLES array.
set -euo pipefail

export LC_ALL=C

cd "$(dirname "$0")/.."

SMOKETEST=scripts/smoketest.sh
EXAMPLES_ROOT=crates/rustc-codegen-cuda/examples

if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required to verify the smoketest example contract" >&2
    echo "       refusing to report success from a check that cannot run" >&2
    exit 1
fi

test -s "${SMOKETEST}"

# The marker and skip patterns below duplicate verdict_standard's greps.  Pin
# that coupling: if the verdict changes its patterns, this guard is stale and
# must say so rather than keep checking the old contract.
if ! grep -Fq "'SUCCESS|PASS|Complete'" "${SMOKETEST}"; then
    echo "error: ${SMOKETEST} no longer greps 'SUCCESS|PASS|Complete' for success" >&2
    echo "       verdict_standard's contract changed; update this guard" >&2
    exit 1
fi
if ! grep -Fq "'^[[:space:]]*(skipping:|pass \\(skipped\\))'" "${SMOKETEST}"; then
    echo "error: ${SMOKETEST} no longer greps the skip declaration this guard expects" >&2
    echo "       verdict_standard's contract changed; update this guard" >&2
    exit 1
fi

python3 - "${SMOKETEST}" "${EXAMPLES_ROOT}" <<'PY'
import glob
import os
import re
import pathlib
import subprocess
import sys
import tempfile
import tomllib

smoketest, examples_root = sys.argv[1], sys.argv[2]
source = open(smoketest, encoding="utf-8").read()

# Exercise the actual CLI/preflight without building or launching an example.
# The Rust parser owns aliases and normalization; this checks that the shell
# trusts its token, honors explicit mode, and refuses a failed/malformed reply.
with tempfile.TemporaryDirectory(prefix="smoketest-debug-policy-") as temp:
    bin_dir = pathlib.Path(temp)
    cargo = bin_dir / "cargo"
    cargo.write_text('''#!/usr/bin/env bash
case "$*" in
    "oxide --help") exit 0 ;;
    "oxide __debug-policy")
        printf '%s\\n' "$SMOKETEST_TEST_POLICY"
        exit "$SMOKETEST_TEST_POLICY_STATUS" ;;
    *) exit 99 ;;
esac
''')
    cargo.chmod(0o755)
    smi = bin_dir / "nvidia-smi"
    smi.write_text("#!/usr/bin/env bash\nexit 1\n")
    smi.chmod(0o755)
    controls = [
        (token, 0, explicit, token != "full" or explicit)
        for token in ("unset", "none", "line-tables", "full", "unrecognized")
        for explicit in (False, True)
    ]
    controls += [
        (token, status, explicit, False)
        for token, status in (("", 0), ("unknown", 0), ("noise\nnone", 0),
                              ("none", 1), ("full", 1), ("", 127))
        for explicit in (False, True)
    ]
    for token, status, explicit, allowed in controls:
        env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                   CUDA_OXIDE_DEBUG="opaque-to-shell",
                   SMOKETEST_TEST_POLICY=token,
                   SMOKETEST_TEST_POLICY_STATUS=str(status),
                   SMOKETEST_LOG_DIR=str(bin_dir / "logs"))
        args = ["bash", smoketest, "--only", "^$", "--no-color"]
        if explicit:
            args.append("--full-debug")
        result = subprocess.run(args, env=env, capture_output=True, text=True)
        reached_selection = "no examples matched the given filters" in result.stderr
        if result.returncode != (1 if allowed else 2) or reached_selection != allowed:
            sys.exit(f"debug-policy preflight failed for {(token, status, explicit)!r}: "
                     f"{result.returncode}, {result.stdout!r}, {result.stderr!r}")

# The full-debug lane exists specifically to avoid conflating a debug-info
# census with the optimized/libNVVM compile-only matrix. Pin that routing
# statically: this guard is fast, needs no toolkit/GPU, and fails if a future
# refactor silently makes --full-debug an alias for --compile-only.
try:
    full_debug_body = source.split("run_full_debug_build() {", 1)[1].split(
        "\n}\n\n# Run cargo oxide", 1
    )[0]
    full_debug_verdict = source.split("verdict_compile() {", 1)[1].split(
        "\n}\n\n# Assert the artifact is full-debug PTX", 1
    )[0]
    full_debug_ptx = source.split("full_debug_ptx_verdict() {", 1)[1].split(
        "\n}\n\n# These direct-LLVM FFI examples", 1
    )[0]
    full_debug_ffi = source.split("full_debug_relocatable_ffi_verdict() {", 1)[1].split(
        "\n}\n\n# The one configured cubin project", 1
    )[0]
    full_debug_cubin = source.split("full_debug_cubin_verdict() {", 1)[1].split(
        "\n}\n\n# ---- Runner", 1
    )[0]
    run_cargo_prefix = source.split("run_cargo() {", 1)[1].split(
        "# Rust `char`", 1
    )[0]
    error_verdict = source.split("verdict_error() {", 1)[1].split(
        "\n}\n\nverdict_tcgen05()", 1
    )[0]
except IndexError:
    sys.exit("parse self-test failed: could not isolate full-debug policy functions")

full_debug_contract = {
    "uses cargo oxide build": 'local -a args=("build" "${ex}" "--device-debug")',
    "dispatches before compile-only special routes": (
        'if [[ ${FULL_DEBUG} -eq 1 ]]; then\n'
        '        run_full_debug_build "${ex}" "${log}" "${cat}"\n'
        '        return'
    ),
    "runs permanent debug-info contracts": 'bash "${debug_info_check}"',
    "runs the debug example's optimization-invariant line contract": (
        'if [[ ${CARGO_EC} -eq 0 && "${ex}" == "debug" ]]; then\n'
        '        if ! bash "${invariant_shape_check}"'
    ),
}
for requirement, needle in full_debug_contract.items():
    haystack = run_cargo_prefix if requirement.startswith("dispatches") else full_debug_body
    if needle not in haystack:
        sys.exit(f"full-debug route contract missing: {requirement}")
if 'local -a args=("emit-ltoir"' in full_debug_body:
    sys.exit("full-debug route contract violated: routed through emit-ltoir/libNVVM")
if 'bash "${shape_check}"' in full_debug_body:
    sys.exit("full-debug route contract violated: runs optimized code-shape checks")
for requirement in (
    '--full-debug)    FULL_DEBUG=1; shift;;',
    'if [[ ${COMPILE_ONLY} -eq 1 && ${FULL_DEBUG} -eq 1 ]]; then',
):
    if requirement not in source:
        sys.exit(f"full-debug CLI contract missing: {requirement}")
if 'local debug_info_check=' not in full_debug_body or source.count('local debug_info_check=') != 1:
    sys.exit("full-debug isolation contract violated: debug-info checks escaped their lane")
for requirement in (
    'full_debug_ptx_verdict "${ex_dir}/${artifact}.ptx" "direct LLVM PTX"',
    'device_ffi_test|mathdx_ffi_test|small_type_ffi_test)',
    'full_debug_relocatable_ffi_verdict',
    '"${ex_dir}/simt/cutile_inter_kernel_simt.ptx"',
    'full_debug_cubin_verdict "${ex_dir}/device/scale_offset_device"',
):
    if requirement not in full_debug_verdict:
        sys.exit(f"full-debug artifact verdict missing: {requirement}")
for requirement in (
    r"\.target[[:space:]]+.*,[[:space:]]*debug",
    r"\.debug_info",
    '[[ -z "${PTXAS_BIN}" ]]',
    'ptxas_verify "${ptx}"',
    '"ptxas gate skipped:"*',
):
    if requirement not in full_debug_ptx:
        sys.exit(f"full-debug PTX assertion missing: {requirement}")
for requirement in (
    'device_ffi_test)',
    'mathdx_ffi_test)',
    'small_type_ffi_test)',
    '"${PTXAS_BIN}" -arch="${arch}" -c',
    "'[.]debug_info'",
    "'[.]debug_line'",
    'DW_TAG_compile_unit',
    'actual_undefined',
    'expected_undefined',
):
    if requirement not in full_debug_ffi:
        sys.exit(f"full-debug relocatable FFI assertion missing: {requirement}")
for requirement in (
    'cuda-oxide-compile-options-v2',
    "'^debug=full$'",
    '[.]debug_info',
    '[.]debug_line',
    'DW_TAG_compile_unit',
    'scale_offset_f32',
):
    if requirement not in full_debug_cubin:
        sys.exit(f"full-debug cubin assertion missing: {requirement}")
zero_exit = 'if [[ ${ec} -eq 0 ]]; then'
generic_success = "if grep -qE 'Device codegen failed|Translation failed|Compilation error|Unsupported construct'"
if zero_exit not in error_verdict or error_verdict.index(zero_exit) > error_verdict.index(generic_success):
    sys.exit("error verdict contract violated: diagnostic text can pass with exit 0")
if not re.search(
    r'\[\[ \$\{FULL_DEBUG\} -eq 0 \]\] && verify_nvvm_in_compile_only', source
):
    sys.exit("full-debug verdict contract missing: libNVVM artifact branch is not excluded")
if "FULL_DEBUG_CONFIGURED_ROUTE_EXAMPLES=(interop_cubin_identity)" not in source:
    sys.exit("full-debug route contract missing: explicit cubin-project accommodation")
if "Full-debug configured route: %d / %d passed" not in source:
    sys.exit("full-debug route contract missing: configured-route summary")

# Same two patterns verdict_standard applies to the run log.
MARKER = re.compile(r"SUCCESS|PASS|Complete")
SKIP = re.compile(r"^[ \t]*(skipping:|pass \(skipped\))", re.IGNORECASE)

lists = re.findall(r"^([A-Z0-9_]+_EXAMPLES)=\(([^)]*)\)", source, re.M)
on_disk = sorted(
    os.path.basename(os.path.dirname(path))
    for path in glob.glob(os.path.join(examples_root, "*", "Cargo.toml"))
)

interop_configs = {}
for path in glob.glob(os.path.join(examples_root, "*", "Cargo.toml")):
    with open(path, "rb") as file:
        manifest = tomllib.load(file)
    cuda_oxide = manifest.get("package", {}).get("metadata", {}).get("cuda-oxide", {})
    device_crates = cuda_oxide.get("device-crates", [])
    if device_crates:
        interop_configs[os.path.basename(os.path.dirname(path))] = device_crates

expected_interop_configs = {
    "cutile_inter_kernel": [
        {"manifest-path": "simt/Cargo.toml", "ptx-dir": "simt"},
    ],
    "interop_cubin_identity": [
        {
            "manifest-path": "device/Cargo.toml",
            "artifact-dir": "device",
            "artifact-name": "scale_offset_device",
            "artifact-kind": "cubin",
            "source-identity": True,
            "bin": "scale-offset-device",
        },
    ],
}
if interop_configs != expected_interop_configs:
    sys.exit(f"full-debug interop artifact map is stale: {interop_configs!r}")

expected_nested_packages = {
    "cutile_inter_kernel": "cutile_inter_kernel_simt",
    "interop_cubin_identity": "interop-cubin-identity-kernels",
}
for project, entries in interop_configs.items():
    nested = os.path.join(examples_root, project, entries[0]["manifest-path"])
    with open(nested, "rb") as file:
        nested_name = tomllib.load(file).get("package", {}).get("name")
    if nested_name != expected_nested_packages[project]:
        sys.exit(
            f"full-debug interop package-name map is stale for {project}: {nested_name!r}"
        )

# Parse self-tests: a guard whose failure mode is "matched nothing" has to
# prove it still reads both inputs before a clean result is believed.
if len(lists) < 9:
    sys.exit(f"parse self-test failed: found {len(lists)} *_EXAMPLES arrays in {smoketest}")
if len(on_disk) < 100:
    sys.exit(f"parse self-test failed: found {len(on_disk)} examples under {examples_root}")

failures = []

known = set(on_disk)
for name, body in lists:
    phantom = [entry for entry in body.split() if entry not in known]
    if phantom:
        failures.append(
            f"{name} names {len(phantom)} example(s) that do not exist: " + " ".join(phantom)
        )

# classify() returns `standard` for anything not claimed by a category array.
# NVVM_VERIFY / NO_LAUNCH / NO_OPT_SHAPE are modifiers, not categories, so
# their members stay `standard` and still need a marker.
CATEGORY_LISTS = (
    "TCGEN05_EXAMPLES",
    "WGMMA_EXAMPLES",
    "BLACKWELL_MMA_EXAMPLES",
    "LTOIR_EXAMPLES",
    "LTOIR_MODERN_EXAMPLES",
    "AUTO_NVVM_EXAMPLES",
    "IKET_EXAMPLES",
    "BLACKWELL_COMPILE_EXAMPLES",
    "SM100_COMPILE_EXAMPLES",
    "ERROR_EXAMPLES",
)
by_name = dict(lists)
missing_arrays = [name for name in CATEGORY_LISTS if name not in by_name]
if missing_arrays:
    sys.exit(
        "parse self-test failed: classify() arrays not found in "
        f"{smoketest}: " + " ".join(missing_arrays)
    )

categorised = {
    entry for name in CATEGORY_LISTS for entry in by_name[name].split()
}

# classify() returns the *first* category that claims an example, so an entry
# in two category arrays silently takes whichever loop runs first, and with it
# the wrong compute-capability gate and the wrong verdict rules. Nothing else
# reports that: the phantom check above passes because both names are real
# examples, and the set comprehension here discards the duplicate by
# construction. Uniqueness has to be asserted against the raw lists.
claimed_by = {}
for name in CATEGORY_LISTS:
    for entry in by_name[name].split():
        claimed_by.setdefault(entry, []).append(name)
for entry, names in sorted(claimed_by.items()):
    if len(names) > 1:
        failures.append(
            f"{entry} is claimed by {len(names)} category arrays "
            f"({', '.join(names)}); classify() checks {names[0]} first "
            "and ignores the rest"
        )

unmarked = []
for example in on_disk:
    if example in categorised:
        continue
    pattern = os.path.join(examples_root, example, "src", "**", "*.rs")
    has_marker = False
    for path in glob.glob(pattern, recursive=True):
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.lstrip().startswith("//"):
                    continue
                if MARKER.search(line) and not SKIP.search(line):
                    has_marker = True
                    break
        if has_marker:
            break
    if not has_marker:
        unmarked.append(example)

if unmarked:
    failures.append(
        "these `standard` examples print no SUCCESS/PASS/Complete marker, so "
        "smoketest reports FAIL (no success marker):\n  "
        + "\n  ".join(unmarked)
    )

if failures:
    print("error: smoketest example contract violated", file=sys.stderr)
    for failure in failures:
        print(f"  {failure}", file=sys.stderr)
    sys.exit(1)

standard = len(on_disk) - len(categorised)
print(
    f"OK: {len(lists)} *_EXAMPLES arrays name only real examples, no example "
    f"is claimed by two categories, and all {standard} standard examples "
    "print a success marker."
)
PY
