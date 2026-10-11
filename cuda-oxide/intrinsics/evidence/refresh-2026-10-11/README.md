This directory records the 2026-10-11 LLVM evidence refresh. It is an audit
attachment; the evidence scanner reads only JSON files directly in the parent
directory.

All **1,699 primary captures across 1,029 intrinsic routes passed**, including
1,506 terminal assemblies. The initial candidate sweep had 192 failures: 148
renderer panics from incomplete candidate records and 44 invalid combined
LLVM target/PTX requests. These were rerun using the existing full-catalog
renderer and the original evidence's separate compiler and assembler targets.
The final captures include **56 cases with different codegen and terminal SMs**:
the 44 invalid requests and 12 cases initially blocked by the renderer. For
example, LLVM `sm_20/+ptx32` output is assembled with `ptxas -arch=sm_75`;
the capture does not claim LLVM supports `sm_75/+ptx32`.

All **377 auxiliary captures passed**: 187 typed declarations, 116 typed codegen
checks, 66 typed assemblies, and four inline FP8 alternatives checked through
both codegen and assembly. Compilation checks used the existing
`probe.rs::assert_intrinsic_declaration_canonicalizes` and
`probe.rs::validate_probe_instructions` validators, including imported
declaration attributes and exact PTX operand/selector checks. Historical
negative experiments and hardware execution were **not rerun**. Their old
compiler provenance and original outcomes remain explicit in the parent
evidence files. All 38 historical stages have outcome `not_revalidated`;
they establish neither rejection nor execution with the refreshed compiler.
The failed typed-route admission checks apply only to unchanged libNVVM
evidence. The LLVM historical negatives do not authorize current routes.

The tables preserve 770 refreshed stage groups and 3,753 artifact proofs:

- `metadata.json` defines the columns, zero-based indexes, exact tools,
  original paths, source identities, and command argument templates.
- `captures.jsonl` contains one row per primary or auxiliary capture. A null
  artifact slot means that capture did not produce that artifact.
- `artifacts.jsonl` deduplicates SHA-256 hashes and byte counts.
- `stages.jsonl` maps the new evidence filename and JSON pointer to capture
  artifacts and the aggregate ledger digest cited by the evidence detail.
- `inputs.jsonl` retains the exact UTF-8 source for all 307 distinct auxiliary
  LLVM inputs. No temporary helper binary is needed to recreate these inputs.

To reproduce a capture, use the recorded tool versions and argument arrays in
`metadata.json`. Primary sources come from the existing
`render::render_probe` implementation in
`crates/cuda-intrinsics-gen/src/render/probes.rs`, through the full catalog;
the generated files are `intrinsics/probes/{id}.ll` or
`intrinsics/probes/{id}.llvm_nvptx.ll`. Every captured primary source was
reconstructed byte-for-byte from these files by restoring its original
catalog-hash and LLVM-revision comments. Those original comments identify
the inputs at capture time, before the evidence refresh changed catalog
provenance. They do not identify the final catalog.

For primary captures, restore the catalog hash from capture column 7 and the
LLVM revision from `primary_header_llvm_revisions`. For auxiliary captures,
find the source in `inputs.jsonl` by its LLVM-IR artifact index. The source's
hash and byte count must then match `artifacts.jsonl`. Derive the original
input and output filenames from `original_paths`; the column definitions
name the values substituted into those path rules. Run the applicable
commands, using `codegen_gpu` for LLVM and `terminal_gpu` for ptxas:

```sh
"$LLVM_BIN/llvm-as" "$INPUT" -o "$BITCODE"
"$LLVM_BIN/llvm-dis" "$BITCODE" -o "$CANONICAL_IR"
"$LLVM_BIN/llc" "$INPUT" -march=nvptx64 -mcpu="$CODEGEN_GPU" \
  -mattr="$PTX_FEATURE" -o "$PTX"
"$PTXAS" -arch="$TERMINAL_GPU" "$PTX" -o "$CUBIN"
```

Run the first two commands only when the capture records canonicalization,
and run ptxas only when a terminal GPU is recorded. Declaration-only auxiliary
captures stop after canonicalization. The original source and bitcode paths
are significant: llvm-as supplies `source_filename`, and llvm-dis prints the
bitcode path as `ModuleID`. Restoring those paths allows byte-for-byte hash
comparison; other paths provide equivalent compilation with different
canonical artifact hashes. No `source_filename` override or path-remapping
flag was used. The normal contributor gates remain:

```sh
cargo run -p cuda-intrinsics-gen -- check
cargo run -p cuda-intrinsics-gen -- probe --all --skip-terminal --per-target
cargo run -p cuda-intrinsics-gen -- check-abi-history --base-ref "$BASE_REF"
```

Extraction used public LLVM revision
`aab11ee0ea5ea9a1d462ad0bc1fae9d9bb643019`, Git tree
`743426f16c6d14d89426972b8aad1540e5d25ab0`, consumed by Rust commit
`6f5da406b56c2db6607202b63daee73aa8c61297`. Build `llvm-tblgen` from that
revision, then extract from the CUDAoxide workspace:

```sh
git clone https://github.com/rust-lang/llvm-project.git "$LLVM_SRC"
git -C "$LLVM_SRC" checkout aab11ee0ea5ea9a1d462ad0bc1fae9d9bb643019
cmake -S "$LLVM_SRC/llvm" -B "$LLVM_BUILD" -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=NVPTX \
  -DLLVM_ENABLE_PROJECTS= -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF
cmake --build "$LLVM_BUILD" --target llvm-tblgen -j8
cargo run -p cuda-intrinsics-gen -- extract --llvm-src "$LLVM_SRC" \
  --llvm-tblgen "$LLVM_BUILD/bin/llvm-tblgen"
```

Both TableGen dump hashes match the previous extraction. The normalized
2,633 declarations retain their semantic facts; the normalized-file hash
changes with source provenance. Exact dump commands, hashes, the local
TableGen binary identity, and the independent LLVM 22 intrinsic-dump
comparison identity are in `metadata.json` and `../../upstream.lock`.
The public clone above is a reproduction alternative; capture used a local
shared checkout followed by a depth-one fetch of the public revision. Its
exact checkout paths and command arguments are also recorded in the metadata.
The local TableGen binary hash identifies that build; matching dump hashes
provide the reproducibility check.

The following check reconstructs every stage digest without temporary
artifacts or helper programs. Run it from this directory:

```python
import hashlib, json
from pathlib import Path

def rows(name):
    return [json.loads(line) for line in Path(name).read_text().splitlines()]

metadata = json.loads(Path("metadata.json").read_text())
captures = rows("captures.jsonl")
artifacts = rows("artifacts.jsonl")
for _, _, _, expected, proofs in rows("stages.jsonl"):
    entries = []
    for capture_index, kind_index in proofs:
        capture = captures[capture_index]
        digest = artifacts[capture[6][kind_index]][0]
        entries.append((capture[0], metadata["artifact_kinds"][kind_index], digest))
    ledger = "".join(f"{name} {kind} {digest}\n" for name, kind, digest in sorted(entries))
    assert hashlib.sha256(ledger.encode()).hexdigest() == expected
print("Verified all 770 stage ledgers")
```
