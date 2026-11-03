# Catalog-Generated Intrinsics

Most cuda-oxide device intrinsics are not implemented by editing compiler
stages independently. They are described as reviewed catalog inputs and
`cuda-intrinsics-gen` generates the repetitive API, dialect, importer,
lowering, target, reference, and probe surfaces from that contract.

Use this workflow when the intrinsic fits the generated catalog model. For an
operation that requires bespoke verification or lowering that the catalog
cannot express, use [Adding New Intrinsics](adding-new-intrinsics.md) instead.

---

## Choose the Right Path First

Before editing anything, find the nearest existing intrinsic with the same
semantic and lowering shape.

```text
Does an existing catalog family model the intrinsic?
    |
    +-- yes -> extend that overlay family and the ABI ledger
    |
    +-- no  -> determine whether the generator needs a new family/shape
               or whether the operation is genuinely hand-written
```

An LLVM declaration by itself is not enough to admit an intrinsic. The catalog
also records cuda-oxide policy such as its Rust API, safety and effects,
expected PTX form, target requirements, ABI identity, and lowering evidence.

The normal contribution path is therefore to extend an existing reviewed
family, not to edit generated Rust files or `intrinsics/catalog.json` directly.

---

## Sources of Truth

The generated intrinsic pipeline is assembled from reviewed inputs and resolved
into a generated catalog:

```text
intrinsics/upstream.lock -> intrinsics/imported.json ----\
intrinsics/overlay.toml -> intrinsics/overlay/*.toml -----+
intrinsics/abi-v1.toml -----------------------------------+--> resolve
intrinsics/evidence/*.json -------------------------------/       |
                                                                  v
                                                     intrinsics/catalog.json
                                                            GENERATED
                                                                  |
                                                                  v
                                                        generated outputs
```

Their roles are different:

| File / artifact | Role |
| :-------------- | :--- |
| `intrinsics/upstream.lock` | Pins the LLVM source/extraction identity used by the imported facts, and the rustc commit whose `llc` the evidence probe must run under. |
| `intrinsics/imported.json` | Records what the pinned LLVM/NVPTX source declares. Normal intrinsic additions consume this committed file; they do not rerun extraction. |
| `intrinsics/overlay.toml` | Explicitly indexes the admitted overlay shards. |
| `intrinsics/overlay/*.toml` | Records cuda-oxide policy for reviewed intrinsic families. |
| `intrinsics/abi-v1.toml` | Append-only stable intrinsic ABI ledger. |
| `intrinsics/evidence/*.json` | Records reviewed lowering evidence when the route requires it. |
| `intrinsics/catalog.json` | Resolved generated plan. Do not edit it by hand. |

Generated sources and probe/reference outputs carry a `DO NOT EDIT` marker.
Change their source inputs and regenerate instead.

### Catalog Metadata

The overlay manifest carries version and backend metadata for the resolved
catalog. These values describe the catalog contract; they are not
per-intrinsic knobs.

There are two distinct `schema` numbers in play, one per side of the
generator:

```text
intrinsics/overlay.toml   schema = 44   (overlay input format)
        | cuda-intrinsics-gen
        v
intrinsics/catalog.json   "schema": 46  (serialized catalog format)
```

- `schema` (in `overlay.toml`) identifies the overlay **input** format the
  generator accepts (`OVERLAY_SCHEMA` in
  `cuda-oxide/crates/cuda-intrinsics-gen/src/resolve/overlay.rs`); the resolver rejects
  a mismatched manifest. It is not the catalog format version.
- `catalog_version` identifies the reviewed catalog contract.
- `intrinsic_abi` identifies the intrinsic ABI generation whose stable entries
  are recorded in `intrinsics/abi-v1.toml`.
- `backend_profile` selects the reviewed backend profile used while resolving
  lowering evidence.

Contributors normally change these values only as part of a deliberate catalog,
ABI, or backend-profile transition. Do not synthesize or edit corresponding
metadata directly in generated `intrinsics/catalog.json`.

### catalog.json Header Fields

The generated catalog stamps its own header at resolve time. For reference,
since the file is generated and otherwise undocumented:

- `schema`: version of the **serialized catalog** format (`CATALOG_SCHEMA` in
  the same `resolve/overlay.rs`). Distinct from the overlay's `schema`; see
  the sketch above.
- `catalog_version`: the reviewed catalog contract, copied from
  `overlay.toml`.
- `intrinsic_abi`: the ABI generation, copied from `overlay.toml`; its stable
  entries live in `intrinsics/abi-v1.toml`.
- `generator_version`: the `cuda-intrinsics-gen` crate version that produced
  the catalog (its Cargo package version). The same value appears in the
  `@generated by cuda-intrinsics-gen <version>` banner of every generated
  file.

---

## The Normal Workflow: Extend an Existing Family

### 1. Find the nearest sibling

Start in `intrinsics/overlay/`, not in generated Rust code and not by scrolling
`intrinsics/imported.json`.

Find an existing entry with the same family and lowering shape, then use the
imported LLVM facts only for the fields that must agree with upstream.

For example:

```bash
grep -R 'id = "<nearest-sibling>"' intrinsics/overlay
rg '<llvm-or-record-name>' intrinsics/imported.json
```

Prefer copying a proven sibling over constructing an entry from scratch. The
resolver is intentionally fail-closed, but matching the established family
shape keeps the review small and makes unexpected generated diffs easier to
spot.

### 2. Add or extend the overlay entry

Edit the appropriate file under:

```text
intrinsics/overlay/<family>.toml
```

The overlay is the hand-edited policy input. Do not edit the corresponding
operation under `cuda-oxide/crates/dialect-nvvm/src/ops/generated/`, the generated
importer dispatch, generated lowering tables, or `intrinsics/catalog.json`.

If the contribution requires a **new overlay shard**, also add that shard to
the sorted `shards` list in:

```text
intrinsics/overlay.toml
```

`generate` and `check` reconcile the index with the overlay directory and
reject an unindexed shard.

### 3. Append the ABI ledger entry

Add the new stable identity at the end of:

```text
intrinsics/abi-v1.toml
```

The ledger is append-only. Existing entries must not be reordered, deleted,
renumbered, or repurposed. The catalog identity, operation key, and raw Rust
signature must agree with the overlay entry.

Treat the ABI ID as a merge-time resource. If another intrinsic PR lands first,
rebase onto the latest `main`, append after the new ledger tail, and regenerate.
Do not preserve an obsolete numeric range merely because the branch used it
earlier.

### 4. Handle lowering evidence only when needed

If the new intrinsic uses an already-reviewed lowering/evidence shape, follow
the existing sibling contract. A normal catalog addition does not necessarily
need a new evidence file.

If the backend route is new and needs candidate evidence, use an explicit
candidate probe:

```bash
HOST="$(rustc -vV | sed -n 's/^host: //p')"
LLC="$(rustc --print sysroot)/lib/rustlib/$HOST/bin/llc"

cargo run -p cuda-intrinsics-gen -- probe --candidate \
  --intrinsic <catalog-id> \
  --llc "$LLC" \
  --gpu-target <sm-target> \
  --ptx-feature <ptx-feature> \
  --ptxas <path-to-ptxas>
```

On a system where terminal assembly is deliberately unavailable, replace
`--ptxas <path-to-ptxas>` with the explicit `--skip-terminal` flag. Candidate
mode requires one of those choices; it does not silently fall back.

Review candidate output before promoting any evidence into the committed
contract.

### 5. Regenerate

Run:

```bash
cargo run -p cuda-intrinsics-gen -- generate
```

Then inspect the working tree and diff before running broader checks:

```bash
git status --short
git diff --stat
git diff -- intrinsics cuda-oxide/crates/cuda-intrinsics cuda-oxide/crates/cuda-device \
  cuda-oxide/crates/dialect-nvvm cuda-oxide/crates/mir-importer cuda-oxide/crates/mir-lower \
  cuda-oxide/crates/cuda-oxide-codegen cuda-oxide/crates/rustc-codegen-cuda
```

Check `git status --short` as well: newly created overlay or evidence files are
untracked until staged and therefore do not appear in `git diff`.

For an addition that fits an existing family, generated changes should follow
the sibling pattern. Unexpected unrelated changes are a reason to stop and
recheck the overlay, ledger, evidence, or branch base.

---

## Validate Before Pushing

The repository provides one local entry point for the generated-intrinsics CI
contract:

```bash
just -f cuda-oxide/Justfile check-intrinsics upstream/main
```

Use the actual PR base if it is not `upstream/main`.

That recipe mirrors the generated-intrinsics CI job and currently covers the
three load-bearing checks:

```text
cuda-intrinsics-gen check
cuda-intrinsics-gen probe --all --skip-terminal --per-target
cuda-intrinsics-gen check-abi-history --base-ref <base>
```

Use `just -f cuda-oxide/Justfile check-intrinsics ...` as the normal contributor interface rather than
copying the underlying commands into scripts. The recipe can evolve when the
CI contract gains additional coverage.

Two environment requirements the gates carry with them:

- `check-abi-history` walks git history back to the base ref, so it needs a
  full clone. CI checks out with `fetch-depth: 0`; a shallow local clone can
  fail to resolve the base.
- The generator shells out to `rustfmt` when it writes generated sources.
  That is why `rust-toolchain.toml` pins the `rustfmt` component; no extra
  install is needed on a normal checkout.

For a focused probe while developing one intrinsic, you can also run:

```bash
cargo run -p cuda-intrinsics-gen -- probe \
  --intrinsic <catalog-id> \
  --skip-terminal
```

Finally, run the normal checks required by the files you changed. For changes
that also update the book:

```bash
just -f cuda-oxide/Justfile book
bash cuda-oxide/scripts/check-book-api-names.sh
git diff --check
```

---

## When the Existing Catalog Cannot Express It

A missing sibling is a design signal. If the intrinsic requires a lowering
shape, operand adapter, result contract, or family structure the generator does
not model, the change is no longer a small catalog admission.

A new generated shape typically involves generator code such as:

```text
cuda-oxide/crates/cuda-intrinsics-gen/src/model/
cuda-oxide/crates/cuda-intrinsics-gen/src/resolve/
cuda-oxide/crates/cuda-intrinsics-gen/src/render/
```

and may also require real compiler lowering changes under `cuda-oxide/crates/mir-lower/`
or `cuda-oxide/crates/dialect-nvvm/`.

Keep that work separate from routine catalog admission when possible. The ABI
ledger and generated-intrinsics validation contract still apply.

If the operation instead needs bespoke verification or lowering by design and
should remain outside the catalog, follow the hand-written workflow in
[Adding New Intrinsics](adding-new-intrinsics.md).

---

## Extraction Is Not Part of a Normal Addition

The `extract` command refreshes imported LLVM facts. It is for an LLVM source or
pin update, not for adding an intrinsic that already exists in the committed
`intrinsics/imported.json`.

Normal additions consume the committed imported facts. This keeps routine
intrinsic work independent of an LLVM checkout and makes the reviewed overlay,
ABI ledger, and evidence the explicit cuda-oxide policy layer.

---

## What Not to Edit by Hand

For a catalog-generated intrinsic, do not directly edit:

- `intrinsics/catalog.json`;
- generated files under `cuda-oxide/crates/cuda-intrinsics/src/generated/`;
- generated files under `cuda-oxide/crates/cuda-device/src/generated/`;
- generated files under `cuda-oxide/crates/dialect-nvvm/src/ops/generated/`;
- generated intrinsic dispatch in `mir-importer`;
- generated intrinsic conversion tables in `mir-lower`;
- generated target/collector metadata;
- generated probe/reference outputs.

If regeneration overwrites a hand edit, the edit was made at the wrong layer.
The source of truth is the reviewed catalog input, ABI, evidence, or generator
logic that owns that output.
