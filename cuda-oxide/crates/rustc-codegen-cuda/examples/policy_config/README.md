# Compile-time policy configuration

A policy is a named bundle of compile-time choices. This example gives one
generic kernel two concrete configurations:

```text
                    SmallTilePolicy     WideTilePolicy
block tile          1024 elements       4096 elements
items per thread    16                  16
element atom        xor one element     xor one element
launch bounds       64 threads, 2/SM    256 threads, 1/SM
loop unroll          2                   4
```

The tile and atom are metadata: they describe the intended shape, layout,
memory space, and cooperating threads. They do not allocate memory, index a
pointer, or implement the operation. The example's `VectorPolicy` trait and
kernel body give those descriptions domain-specific meaning.

```rust,ignore
trait VectorPolicy: Policy {
    type BlockTile: TileSpec;
    type ElementAtom: AtomSpec;
    const MAX_THREADS: u32;
    const ITEMS_PER_THREAD: u32;
    const UNROLL: u32;
}

#[launch_bounds(P::MAX_THREADS, P::MIN_BLOCKS)] // maximum, not exact
#[launch_contract(domain = 1)]
fn transform<P: VectorPolicy>() {
    #[unroll(P::UNROLL)] // Copies this many loop iterations at a time.
    while lane < runtime_count { /* ... */ }
}
```

`MAX_THREADS` is the largest block the specialization accepts; it does not
force every launch to use that many threads. `UNROLL = 4` does not run the loop
only four times. It groups four iterations and still handles any remainder.

rustc specializes the kernel once for each policy type. cuda-oxide reads the
concrete constants from monomorphized MIR, so there is no runtime policy
argument or policy-selection branch:

```text
transform::<SmallTilePolicy> -> transform_TID_<A> -> .maxntid 64
transform::<WideTilePolicy>  -> transform_TID_<B> -> .maxntid 256
```

`PolicyId` is the explicit library-facing stable identity. The `_TID_` entry
name is the compiler-generated identity that the host and device sides agree
on for the concrete Rust specialization.

The same concrete maximum is also used by the prepared host launch:

```rust,ignore
let launch = module.prepare_transform::<SmallTilePolicy>(
    LaunchConfig1D::new(1, 64, 0),
)?;

// Still unsafe because this example passes raw pointers; the launch geometry
// itself was checked before the driver call.
unsafe { module.transform::<SmallTilePolicy>(&stream, &launch, input, output, count)? };
```

For `SmallTilePolicy`, preparation rejects blocks over 64 threads. For
`WideTilePolicy`, it rejects blocks over 256 threads.

Build and inspect the generated PTX without a GPU:

```bash
cargo oxide build policy_config
cargo run --release \
  --manifest-path cuda-oxide/crates/rustc-codegen-cuda/examples/policy_config/Cargo.toml \
  -- --verify-ptx
```

The CPU verifier checks both metadata descriptions, stable policy IDs, two
distinct host-addressable PTX entries from the same release build, different
launch directives, and folded policy tags. It also inspects the raw LLVM IR,
before LLVM optimization, to prove that the two policy unroll factors produced
two and four grouped-loop body copies. The loop bound is runtime data, so this
shape cannot come from folding a known source trip count.
Invalid or unevaluatable attribute constants fail compilation instead of
silently choosing defaults. Use `--release` for the verifier because Rust's
compiler-generated type identity can differ between build profiles; the
explicit `PolicyId` is the profile-independent identity.

The descriptor traits are open because they are metadata, not capabilities. A
kernel library can add a knob such as `Swizzled<128>` and use it in a normal
tile without changing cuda-oxide:

```rust,ignore
enum Swizzled<const BYTES: usize> {}
impl<const BYTES: usize> Layout for Swizzled<BYTES> {}

type TileA = Tile<Shape2<64, 128>, Swizzled<128>, Shared, Block>;
```

## Current boundary

Policy expressions are supported for `launch_bounds` and loop `unroll` first.
Generic expressions currently need Rust's `generic_const_exprs` feature. Host
`launch_contract` fields, cluster dimensions, and dynamic shared-memory sizes
remain literal. A policy-based `launch_bounds` maximum composes with a host
launch contract and is checked separately for each specialization. The
minimum-block value remains a compiler occupancy hint; it is not launch
geometry. Other named constants used for that maximum must be visible at module
scope because the host contract is generated beside the kernel function.

The example needs no GPU by default. Passing `--launch` is an optional runtime
check that allocates both tile sizes and executes both specializations.
