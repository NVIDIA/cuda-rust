# Kernel Families

GPU kernels are frequently specialized for different problem shapes, hardware
capabilities, resource limits, or tuning decisions. A single logical operation
may therefore have several ahead-of-time compiled entry points.

`KernelFamily` represents such a set of compiled variants and provides a
validated host-side selection mechanism.

It separates two independent decisions:

```text
Eligibility: can this variant safely execute this problem?
Preference:  which eligible variant should execute?
```

This distinction is important. Performance policy must never select a variant
that violates its launch, shape, hardware, or resource requirements.

`KernelFamily` performs selection entirely on the CPU. It does not compile
kernels, query CUDA devices, benchmark entries, create threads, or access the
filesystem. The application supplies all relevant problem and hardware facts.

## When to use a kernel family

Use `KernelFamily` when an operation has a small, fixed set of already compiled
implementations, such as:

* different output tile sizes
* different block dimensions
* specialized tensor-core kernels for particular compute capabilities
* small-input and large-input implementations
* kernels with different shared-memory requirements
* kernels tuned for different problem-shape ranges
* a generic fallback plus one or more optimized variants

A kernel family is not a runtime registry or a JIT compiler. Its number of
variants is a const generic and is therefore part of the Rust type:

```rust
KernelFamily<Id, Entry, Meta, const N: usize>
```

The family owns exactly `N` variants.

This bounded representation is useful when the available kernels are determined
at build time and should remain explicit and auditable.

## Selection model

The selection flow has two modes:

```text
Force(id) -> resolve variant -> validate ----------------------> Override

Auto      -> cache lookup -> resolve -> validate --------------> Cache
          -> filter eligible variants
          -> selector chooses an ID
          -> validate selector output
          -> cache selected ID --------------------------------> Selector
```

`SelectionMode::Force(id)` is a validated manual override. It bypasses the cache
and selector, but it does not bypass eligibility validation.

`SelectionMode::Auto` first attempts to use a cached stable variant ID. Unknown
or newly ineligible cache entries are treated as misses. The selector then sees
only eligible variants.

Every successful selection reports its provenance:

```rust
pub enum SelectionSource {
    Override,
    Cache,
    Selector,
}
```

This lets applications distinguish operator overrides, cached decisions, and
new selector decisions in logs and diagnostics.

## Family anatomy

A family contains a stable family identity and an array of variants:

```rust
pub struct KernelFamily<Id, Entry, Meta, const N: usize> {
    // Private fields
}
```

Each variant contains three caller-defined values:

```rust
pub struct KernelVariant<Id, Entry, Meta> {
    // Private fields
}
```

| Component | Purpose                                                |
| :-------- | :----------------------------------------------------- |
| `Id`      | Stable semantic identity for the variant               |
| `Entry`   | Callable entry or launcher associated with the variant |
| `Meta`    | Eligibility and preference metadata                    |
| `N`       | Compile-time number of variants                        |

`Entry` is intentionally generic. It can be a function pointer, a typed launch
wrapper, an enum describing generated methods, or another application-defined
value.

`Meta` is also application-defined. It commonly stores requirements and tuning
facts such as:

* minimum compute capability
* tile dimensions
* alignment requirements
* shared-memory usage
* supported data types
* preferred problem-size range
* launch geometry
* tuning score or priority

## Defining stable variant IDs

Every variant requires an explicit ID:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VariantId {
    Scalar,
    Chunked,
}
```

IDs are used by:

* manual overrides
* caches
* selectors
* diagnostics
* configuration files
* environment variables

An ID must represent the semantic identity of a compiled variant.

Good IDs include enum values or stable strings whose meaning is controlled by
the application:

```rust
enum GemmVariantId {
    M256xN256,
    M512xN256,
}
```

Do not use values whose identity may change between builds or processes:

* array indices
* pointers
* `TypeId`
* formatted debug output
* addresses of generated functions
* hashes whose algorithm or inputs are not stable

Array position is not a substitute for a stable ID. Declaration order may
affect selection policy, but it must not define cache identity.

## Constructing a family

Create variants with `KernelVariant::new`:

```rust
let variant = KernelVariant::new(
    VariantId::Chunked,
    chunked_entry,
    VariantMetadata {
        min_len: 1024,
        chunk_width: 256,
    },
);
```

Create the family with `KernelFamily::try_new`:

```rust
let family = KernelFamily::try_new(
    "vector-copy",
    1,
    [
        scalar_variant,
        chunked_variant,
    ],
)?;
```

The arguments are:

```rust
KernelFamily::try_new(
    name: &'static str,
    revision: u32,
    variants: [KernelVariant<Id, Entry, Meta>; N],
)
```

Construction validates three structural invariants:

1. The family contains at least one variant.
2. The family name is not empty or whitespace-only.
3. Variant IDs are unique within the family.

Construction can return:

```rust
pub enum KernelFamilyBuildError {
    EmptyFamily,
    EmptyFamilyName,
    DuplicateVariantId {
        first_index: usize,
        duplicate_index: usize,
    },
}
```

## Family identity and revisioning

The family name and revision form its selection and cache namespace:

```rust
let id = family.id();

println!("{}", id.name());
println!("{}", id.revision());
println!("{id}"); // vector-copy@1
```

A persistent or in-memory cache must include both values in its key.

Increment the revision whenever an existing cached decision might no longer
mean the same thing.

Revision changes are required when modifying:

* variant membership
* the meaning of a variant ID
* eligibility rules
* selector or preference policy
* tuning methodology
* metadata used by selection
* declaration order, unless selection and tuning are explicitly
  order-independent

For example:

```rust
KernelFamily::try_new("vector-copy", 1, variants_v1)?;
KernelFamily::try_new("vector-copy", 2, variants_v2)?;
```

Changing the revision prevents decisions made under the first contract from
being interpreted under the second contract.

The family name should describe the logical operation and specialization axis:

```text
image-filter/radius
gemm/output-tile
reduction/block-size
attention/head-dimension
```

Avoid names based on transient implementation locations or process-specific
state.

## Eligibility with `KernelProblem`

Eligibility answers whether a specific variant can safely handle a problem.

Implement:

```rust
pub trait KernelProblem<Variant> {
    type Rejection: std::error::Error + Send + Sync + 'static;

    fn validate(&self, variant: &Variant) -> Result<(), Self::Rejection>;
}
```

The problem type should contain every fact required to validate the launch.

For example:

```rust
#[derive(Clone, Copy, Debug)]
struct CopyProblem {
    len: usize,
}
```

Validation may consider:

* problem dimensions
* divisibility requirements
* buffer alignment
* data type
* device compute capability
* shared-memory capacity
* grid and block limits
* cooperative-launch support
* cluster-launch support
* integer conversion and allocation overflow
* application-specific semantic constraints

Implementations should be deterministic and free of side effects.

Eligibility is a safety and correctness boundary. It should not encode only
performance preference.

For example, a variant requiring a length divisible by 256 should reject an
incompatible problem even if another layer is expected to avoid selecting it.

## Preference with `KernelSelector`

A selector chooses among variants that have already passed eligibility
validation:

```rust
pub trait KernelSelector<Problem, Variant, Id> {
    type Error: std::error::Error + Send + Sync + 'static;

    fn select(
        &mut self,
        family: KernelFamilyId,
        problem: &Problem,
        eligible: &[&Variant],
    ) -> Result<Id, Self::Error>;
}
```

The `eligible` slice has two guarantees:

1. It is not empty.
2. It preserves family declaration order.

The selector returns a stable variant ID, not an array index or entry reference.

Selectors can implement policies such as:

* choose the largest tile supported by the problem
* choose a small-input kernel below a threshold
* choose using offline benchmark results
* choose using hardware generation
* choose the first eligible variant
* choose from an application configuration
* invoke an external tuning service

A selector must return an ID from the provided eligible slice. `KernelFamily`
checks this requirement and rejects invalid selector output.

Closures can also act as selectors when their signature matches the trait.
Annotate every closure parameter type. Without the annotations, closure
lifetime inference can fix the borrows to one specific call and rustc rejects
the closure with "implementation of `FnMut` is not general enough". A
selector that cannot fail can use `std::convert::Infallible` as its error
type:

```rust
let mut selector =
    |_: KernelFamilyId, _: &CopyProblem, eligible: &[&Variant]| {
        Ok::<_, Infallible>(*eligible[0].id())
    };
```

## Automatic selection

Automatic selection uses:

```rust
SelectionMode::Auto
```

The full call is:

```rust
let selected = family.select(
    &problem,
    SelectionMode::Auto,
    &mut selector,
    &mut cache,
)?;
```

The automatic path performs these operations:

1. Ask the cache for a stable variant ID.
2. Resolve the cached ID within the family.
3. Revalidate the cached variant against the current problem.
4. Return a valid cache hit immediately.
5. Otherwise, validate every family member.
6. Preserve eligible variants in declaration order.
7. Fail if no variant is eligible.
8. Ask the selector to choose from the eligible slice.
9. Verify that the selector returned a family member.
10. Verify that the returned member was in the eligible slice.
11. Store the validated selected ID in the cache.
12. Return the selected variant with `SelectionSource::Selector`.

Cached IDs are hints, not authority.

A cache entry may become invalid because:

* the cache was written by an older application version
* eligibility depends on hardware
* problem constraints changed
* a persistent cache is corrupted
* a variant was removed
* a previous policy used a different revision incorrectly

For these reasons, every cached ID is resolved and validated before use.

## Forced selection

Use:

```rust
SelectionMode::Force(id)
```

when a caller explicitly requests a variant.

Typical sources include:

* command-line options
* environment variables
* test configuration
* debugging controls
* performance experiments
* operator policy

Example:

```rust
let selected = family.select(
    &problem,
    SelectionMode::Force(VariantId::Scalar),
    &mut selector,
    &mut cache,
)?;
```

Forced selection:

* bypasses cache lookup
* bypasses selector invocation
* does not update the cache
* verifies that the requested ID belongs to the family
* validates that the requested variant is eligible

An unknown forced ID returns `UnknownForcedVariant`.

A known but incompatible variant returns `IneligibleForcedVariant`.

A manual override therefore cannot silently bypass launch requirements.

## Selection results

A successful call returns a borrowed `SelectedVariant`:

```rust
pub struct SelectedVariant<'family, Id, Entry, Meta> {
    // Private fields
}
```

Access the variant and provenance with:

```rust
let variant = selected.variant();
let source = selected.source();

println!("selected {:?} from {}", variant.id(), source);
```

The selected value borrows from the family. It does not copy or move the
variant entry or metadata.

The variant provides:

```rust
variant.id();
variant.entry();
variant.metadata();
```

`SelectionSource` implements `Display`:

```text
override
cache
selector
```

## Complete example

The following example uses ordinary CPU functions as entries. The same
selection structure applies when entries are typed wrappers around compiled
CUDA kernels.

```rust
use cuda_host::{
    KernelFamily, KernelFamilyBuildError, KernelFamilyId, KernelProblem, KernelSelector,
    KernelVariant, NoKernelSelectionCache, SelectionMode, SelectionSource,
};
use std::convert::Infallible;
use std::error::Error;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VariantId {
    Scalar,
    Chunked,
}

type Entry = fn(&[f32], &mut [f32]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VariantMetadata {
    min_len: usize,
    chunk_width: usize,
}

type Variant = KernelVariant<VariantId, Entry, VariantMetadata>;
type CopyFamily = KernelFamily<VariantId, Entry, VariantMetadata, 2>;

fn scalar_copy(input: &[f32], output: &mut [f32]) {
    output.copy_from_slice(input);
}

fn chunked_copy(input: &[f32], output: &mut [f32]) {
    const WIDTH: usize = 256;

    for (input_chunk, output_chunk) in input
        .chunks_exact(WIDTH)
        .zip(output.chunks_exact_mut(WIDTH))
    {
        output_chunk.copy_from_slice(input_chunk);
    }
}

fn copy_family() -> Result<CopyFamily, KernelFamilyBuildError> {
    KernelFamily::try_new(
        "book/vector-copy",
        1,
        [
            KernelVariant::new(
                VariantId::Scalar,
                scalar_copy as Entry,
                VariantMetadata {
                    min_len: 0,
                    chunk_width: 1,
                },
            ),
            KernelVariant::new(
                VariantId::Chunked,
                chunked_copy as Entry,
                VariantMetadata {
                    min_len: 1024,
                    chunk_width: 256,
                },
            ),
        ],
    )
}

#[derive(Clone, Copy, Debug)]
struct CopyProblem {
    len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CopyVariantRejection {
    variant: VariantId,
    required_min_len: usize,
    required_multiple: usize,
    actual_len: usize,
}

impl fmt::Display for CopyVariantRejection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:?} requires len >= {} and len divisible by {}, but len is {}",
            self.variant,
            self.required_min_len,
            self.required_multiple,
            self.actual_len,
        )
    }
}

impl Error for CopyVariantRejection {}

impl KernelProblem<Variant> for CopyProblem {
    type Rejection = CopyVariantRejection;

    fn validate(&self, variant: &Variant) -> Result<(), Self::Rejection> {
        let metadata = variant.metadata();

        if self.len < metadata.min_len
            || !self.len.is_multiple_of(metadata.chunk_width)
        {
            return Err(CopyVariantRejection {
                variant: *variant.id(),
                required_min_len: metadata.min_len,
                required_multiple: metadata.chunk_width,
                actual_len: self.len,
            });
        }

        Ok(())
    }
}

struct CopySelector;

impl KernelSelector<CopyProblem, Variant, VariantId> for CopySelector {
    type Error = Infallible;

    fn select(
        &mut self,
        _family: KernelFamilyId,
        _problem: &CopyProblem,
        eligible: &[&Variant],
    ) -> Result<VariantId, Self::Error> {
        let preferred = eligible
            .iter()
            .copied()
            .max_by_key(|variant| variant.metadata().chunk_width)
            .expect("eligible is guaranteed to be non-empty");

        Ok(*preferred.id())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let family = copy_family()?;

    let input = vec![1.0f32; 4096];
    let mut output = vec![0.0f32; input.len()];
    let problem = CopyProblem { len: input.len() };

    let mut selector = CopySelector;
    let mut cache = NoKernelSelectionCache;

    let selected = family.select(
        &problem,
        SelectionMode::Auto,
        &mut selector,
        &mut cache,
    )?;

    println!(
        "selected {:?} from {}",
        selected.variant().id(),
        selected.source(),
    );

    assert_eq!(selected.variant().id(), &VariantId::Chunked);
    assert_eq!(selected.source(), SelectionSource::Selector);

    let entry = *selected.variant().entry();
    entry(&input, &mut output);

    assert_eq!(input, output);

    let forced = family.select(
        &problem,
        SelectionMode::Force(VariantId::Scalar),
        &mut selector,
        &mut cache,
    )?;

    assert_eq!(forced.variant().id(), &VariantId::Scalar);
    assert_eq!(forced.source(), SelectionSource::Override);

    Ok(())
}
```

For an input length of 4096, both variants are eligible. The selector prefers
the variant with the larger `chunk_width`.

For an input length of 1000, only `Scalar` is eligible because `Chunked`
requires at least 1024 elements and a length divisible by 256.

The selection mechanism does not know what `chunk_width` means. Eligibility and
preference semantics belong entirely to the application.

## Adding a cache

Use `NoKernelSelectionCache` when selection should run every time:

```rust
let mut cache = NoKernelSelectionCache;
```

For repeated problem classes, implement `KernelSelectionCache`:

```rust
pub trait KernelSelectionCache<Problem, Id> {
    type Error: std::error::Error + Send + Sync + 'static;

    fn lookup(
        &mut self,
        family: KernelFamilyId,
        problem: &Problem,
    ) -> Result<Option<Id>, Self::Error>;

    fn store(
        &mut self,
        family: KernelFamilyId,
        problem: &Problem,
        variant: &Id,
    ) -> Result<(), Self::Error>;
}
```

A simple in-memory cache for the previous example can use problem length as
part of its key:

```rust
use cuda_host::{KernelFamilyId, KernelSelectionCache};
use std::collections::HashMap;
use std::convert::Infallible;

#[derive(Default)]
struct MemoryCache {
    values: HashMap<(KernelFamilyId, usize), VariantId>,
}

impl KernelSelectionCache<CopyProblem, VariantId> for MemoryCache {
    type Error = Infallible;

    fn lookup(
        &mut self,
        family: KernelFamilyId,
        problem: &CopyProblem,
    ) -> Result<Option<VariantId>, Self::Error> {
        Ok(self.values.get(&(family, problem.len)).copied())
    }

    fn store(
        &mut self,
        family: KernelFamilyId,
        problem: &CopyProblem,
        variant: &VariantId,
    ) -> Result<(), Self::Error> {
        self.values.insert((family, problem.len), *variant);
        Ok(())
    }
}
```

The application decides which problem properties define cache equivalence.

For example, a GEMM cache key may include:

```text
family name
family revision
compute capability
M
N
K
input type
output type
layout
alignment class
```

Do not omit a property when it can change eligibility or preference.

A cache implementation may be:

* an in-memory map
* a process-wide cache
* a database
* a configuration file
* a benchmark result store
* a remote tuning service

Cache errors are explicit. If `lookup` or `store` returns an error, selection
stops and returns `CacheLookupFailed` or `CacheStoreFailed`.

A best-effort cache may internally translate storage failures into misses or
no-op stores when that policy is appropriate.

## Error handling

`KernelFamily::select` can report errors from four layers:

1. forced selection
2. eligibility validation
3. selector execution or output validation
4. cache operations

The principal variants are:

| Error                               | Meaning                                                     |
| :---------------------------------- | :---------------------------------------------------------- |
| `UnknownForcedVariant`              | A forced ID is not a family member                          |
| `IneligibleForcedVariant`           | A forced family member cannot handle the problem            |
| `NoEligibleVariants`                | Every family member failed validation                       |
| `SelectorFailed`                    | The selector returned its own policy error                  |
| `SelectorReturnedUnknownVariant`    | The selector returned an ID outside the family              |
| `SelectorReturnedIneligibleVariant` | The selector returned a family member that was not eligible |
| `CacheLookupFailed`                 | Cache lookup returned an error                              |
| `CacheStoreFailed`                  | Cache storage returned an error                             |

Unknown or ineligible cached IDs are not selection errors. They are treated as
cache misses because cache contents are untrusted hints.

Selector mistakes are errors because the selector receives the authoritative
eligible slice during the current selection call.

## Declaration order

Family variants retain declaration order:

```rust
KernelFamily::try_new(
    "operation",
    1,
    [
        fallback,
        medium,
        aggressive,
    ],
)?;
```

The selector receives eligible variants in this same relative order.

This supports deterministic policies such as:

```rust
Ok(*eligible[0].id())
```

or priority encoded through declaration order.

If declaration order affects selection or tuning, changing the order requires a
family revision bump.

Prefer explicit metadata when the policy is more complex than a simple ordered
fallback. Metadata makes the policy easier to inspect, test, and evolve.

## Testing a family

Kernel-family policy is host-side logic and can be tested without a CUDA
device.

Tests should verify at least:

* empty families are rejected
* blank names are rejected
* duplicate IDs are rejected
* forced selection bypasses selector and cache
* forced selection still validates eligibility
* valid cache hits bypass the selector
* unknown cache IDs fall back to the selector
* newly ineligible cache IDs fall back to the selector
* stale cache entries are replaced
* selectors receive only eligible variants
* declaration order is preserved
* unknown selector output is rejected
* ineligible selector output is rejected
* no eligible variants produces an error
* cache lookup failures propagate
* cache store failures propagate
* family revisions isolate cached decisions

The repository integration tests provide executable examples:

```text
cuda-oxide/crates/cuda-host/tests/kernel_family.rs
```

Run them with:

```text
cargo test -p cuda-host --test kernel_family
```

## Using generated CUDA launchers

In a real application, `Entry` commonly represents a concrete typed launcher.

For example:

```rust
type KernelLauncher = unsafe fn(
    &kernels::LoadedModule,
    &CudaStream,
    LaunchConfig,
    &DeviceBuffer<f32>,
    &mut DeviceBuffer<f32>,
) -> Result<(), cuda_core::DriverError>;
```

The family then stores one launcher per compiled specialization:

```rust
type Family =
    KernelFamily<VariantId, KernelLauncher, VariantMetadata, 2>;
```

Selection returns the launcher and its metadata:

```rust
let selected = family.select(
    &problem,
    mode,
    &mut selector,
    &mut cache,
)?;

let launch = *selected.variant().entry();
let metadata = selected.variant().metadata();

println!(
    "launching {:?} from {}",
    selected.variant().id(),
    selected.source(),
);

unsafe {
    launch(
        &module,
        &stream,
        launch_config,
        &input,
        &mut output,
    )?;
}
```

`KernelFamily` validates variant selection. It does not make an otherwise unsafe
raw launch safe.

Launch geometry, pointer validity, buffer sizes, context identity, and kernel
ABI obligations still belong to the launch API. Prefer prepared launch
contracts when the kernel geometry can be expressed statically. See
[Launching Kernels](launching-kernels.md).

## Real-world example

The `gemm_sol_final` example uses a two-entry family for two Blackwell GEMM
output-tile specializations:

```text
M256xN256
M512xN256
```

Its family metadata includes:

* kernel name
* output tile
* M tile size
* N tile size
* K tile multiple
* problem-size preference threshold

Eligibility checks validate matrix dimensions, tile divisibility, integer
ranges, allocation-size overflow, and CUDA grid limits.

The selector chooses the largest preferred tile whose threshold is compatible
with the problem.

The `GEMM_SOL_VARIANT` environment variable exposes:

```text
auto
m256xn256
m512xn256
```

The forced values remain validated overrides. They cannot select a tile whose
compiled shape is incompatible with the matrix dimensions.

See:

* [`gemm_sol_final`](https://github.com/NVlabs/cuda-oxide/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/gemm_sol_final)
* [`KernelFamily` integration tests](https://github.com/NVlabs/cuda-oxide/blob/main/cuda-oxide/crates/cuda-host/tests/kernel_family.rs)
* [`cuda-host` kernel-family implementation](https://github.com/NVlabs/cuda-oxide/blob/main/cuda-oxide/crates/cuda-host/src/kernel_family.rs)
* [`cuda-host` README](https://github.com/NVlabs/cuda-oxide/blob/main/cuda-oxide/crates/cuda-host/README.md)

## Design guidelines

Use the following rules when introducing a new family.

### Keep the family bounded

Prefer a small set of meaningful compiled variants.

A large dynamically changing set usually indicates that a registry, generated
dispatch table, or tuning database is a better abstraction.

### Make IDs semantic

The ID should describe which compiled contract is selected, not where it happens
to appear in an array.

### Put correctness in eligibility

Any condition required for safe or correct execution belongs in
`KernelProblem::validate`.

Do not rely on the selector to avoid unsafe variants.

### Put performance in preference

The selector should decide among already valid choices.

This keeps tuning policy independent from safety validation.

### Treat cache entries as untrusted

Always allow `KernelFamily` to resolve and revalidate cached IDs.

Do not launch directly from a cached ID without family validation.

### Version policy changes

Increment the family revision when a cached choice may have different meaning
under the new code.

### Preserve observability

Record `SelectionSource`, family identity, selected ID, and relevant problem
facts in diagnostics.

This is particularly useful when comparing operator overrides, cached tuning,
and current selector behavior.

## Summary

`KernelFamily` provides a bounded and validated dispatch layer for
ahead-of-time compiled kernel specializations.

Its core properties are:

* fixed family membership
* stable semantic variant IDs
* caller-defined entries and metadata
* explicit eligibility validation
* independent preference selection
* validated manual overrides
* revalidated cache entries
* explicit family revisioning
* observable selection provenance
* CPU-only, device-independent policy logic

The abstraction does not decide how kernels should be tuned. It provides the
invariants required to make that tuning policy explicit, testable, and safe to
integrate with compiled CUDA launchers.
