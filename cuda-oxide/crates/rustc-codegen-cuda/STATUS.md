# rustc-codegen-cuda: error example status

Examples named `error*` fall into two kinds:

- **diagnostics-fixture**: intentional negative test. The compiler is supposed
  to reject this. It is not a gap.
- **support-gap**: a real Rust feature that the compiler does not yet handle.
  Kept as an expected-failure regression test until it is implemented.

When adding a new `error*` example, update this table and the
`ERROR_EXAMPLES` array in `cuda-oxide/scripts/smoketest.sh` in the same commit.
Run `cuda-oxide/scripts/check-error-example-status.sh` to verify both are in sync.

| Example                                | Kind                | Fails at                             |
| :------------------------------------- | :------------------ | :----------------------------------- |
| `error`                                | diagnostics-fixture | `core::fmt` reachable from device    |
| `error_enum_bool_payload_addr`         | diagnostics-fixture | `&mut` to a canonical-byte payload   |
| `error_enum_pointer_overlap`           | support-gap         | Overlaid pointer/integer payload     |
| `error_enum_shared_pointer_layout`     | support-gap         | Oversized AS3 pointer arrays/vectors |
| `error_generated_intrinsic_abi`        | diagnostics-fixture | Unsupported raw intrinsic ABI        |
| `error_generated_intrinsic_callable`   | diagnostics-fixture | Raw intrinsic passed through `Fn`    |
| `error_generated_intrinsic_fn_pointer` | diagnostics-fixture | Raw intrinsic made into `fn` pointer |
| `error_generated_intrinsic_unknown_id` | diagnostics-fixture | Unknown ID in a supported ABI        |
| `error_heap_alloc`                     | diagnostics-fixture | `__rust_alloc` reachable (#108)      |
| `error_host_arch_intrinsic`            | diagnostics-fixture | `core::arch::<host>` intrinsic reached |
| `error_host_target_feature`            | diagnostics-fixture | `#[target_feature]` fn reached       |
| `error_kernel_shared_param`            | diagnostics-fixture | AS3/AS5 pointer as kernel parameter  |
| `error_missing_device_attr`            | diagnostics-fixture | `thread::index_*` stub (#76)         |
| `error_set_discriminant_uninhabited`   | diagnostics-fixture | Invalid enum variant selection       |

Drops whose monomorphized glue is provably a no-op (e.g. the
`core::array::IntoIter` behind `for x in arr` with Copy elements) lower
to a plain branch since issue #138. When the no-op proof fails, the drop
lowers to a device-side call to the monomorphized `drop_in_place::<T>`
shim, which the collector gathers and the pipeline compiles like any
other device function; see the `drop_glue` example. This covers direct
`impl Drop` types reachable from kernels. The shim and every
`Drop::drop` body it reaches go through the same device pipeline, so
drop glue whose MIR uses constructs the pipeline cannot yet translate
(e.g. slice/`Vec` element drops, panic formatting) still fails
compilation with a diagnostic. Destructors are never silently skipped:
a drop is either proven unobservable or its call is emitted.

The remaining enum storage fixtures deliberately fail closed. Pointer and
integer payloads cannot share one lowered slot without erasing LLVM pointer
provenance. Shared-memory pointers are 64 bits in PTX and legacy NVVM output
but 32 bits in modern NVVM output. Direct fields and pointer leaves nested
through structs/tuples use target-stable generic physical storage, with
recursive reconstruction at enum construction and extraction boundaries.
Arrays use the same reconstruction when the payload's recursive expansion
contains at most 16 array-expanded shared-pointer leaves in total, whether
they come from one array or from several arrays nested through structs.
Larger expansions remain rejected to keep generated code shape bounded, while
pointer vectors still need separate ABI and address-space-cast semantics.
(Bool leaves use canonical i8 physical bytes and are converted recursively at
the same boundary.)
