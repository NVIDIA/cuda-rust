# dialect-iket

`dialect-iket` is the semantic, compiler-facing representation of
In-Kernel Event Tracing in CUDA Oxide.

```text
Rust device API
    -> iket.mark / iket.range_start / iket.range_end
       iket.sentinel_token / iket.range_push / iket.range_pop
    -> IKET lowering policy
    -> NativeDump or ExtendedNativeDump placeholders + CUBIN metadata
    -> IKET runtime
```

The dialect owns source intent only:

- event and range names are arbitrary-length strings without a 32-character
  source restriction;
- token ranges use a first-class `!iket.range_token` SSA type, corresponding
  to CuTe DSL's `!iket.range.token` (Pliron type mnemonics cannot contain a
  second dot);
- payload signedness and width remain explicit in `#iket.payload_kind`;
- operations preserve only event/range intent and payload semantics; physical
  instrumentation resources and method selection belong to lowering.

## Lowering policy

The lowering pass is intentionally separate from this crate. Its public
configuration should support `auto`, `native`, and `extended`:

- `auto` selects NativeDump while the module fits its event-ID budget and
  switches to ExtendedNativeDump when it does not;
- `native` requests the lower-overhead 4-byte event encoding and fails with a
  compile-time diagnostic if the module exceeds the runtime-compatible budget;
- `extended` requests the wider 8-byte event encoding even for a small module.

The event-ID budget must come from the selected IKET compatibility profile,
not from a dialect constant. IKET uses a reserved range-pop ID and distinct
NativeDump/ExtendedNativeDump limits.

Long names are lowered using the CUDA C++ IKET convention: a fixed-size event
attribute contains either the short name or an `h<16hex>` FNV-1a placeholder,
and a `__iket_string_decl_*_str` global stores the full string. IKET resolves
that table at CUBIN load time.

## Runtime boundary

The compiler repository owns IKET annotations and lowering, but does not
vendor the IKET profiler. Runtime controls such as buffer auto-sizing, warp
trace, and Perfetto export remain owned by `run-iket`.
