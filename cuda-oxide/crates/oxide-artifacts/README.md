# oxide-artifacts

`oxide-artifacts` defines the embedded device-artifact container used by
`cuda-oxide`.

The crate is deliberately accelerator-neutral. It knows how to describe bundles
of generated device-code payloads, but it does not know how to compile or
launch them. Runtime crates can parse bundles and decide whether they can
consume them.

`oxide-artifacts` also does not decide target policy. Choices such as PTX versus
LTOIR, cubin versus fatbin, or single-arch versus multi-arch packaging belong to
the compiler and runtime loader layers. The container only records the payloads
and entry metadata it is given, so higher-level launch APIs can stay stable as
the payload strategy evolves.

## Design

An artifact bundle is a small binary blob stored in a retained host-object
section named `.oxart`.

Each bundle contains:

- a bundle name, normally the producing Rust crate name
- a device target string, such as `sm_90`
- zero or more entry records, such as CUDA kernels
- one or more payload records, such as generated PTX, cubin, or IR bytes

Multiple bundle blobs may be concatenated in the same section. Parsers walk the
section by reading each blob's `total_len` field.

The section-object writer is behind the `object-write` feature and uses the
Rust `object` crate rather than hand-writing ELF. That keeps the common
infrastructure portable across the CUDA host platforms this crate supports
today: Linux on AMD64 and ARM64.

## Wire Format

All integer fields are little-endian. Offsets are relative to the start of the
bundle blob, not the host object file or section.

```text
Artifact Blob

  0                   1                   2                   3
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Magic "OXIDEART"                      |
 |                                                               |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |          Version              |         Header Length         |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Total Length                          |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |        Name Length            |        Target Length          |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |       Payload Count           |        Entry Count            |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |        Compile Options (v2/v3) / Reserved Zero (v1)              |
 |                                                               |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Bundle Name ...                      /
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Target String ...                    /
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Payload Records ...                  /
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Entry Records ...                    /
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Payload Names/Data and Entry Symbols  /
 |                         (each variable item is 8-byte aligned)/
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

Header size is currently 32 bytes. The final field occupies bytes 24 through
31 (the half-open range `[24, 32)`) and is a 64-bit compile-options bitset in
versions 2 and 3. Bit 0 disables FMA contraction; unknown option bits are rejected.
Version-1 bundles define the same bytes as reserved zero and use the historical
default policy, where FMA contraction is allowed.

Version 3 carries finalized kernel launch facts. Bundles without launch facts
retain their existing encoding: version 1 for default compile policy and version
2 for non-default compile policy. A reader that does not support version 3
rejects those bundles before loading code with unknown launch requirements.

```text
Payload Record (24 bytes)

  0                   1                   2                   3
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |       Payload Kind            |          Flags (zero)         |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Data Offset                           |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Data Length                           |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Name Offset                           |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |        Name Length            |          Reserved             |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Reserved                              |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

Payload kind values:

- `0x0100`: PTX
- `0x0110`: NVVM IR
- `0x0120`: LTOIR
- `0x0200`: cubin

```text
Entry Record (24 bytes in v1/v2; 48 bytes in v3)

  0                   1                   2                   3
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |         Entry Kind            |             Flags             |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Metadata                              |
 |                                                               |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Symbol Offset                         |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |       Symbol Length           |          Reserved             |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                         Reserved                              |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

Entry kind values:

- `1`: kernel
- `2`: device function

If bit 0 of `Flags` is set, `Metadata` is present and interpreted by the
consumer for that entry kind. This opaque metadata is independent of typed
launch facts.

Version 3 adds the following fields to every entry record:

| Byte offset | Field | Type |
| --- | --- | --- |
| 24 | Exact block X | u32 |
| 28 | Exact block Y | u32 |
| 32 | Exact block Z | u32 |
| 36 | Finalized dynamic shared-memory bytes per block | u32 |
| 40 | Dynamic shared-memory alignment | u32 |
| 44 | Reserved zero | u32 |

Flag bit 1 indicates that launch facts are present; bit 2 indicates an exact
block shape and requires bit 1. Facts are valid only for kernel entries. Exact
block dimensions must be nonzero and alignment must be a nonzero power of two.
Unknown version-3 entry flags are rejected. Absent fields are written as zero.

Each record describes its concrete entry symbol, including separate generic
monomorphizations. A zero-byte allocation is distinct from absent facts. The
byte count excludes static shared memory and driver-reserved memory. The
producer supplies a finalized allocation and a matching generated alignment;
the container validates their representation but does not derive requirements
from device code or prove that the producer's claims are true.

## Object Storage

When `object-write` is enabled, `build_host_object_for_target` creates a
relocatable object with a single `.oxart` data section.

The writer marks the ELF section as retained with
`SHF_ALLOC | SHF_GNU_RETAIN`.

The rustc CUDA backend writes the generated device artifact into one bundle
blob, emits a host object for the current host target, and appends that object
to rustc's compiled module list before linking. At runtime, `cuda-host` can read
the executable's object sections and load PTX/cubin payloads directly or build a
cubin from embedded NVVM IR/LTOIR before loading.

## Constraints

- The latest supported format version is `3`; bundles without launch facts
  retain their version `1` or `2` encoding.
- All numeric fields are little-endian.
- The blob header is fixed at 32 bytes.
- Payload records are 24 bytes; entry records are 24 bytes in v1/v2 and 48 in v3.
- String and payload offsets are 32-bit, so a single blob must fit in `u32::MAX`
  bytes.
- String lengths and record counts are 16-bit.
- Bundle names, target strings, payload names, and entry symbols must be UTF-8.
- Payloads must be non-empty.
- Unknown payload or entry kind values are rejected by the current parser.
- Compression is not part of the current wire format.

## Feature Flags

- `object-read`: parse artifact bundles out of host object/executable bytes.
- `object-write`: emit host relocatable objects with an `.oxart`
  section.
- `object`: enables both read and write support.

## Compression Policy

Compression is intentionally not part of the base artifact wire format.

`oxide-artifacts` stores device-code payload bytes as provided by the compiler
or packaging layer. This keeps the crate as a neutral container for PTX, NVVM IR,
LTOIR, cubin, and entry metadata, while allowing runtime or loader layers to
decide how those payloads should be consumed.

If embedded payload size becomes a measured problem, compression should first be
handled by a higher-level packaging, distribution, or loader layer. Per-payload
compression should only be added to this format as an explicit versioned
extension with codec and decompressed-size metadata.

## TODO

- Windows support is tracked in [#139](https://github.com/NVlabs/cuda-oxide/issues/139).
