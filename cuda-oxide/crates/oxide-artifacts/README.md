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
today: Linux ELF on AMD64/ARM64 and Windows COFF on MSVC AMD64/ARM64.

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
 |        Compile Options (v2) / Reserved Zero (v1)              |
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
version 2. Bit 0 disables FMA contraction; unknown option bits are rejected.
Version-1 bundles define the same bytes as reserved zero and use the historical
default policy, where FMA contraction is allowed.

Version 2 is the latest supported format. For backward compatibility, the
writer still emits version 1 when the compile-options bitset is zero. It emits
version 2 only when a bundle carries a non-default compile policy, ensuring
that an older reader rejects policy it would otherwise silently ignore.

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
Entry Record (24 bytes)

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
consumer for that entry kind. No metadata interpretation is currently required
for PTX kernel loading.

## Object Storage

When `object-write` is enabled, `build_host_object_for_target` creates a
relocatable object with a single `.oxart` data section.

The writer marks the ELF section as retained with
`SHF_ALLOC | SHF_GNU_RETAIN`.

For Windows MSVC host targets, the writer emits a COFF object and marks `.oxart`
as initialized readable data with `IMAGE_SCN_CNT_INITIALIZED_DATA |
IMAGE_SCN_MEM_READ`. The CUDA rustc backend writes those Windows artifact
objects with the `.embed.obj` suffix; non-Windows targets continue to use
`.embed.o`.

The rustc CUDA backend writes the generated device artifact into one bundle
blob, emits a host object for the current host target, and appends that object
to rustc's compiled module list before linking. At runtime, `cuda-host` can read
the executable's object sections and load PTX/cubin payloads directly or build a
cubin from embedded NVVM IR/LTOIR before loading.

## Constraints

- The latest supported format version is `2`; default-policy bundles are still
  emitted as version `1` for backward compatibility.
- All numeric fields are little-endian.
- The blob header is fixed at 32 bytes.
- Payload and entry records are fixed at 24 bytes each.
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
