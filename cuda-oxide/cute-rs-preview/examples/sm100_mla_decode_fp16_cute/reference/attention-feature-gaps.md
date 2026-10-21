# B200/SM100 attention: omitted functionality and compiler gaps

Most omitted attention functionality requires kernel implementation, host API
changes, and validation. Some features also require extensions to the CuTe-RS
API, dialect contracts, or the selected CUTLASS MLIR exporter.

This follow-up audit covers the [FMHA prefill](../../sm100_fmha_prefill_cute/README.md)
and [FP16 MLA decode](../README.md) examples
against their pinned CUTLASS v4.7.0 references at commit
`dcf215af68a2d08d305076c152a06f201728cd53`. It records remaining work after the
initial matched FP16 profiles; the example READMEs list the implemented
configurations.

## Capability gaps

| Capability | Kernel and host API work | Compiler or dialect work |
|---|---|---|
| Variable sequence lengths and split counts | Add length, prefix-offset, and split arrays; update addressing, scheduling, masking, workspace indexing, and reduction. | Existing operations largely suffice. |
| Sliding windows and bottom-right causal masks | Implement mask policies, aligned coordinates, consistent pipeline trip counts, and fully masked row handling. | Existing comparisons, arithmetic, and control flow suffice. |
| Attention sinks and prefill LSE output | Add arguments and output storage; implement denominator/LSE calculations and reference checks. | Existing FP32 arithmetic, loads, and stores suffice. |
| Prefill head dimensions 32/64; decode query lengths 2–4 and supported smaller head counts | Retile storage, update indexing, descriptors, scheduling, and launch configuration. | Existing FP16 MMA operations support these configurations. |
| Smaller decode pages | Handle multiple page lookups and copies, shared-memory subviews, and barrier transaction counts. | Extend typed layout/TMA support, particularly for 2- and 4-token pages and divided shared-memory views. |
| BF16 prefill | Add BF16 storage, probability conversion, descriptor selection, and validation. | Generalize the FP16-specific CuTe operands, type contracts, importer, and lowering. Add packed BF16 conversion export. Underlying BF16 primitives already exist. |
| FP8 prefill | Implement byte-sized layouts, probability packing/scaling, revised staging, host conversion, and FP8-aware validation. | Extend element-type and element-width support. Export the existing FP8 MMA kind and required packed conversions through the selected backend. |
| Softmax skipping and approximate correction skipping | Coordinate decision flags, scaling, metadata, and pipeline progress; add approximation-aware tests. | Mostly existing operations. Source-matched popcount and optional statistics counters need exporter coverage. Exact unchanged-maximum correction skipping already exists. |
| Exp2 emulation | Implement the reference polynomial, exponent reconstruction, and instruction interleaving. | Add packed round-down addition support in the API/IR and backend. Packed subtraction already exists but needs selected-backend export. |
| Programmatic dependent launch (PDL) | Wire dependency waits/triggers and the appropriate host launch protocol. | Host and native NVVM APIs already exist. The selected CUTLASS MLIR exporter needs grid-dependency mappings. |

## Scope

- BF16 and FP8 are supported options of the pinned prefill reference; the
  selected `mla_decode_fp16.py` reference remains FP16-only. The prefill FP8
  profile uses E4M3FN. Its validation requires FP8 for head dimension 192.
- The decode reference fixes latent dimension 512 and RoPE dimension 64.
  Changing those dimensions would extend beyond the selected reference.
  It permits query lengths 1–4 and head counts up to 128; fewer than 128
  heads require unsplit execution.
- Windows, bottom-right causal alignment, sinks, softmax skipping, exp2
  emulation, and PDL are prefill-reference options. Adding them to the selected
  decode reference would be a separate extension. Configurable approximate
  correction skipping is available in both references.
- Having an operation in the native NVVM path does not guarantee that the
  selected CUTLASS MLIR exporter supports it. BF16 uses the existing `F16`
  MMA instruction family with different element selectors; FP8 needs the
  distinct `F8f6f4` kind exported through that backend.
- This audit identifies implementation requirements; it does not establish
  that the omitted configurations have compiled or passed GPU validation.

## Implementation entry points

- [Attention types and layout constraints](../../../cute-rs/src/attention/sm100/mod.rs),
  [host TMA descriptors](../../../cute-rs/src/attention/sm100/host.rs), and
  [CuTe operation contracts](../../../dialect-cute/src/ops/sm100_attention.rs)
  currently specialize the attention path to FP16.
- [CuTe importing](../../../../crates/mir-importer/src/translator/cute/sm100_attention.rs)
  and [attention export](../../../../crates/cuda-oxide-mlir-export/src/cute/independent_sm100.rs)
  must preserve any new element and layout information through translation.
- [Raw MMA export](../../../../crates/cuda-oxide-mlir-export/src/nvvm/tcgen05.rs)
  and [numeric export](../../../../crates/cuda-oxide-mlir-export/src/nvvm/numeric.rs)
  are the entry points for additional MMA kinds, conversions, and packed
  arithmetic. Review the [selected-backend registry](../../../../crates/cuda-oxide-mlir-export/src/profile.rs)
  when adding an operation that already has a native NVVM route.
- [Device dependency operations](../../../../crates/cuda-device/src/generated/grid_dependency.rs)
  and the [host PDL launch API](../../../../../cuda-core/src/api.rs) already exist;
  PDL work includes their selected-backend mappings and example integration.
- The pinned [prefill source](../../sm100_fmha_prefill_cute/reference/cute/blackwell/kernel/attention/fmha/fmha.py),
  [mask and exp2 helpers](../../sm100_fmha_prefill_cute/reference/helpers/fmha_helpers.py),
  and [decode source](cute/blackwell/kernel/attention/mla/mla_decode_fp16.py)
  define the behaviors and supported configurations to match.

## Implementation order

1. Add FP16 functionality using existing operations: masks, sinks, LSE,
   variable lengths/splits, and supported dimension variants.
2. Extend typed layouts and TMA subviews for smaller pages.
3. Generalize element types and conversions for BF16.
4. Add FP8 layout, packing, MMA export, scaling, and staging support.
5. Complete approximation features, exp2 emulation, and PDL exporter integration.

Validate each configuration against the pinned CuTeDSL reference before
comparing performance. This work can remain within the CuTe + NVVM programming
model; LIR is not required.
