# nvls_all_reduce

Single-process NVLS (NVLink SHARP) all-reduce across every multicast-capable
GPU in the machine, built on the `cuda_core::simt::vmm` multicast wrappers
(`MulticastObject`, `Mapping::new_multicast`).

## What it demonstrates

- Building a multicast team: `cuMulticastCreate`, `cuMulticastAddDevice`,
  then per-GPU `cuMulticastBindMem`, all through safe RAII wrappers.
- Mapping both views of the bound region on each GPU: unicast (that GPU's
  own physical copy, staged and read with normal memcpys) and multicast
  (the switch-backed view for `multimem.*` instructions).
- A device kernel that all-reduces in one switch round-trip per element:
  `multimem.ld_reduce.relaxed.sys.global.add.v4.f32` makes the NVSwitch sum
  a float4 across all bound GPUs, and `multimem.st` broadcasts the result
  back into every copy. No cross-GPU barriers, no P2P copies.

The `multimem` instructions are emitted with `ptx_asm!`; codegen detects the
family and raises the module target to sm_90 / PTX 8.6 automatically. A
typed `cuda_device` multimem intrinsic family is the natural follow-up, and
this kernel doubles as its reference PTX.

## Requirements

- An NVLink-switch system: HGX/DGX H100 or B200 (the driver reports
  `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`). PCIe-only or NVLink
  point-to-point boxes do not qualify.
- 2+ GPUs, CUDA 13.x driver (R580+).

On any other machine the example prints `skipping: ...` and exits cleanly
(so CI smoketests pass everywhere).

## Run

```bash
cargo oxide run nvls_all_reduce
```

Expected output on an 8-GPU HGX box:

```
nvls_all_reduce example
Team: 8 multicast-capable GPUs, 1048576 f32 elements
GPU 0: all 1048576 elements correct
...
GPU 7: all 1048576 elements correct
SUCCESS: NVLS all-reduce across 8 GPUs verified
```

Each GPU `g` stages `input[i] = (g + 1) + (i % 1024)`; after the reduce,
every GPU's copy must hold the exact f32 sum across all GPUs (small
integers, so the sums are exact and order-independent).

## See also

- `cuda-core/tests/simt_vmm_multicast.rs` in cutile-rs: host-side team plumbing test
  (bind/map/unbind lifecycle) that runs without issuing `multimem`.
- PTX ISA section 9.7.13.4 (`multimem.ld_reduce`, `multimem.st`,
  `multimem.red`).
