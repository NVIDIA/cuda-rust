# Explicit cache-hint probe

The `L2::128B` experiment showed no consistent meaningful benefit and was not
retained. The native NVVM exporter was restored. These single-round exploratory
measurements do not replace the repeated final performance results.

The saved CUTLASS reference PTX uses plain `cp.async.cg.shared.global`; its
`LTC128B` SASS modifier is selected by the assembler. The exact condition that
selects it remains unknown. The probe compared plain inline PTX with an otherwise
identical `L2::128B` form, preserving the dynamic source-byte operand and its
partial-atom and zero-byte copy behavior. NVIDIA documents the qualifier as a
prefetch-size performance hint, introduced in PTX 7.4 and supported on `sm_80`
and newer. This experiment used the existing `sm_120a` target.
[PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async).

In all four kernels, plain and hinted inline PTX differ only in the `LTC128B`
bit at 24 static LDGSTS instructions. Raw ELF `.text` comparisons confirm one
changed bit per copy and identical remaining code bytes. The retained native
and plain inline-PTX static kernels also match exactly. Runtime kernels change
register allocation and scheduling when switching to plain inline PTX:
register counts change from 128 to 127 for causal and 136 to 128 for noncausal.
A direct native-to-hinted runtime comparison therefore cannot isolate the hint;
the plain inline-PTX timing controls are necessary.

| Probe | Mode | cuda-oxide median, µs | Matched reference median, µs |
| --- | --- | ---: | ---: |
| Hinted | Static causal | 15.488 | 15.424 |
| Hinted | Static noncausal | 15.616 | 15.456 |
| Hinted | Runtime causal | 16.096 | 15.648 |
| Hinted | Runtime noncausal | 16.096 | 15.680 |
| Plain inline PTX | Runtime causal | 16.095 | 15.679 |
| Plain inline PTX | Runtime noncausal | 16.095 | 15.679 |

Each case used one round of 31 measured launches after 100 warmups, with
`seq_q = seq_k = 1024`, eight heads, CUTLASS 4.7, and physical GPU 1. Timings
are Nsight Systems kernel-active durations. The runtime hint and control
medians differ by only 0.001 µs; their separate matched reference medians also
vary. These observations do not establish a useful hint effect.

[The complete record](cache-hint-probe.json) preserves all six reports, actual
samples, exact commands, trace metadata, correctness results, binary and CUBIN
hashes, build manifests, and the instruction-by-instruction comparison. Build
manifests identify compiled sources; report source snapshots describe files at
measurement time. Disassembly was run sequentially after discarding outputs
affected by a concurrent cuobjdump temporary-file collision.
