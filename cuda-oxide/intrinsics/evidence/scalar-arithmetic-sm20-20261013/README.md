# Scalar arithmetic floor verification (2026-10-13)

The 64 scalar arithmetic variants were previously assigned PTX 7.0 / sm_80
for both backends. That was the original probe target, not an instruction
requirement. The generator now separates a conservative family-wide native
contract (PTX 2.0 / sm_20) from LLVM's verified sm_20 / PTX 3.2 route and CUDA
13.3 libNVVM's verified compute_75 route. The compiler's existing recorded
target vocabulary is unchanged.

The native contract covers every variant, including preserved f32 subnormals,
directed division, and f32 FMA. The instruction requirements are documented
in the PTX ISA sections for [mul](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-mul),
[add](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add),
[div](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-div),
and [fma](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-fma).

`captures.json` records 192 successful LLVM codegen/assembly checks: all 64
variants at sm_20/+ptx32, sm_75/+ptx63, and sm_80/+ptx70. CUDA 13.3 no longer
assembles sm_20 cubins, so the sm_20 PTX is assembled for sm_75. This is an
instruction/lowering check, not a claim of sm_20 hardware execution.
`inputs.jsonl` preserves the exact probe text and the auxiliary typed
declarations. All 64 declarations were canonicalized with llvm-as/llvm-dis;
their required imported IR properties were checked. Canonical IR hashes can
change with input and output paths. The captures identify the actual tools
and artifacts, including the nightly-2026-10-11 LLVM 23.1.3 binary.

Run from `cuda-oxide/` with that nightly's LLVM tools and CUDA 13.3 ptxas:

```sh
python scripts/probe_scalar_llvm_floors.py --llc /path/to/llc \
  --ptxas /path/to/ptxas --out target/scalar-llvm-floors
```

The script restores either LF or CRLF to match every recorded input hash,
checks the exact retained opcode and absence of unresolved extern functions,
and assembles every PTX output. Its output ledger contains fresh tool and
artifact hashes. No driver or GPU is required.

The companion `../scalar-arithmetic-sm75-20261013/` capture uses libNVVM IR
2.0, validates all 64 exact inline-PTX operations for compute_75, assembles
the resulting PTX for sm_75, and compiles/links LTO IR with nvJitLink. Its
fixture uses `llvm.used` so optimization cannot discard the kernel entries.
Both cubins were checked for all 64 kernel text sections. libNVVM's
architecture query accepts compute_75/compute_80 and rejects compute_70 and
compute_72. The admitted libNVVM floor is therefore sm_75 / PTX 6.3; the
current toolkit emits PTX 9.3. Run with the actual CUDA 13.3 libraries:

```sh
python scripts/probe_scalar_nvvm_floors.py --libnvvm /path/to/libnvvm \
  --nvjitlink /path/to/libnvJitLink --ptxas /path/to/ptxas \
  --out target/scalar-nvvm-floors
```

On Windows these are `nvvm/bin/x64/nvvm64_40_0.dll`,
`bin/x64/nvJitLink_130_0.dll`, and `bin/ptxas.exe` in the CUDA toolkit.
The script checks the exact operations and both cubin entry sets, records
the library identities and IR/API versions, and leaves runtime validation
unexecuted. These captures do not claim execution of the 64 kernels on a GPU.

Regressions cover all 64 ABI markers through the generator and the compiler's
real target validation, including the distinct LLVM/libNVVM boundary.
Other instruction families and all 1029 existing ABI identities are unchanged.
