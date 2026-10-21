# FP16 comparison implementation

These Python sources are cuda-oxide code under **Apache-2.0**. The kernel
implements the example's FP16 matrix product, 256×352 output tile, two-CTA
schedule, and optional row bias using public CUTLASS APIs. The schedule also
follows the repository’s [Apache-2.0 Rust kernel](../src/kernel.rs). The installed
CUTLASS toolchain retains its
[separate license](https://docs.nvidia.com/cutlass/4.7.0/media/docs/pythonDSL/license.html).

## Public API references

The dependency is `nvidia-cutlass-dsl[cu13]==4.7.0` from PyPI:

- [Instruction primitives](https://docs.nvidia.com/cutlass/4.7.0/media/docs/pythonDSL/primitives.html): TMA, barriers, MMA, and TMEM.
- [CUDA utilities](https://docs.nvidia.com/cutlass/4.7.0/media/docs/pythonDSL/cuda.html): tensor-map construction.
- [CuTe runtime](https://docs.nvidia.com/cutlass/4.7.0/media/docs/pythonDSL/cute_dsl_api/cute_runtime.html): tensor views and compilation inputs.

The `experimental.primitives` and `experimental.cuda` namespaces are public
but experimental. Keep the version pin until a replacement passes the SM100
checks. Calling these APIs does not include their implementation in this tree.

The independent PyTorch oracle uses public
[`torch.bmm`](https://docs.pytorch.org/docs/stable/generated/torch.bmm.html)
for FP32 accumulation, adds row bias, and rounds once to FP16. It disables
TF32 and autocast and checks `atol=0.1`, `rtol=1e-5`.

## Local originals and validation

The sibling `../reference/` directory is ignored and reserved for existing
local kernels. Default comparisons import only this `public_reference`
package, so a fresh checkout has the same public implementation.

The [validation notes](README.md#validation-and-known-limitation) record passing
SM100 numerical checks and the unresolved Python memcheck diagnostic.
[Current performance results](performance.md) compare this public kernel with
both Rust implementations. Earlier timings of the local reference are not
measurements of this kernel.
