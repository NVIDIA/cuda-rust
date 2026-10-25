# External device LTO-IR

This example calls a CUDA C++ device function from a Rust kernel. The C++
function calls another function in a second LTO-IR file. `cargo oxide` links
both files into the embedded cubin during the build, and the program uses the
ordinary `kernels::load` API. It checks `output[i] = 2 * input[i] + 3` on the GPU.

From `cuda-oxide/`, with the CUDA Toolkit and a matching GPU available:

```bash
example=crates/rustc-codegen-cuda/examples/external_ltoir
mkdir -p build/device-libs
nvcc --ltoir -dlto -arch=sm_120 \
  "$example/scale.cu" -o build/device-libs/scale.ltoir
nvcc --ltoir -dlto -arch=sm_120 \
  "$example/add.cu" -o build/device-libs/add.ltoir
cargo oxide run external_ltoir --arch sm_120 --materialize-cubin \
  --link-ltoir build/device-libs/scale.ltoir \
  --link-ltoir build/device-libs/add.ltoir
```

Use the same architecture for both compilers. Each `--link-ltoir` adds one
file in argument order; paths are relative to the command's working directory.
The inputs must be raw LTO-IR, not object files or fatbins. `--ltoir -dlto`
writes the explicit output path, including with architecture names such as
`sm_90a`. All supplied files are linked into each materialized device artifact
in the Cargo invocation. The executable contains the finished cubin and needs no external `.ltoir` files
or runtime link step.

`scripts/check-external-ltoir.sh sm_120` exercises run, build and test, repeated
build freshness, content changes at the same path, a missing transitive symbol,
execution with the link inputs removed, corrupt LTO-IR, and rejection of an
input/output collision without changing either file. It prints the
temporary directory containing its logs and generated inputs; it uses the
same nvcc commands with those temporary output paths. The check requires a GPU
matching the requested architecture.
