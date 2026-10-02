# Legacy NVVM pointer shapes

This example checks pointer handling in LLVM 7 NVVM IR.

cuda-oxide uses opaque pointers inside the compiler. Older libNVVM targets need
typed pointers in the final LLVM text:

```text
internal `ptr`
     │
     ▼
neutral LLVM 7 `i8*`
     │
     ├──► `float*` when reading an f32
     └──► `i32*`   when reading its bits
```

The kernel covers:

- pointers chosen by branches;
- a pointer carried through a loop;
- pointers passed to and returned from functions;
- pointers inside nested structs;
- an array of pointers;
- one address read as both `f32` and `u32`.

Run it normally on any supported GPU:

```bash
cargo oxide run legacy_nvvm_pointer_shapes
```

To verify and compile the LLVM 7 path on any machine, choose a pre-Blackwell
target:

```bash
cargo oxide emit-ltoir legacy_nvvm_pointer_shapes --arch sm_86 \
  --output /tmp/legacy_nvvm_pointer_shapes-sm86.ltoir
```

On Blackwell, the same legacy target can also run through the forward PTX
compatibility path:

```bash
cargo oxide run legacy_nvvm_pointer_shapes --arch sm_86
```

This path asks the installed CUDA driver to JIT PTX produced by the selected
toolkit. The driver must support that toolkit's PTX version. If several CUDA
toolkits are installed, select a compatible one explicitly:

```bash
CUDA_TOOLKIT_PATH=/usr/local/cuda-13.0 \
  cargo oxide run legacy_nvvm_pointer_shapes --arch sm_86
```

CUDA error 222 means the selected toolkit produced a newer PTX version than
the installed driver can JIT. Use an older compatible toolkit or upgrade the
driver.
