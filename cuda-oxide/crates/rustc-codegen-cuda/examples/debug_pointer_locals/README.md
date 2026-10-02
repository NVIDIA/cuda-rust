# debug_pointer_locals

This fixture checks three pointer locals:

```text
ptr      -> 41
fptr     -> 13.0
null_ptr -> null
```

The live pointers use distinct nonzero addresses. The null control catches
missing debug storage that merely reads as zero.

From the cuda-oxide repository root:

```bash
CUDA_OXIDE_DEBUG=full \
  cargo oxide build debug_pointer_locals
crates/rustc-codegen-cuda/examples/debug_pointer_locals/verify-debug-info.sh
scripts/debug-smoketest.sh debug_pointer_locals
```

`cargo oxide debug` detects the local GPU architecture. If detection is not
available, append an explicit architecture such as `--arch sm_120a`.

For an interactive session,
`CUDA_OXIDE_DEBUG=full cargo oxide debug debug_pointer_locals` opens cuda-gdb
with the full-debug MIR policy. The environment assignment on the earlier
build command applies only to that command. Use the marker rather than a
hard-coded source line:

```gdb
set pagination off
break src/main.rs:<CUDA_OXIDE_DEBUG_POINTER_BREAKPOINT line>
run
backtrace
frame 0
info args
info locals
print ptr
print fptr
print *ptr
print *fptr
print null_ptr
whatis ptr
whatis fptr
continue
```

For CUDA thread 0, healthy debugger values are:

```text
input != 0x0
fdata != 0x0
out.ptr != 0x0
ptr == input
fptr == fdata
*ptr == 41
*fptr == 13.0
null_ptr == 0x0
```

The regression is present when the kernel arguments remain non-null but
cuda-gdb reports `ptr = 0x0` and `fptr = 0x0`.

After `ReferencePropagation`, rustc associates `ptr` and `fptr` with the
surviving `&i32` and `&f32` MIR locals. The structural test pins those types
and proves their debugger slots are written. CUDA-GDB may display NVPTX
pointers as `*mut`, so the live test pins the pointee types and values rather
than the mutability spelling.
