# manual_launch_libdevice

Manual low-level launch regression for NVVM IR/libdevice artifacts.

This example intentionally uses `load_kernel_module` and the unsafe
`cuda_launch!` macro (wrapped in `unsafe { }` with a SAFETY comment) instead
of `#[cuda_module]`. It keeps the explicit sidecar-loading API covered for the
NVVM IR path while the primary examples use typed embedded modules.

Run it with:

```bash
cargo oxide run manual_launch_libdevice --emit-nvvm-ir --arch sm_120
```
