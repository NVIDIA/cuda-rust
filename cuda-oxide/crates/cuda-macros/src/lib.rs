/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Procedural macros for CUDA kernel development: `#[kernel]`, `#[device]`,
//! `#[cuda_module]`, `gpu_printf!`, `ptx_asm!`, and friends.
//!
//! # The `host` feature in mixed build graphs
//!
//! The default-on `host` cargo feature makes `#[kernel]` and `#[cuda_module]`
//! emit the generated host surface (the `LoadedModule` loader and launchers,
//! the `CudaKernel` marker impls), all of which names `::cuda_host` /
//! `::cuda_core`. A crate that only compiles kernels takes this crate with
//! `default-features = false` and drops the host dependency stack.
//!
//! Proc-macro features unify globally per build graph. If any crate in the
//! graph enables `cuda-macros/host`, every crate expanding these macros gets
//! the host-emitting expansion, including a device-only kernel crate; its
//! expansion then names `cuda_host`, which it cannot resolve (E0433). A
//! device-only kernel crate consumed by a host application must therefore
//! forward the feature itself:
//!
//! ```toml
//! [features]
//! host = ["dep:cuda-host", "cuda-macros/host"]
//! ```
//!
//! so the same switch that turns host emission on also adds the `cuda-host`
//! dependency that resolves it.

#![feature(proc_macro_def_site, proc_macro_tracked_env)]

mod common;
mod cuda_module;
mod device;
mod kernel;
mod launch;
mod launch_attrs;
mod printf;
mod ptx_asm;

#[cfg(test)]
mod tests;

use proc_macro::TokenStream;

use crate::common::track_codegen_environment;
use crate::launch::{
    CudaLaunchAsyncInput, CudaLaunchInput, expand_cuda_launch, expand_cuda_launch_async,
};
use syn::parse_macro_input;

/// GPU printf macro for formatted output from GPU kernels.
///
/// This macro translates Rust-style format strings to C-style and calls
/// CUDA's `vprintf` function.
///
/// # Usage
///
/// ```ignore
/// use cuda_device::gpu_printf;
///
/// #[kernel]
/// fn my_kernel() {
///     let tid = thread::index_1d().get();
///     gpu_printf!("Thread {}: Hello from GPU!\n", tid);
/// }
/// ```
///
/// # Format Specifiers
///
/// | Specifier | Description     | Example                                        |
/// |-----------|-----------------|------------------------------------------------|
/// | `{}`      | Default format  | `gpu_printf!("{}", 42)`                        |
/// | `{:x}`    | Hex (lower)     | `gpu_printf!("{:x}", 255)` → "ff"              |
/// | `{:X}`    | Hex (upper)     | `gpu_printf!("{:X}", 255)` → "FF"              |
/// | `{:#x}`   | Hex with prefix | `gpu_printf!("{:#x}", 255)` → "0xff"           |
/// | `{:o}`    | Octal           | `gpu_printf!("{:o}", 8)` → "10"                |
/// | `{:e}`    | Scientific      | `gpu_printf!("{:e}", 1000.0)` → "1.000000e+03" |
/// | `{:.N}`   | Precision       | `gpu_printf!("{:.2}", 3.14159)` → "3.14"       |
/// | `{:N}`    | Width           | `gpu_printf!("{:8}", 42)` → "      42"         |
/// | `{:0N}`   | Zero-pad        | `gpu_printf!("{:08}", 42)` → "00000042"        |
///
/// # Returns
///
/// The number of arguments (i32), or negative on error.
/// Note: CUDA vprintf returns arg count, not character count.
#[proc_macro]
pub fn gpu_printf(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as printf::GpuPrintfInput);
    printf::gpu_printf_impl(input).into()
}

/// Inline PTX assembly for CUDA device code.
///
/// The macro accepts the `%0` operand placeholders used by CUDA inline PTX,
/// plus `in`, `out`, and `inout` operands with PTX constraint strings.
/// Supported register constraints are `"h"`, `"r"`, `"l"`, `"q"`, `"f"`, and
/// `"d"`, plus `"n"` for immediate integer inputs:
///
/// ```ignore
/// let y: u32;
/// unsafe {
///     ptx_asm!(
///         "add.u32 %0, %1, %2;",
///         out("=r") y,
///         in("r") x,
///         in("r") z,
///     );
/// }
/// ```
///
/// Read-write operands use `inout` with a `+`-prefixed register constraint.
/// The current Rust value initializes the PTX output register and the final
/// register value is written back to the same place:
///
/// ```ignore
/// let mut accumulator = initial;
/// unsafe {
///     ptx_asm!(
///         "add.u32 %0, %0, %1;",
///         inout("+r") accumulator,
///         in("r") increment,
///         options(register_only),
///     );
/// }
/// ```
///
/// Literal PTX registers that begin with `%` must be escaped as `%%`, matching
/// CUDA C++ inline PTX. Literal `$` labels can be written normally.
///
/// The surface supports up to 64 output operands across `out` and `inout`,
/// up to 16 explicit `in` operands, `clobber("memory")`,
/// `options(register_only)`, and the explicit
/// `options(register_only, may_diverge)` opt-in. `out` constraints use an `=`
/// prefix, such as `"=r"`, while `inout` constraints use a `+` prefix, such as
/// `"+r"`. All output operands must appear before explicit inputs.
///
/// With two or more output operands, including any mixture of `out` and
/// `inout`, the marker returns a tuple under the hood and the macro writes its
/// elements back to the output places in declaration order:
///
/// ```ignore
/// let sum: u32;
/// let prod: u32;
/// unsafe {
///     ptx_asm!(
///         "add.u32 %0, %2, %3; mul.lo.u32 %1, %2, %3;",
///         out("=r") sum,
///         out("=r") prod,
///         in("r") x,
///         in("r") y,
///     );
/// }
/// ```
///
/// By default, snippets are treated as side-effecting and stay inside their
/// current control flow. Use `options(register_only)` only for snippets that
/// read explicit operands and write explicit outputs. **Never** use
/// `may_diverge` for `.sync` instructions, collectives, or any snippet whose
/// participating lanes matter.
#[proc_macro]
pub fn ptx_asm(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as ptx_asm::PtxAsmInput);
    ptx_asm::ptx_asm_impl(input).into()
}

/// Generates a typed host-side loader and launch surface for the kernels in an
/// inline module.
///
/// The generated API loads the current crate's embedded artifact bundle and
/// exposes synchronous methods per `#[kernel]` function. When `cuda-host` is
/// built with its `async` feature, the macro also emits borrowed async and owned
/// async methods. Kernel parameter types are mapped to host-side launch types:
///
/// - `&[T]` -> `&cuda_core::DeviceBuffer<T>`
/// - `&mut [T]` -> `&mut cuda_core::DeviceBuffer<T>`
/// - `DisjointSlice<T>` -> `&mut cuda_core::DeviceBuffer<T>`
/// - `#[grid_constant] value: &T` -> a by-value `T` (`T: Copy`); device code
///   borrows read-only launch-parameter storage, without a device allocation
/// - `Copy` scalar/struct/closure/raw-pointer arguments keep their original
///   type and pass through `cuda_host::KernelScalar`
///
/// # Nested modules
///
/// Kernels may be organized in inline modules nested inside the annotated
/// module. Each namespace gets its own `LoadedModule` view, so launcher
/// signatures resolve types, private generated helpers, and raw module
/// identifiers beside the source kernel:
///
/// ```text
/// kernels::LoadedModule
///     -> kernels::stage::LoadedModule::from_parent(&kernels)
///         -> stage.step(...)
/// ```
///
/// All views share one loaded CUDA module and one generic-function cache.
/// Create deeper views from their immediate parent in the same way.
/// The name `LoadedModule` is therefore reserved in every inline namespace
/// that owns kernels or contains a deeper kernel namespace.
/// The generated method name `as_cuda_module` is reserved in every kernel
/// namespace, and `from_parent` is additionally reserved in nested namespaces.
///
/// Procedural macros cannot see the contents of `mod child;` or `include!`.
/// Those items are preserved, but kernels behind either boundary do not get
/// generated launchers. Keep auto-launched nested kernels in inline modules.
///
/// Launcher methods are namespace-qualified, but PTX entry symbols are still
/// bare function names. Kernel names must therefore be unique throughout one
/// `#[cuda_module]` tree, including cfg-gated alternatives; the macro rejects
/// duplicates rather than risk loading the wrong entry.
/// Raw module identifiers are supported. Raw kernel function identifiers are
/// not: their PTX-name contract predates nested-module support and remains an
/// unsupported edge for a future backend-wide naming change.
///
/// # Example
///
/// ```ignore
/// #[cuda_module]
/// mod kernels {
///     use super::*;
///
///     #[kernel]
///     pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
///         // ...
///     }
/// }
///
/// let module = kernels::load(&ctx)?;
/// // SAFETY: the raw launch is fully 1-D and matches vecadd's resources.
/// unsafe {
///     module.vecadd(&stream, LaunchConfig::for_num_elems(n), &a, &b, &mut c)?;
/// }
///
/// let module = kernels::load_async(0)?;
/// // SAFETY: the raw launch is fully 1-D and matches vecadd's resources.
/// unsafe {
///     module.vecadd_async(LaunchConfig::for_num_elems(n), &a, &b, &mut c)
/// }?.sync()?;
///
/// // SAFETY: the raw launch is fully 1-D and matches vecadd's resources.
/// let (a, b, c) = unsafe {
///     module.vecadd_async_owned(LaunchConfig::for_num_elems(n), a, b, c)
/// }?
///     .await?;
/// ```
///
/// # Raw launch safety
///
/// A raw [`LaunchConfig`](https://docs.rs/cuda-core/latest/cuda_core/simt/launch/struct.LaunchConfig.html)
/// is not tied to the kernel's indexing model. The generated raw synchronous,
/// borrowed-async, and owned-async methods are therefore `unsafe`: callers must
/// prove that the chosen dimensions and resources satisfy the kernel. Add
/// `#[launch_contract(...)]` to generate a prepared-launch path that is safe
/// when the source kernel itself is safe and has no `#[grid_constant]`
/// parameters. Grid-constant host launchers remain unsafe because copying the
/// parameter does not prove device validity of values reached through it.
#[proc_macro_attribute]
pub fn cuda_module(attr: TokenStream, item: TokenStream) -> TokenStream {
    cuda_module::cuda_module_entry(attr, item)
}

/// Marks a function as a CUDA kernel.
///
/// This attribute:
/// 1. Adds `#[no_mangle]` to preserve the function name in the binary
/// 2. Marks the function for detection by the `rustc-codegen-cuda` backend
///
/// # Grid-constant parameters
///
/// `#[grid_constant] descriptor: &T` passes the complete `T` by value in the
/// launch parameters and gives device code a reference to that read-only
/// storage. `T` must be sized, nonzero, and have no interior mutability
/// (`UnsafeCell`, including cells or atomics stored within the value). The
/// compiler checks this after generic specialization. References or raw
/// pointers stored inside `T` still refer to separate allocations; those
/// allocations must satisfy the usual device-access and lifetime requirements.
///
/// Use `&T` or `&'_ T`: the storage lasts only until this launch completes, so
/// named and `'static` outer lifetimes are rejected. A pointer into the value
/// must not be used after the launch. `#[cuda_module]` generates a host
/// argument of type `T` and requires `T: Copy`; low-level launches must likewise
/// pass the value's bytes, rather than a pointer to a device allocation.
///
/// Every generated host launcher with a grid-constant parameter is `unsafe`,
/// including prepared synchronous, borrowed-async, and owned-async methods.
/// `Copy` permits copying the bytes, and the compiler's immutability check
/// protects the copied storage; neither proves that nested references are
/// valid on the GPU. The caller must ensure that references within the value
/// are device-valid and that memory reached through it remains accessible,
/// alive, and correctly synchronized until device execution completes.
/// Submitting a launch or constructing an async operation does not end these
/// obligations, and copying the value does not retain separate allocations.
/// Prepared launch checks cover geometry and declared resources, not these
/// nested-allocation requirements. This conservative rule also applies to
/// payloads containing only numbers. It does not change the source kernel's
/// Rust safety or the ordinary signature of a generic device helper.
///
/// ```ignore
/// #[kernel]
/// pub fn copy(#[grid_constant] descriptor: &TensorMap, output: *mut u32) {
///     // descriptor borrows one read-only value shared by the entire grid.
/// }
/// ```
///
/// # Generic Kernels
///
/// For generic kernels (like `template<class F> __global__` in CUDA C++),
/// specify the types to instantiate:
///
/// ```ignore
/// #[kernel(Scale, Fma, Square)]
/// pub fn map<F: GpuFn>(f: F, input: &[i32], output: DisjointSlice<i32>) {
///     // ...
/// }
/// ```
///
/// This generates three PTX entry points: `map_Scale`, `map_Fma`, `map_Square`.
/// Each is a monomorphized version of the generic kernel.
///
/// # Example (non-generic)
///
/// ```ignore
/// #[kernel]
/// pub fn simple_kernel(data: &mut [i32]) {
///     // ...
/// }
/// ```
///
/// Kernels that use contract-backed fast coordinates name their launch
/// capability explicitly. It is a local device-only binding, not an ABI
/// parameter:
///
/// ```ignore
/// #[kernel(launch_context = launch_context)]
/// #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
/// pub fn map(mut output: DisjointSlice<u32>) {
///     let index = thread::index_1d_u32(launch_context);
///     // ...
/// }
/// ```
///
/// # Unchecked indexing (opt-in, trust-me contract)
///
/// `#[kernel(unchecked_indexing)]` tells the compiler to **delete the safety
/// check behind slice/array indexing** (`a[i]`) in this kernel. Normally
/// every `a[i]` compiles to "is `i` inside the slice? if not, stop the
/// kernel"; with this flag the access happens unconditionally. You are
/// promising that every index is in bounds, the same promise as
/// [`slice::get_unchecked`]. If the promise is wrong, the result is
/// **undefined behavior**: no guard, no trap, possibly silent corruption of
/// unrelated data. Turn it on only after the checked build is proven correct
/// (for example under `compute-sanitizer`).
///
/// ```ignore
/// #[kernel(unchecked_indexing)]
/// pub fn hot_loop(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
///     let idx = thread::index_1d().get();
///     if let Some(out) = c.get_mut(thread::index_1d()) {
///         *out = a[idx] + b[idx]; // no bounds guard or trap emitted
///     }
/// }
/// ```
///
/// The flag composes with the other kernel arguments, e.g.
/// `#[kernel(f32, unchecked_indexing)]` or
/// `#[kernel(launch_context = lc, unchecked_indexing)]`.
///
/// The bare word `unchecked_indexing` is **reserved** in the `#[kernel]`
/// argument list: it always parses as this flag, never as a legacy
/// instantiation type. A user type literally named `unchecked_indexing` can
/// still be instantiated by spelling it as a path, e.g.
/// `#[kernel(self::unchecked_indexing)]` or
/// `#[kernel(crate::unchecked_indexing)]`, which parses as a type.
///
/// Scope and caveats:
///
/// - Only the checks behind indexing (`a[i]`) are removed. Arithmetic
///   overflow, division/remainder by zero, misaligned-pointer checks, and
///   every other safety check keep their normal compare-and-trap behavior.
/// - Range-indexing failures (`&a[i..j]`) and explicit panics arrive as calls
///   to `core::panicking::*`, not as MIR asserts, so they still trap.
/// - The flag covers the kernel's translated MIR body, including everything
///   rustc MIR-inlined into it. Separately translated `#[device]` functions
///   are **not** covered; the whole-build switch
///   `CUDA_OXIDE_UNCHECKED_INDEXING=1` (or `cargo oxide ...
///   --unchecked-indexing`) covers every translated body.
/// - Elision never leaks out of the opted kernel. For generic kernels
///   (including legacy `#[kernel(T, ...)]` instantiation), the expansion
///   keeps the elision marker on the generated entry function plus a hidden
///   unchecked twin of the implementation that only that entry calls. The
///   user-named implementation function stays ordinary bounds-checked Rust,
///   so a different kernel or `#[device]` function that calls it keeps its
///   own bounds checks (fail-closed).
///
/// # Loop unrolling
///
/// Put `#[unroll]` on a loop with a compile-time-known trip count to request full
/// unrolling. Use `#[unroll(N)]`, where `N >= 2`, to request `N` iterations of
/// work per trip; a remainder loop handles any leftovers.
///
/// ```ignore
/// #[kernel]
/// pub fn example(n: u32) {
///     let mut i = 0;
///     #[unroll]
///     while i < 4 { work(i); i += 1; }
///
///     let mut j = 0;
///     #[unroll(4)]
///     while j < n { work(j); j += 1; }
/// }
/// ```
///
/// A factor may also be a typed `u32` policy constant, such as
/// `#[unroll(P::UNROLL)]`. A generic expression currently requires Rust's
/// `generic_const_exprs` feature. Partial factors must be in `2..=1024`; an
/// invalid specialization fails compilation instead of becoming a no-op.
///
/// The pass currently recognizes explicit counted `while` loops. Range-based
/// `for` loops are not yet recognized.
///
/// Only the annotated loop is unrolled. Inner loops are copied intact unless
/// they carry their own annotation. Several `continue` paths (multiple
/// back-edges) are supported.
///
/// Full `#[unroll]` preserves `break` paths and multiple exit targets. Partial
/// `#[unroll(N)]` currently requires the loop condition to be the only exit. If
/// the loop has a `break` or another extra exit, the compiler warns and skips
/// partial unrolling for that loop.
///
/// Partial unrolling also requires a positive counter step, a `<` or `<=` test,
/// and a limit that does not change inside the loop. One annotation may create
/// at most 1,024 body copies, 8,192 cloned basic blocks, and 65,536 cloned
/// operations. Factors above 1,024 are rejected; unsupported loop shapes warn
/// and are not unrolled.
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    kernel::kernel_entry(attr, item)
}

/// Mark a module-scope `static` as a CUDA constant-memory global.
///
/// The static must be typed `ConstantMemory<T>` (see
/// [`cuda_device::ConstantMemory`](../../cuda_device/struct.ConstantMemory.html)). The
/// `ConstantMemory<T>` wrapper has `UnsafeCell<T>` semantics on the device,
/// preventing the compiler from constant-folding the initializer and making
/// the host's `cuMemcpyHtoD` updates observable from kernels.
///
/// The macro adds a reserved `#[unsafe(export_name = "cuda_oxide_const_246e25db_...")]`
/// so the PTX symbol carries a name the host can resolve via
/// `cuModuleGetGlobal`. When used inside `#[cuda_module]`, the generated name
/// includes module/source-location context to avoid collisions between constants
/// that share the same Rust identifier. The host-side `#[cuda_module]` expansion
/// separately generates per-constant setter methods on the loaded module:
///
/// - `module.set_<name>(&stream, &value)` — stream-ordered async write
///   (recommended; orders correctly against surrounding kernel launches).
/// - `module.set_<name>_blocking(&value)` — synchronous `cuMemcpyHtoD`
///   for one-shot initialization where no stream is in scope.
///
/// # Restrictions
///
/// - The static must be typed `ConstantMemory<T>`.
/// - The initializer must be `ConstantMemory::UNINIT` (or any other all-zeros
///   value). Honoring arbitrary non-zero initializers is not yet
///   implemented; populate from the host before any kernel reads the value.
/// - The attribute must appear inside a `#[cuda_module]`. Placed elsewhere
///   it silently produces an unreachable symbol (no setter is generated).
/// - The identifier must not start with the reserved cuda-oxide prefix.
///
/// # Example
///
/// ```ignore
/// #[cuda_module]
/// mod kernels {
///     #[constant]
///     static COEFFS: ConstantMemory<[f32; 4]> = ConstantMemory::UNINIT;
///
///     #[kernel]
///     pub fn compute(mut out: DisjointSlice<f32>) {
///         let c = COEFFS.get();
///         let i = thread::index_1d().get();
///         if let Some(e) = out.get_mut(thread::index_1d()) {
///             *e = c[0] * (i as f32) + c[1];
///         }
///     }
/// }
///
/// // Host
/// module.set_coeffs(&stream, &[1.0, 2.0, 3.0, 4.0])?;
/// ```
#[proc_macro_attribute]
pub fn constant(attr: TokenStream, item: TokenStream) -> TokenStream {
    device::constant_entry(attr, item)
}

/// Specifies launch bounds for a kernel (max threads per block, min blocks per SM).
///
/// This attribute sets kernel launch bounds at compile time by emitting `.maxntid`
/// and `.minnctapersm` PTX directives. This helps the CUDA compiler optimize
/// register allocation and occupancy.
///
/// `.maxntid` bounds the product `x * y * z`, so a 256-thread maximum admits
/// `(256, 1, 1)`, `(16, 16, 1)` and `(4, 8, 8)` alike. Use
/// `#[launch_contract(block = (x, y, z))]` to require one exact shape; that
/// emits `.reqntid` instead, which the driver enforces per axis. A kernel
/// carrying both attributes emits `.reqntid` alone, because ptxas rejects an
/// entry declaring both directives. `.minnctapersm` composes with either.
///
/// # Usage
///
/// ```ignore
/// use cuda_device::{kernel, launch_bounds, DisjointSlice};
///
/// #[kernel]
/// #[launch_bounds(256)]              // Max 256 threads per block
/// pub fn simple_kernel(output: DisjointSlice<f32>) { ... }
///
/// #[kernel]
/// #[launch_bounds(256, 2)]           // Max 256 threads, min 2 blocks per SM
/// pub fn optimized_kernel(output: DisjointSlice<f32>) { ... }
///
/// trait Policy {
///     const MAX_THREADS: u32;
///     const MIN_BLOCKS: u32;
/// }
///
/// #[kernel]
/// #[launch_bounds(P::MAX_THREADS, P::MIN_BLOCKS)]
/// pub fn configured<P: Policy>(output: DisjointSlice<f32>) { ... }
/// ```
///
/// # Parameters
///
/// - First parameter (required): Maximum threads per block
/// - Second parameter (optional): Minimum blocks per SM for occupancy hints
/// - Both parameters may be typed `u32` const expressions. Expressions that
///   depend on a generic parameter currently require Rust's
///   `generic_const_exprs` feature.
///
/// The first value is a maximum, not an exact launch shape. When a kernel also
/// has `#[launch_contract(domain = ...)]`, each policy specialization carries
/// its evaluated maximum into the host contract. Preparation rejects a larger
/// block before the CUDA launch call.
///
/// # Requirements
///
/// - Must be used WITH `#[kernel]` (not standalone)
/// - May appear before or after `#[kernel]`; generic entry generation forwards
///   an already-expanded compiler marker to each entry wrapper
///
/// # Performance Impact
///
/// Launch bounds help the compiler:
/// - Allocate registers more efficiently
/// - Optimize occupancy (threads per SM)
/// - Make better scheduling decisions
///
/// # PTX Output
///
/// ```ptx
/// .entry my_kernel .maxntid 256 .minnctapersm 2 { ... }
/// ```
#[proc_macro_attribute]
pub fn launch_bounds(attr: TokenStream, item: TokenStream) -> TokenStream {
    launch_attrs::launch_bounds_entry(attr, item)
}

/// Declares the host launch geometry and resource contract for a kernel.
///
/// `#[cuda_module]` uses this opt-in declaration to generate a prepared,
/// kernel-branded launch path. A prepared launch can be reused without
/// repeating CUDA capability and function-resource queries, while raw
/// [`LaunchConfig`](https://docs.rs/cuda-core/latest/cuda_core/simt/launch/struct.LaunchConfig.html)
/// remains available through an explicitly unsafe generated method.
///
/// ```ignore
/// use cuda_device::{kernel, launch_bounds, launch_contract, DisjointSlice};
///
/// #[kernel(launch_context = launch_context)]
/// #[launch_bounds(256)]
/// #[launch_contract(
///     domain = 1,
///     coordinates = u32,
///     block = (256, 1, 1),
///     dynamic_shared = 0,
///     min_compute_capability = (8, 0),
/// )]
/// pub fn map(mut output: DisjointSlice<f32>) {
///     let index = thread::index_1d_u32(launch_context);
///     // use `index` with a proof-carrying view...
/// }
/// ```
///
/// `domain` is an author declaration, not body inference: helper calls can
/// hide which hardware indices a kernel reads. For obvious mismatches,
/// `#[cuda_module]` cross-checks the declaration against `DisjointSlice` index
/// spaces and the fixed cluster shape.
///
/// `coordinates = u32` opts into narrow, proof-carrying coordinate APIs such
/// as `thread::index_1d_u32(launch_context)`. The generated prepared launch checks each
/// axis:
///
/// ```text
/// grid_axis * block_axis <= 2^32
/// ```
///
/// This keeps zero-based global coordinates representable by `u32`. Generated
/// raw launch methods remain unsafe because their caller must uphold this and
/// the rest of the declared contract without preparation.
///
/// Dynamic shared memory may be fixed with `dynamic_shared = BYTES` or bounded
/// with `dynamic_shared_range = (MIN, MAX)`. The byte extent remains an author
/// contract because arbitrary pointer arithmetic cannot be inferred. When the
/// maximum is non-zero, the macro injects a compiler marker whose call is
/// removed before code generation. The declared `dynamic_shared_alignment`
/// therefore becomes a minimum alignment in generated PTX without adding
/// kernel hot-path instructions.
///
/// A `#[launch_bounds(P::MAX_THREADS, ...)]` maximum may depend on a generic
/// policy. The generated host contract evaluates it separately for each
/// specialization. An explicit `block = (x, y, z)` remains exact and must fit
/// within every policy maximum used with that specialization.
/// A non-policy constant used for that maximum must be visible at module scope,
/// because the host contract is generated beside the kernel function.
///
/// # Size requirements: `requires`
///
/// `requires = (relation, ...)` writes down, next to the kernel, the size
/// relationships its buffers and scalars must satisfy for every access to be
/// in bounds. The generated launcher checks them once per launch, on the
/// CPU, and refuses to launch (with a typed error naming the failed
/// relation) instead of letting an undersized buffer fault mid-kernel:
///
/// ```ignore
/// #[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1),
///     requires = (a.len() >= m * k, b.len() >= k * n, c.len() >= m * n))]
/// ```
///
/// Each relation is one comparison (`>=`, `>`, `<=`, `<`, `==`, `!=`) between
/// expressions built from slice or `DisjointSlice` parameters as
/// `<param>.len()`, unsigned integer scalar parameters (`u8`/`u16`/`u32`/
/// `u64`/`usize`) used directly, integer literals, parentheses, and the
/// arithmetic operators `+`, `-`, `*`. Comma-separated relations are
/// implicitly ANDed; each is checked separately with its own error.
///
/// The grammar is deliberately tiny rather than arbitrary Rust, for two
/// reasons. Every identifier is validated against the kernel's actual
/// parameter list, so a typo is a compile error instead of a check against
/// the wrong value. And every `+`/`-`/`*` compiles to checked `u64`
/// arithmetic, so a huge `m * k` cannot wrap around to a small number and
/// falsely pass.
///
/// Every checked launcher generated by `#[cuda_module]` (the sync prepared
/// launcher and, with the `async` feature, the `_async` and `_async_owned`
/// twins) evaluates every relation once at launch time, before any argument
/// is handed to the driver. Evaluation widens every operand to `u64` and uses checked
/// arithmetic: a false relation fails with
/// `LaunchContractError::SizeRequirementViolated` (carrying the relation's source
/// text and both evaluated sides) and arithmetic leaving the `u64` range
/// fails with `LaunchContractError::SizeRequirementOverflow`, so a bad launch is
/// rejected on the CPU instead of trapping mid-kernel. The generated
/// `_unchecked` escape hatches intentionally skip these checks, exactly as
/// they skip the geometry checks; their safety contract passes the
/// obligation to the caller.
///
/// The relations are **enforced only by `#[cuda_module]`-generated
/// launchers**. On a kernel outside any `#[cuda_module]`, this attribute
/// still validates every relation for well-formedness at compile time
/// (unknown identifiers and unsupported grammar are errors at the attribute
/// site), but no launcher exists to evaluate the relations at runtime. That
/// is the same enforcement story as `dynamic_shared`.
///
/// An exact `block` is the one key that also holds without a generated
/// launcher. The shape reaches the device compiler as `.reqntid x, y, z`, and
/// the CUDA driver refuses any launch whose block differs on any axis, so a
/// standalone contracted kernel and an `_unchecked` raw launch are both
/// covered. `.reqntid` and `.maxntid` cannot appear on one entry, so a kernel
/// declaring an exact `block` emits `.reqntid` in place of the maximum that
/// `#[launch_bounds]` would otherwise contribute.
#[proc_macro_attribute]
pub fn launch_contract(attr: TokenStream, item: TokenStream) -> TokenStream {
    launch_attrs::launch_contract_entry(attr, item)
}

/// Specifies compile-time cluster dimensions for a kernel.
///
/// This attribute sets the thread block cluster size at compile time by emitting
/// the `.reqnctapercluster` PTX directive. When used, the kernel will automatically
/// launch with the specified cluster configuration.
///
/// Note: Named `cluster_launch` (not `cluster`) to avoid conflict with `cuda_device::cluster` module.
///
/// # Usage
///
/// ```ignore
/// use cuda_device::{kernel, cluster, cluster_launch, DisjointSlice};
///
/// #[kernel]
/// #[cluster_launch(4, 1, 1)]  // 4 blocks per cluster in X dimension
/// pub fn my_cluster_kernel(output: DisjointSlice<u32>) {
///     let rank = cluster::block_rank();
///     // ...
/// }
/// ```
///
/// # Cluster Dimensions
///
/// - `#[cluster_launch(n)]` - 1D cluster with n blocks
/// - `#[cluster_launch(x, y)]` - 2D cluster with x*y blocks
/// - `#[cluster_launch(x, y, z)]` - 3D cluster with x*y*z blocks
///
/// Maximum cluster size is typically 16 blocks (hardware dependent).
///
/// # Requirements
///
/// - Must be used WITH `#[kernel]` (not standalone)
/// - Requires sm_90+ (Hopper) or newer GPU
/// - May appear before or after `#[kernel]`; generic entry generation forwards
///   an already-expanded compiler marker to each entry wrapper
///
/// # How It Works
///
/// The macro injects `cuda_device::cluster::__cluster_config::<X, Y, Z>()` at the
/// start of the kernel. The compiler:
/// 1. Detects this marker during MIR translation
/// 2. Extracts the const generic parameters (X, Y, Z)
/// 3. Emits `!nvvm.annotations` metadata with cluster dimensions
/// 4. LLVM NVPTX backend generates `.reqnctapercluster X, Y, Z` in PTX
///
/// # PTX Output
///
/// ```ptx
/// .entry my_cluster_kernel .reqnctapercluster 4, 1, 1 { ... }
/// ```
///
/// # Compile-Time vs Runtime Clusters
///
/// | Method | Pros | Cons |
/// |--------|------|------|
/// | `#[cluster_launch(x,y,z)]` (compile-time) | Simple, no special launch API | Fixed at compile time |
/// | `cuLaunchKernelEx` (runtime) | Dynamic cluster sizes | Requires FFI, complex setup |
#[proc_macro_attribute]
pub fn cluster_launch(attr: TokenStream, item: TokenStream) -> TokenStream {
    launch_attrs::cluster_launch_entry(attr, item)
}

/// Marks a kernel for cooperative launch (`CU_LAUNCH_ATTRIBUTE_COOPERATIVE`).
///
/// A cooperative launch guarantees that every block in the grid is
/// co-resident on the device, which is the precondition for grid-wide
/// barriers: without it, `cuda_device::grid::sync()` deadlocks (or reads a
/// null grid-workspace pointer) because blocks that have not been scheduled
/// yet can never reach the barrier.
///
/// Unlike `#[cluster_launch]`, this attribute changes nothing in the
/// generated PTX. Cooperative-ness is purely a launch-time property: the
/// `#[cuda_module]` macro reads this marker and routes every generated
/// launch method through `cuLaunchKernelEx` with the cooperative attribute
/// set, instead of plain `cuLaunchKernel`.
///
/// # Usage
///
/// ```ignore
/// use cuda_device::{cooperative_launch, grid, kernel, DisjointSlice};
///
/// #[kernel]
/// #[cooperative_launch]
/// pub fn my_grid_sync_kernel(mut out: DisjointSlice<u32>) {
///     // ... per-block work ...
///     grid::sync();
///     // ... grid-wide post-barrier work ...
/// }
/// ```
///
/// # Requirements
///
/// - Must be used WITH `#[kernel]` (not standalone), on a kernel inside a
///   `#[cuda_module]` module
/// - May appear before or after `#[kernel]`; `#[cuda_module]` records this
///   launch-time setting before nested function attributes expand
/// - The device must support cooperative launch
///   (`CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH`)
/// - The grid must fit on the device in one wave, otherwise the driver
///   rejects the launch with `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`
///
/// May be combined with `#[cluster_launch(x, y, z)]`; both launch
/// attributes are then passed to `cuLaunchKernelEx` in the same call.
///
/// Outside `#[cuda_module]`, the legacy (caller-unsafe) `cuda_launch!`
/// macro offers the same behaviour through its `cooperative: true` field.
#[proc_macro_attribute]
pub fn cooperative_launch(attr: TokenStream, item: TokenStream) -> TokenStream {
    launch_attrs::cooperative_launch_entry(attr, item)
}

/// Marks a function as a CUDA device function.
///
/// Device functions run on the GPU and can be called from kernels or other device functions,
/// but cannot be called from host code.
///
/// This attribute:
/// 1. Adds `#[no_mangle]` to preserve the function name in the binary
/// 2. Renames the function into the reserved `cuda_oxide_device_<hash>_` namespace
///    for detection by the codegen backend (the prefix lives in
///    `cuda-oxide/crates/reserved-oxide-symbols/`)
/// 3. Marks the function for extraction by the `rustc-codegen-cuda` backend
///
/// Device functions can:
/// - Return values (unlike kernels which must return `()`)
/// - Be called from kernels and other device functions
/// - Use generics (each monomorphization becomes a separate device function)
/// - Use per-loop `#[unroll]` and `#[unroll(N)]` annotations
///
/// # Loop unrolling
///
/// Loop annotations work the same way in device function definitions as they do
/// in kernels. Use an explicit counted `while` loop; range-based `for` loops are
/// not yet recognized. Partial factors must be `N >= 2`. Multiple `continue`
/// paths are supported; full unrolling preserves `break` and multiple exit
/// targets. Partial unrolling requires a positive counter step, a `<` or `<=`
/// test, an unchanging limit, and no exit besides the normal header test.
///
/// One annotation may create at most 1,024 body copies, 8,192 cloned basic
/// blocks, and 65,536 cloned operations. Factors above 1,024 are rejected;
/// unsupported loop shapes warn and are not unrolled.
///
/// # Example: Device Function Definition
///
/// ```ignore
/// use cuda_device::device;
///
/// #[device]
/// pub fn helper(x: f32, y: f32) -> f32 {
///     x * x + y * y
/// }
///
/// #[kernel]
/// pub fn my_kernel(data: *mut f32) {
///     let result = helper(1.0, 2.0);
///     unsafe { *data = result; }
/// }
/// ```
///
/// # Example: External Device Function Declaration (FFI)
///
/// ```ignore
/// use cuda_device::{device, convergent};
///
/// // Declare external device functions from LTOIR (e.g., CCCL)
/// #[device]
/// extern "C" {
///     #[convergent]
///     fn cub_block_reduce_sum_f32(input: f32, temp: *mut u8) -> f32;
///
///     fn fast_math_helper(x: f32) -> f32;
/// }
///
/// #[kernel]
/// pub fn my_kernel(data: *mut f32) {
///     let result = unsafe { cub_block_reduce_sum_f32(*data, temp_ptr) };
/// }
/// ```
#[proc_macro_attribute]
pub fn device(_attr: TokenStream, item: TokenStream) -> TokenStream {
    device::device_entry(_attr, item)
}

// ============================================================================
// NVVM Attributes for Device FFI
// ============================================================================

/// Marks a device function as convergent.
///
/// Convergent functions must be called by all threads in a warp/block together.
/// This prevents the optimizer from moving calls across control flow boundaries.
///
/// # When to Use
///
/// - Synchronization primitives (`__syncthreads`, barriers)
/// - Warp-collective operations (`__shfl_*`, warp vote, warp reduce)
/// - Block-collective operations (CUB block reduce/scan)
///
/// # Example
///
/// ```ignore
/// #[device]
/// extern "C" {
///     #[convergent]
///     fn cub_block_reduce_sum(input: f32, temp: *mut u8) -> f32;
/// }
/// ```
///
/// # Generated LLVM IR
///
/// ```llvm
/// declare float @cub_block_reduce_sum(float, ptr) #0
/// attributes #0 = { convergent nounwind }
/// ```
#[proc_macro_attribute]
pub fn convergent(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // This is a marker attribute - just pass through the item unchanged.
    // The collector will read this attribute and apply the LLVM convergent attribute.
    item
}

/// Marks a device function as pure (no side effects).
///
/// Pure functions only depend on their inputs and have no side effects.
/// This enables aggressive optimizations like CSE and dead code elimination.
///
/// # When to Use
///
/// - Math functions that don't access memory
/// - Functions that compute results purely from input arguments
///
/// # Example
///
/// ```ignore
/// #[device]
/// extern "C" {
///     #[pure]
///     fn fast_rsqrt(x: f32) -> f32;
/// }
/// ```
///
/// # Generated LLVM IR
///
/// ```llvm
/// declare float @fast_rsqrt(float) #0
/// attributes #0 = { nounwind readnone }
/// ```
#[proc_macro_attribute]
pub fn pure(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Marker attribute - collector will read and apply LLVM readnone attribute
    item
}

/// Marks a device function as read-only.
///
/// Read-only functions may read memory but never write to it.
/// This enables optimizations like load hoisting and caching.
///
/// # When to Use
///
/// - Lookup table functions
/// - Functions that only read from input arrays
///
/// # Example
///
/// ```ignore
/// #[device]
/// extern "C" {
///     #[readonly]
///     fn lookup_table(table: *const f32, idx: i32) -> f32;
/// }
/// ```
///
/// # Generated LLVM IR
///
/// ```llvm
/// declare float @lookup_table(ptr, i32) #0
/// attributes #0 = { nounwind readonly }
/// ```
#[proc_macro_attribute]
pub fn readonly(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Marker attribute - collector will read and apply LLVM readonly attribute
    item
}

/// Launch a CUDA kernel synchronously on a given stream. **Unsafe**: the
/// expansion calls the unsafe `cuda_core` launch functions without wrapping
/// them, so every use must appear inside an `unsafe { }` block.
///
/// Uses the `CudaKernel` trait (generated by `#[kernel]`) to look up the PTX
/// entry point name. Arguments are marshaled into a `Vec<*mut c_void>` and
/// passed directly to `cuda_core::launch_kernel` (`cuLaunchKernel`).
///
/// # Safety
///
/// This macro cannot check the kernel's signature. It hands the driver a raw
/// array of argument pointers and trusts you completely. By wrapping the
/// macro in `unsafe { }`, the caller promises:
///
/// - the argument **count and order** match the kernel's actual parameter
///   list (with each `slice(..)` / `slice_mut(..)` counting as two
///   parameters: pointer then length);
/// - each argument's **type, size, and alignment** match the corresponding
///   kernel parameter;
/// - every pointer argument is **device-accessible** (a valid device
///   allocation, or host memory reachable via HMM/unified memory) and stays
///   alive until the kernel finishes;
/// - the grid, block, dynamic-shared-memory size, cluster dimensions, and
///   cooperative mode satisfy every indexing, resource, and synchronization
///   assumption made by the kernel.
///
/// A mismatch is undefined behavior, not a runtime error: too few or
/// mistyped arguments make the driver read past the end of the args array,
/// and a bad pointer makes the device dereference junk.
///
/// For kernels embedded in your own crate, prefer `#[cuda_module]` with a
/// launch contract: it checks the signature and prepares a kernel-branded
/// geometry/resource proof. This macro's remaining niche is modules loaded at
/// **runtime by name** (e.g. external PTX files), where no compile-time
/// contract exists to check.
///
/// # Usage
///
/// ```ignore
/// // SAFETY: argument count, order, and types match `vecadd`; the buffers stay
/// // live, and the raw configuration matches vecadd's 1-D index space.
/// unsafe {
///     cuda_launch! {
///         kernel: vecadd,
///         stream: stream,
///         module: module,
///         config: LaunchConfig::for_num_elems(n as u32),
///         args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
///     }
/// }
/// ```
///
/// # Fields
///
/// | Field         | Type              | Description                                   |
/// |---------------|-------------------|-----------------------------------------------|
/// | `kernel`      | path              | `#[kernel]` function name (may be generic)    |
/// | `stream`      | `Arc<CudaStream>` | Stream to launch on                           |
/// | `module`      | `Arc<CudaModule>` | Loaded PTX module containing the kernel       |
/// | `config`      | `LaunchConfig`    | Grid/block dimensions, shared memory          |
/// | `cluster_dim` | `(u32,u32,u32)`   | *(optional)* Cluster dims for `cuLaunchKernelEx` |
/// | `cooperative` | `bool`            | *(optional)* Set `true` to launch via `cuLaunchKernelEx` with `CU_LAUNCH_ATTRIBUTE_COOPERATIVE` (required for `grid::sync()`) |
/// | `args`        | `[arg, ...]`      | Kernel arguments (see below)                  |
///
/// `cluster_dim` and `cooperative` may be combined. When both are set and
/// `cooperative` is `true`, the expansion calls
/// `cuda_core::launch_kernel_ex_cooperative_on_stream`.
///
/// # Argument forms
///
/// - `expr` -- scalar or pointer passed directly
/// - `slice(buf)` -- immutable device buffer; pushes `(cu_deviceptr, len)` as two args
/// - `slice_mut(buf)` -- mutable device buffer; same as `slice` but borrows `&mut`
/// - `move |captures| body` -- closure whose captures are marshaled individually
/// - `|captures| body` -- non-move closure; captures passed as raw pointers (HMM)
///
/// # Returns
///
/// `Result<(), cuda_core::DriverError>` -- the launch is asynchronous, so
/// a successful return only means the launch was enqueued.  Call
/// `stream.synchronize()` to wait for completion.
#[proc_macro]
pub fn cuda_launch(input: TokenStream) -> TokenStream {
    track_codegen_environment();
    let input = parse_macro_input!(input as CudaLaunchInput);
    expand_cuda_launch(input).into()
}

/// Launch a CUDA kernel asynchronously, returning a lazy `AsyncKernelLaunch`.
///
/// Unlike [`cuda_launch!`], this macro does **not** take a `stream:` parameter. The
/// CUDA stream is assigned later by the active `SchedulingPolicy` when the returned
/// operation is `.sync()`'d or `.await`'d. This enables lazy composition: multiple
/// launches can be chained with `.and_then()`, run in parallel with `zip!()`, or
/// awaited individually.
///
/// # Fields
///
/// | Field    | Type                | Description                                |
/// |----------|---------------------|--------------------------------------------|
/// | `kernel` | path                | `#[kernel]` function name (may be generic) |
/// | `module` | `Arc<CudaModule>`   | Loaded PTX module containing the kernel    |
/// | `config` | `LaunchConfig`      | Grid/block dimensions, shared memory       |
/// | `args`   | `[arg, ...]`        | Kernel arguments (see below)               |
///
/// # Argument forms
///
/// - `slice(x)` -- immutable device slice; pushes `(ptr, len)` as two kernel args
/// - `slice_mut(x)` -- mutable device slice; same as `slice` but takes `&mut`
/// - `expr` -- scalar or device pointer passed directly
/// - `|captures| body` -- closure environment passed by value
///
/// # Returns
///
/// An `AsyncKernelLaunch` implementing `DeviceOperation`. No GPU work is enqueued
/// until the caller schedules it.
///
/// # Safety
///
/// This is a raw launch API. The caller must ensure that:
///
/// - argument count, order, type, size, and alignment match the kernel ABI;
/// - referenced allocations remain valid and correctly aliased until the lazy
///   operation completes;
/// - the scheduled stream belongs to the function's CUDA context; and
/// - dimensions, block shape, shared memory, and launch mode satisfy every
///   indexing, resource, and synchronization assumption made by the kernel.
///
/// The macro invocation must therefore be inside an `unsafe` block.
///
/// # Usage
///
/// ```ignore
/// use cuda_host::cuda_launch_async;
/// use cuda_core::simt::LaunchConfig;
///
/// // SAFETY: ABI, lifetimes, geometry, and resources match vecadd.
/// let op = unsafe {
///     cuda_launch_async! {
///         kernel: vecadd,
///         module: module,
///         config: LaunchConfig::for_num_elems(N as u32),
///         args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
///     }
/// };
///
/// // Synchronous (blocks calling thread):
/// op.sync()?;
///
/// // Or asynchronous (suspends the async task):
/// // op.await?;
///
/// // Or compose before executing:
/// // let chained = op.and_then(|()| another_op);
/// // chained.sync()?;
/// ```
#[proc_macro]
pub fn cuda_launch_async(input: TokenStream) -> TokenStream {
    track_codegen_environment();
    let input = parse_macro_input!(input as CudaLaunchAsyncInput);
    expand_cuda_launch_async(input).into()
}
