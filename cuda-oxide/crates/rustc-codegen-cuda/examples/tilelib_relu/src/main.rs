/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! tilelib_relu: a tile library written in Rust, checked on the GPU.
//!
//! The device half is the three `#[tile_export]` functions below. Each one
//! becomes an implementation a cuTile program calls through `foreign_call`,
//! on the native SIMT ABI: `in` scalars by value, everything else behind a
//! pointer, the return value as a trailing `out` pointer.
//!
//! The host half proves that ABI without a CUDA 13.5 or newer tileiras. It reads the
//! descriptions the backend wrote, generates `mock/caller.cu` with the
//! prototypes the Tile compiler derives, compiles it with nvcc, links it with
//! the Rust LTOIR through nvJitLink, and checks the numbers on the GPU:
//!
//!   cargo oxide run tilelib_relu --emit-nvvm-ir --arch sm_120

use cuda_artifact_finalizer::{
    CudaArch, DebugPolicy, FinalizationOptions, FinalizerOutput, LtoLinker, NamedInput,
    NvvmCompiler,
};
use cuda_core::embedded::{ArtifactCompileOptions, ArtifactDebugPolicy};
use cuda_core::{CudaContext, CudaModule, DeviceBuffer, launch_kernel_on_stream};
use cuda_device::{tile_export, warp};
use oxide_tile_export::{
    AbiParamKind, DESCRIPTIONS_SUFFIX, TileAccess, TileElem, TileExportDesc, descriptions_from_json,
};
use std::error::Error;
use std::ffi::c_void;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

// ============================================================================
// Device half: the tile library
// ============================================================================

/// Elementwise `max(x, 0)`.
#[tile_export(tile_shape = 128)]
pub fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

/// Scale a tile in place. The cuTile side knows it as `scale_in_place`.
#[tile_export(
    tile_shape = 128,
    name = "scale_in_place",
    symbol = "scale_in_place_impl"
)]
pub fn scale(x: &mut f32, factor: f32) {
    *x *= factor;
}

/// Sum over the 32 lanes of each warp; every lane gets the warp total.
#[tile_export(tile_shape = 128, sync = "warp")]
pub fn warp_sum(x: f32) -> f32 {
    let mut total = x;
    let mut offset = 16;
    while offset > 0 {
        total += warp::shuffle_xor_f32_sync(u32::MAX, total, offset);
        offset /= 2;
    }
    total
}

// ============================================================================
// Host half: prove the ABI through a generated CUDA C++ caller
// ============================================================================

const CRATE: &str = "tilelib_relu";

type Fallible<T> = Result<T, Box<dyn Error>>;

/// How the run ended when nothing went wrong.
enum Outcome {
    /// The exports that were called through the native ABI and checked, plus
    /// the ones the generated caller cannot declare, each with its reason.
    Verified {
        verified: usize,
        skipped: Vec<(String, String)>,
    },
    /// The check cannot run on this machine; the reason is printed.
    Skipped(String),
}

fn main() {
    match run() {
        Ok(Outcome::Verified { verified, skipped }) => {
            let total = verified + skipped.len();
            let mut line =
                format!("SUCCESS: {verified}/{total} tile exports verified through the native ABI");
            if !skipped.is_empty() {
                let names: Vec<&str> = skipped.iter().map(|(decl, _)| decl.as_str()).collect();
                let _ = write!(line, ", {} skipped ({})", skipped.len(), names.join(", "));
            }
            println!("{line}");
        }
        Ok(Outcome::Skipped(reason)) => println!("skipping: {reason}"),
        Err(error) => {
            println!("FAIL: {error}");
            std::process::exit(1);
        }
    }
}

fn run() -> Fallible<Outcome> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let (exports, skipped) = partition_callable(read_descriptions(dir)?);
    if exports.is_empty() {
        return Ok(Outcome::Skipped(
            "every export uses half precision, which the generated caller cannot declare"
                .to_string(),
        ));
    }
    let options = read_build_policy(dir)?;
    let Some(nvcc) = find_nvcc() else {
        return Ok(Outcome::Skipped("nvcc not found".to_string()));
    };

    let caller_cu = dir.join("mock").join("caller.cu");
    write_if_changed(&caller_cu, &caller_source(&exports)?)?;

    let rust_ltoir = rust_ltoir(dir, &options)?;
    let caller_ltoir = compile_caller(&nvcc, &caller_cu, options.target())?;
    // Both halves are LTOIR, so nvJitLink resolves the implementation symbols
    // at link time, the same way a real tile library is consumed.
    let cubin = LtoLinker::discover()?.link_ltoir(
        &[
            NamedInput::new("tilelib_relu.ltoir", &rust_ltoir),
            NamedInput::new("caller.ltoir", &caller_ltoir),
        ],
        &options,
        FinalizerOutput::Cubin,
    )?;
    println!(
        "nvJitLink: {} bytes of cubin for {}",
        cubin.len(),
        options.target().sm()
    );

    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(error) => return Ok(Outcome::Skipped(format!("no CUDA device: {error}"))),
    };
    let (major, minor) = ctx.compute_capability()?;
    let gpu = u32::try_from(major * 10 + minor)?;
    if gpu != options.target().capability() {
        return Ok(Outcome::Skipped(format!(
            "the device code targets {}, this GPU is sm_{gpu}",
            options.target().sm()
        )));
    }
    let module = ctx.load_module_from_image(&cubin)?;
    for desc in &exports {
        check_export(&ctx, &module, desc)?;
    }
    Ok(Outcome::Verified {
        verified: exports.len(),
        skipped,
    })
}

/// Split the exports into the ones the generated caller can declare and the
/// ones it cannot, printing one `SKIP` line per export left out. nvcc has no
/// scalar type that lowers to LLVM `half` or `bfloat` (its `__half` is a
/// struct), so a half-precision export is reported and counted rather than
/// failed: the ABI defect that would be is in the mock, not in the export.
fn partition_callable(
    exports: Vec<TileExportDesc>,
) -> (Vec<TileExportDesc>, Vec<(String, String)>) {
    let mut callable = Vec::with_capacity(exports.len());
    let mut skipped = Vec::new();
    for desc in exports {
        match desc
            .params
            .iter()
            .find_map(|param| c_type(param.elem).err())
        {
            Some(reason) => {
                println!("SKIP {}: {reason}", desc.decl);
                skipped.push((desc.decl, reason));
            }
            None => callable.push(desc),
        }
    }
    (callable, skipped)
}

/// The descriptions the backend wrote next to the device artifact, validated
/// so the generated C++ can rely on the ABI rules.
fn read_descriptions(dir: &Path) -> Fallible<Vec<TileExportDesc>> {
    let path = dir.join(format!("{CRATE}.{DESCRIPTIONS_SUFFIX}"));
    let text = std::fs::read_to_string(&path).map_err(|error| {
        format!(
            "could not read {} (build with `cargo oxide run {CRATE} --emit-nvvm-ir --arch sm_120`): {error}",
            path.display()
        )
    })?;
    let exports = descriptions_from_json(&text)?;
    if exports.is_empty() {
        return Err(format!("{} describes no tile exports", path.display()).into());
    }
    for desc in &exports {
        desc.validate()
            .map_err(|error| format!("description of `{}`: {error}", desc.decl))?;
    }
    Ok(exports)
}

/// The target and compile policy recorded with the NVVM IR. The link must use
/// the same architecture and FMA setting as the build, or nvJitLink rejects
/// the input or changes the arithmetic.
fn read_build_policy(dir: &Path) -> Fallible<FinalizationOptions> {
    let target_path = dir.join(format!("{CRATE}.target"));
    let text = std::fs::read_to_string(&target_path)
        .map_err(|error| format!("could not read {}: {error}", target_path.display()))?;
    let target: CudaArch = text.lines().next().unwrap_or_default().trim().parse()?;

    let options_path = dir.join(format!("{CRATE}.options"));
    let compile = match std::fs::read_to_string(&options_path) {
        Ok(text) => ArtifactCompileOptions::from_sidecar_text(&text)
            .map_err(|error| format!("invalid {}: {error}", options_path.display()))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => ArtifactCompileOptions::new(),
        Err(error) => {
            return Err(format!("could not read {}: {error}", options_path.display()).into());
        }
    };
    let debug = match compile.debug_policy() {
        ArtifactDebugPolicy::None => DebugPolicy::None,
        ArtifactDebugPolicy::LineTables => DebugPolicy::LineTables,
        ArtifactDebugPolicy::Full => DebugPolicy::Full,
    };
    Ok(FinalizationOptions::new(target)
        .with_fma_contraction(compile.fma_contraction_enabled())
        .with_debug_policy(debug))
}

/// The Rust half as LTOIR. `cargo oxide tilelib` and `emit-ltoir` leave a
/// `.ltoir` next to the `.ll`; a plain `--emit-nvvm-ir` build leaves only the
/// `.ll`, so it is compiled here with libNVVM, the call cargo oxide makes. A
/// `.ltoir` older than the `.ll` is stale and gets recompiled too.
fn rust_ltoir(dir: &Path, options: &FinalizationOptions) -> Fallible<Vec<u8>> {
    let ll = dir.join(format!("{CRATE}.ll"));
    let ltoir = dir.join(format!("{CRATE}.ltoir"));
    if ltoir.is_file() && !older_than(&ltoir, &ll) {
        println!("libNVVM: using {}", ltoir.display());
        return Ok(std::fs::read(&ltoir)?);
    }
    let ir = std::fs::read(&ll).map_err(|error| {
        format!(
            "could not read {} (build with `cargo oxide run {CRATE} --emit-nvvm-ir --arch sm_120`): {error}",
            ll.display()
        )
    })?;
    let ltoir =
        NvvmCompiler::discover()?.compile_nvvm_ir_to_ltoir(&format!("{CRATE}.ll"), &ir, options)?;
    println!(
        "libNVVM: {} compiled to LTOIR ({} bytes)",
        ll.display(),
        ltoir.len()
    );
    Ok(ltoir)
}

fn older_than(file: &Path, other: &Path) -> bool {
    let modified = |path: &Path| std::fs::metadata(path).and_then(|meta| meta.modified());
    match (modified(file), modified(other)) {
        (Ok(file_time), Ok(other_time)) => file_time < other_time,
        _ => false,
    }
}

/// nvcc from `PATH`, then from the toolkit roots libNVVM is also looked up in.
fn find_nvcc() -> Option<PathBuf> {
    let on_path = std::env::var_os("PATH")
        .map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
        .unwrap_or_default()
        .into_iter()
        .map(|dir| dir.join("nvcc"));
    let in_toolkit = ["CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_PATH"]
        .into_iter()
        .filter_map(std::env::var_os)
        .map(PathBuf::from)
        .chain([PathBuf::from("/usr/local/cuda"), PathBuf::from("/opt/cuda")])
        .map(|root| root.join("bin").join("nvcc"));
    on_path
        .chain(in_toolkit)
        .find(|candidate| candidate.is_file())
}

/// Generate `mock/caller.cu`: the implementation prototypes the Tile compiler
/// derives from the descriptions, and one kernel per export that calls the
/// implementation the way the compiler does over one tile.
fn caller_source(exports: &[TileExportDesc]) -> Fallible<String> {
    let mut out = String::new();
    out.push_str(
        "// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n\
         // SPDX-License-Identifier: Apache-2.0\n\
         //\n\
         // Generated by the tilelib_relu host program from tilelib_relu.tileexports.json.\n\
         // Do not edit: it is rewritten whenever the descriptions change.\n\
         //\n\
         // Each kernel stands in for the Tile compiler: one thread per invocation of\n\
         // the implementation, launched as one block of tile-size / footprint threads.\n\
         // Scalar `in` parameters arrive by value, every other parameter as a pointer\n\
         // to the first of the invocation's footprint elements. A `sync = warp` export\n\
         // is called by every lane of each warp, which the full block provides.\n\n",
    );
    for desc in exports {
        // `validate` holds every parameter of an export to one footprint, so
        // the first parameter's is the export's.
        let footprint = desc.params[0].footprint;
        let abi = desc.abi_params();
        let mut prototype = Vec::new();
        let mut kernel_params = Vec::new();
        let mut call_args = Vec::new();
        for (param, info) in abi.iter().zip(&desc.params) {
            let c_type = c_type(param.kind.elem()).map_err(|reason| {
                format!("`{}`: parameter `{}`: {reason}", desc.decl, param.name)
            })?;
            let constness = if info.access == TileAccess::In {
                "const "
            } else {
                ""
            };
            // Every parameter is a tile-size buffer to the kernel; by-value
            // parameters are loaded from it, pointer parameters point into it.
            kernel_params.push(format!("{constness}{c_type} *{}", param.name));
            match param.kind {
                AbiParamKind::Scalar(_) => {
                    prototype.push(format!("{c_type} {}", param.name));
                    call_args.push(format!("{}[i]", param.name));
                }
                AbiParamKind::Pointer(_) => {
                    prototype.push(format!("{constness}{c_type} *{}", param.name));
                    call_args.push(format!("&{}[i]", param.name));
                }
            }
        }
        writeln!(
            out,
            "// {}: {} elements, footprint {footprint}, sync {}",
            desc.decl,
            desc.shape[0],
            desc.sync.tile_ir_name()
        )?;
        writeln!(
            out,
            "extern \"C\" __device__ void {}({});",
            desc.symbol,
            prototype.join(", ")
        )?;
        writeln!(
            out,
            "extern \"C\" __global__ void call_{}({}) {{",
            desc.decl,
            kernel_params.join(", ")
        )?;
        if footprint == 1 {
            writeln!(
                out,
                "  const int i = blockIdx.x * blockDim.x + threadIdx.x;"
            )?;
        } else {
            // Thread t owns elements [t * F, (t + 1) * F).
            writeln!(
                out,
                "  const int i = (blockIdx.x * blockDim.x + threadIdx.x) * {footprint};"
            )?;
        }
        writeln!(out, "  {}({});", desc.symbol, call_args.join(", "))?;
        writeln!(out, "}}\n")?;
    }
    Ok(out)
}

/// The C type of a tile element, or why nvcc has none: it has no scalar type
/// that lowers to LLVM `half` or `bfloat` (its `__half` is a struct), so
/// those two cannot be declared in the generated caller.
fn c_type(elem: TileElem) -> Result<&'static str, String> {
    Ok(match elem {
        TileElem::F32 => "float",
        TileElem::F64 => "double",
        TileElem::I8 => "signed char",
        TileElem::I16 => "short",
        TileElem::I32 => "int",
        TileElem::I64 => "long long",
        TileElem::U8 => "unsigned char",
        TileElem::U16 => "unsigned short",
        TileElem::U32 => "unsigned int",
        TileElem::U64 => "unsigned long long",
        TileElem::F16 | TileElem::BF16 => {
            return Err(format!(
                "{} has no by-value C type in nvcc; the mock covers f32, f64 and the integers",
                elem.rust_name()
            ));
        }
    })
}

/// Write `text` to `path` only when it differs, so a run that changes nothing
/// leaves the generated file alone.
fn write_if_changed(path: &Path, text: &str) -> Fallible<()> {
    if std::fs::read_to_string(path).is_ok_and(|current| current == text) {
        println!("caller: {} is current", path.display());
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, text)?;
    println!("caller: wrote {}", path.display());
    Ok(())
}

/// Compile the generated caller to LTOIR for `target` in a scratch directory,
/// removed on success and kept for inspection on failure. `-dc` leaves the
/// Rust symbols unresolved for the link; `--keep` makes nvcc leave the
/// `.ltoir` behind.
fn compile_caller(nvcc: &Path, caller_cu: &Path, target: &CudaArch) -> Fallible<Vec<u8>> {
    let work = std::env::temp_dir().join(format!("{CRATE}-nvcc-{}", std::process::id()));
    std::fs::create_dir_all(&work)?;
    let sm = target.sm();
    let code = sm.trim_start_matches("sm_");

    let mut command = Command::new(nvcc);
    // The same host-compiler override the other LTOIR examples honor.
    if let Some(ccbin) = std::env::var_os("NVCC_CCBIN").or_else(|| std::env::var_os("CUDAHOSTCXX"))
    {
        command.arg("-ccbin").arg(ccbin);
    }
    let output = command
        .args(["-dc", "--keep", "--keep-dir"])
        .arg(&work)
        .arg("-gencode")
        .arg(format!("arch={},code=lto_{code}", target.compute()))
        .arg(caller_cu)
        .arg("-o")
        .arg(work.join("caller.o"))
        .current_dir(&work)
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "nvcc failed ({}); intermediates kept in {}\n{}",
            output.status,
            work.display(),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let ltoir = std::fs::read(work.join("caller.ltoir"))?;
    std::fs::remove_dir_all(&work)?;
    println!("nvcc: caller.ltoir ({} bytes, lto_{code})", ltoir.len());
    Ok(ltoir)
}

/// The data fed to one export and the result expected back.
struct HostCase {
    /// One tile of values per ABI parameter, in description order; result
    /// slots start zeroed.
    buffers: Vec<Vec<f32>>,
    /// Index of the parameter whose buffer is read back and checked.
    checked: usize,
    expected: Vec<f32>,
    what: &'static str,
}

/// What the host feeds each export and what it expects back. Every case is
/// exact in f32: relu copies or zeroes, 0.5 halves without rounding, and the
/// warp sums are small integers.
fn host_model(desc: &TileExportDesc, n: usize) -> Fallible<HostCase> {
    // A ramp from -4 to 4 crosses zero without landing on it.
    let last = (n.max(2) - 1) as f32;
    let ramp: Vec<f32> = (0..n).map(|i| -4.0 + 8.0 * (i as f32) / last).collect();
    let case = match desc.decl.as_str() {
        "relu" => HostCase {
            buffers: vec![ramp.clone(), vec![0.0; n]],
            checked: 1,
            expected: ramp
                .iter()
                .map(|&x| if x > 0.0 { x } else { 0.0 })
                .collect(),
            what: "relu of a ramp from -4 to 4",
        },
        "scale_in_place" => HostCase {
            buffers: vec![ramp.clone(), vec![0.5; n]],
            checked: 0,
            expected: ramp.iter().map(|x| x * 0.5).collect(),
            what: "the ramp scaled in place by 0.5",
        },
        "warp_sum" => {
            let lanes: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let expected = (0..n)
                .map(|i| {
                    let first = i - i % 32;
                    lanes[first..(first + 32).min(n)].iter().sum()
                })
                .collect();
            HostCase {
                buffers: vec![lanes, vec![0.0; n]],
                checked: 1,
                expected,
                what: "32-lane sums over 0..128, every lane holding its warp's total",
            }
        }
        other => return Err(format!("no host model for export `{other}`").into()),
    };
    if case.buffers.len() != desc.params.len() || !desc.params[case.checked].access.is_output() {
        return Err(format!(
            "`{}`: the host model does not match the described parameters",
            desc.decl
        )
        .into());
    }
    Ok(case)
}

/// Run `call_<decl>` over one tile, one thread per footprint, and compare
/// every element of the checked buffer with the host model.
fn check_export(
    ctx: &Arc<CudaContext>,
    module: &Arc<CudaModule>,
    desc: &TileExportDesc,
) -> Fallible<()> {
    let n = usize::try_from(desc.shape[0])?;
    let footprint = usize::try_from(desc.params[0].footprint)?;
    if n % footprint != 0 {
        return Err(format!(
            "`{}`: footprint {footprint} does not divide the tile size {n}",
            desc.decl
        )
        .into());
    }
    let invocations = n / footprint;
    let case = host_model(desc, n)?;

    let stream = ctx.default_stream();
    let buffers = case
        .buffers
        .iter()
        .map(|data| DeviceBuffer::from_host(&stream, data))
        .collect::<Result<Vec<_>, _>>()?;
    let mut pointers: Vec<_> = buffers.iter().map(DeviceBuffer::cu_deviceptr).collect();
    let mut args: Vec<*mut c_void> = pointers
        .iter_mut()
        .map(|pointer| std::ptr::from_mut(pointer).cast::<c_void>())
        .collect();
    let function = module.load_function(&format!("call_{}", desc.decl))?;
    // SAFETY: the generated kernel takes one device pointer per ABI parameter
    // in description order, which is how `args` was built; every buffer holds
    // `n` elements, one footprint for each of the `n / footprint` threads
    // launched, and outlives the launch; the stream belongs to the module's
    // context.
    unsafe {
        launch_kernel_on_stream(
            &function,
            (1, 1, 1),
            (u32::try_from(invocations)?, 1, 1),
            0,
            &stream,
            &mut args,
        )?;
    }

    let got = buffers[case.checked].to_host_vec(&stream)?;
    let mismatches: Vec<_> = got
        .iter()
        .zip(&case.expected)
        .enumerate()
        .filter(|(_, (got, expected))| got != expected)
        .collect();
    if let Some((index, (got, expected))) = mismatches.first() {
        return Err(format!(
            "`{}`: {} of {n} elements differ, first at [{index}]: got {got}, expected {expected}",
            desc.decl,
            mismatches.len()
        )
        .into());
    }
    let footprints = if footprint == 1 {
        String::new()
    } else {
        format!(" in {invocations} footprints of {footprint}")
    };
    println!(
        "PASS {}: {n} elements{footprints} through `{}`, {}",
        desc.decl, desc.symbol, case.what
    );
    Ok(())
}
