// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    env,
    error::Error,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::exit,
};

use bindgen::CodegenConfig;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Expr, Fields, File, GenericArgument, Item, ItemStruct, PathArguments, ReturnType, Type,
    TypeBareFn,
};

const MIN_CUDA_VERSION: u32 = 13000;
/// Environment variables consulted (in order) to locate the CUDA toolkit
/// root. `CUDA_HOME` is the conventional name used by nvcc wrappers and CI
/// images.
const TOOLKIT_ENV_VARS: [&str; 2] = ["CUDA_TOOLKIT_PATH", "CUDA_HOME"];
const SETUP_DIAGNOSTICS_ENV: &str = "CUTILE_SETUP_DIAGNOSTICS";

/// Environment variable naming the CUDA `targets/<dir>` tree to use,
/// overriding the arch+OS table in `toolkit_target.rs`. The value is a
/// directory name under `{toolkit}/targets/` (e.g. `aarch64-linux`), the
/// same value nvcc's `-target-dir` flag takes; CMake exposes the equivalent
/// selection as `CUDAToolkit_TARGET_DIR`. When set, the named tree is the
/// only include candidate: the top-level `include/` (on standard installs a
/// symlink to the host architecture's `targets/` tree) is not consulted, so
/// a cross-build cannot silently bind the host's headers. The directory is
/// still probed for `cuda.h`, so a wrong value fails with the clear
/// discovery error instead of silently falling back.
const TOOLKIT_TARGET_DIR_ENV: &str = "CUDA_TOOLKIT_TARGET_DIR";

struct ApiSpec {
    virtual_header: &'static str,
    header_contents: &'static str,
    generated_api: &'static str,
    generated_shims: &'static str,
    library_name: &'static str,
    api_type: &'static str,
    loader_fn: &'static str,
    /// Returned when the library itself could not be loaded.
    not_loaded_expr: &'static str,
    /// Returned when the library loaded but this symbol is absent
    /// (older driver). Matches cuGetProcAddress's own convention of
    /// reporting absent symbols distinctly.
    missing_symbol_expr: &'static str,
    function_pattern: &'static str,
}

const API_SPECS: &[ApiSpec] = &[
    ApiSpec {
        virtual_header: "cuda_driver_wrapper.h",
        header_contents: "#include <cuda.h>\n",
        generated_api: "cuda_driver_api.rs",
        generated_shims: "cuda_driver_shims.rs",
        library_name: "CudaDriverApi",
        api_type: "CudaDriverApi",
        loader_fn: "cuda_driver",
        not_loaded_expr: "cudaError_enum_CUDA_ERROR_NOT_INITIALIZED",
        missing_symbol_expr: "cudaError_enum_CUDA_ERROR_NOT_FOUND",
        function_pattern: "^cu.*",
    },
    ApiSpec {
        virtual_header: "curand_wrapper.h",
        header_contents: "#include <curand.h>\n",
        generated_api: "curand_api.rs",
        generated_shims: "curand_shims.rs",
        library_name: "CurandApi",
        api_type: "CurandApi",
        loader_fn: "curand_api",
        // curand has no NOT_FOUND analogue; NOT_INITIALIZED covers both.
        not_loaded_expr: "curandStatus_CURAND_STATUS_NOT_INITIALIZED",
        missing_symbol_expr: "curandStatus_CURAND_STATUS_NOT_INITIALIZED",
        function_pattern: "^curand.*",
    },
];

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=wrapper.h");
    // Emitting any rerun-if-changed disables cargo's default "rerun on any
    // package change", so the `include!`d selection table needs naming.
    println!("cargo:rerun-if-changed=toolkit_target.rs");
    for var in TOOLKIT_ENV_VARS {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rerun-if-env-changed={TOOLKIT_TARGET_DIR_ENV}");
    println!("cargo:rerun-if-env-changed={SETUP_DIAGNOSTICS_ENV}");

    let toolkit = resolve_cuda_toolkit()?;
    println!(
        "cargo:rustc-env=CUTILE_RESOLVED_CUDA_TOOLKIT_PATH={}",
        toolkit.root.display()
    );

    // CUDA 12.8 renamed the event elapsed-time entry point to
    // cuEventElapsedTime_v2; earlier toolkits only declare
    // cuEventElapsedTime. Probe the resolved headers so src/lib.rs can
    // dispatch to whichever symbol this build's toolkit declares.
    println!("cargo::rustc-check-cfg=cfg(cuda_has_cuEventElapsedTime_v2)");
    println!("cargo::rustc-check-cfg=cfg(cuda_has_cuLaunchHostFunc_v2)");
    let resolved_cuda_h = toolkit.include_dir.join("cuda.h");
    if let Ok(header) = fs::read_to_string(&resolved_cuda_h) {
        if header.contains("cuEventElapsedTime_v2") {
            println!("cargo:rustc-cfg=cuda_has_cuEventElapsedTime_v2");
        }
        if header.contains("cuLaunchHostFunc_v2") {
            println!("cargo:rustc-cfg=cuda_has_cuLaunchHostFunc_v2");
        }
    }
    let out_dir = env::var("OUT_DIR")?;
    let out_dir = Path::new(&out_dir);

    generate_type_bindings(&toolkit, out_dir)?;
    for spec in API_SPECS {
        generate_dynamic_api(&toolkit, out_dir, spec)?;
    }

    Ok(())
}

/// A validated CUDA toolkit: its install root and the include directory that
/// actually contains `cuda.h` (`{root}/include` for standard installs,
/// `{root}/targets/<dir>/include` for redistributable and Tegra/sbsa layouts
/// that ship no top-level `include/`).
struct ResolvedToolkit {
    root: PathBuf,
    include_dir: PathBuf,
}

fn resolve_cuda_toolkit() -> Result<ResolvedToolkit, Box<dyn Error>> {
    for var in TOOLKIT_ENV_VARS {
        let Some(value) = env::var_os(var) else {
            continue;
        };
        if value.is_empty() {
            return Err(format!(
                "{var} is set to an empty string. Set it to a CUDA 13.0+ toolkit directory."
            )
            .into());
        }
        return resolve_explicit_cuda_toolkit(var, value);
    }
    find_default_cuda_toolkit()
}

fn resolve_explicit_cuda_toolkit(
    var: &str,
    value: OsString,
) -> Result<ResolvedToolkit, Box<dyn Error>> {
    let root = PathBuf::from(value);
    let (version, include_dir) = validate_cuda_toolkit(&root)
        .map_err(|error| format!("{var}={} is invalid: {error}", root.display()))?;
    emit_setup_diagnostic(format_args!(
        "using {var}={} (CUDA {})",
        root.display(),
        format_cuda_version(version)
    ));
    Ok(ResolvedToolkit { root, include_dir })
}

fn find_default_cuda_toolkit() -> Result<ResolvedToolkit, Box<dyn Error>> {
    let mut rejected = Vec::new();
    for root in default_cuda_toolkit_candidates() {
        match validate_cuda_toolkit(root) {
            Ok((version, include_dir)) => {
                emit_setup_diagnostic(format_args!(
                    "CUDA_TOOLKIT_PATH/CUDA_HOME are unset; using discovered CUDA toolkit {} (CUDA {})",
                    root.display(),
                    format_cuda_version(version)
                ));
                return Ok(ResolvedToolkit {
                    root: root.clone(),
                    include_dir,
                });
            }
            Err(error) => {
                emit_setup_diagnostic(format_args!(
                    "CUDA_TOOLKIT_PATH/CUDA_HOME are unset; skipping {}: {error}",
                    root.display()
                ));
                rejected.push(format!("  - {}: {error}", root.display()));
            }
        }
    }

    Err(format!(
        "Neither CUDA_TOOLKIT_PATH nor CUDA_HOME is set, and no CUDA 13.0+ toolkit was found in default locations:\n{}\nSet CUDA_TOOLKIT_PATH or CUDA_HOME to a CUDA 13.0+ toolkit directory.",
        rejected.join("\n")
    )
    .into())
}

// The `targets/<dir>` selection table, shared verbatim with
// `tests/toolkit_target.rs` so it can be unit tested.
include!("toolkit_target.rs");

/// [`resolve_toolkit_include_candidates`] for the target cargo is building
/// for: the [`TOOLKIT_TARGET_DIR_ENV`] override when set, otherwise the
/// table candidates for `CARGO_CFG_TARGET_ARCH` / `CARGO_CFG_TARGET_OS`.
/// Just the plain `{toolkit}/include` when the override is absent and
/// either cfg is unset, which keeps discovery from guessing.
fn build_include_candidates(toolkit: &Path) -> Vec<PathBuf> {
    let override_dir = env::var(TOOLKIT_TARGET_DIR_ENV).ok();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    resolve_toolkit_include_candidates(toolkit, override_dir.as_deref(), &arch, &os)
}

fn default_cuda_toolkit_candidates() -> &'static [PathBuf] {
    static CANDIDATES: std::sync::OnceLock<Vec<PathBuf>> = std::sync::OnceLock::new();
    CANDIDATES.get_or_init(|| {
        #[cfg(windows)]
        let candidates = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2",
        ];
        #[cfg(not(windows))]
        let candidates = [
            "/usr/local/cuda-13.3",
            "/usr/local/cuda-13.2",
            "/usr/local/cuda-13",
            "/usr/local/cuda",
            "/opt/cuda",
        ];

        candidates.into_iter().map(PathBuf::from).collect()
    })
}

/// Validates a toolkit root: probes the standard `include/` and the
/// `targets/<dir>/include/` layouts for `cuda.h` (only the directories
/// [`build_include_candidates`] yields for the build target's own
/// architecture, never all of `targets/*`), then checks the version floor.
/// Returns the version and the include directory that contains `cuda.h`.
fn validate_cuda_toolkit(cuda_toolkit: &Path) -> Result<(u32, PathBuf), String> {
    if !cuda_toolkit.is_dir() {
        return Err(format!("{} is not a directory", cuda_toolkit.display()));
    }

    let candidates = build_include_candidates(cuda_toolkit);
    let Some(include_dir) = select_include_dir(&candidates) else {
        let probed: Vec<String> = candidates
            .iter()
            .map(|dir| format!("    {}", dir.join("cuda.h").display()))
            .collect();
        return Err(format!(
            "{} does not contain cuda.h. Probed:\n{}",
            cuda_toolkit.display(),
            probed.join("\n")
        ));
    };

    let cuda_h = include_dir.join("cuda.h");
    let version = cuda_version_from_header(&cuda_h).map_err(|error| error.to_string())?;
    if version < MIN_CUDA_VERSION {
        return Err(format!(
            "CUDA toolkit {} is too old. The CUDA host-side crates require CUDA 13.0+ (the Tile compiler needs 13.2+)",
            format_cuda_version(version)
        ));
    }

    Ok((version, include_dir.clone()))
}

fn cuda_version_from_header(cuda_h: &Path) -> Result<u32, Box<dyn Error>> {
    let source = fs::read_to_string(cuda_h)?;
    source
        .lines()
        .find_map(|line| {
            let mut parts = line.split_whitespace();
            match (parts.next(), parts.next(), parts.next()) {
                (Some("#define"), Some("CUDA_VERSION"), Some(version)) => version.parse().ok(),
                _ => None,
            }
        })
        .ok_or_else(|| {
            format!(
                "could not find CUDA_VERSION in {}. Set CUDA_TOOLKIT_PATH or CUDA_HOME to a CUDA 13.0+ toolkit directory.",
                cuda_h.display()
            )
            .into()
        })
}

fn format_cuda_version(version: u32) -> String {
    format!("{}.{}", version / 1000, (version % 1000) / 10)
}

fn setup_diagnostics_enabled() -> bool {
    env::var(SETUP_DIAGNOSTICS_ENV)
        .map(|value| {
            matches!(
                value.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(false)
}

fn emit_setup_diagnostic(args: std::fmt::Arguments<'_>) {
    if setup_diagnostics_enabled() {
        println!("cargo:warning={args}");
    }
}

fn generate_type_bindings(toolkit: &ResolvedToolkit, out_dir: &Path) -> Result<(), Box<dyn Error>> {
    let bindings = bindgen_builder(toolkit)
        .header("wrapper.h")
        .blocklist_function(".*")
        .generate()?;
    let source = bindings.to_string();
    emit_layout_cfgs(&source);
    fs::write(out_dir.join("types.rs"), source)?;
    Ok(())
}

// Detect whether bindgen wrapped `CUmemLocation_st::id` in an anonymous union (e.g. CUDA 13.2+)
// or left it as a plain `int` (e.g. 13.0/13.1), so the helper in lib.rs can pick the right access path.
fn emit_layout_cfgs(generated_source: &str) {
    println!("cargo:rustc-check-cfg=cfg(cu_mem_location_anon_union)");
    if cu_mem_location_uses_anon_union(generated_source) {
        println!("cargo:rustc-cfg=cu_mem_location_anon_union");
    }
}

fn cu_mem_location_uses_anon_union(generated_source: &str) -> bool {
    let Ok(file) = syn::parse_file(generated_source) else {
        return false;
    };
    file.items.iter().any(|item| match item {
        Item::Struct(item_struct) if item_struct.ident == "CUmemLocation_st" => {
            let Fields::Named(fields) = &item_struct.fields else {
                return false;
            };
            fields
                .named
                .iter()
                .any(|f| f.ident.as_ref().is_some_and(|i| i == "__bindgen_anon_1"))
        }
        _ => false,
    })
}

fn generate_dynamic_api(
    toolkit: &ResolvedToolkit,
    out_dir: &Path,
    spec: &ApiSpec,
) -> Result<(), Box<dyn Error>> {
    let bindings = bindgen_builder(toolkit)
        .header_contents(spec.virtual_header, spec.header_contents)
        .dynamic_library_name(spec.library_name)
        .dynamic_link_require_all(false)
        .allowlist_function(spec.function_pattern)
        .with_codegen_config(CodegenConfig::FUNCTIONS)
        .generate()?;

    let generated_source = bindings.to_string();
    fs::write(out_dir.join(spec.generated_api), &generated_source)?;
    generate_shims_from_generated_api(&generated_source, out_dir.join(spec.generated_shims), spec)?;
    Ok(())
}

fn bindgen_builder(toolkit: &ResolvedToolkit) -> bindgen::Builder {
    bindgen::builder()
        .clang_arg(format!("-I{}", toolkit.include_dir.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
}

fn generate_shims_from_generated_api(
    generated_source: &str,
    output_path: impl AsRef<Path>,
    spec: &ApiSpec,
) -> Result<(), Box<dyn Error>> {
    let file = syn::parse_file(generated_source)?;
    let item_struct = find_api_struct(&file, spec.api_type).ok_or_else(|| {
        format!(
            "no generated struct found for {} in {}",
            spec.api_type, spec.generated_api
        )
    })?;

    let loader_fn = format_ident!("{}", spec.loader_fn);
    let not_loaded_expr = syn::parse_str::<Expr>(spec.not_loaded_expr)?;
    let missing_symbol_expr = syn::parse_str::<Expr>(spec.missing_symbol_expr)?;
    let shims = generate_field_shims(
        item_struct,
        &loader_fn,
        &not_loaded_expr,
        &missing_symbol_expr,
    );
    let shim_file = syn::parse2::<File>(quote!(#shims))?;

    fs::write(output_path, prettyplease::unparse(&shim_file))?;
    Ok(())
}

fn find_api_struct<'a>(file: &'a File, api_type: &str) -> Option<&'a ItemStruct> {
    file.items.iter().find_map(|item| match item {
        Item::Struct(item_struct) if item_struct.ident == api_type => Some(item_struct),
        _ => None,
    })
}

fn generate_field_shims(
    item_struct: &ItemStruct,
    loader_fn: &proc_macro2::Ident,
    not_loaded_expr: &Expr,
    missing_symbol_expr: &Expr,
) -> TokenStream {
    let Fields::Named(fields) = &item_struct.fields else {
        return TokenStream::new();
    };

    let shim_fns = fields.named.iter().filter_map(|field| {
        let field_name = field.ident.as_ref()?;
        let bare_fn = result_bare_fn(&field.ty)?;
        let args = bare_fn
            .inputs
            .iter()
            .enumerate()
            .map(|(index, arg)| {
                let name = arg
                    .name
                    .as_ref()
                    .map(|(ident, _)| ident.clone())
                    .unwrap_or_else(|| format_ident!("arg_{index}"));
                let ty = &arg.ty;
                (name.clone(), quote!(#name: #ty))
            })
            .collect::<Vec<_>>();
        let arg_names = args.iter().map(|(name, _)| name).collect::<Vec<_>>();
        let arg_defs = args.iter().map(|(_, def)| def).collect::<Vec<_>>();
        let ret = match &bare_fn.output {
            ReturnType::Default => quote!(),
            ReturnType::Type(_, ty) => quote!(-> #ty),
        };

        // `extern "C"` so a wrapper item coerces to a C function pointer,
        // matching cuda-oxide's original bindings surface (deliberately not
        // "C-unwind": that is a different fn-pointer type and breaks the
        // coercion). Unwinding out of extern "C" aborts, so these bodies
        // must be panic-free: both failure paths return the API's own
        // error codes instead.
        Some(quote! {
            #[allow(non_snake_case)]
            #[allow(clippy::missing_safety_doc)]
            #[allow(clippy::too_many_arguments)]
            #[allow(clippy::missing_inline_in_public_items)]
            #[inline]
            pub unsafe extern "C" fn #field_name(#(#arg_defs),*) #ret {
                match #loader_fn() {
                    Ok(api) => match &api.#field_name {
                        Ok(loaded_fn) => unsafe { (*loaded_fn)(#(#arg_names),*) },
                        Err(_) => #missing_symbol_expr,
                    },
                    Err(_) => #not_loaded_expr,
                }
            }
        })
    });

    quote! {
        #(#shim_fns)*
    }
}

fn result_bare_fn(ty: &Type) -> Option<&TypeBareFn> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    if segment.ident != "Result" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    let Some(GenericArgument::Type(Type::BareFn(bare_fn))) = args.args.first() else {
        return None;
    };
    Some(bare_fn)
}
