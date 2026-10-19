/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The callbacks the assembler drives, and the entry point that installs them.
//!
//! The protocol, in the order the assembler uses it:
//!
//! ```text
//! tileir_create_linkage_provider(version)      -> handle + vtable
//! provide_impls(one call site)                 -> implementation catalog bytecode
//! filter_and_rank(hard-compatible candidates)  -> keep all, score 0
//! instantiate(selected calls)                  -> device code exporting the asked symbols
//! free_impls / free_module / free_error        -> give the buffers back
//! destroy                                      -> drop the handle
//! ```
//!
//! Every callback catches panics: a Rust panic must not unwind into the
//! assembler. The provider state is immutable after creation, so the
//! callbacks need no locking.

use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::error::{ErrorStorage, view};
use crate::module::{CodeBufferSpec, ModuleStorage};
use crate::payload::{DeviceCode, Payload, PayloadError};
use crate::sys::{
    ABI_VERSION, TileIRForeignModule, TileIRLPError, TileIRLPFilterAndRankRequest,
    TileIRLPProvideImplsRequest, TileIRLPRankDisposition, TileIRLPRankResult, TileIRLPResult,
    TileIRLinkageProviderAPIVersion, TileIRLinkageProviderHandle, TileIRLinkageProviderInstance,
    TileIRLinkageProviderVTable, TileIROutputType, TileIRResolvedCall,
};
use crate::view::{callsite, resolved, slice};

/// Set this variable to make a provider describe every callback on stderr.
pub const LOG_ENV: &str = "OXIDE_TILE_PROVIDER_LOG";

/// The request envelope version this runtime understands.
const REQUEST_FORMAT_VERSION: u32 = 1;

/// Provider state behind the opaque handle.
pub(crate) struct Provider {
    payload: Payload,
    /// The diagnostic name; the instance's `name` view points into it.
    name: Box<str>,
    log: bool,
}

impl Provider {
    fn log(&self, message: impl FnOnce() -> String) {
        if self.log {
            eprintln!("[oxide-tile-provider {}] {}", self.name, message());
        }
    }
}

/// Install the provider: the body of a shared object's
/// `tileir_create_linkage_provider`. `build` makes the payload; it runs once,
/// after the version check.
///
/// # Safety
/// `out_instance` must be null or point at writable storage for an instance.
pub unsafe fn entry(
    compiler_api_version: TileIRLinkageProviderAPIVersion,
    out_instance: *mut TileIRLinkageProviderInstance,
    build: impl FnOnce() -> Result<Payload, PayloadError>,
) -> TileIRLPResult {
    let log = std::env::var_os(LOG_ENV).is_some_and(|value| !value.is_empty());
    let supported = (ABI_VERSION.major, ABI_VERSION.minor, ABI_VERSION.patch);
    let offered = (
        compiler_api_version.major,
        compiler_api_version.minor,
        compiler_api_version.patch,
    );
    if offered != supported {
        if log {
            eprintln!(
                "[oxide-tile-provider] the assembler speaks linkage provider ABI {}.{}.{}, this \
                 provider was built for {}.{}.{}",
                offered.0, offered.1, offered.2, supported.0, supported.1, supported.2
            );
        }
        return TileIRLPResult::TILEIR_LP_ERROR_UNSUPPORTED_VERSION;
    }
    if out_instance.is_null() {
        return TileIRLPResult::TILEIR_LP_ERROR_INVALID_ARGUMENT;
    }
    let payload = match catch_unwind(AssertUnwindSafe(build)) {
        Ok(Ok(payload)) => payload,
        Ok(Err(error)) => {
            eprintln!("[oxide-tile-provider] the embedded payload is unusable: {error}");
            return TileIRLPResult::TILEIR_LP_ERROR;
        }
        Err(_) => {
            eprintln!("[oxide-tile-provider] building the payload panicked");
            return TileIRLPResult::TILEIR_LP_ERROR;
        }
    };
    let provider = Box::new(Provider {
        name: payload.name.clone().into_boxed_str(),
        payload,
        log,
    });
    provider.log(|| {
        format!(
            "created for ABI {}.{}.{}; exports {}",
            offered.0,
            offered.1,
            offered.2,
            provider
                .payload
                .exports
                .iter()
                .map(|export| format!("{} -> {}", export.decl, export.symbol))
                .collect::<Vec<_>>()
                .join(", ")
        )
    });
    let name = view(&provider.name);
    let handle: TileIRLinkageProviderHandle = Box::into_raw(provider).cast();
    // SAFETY: `out_instance` is non-null and writable per the contract.
    unsafe {
        out_instance.write(TileIRLinkageProviderInstance {
            handle,
            vtable: TileIRLinkageProviderVTable {
                api_version: ABI_VERSION,
                provide_impls: Some(provide_impls),
                filter_and_rank: Some(filter_and_rank),
                instantiate: Some(instantiate),
                free_impls: Some(free_impls),
                free_module: Some(free_module),
                free_error: Some(free_error),
                destroy: Some(destroy),
            },
            name,
        });
    }
    TileIRLPResult::TILEIR_LP_SUCCESS
}

/// # Safety
/// `handle` must be one `entry` produced and `destroy` has not consumed.
unsafe fn provider<'a>(handle: TileIRLinkageProviderHandle) -> Option<&'a Provider> {
    // SAFETY: as documented; the box lives until `destroy`.
    unsafe { handle.cast::<Provider>().as_ref() }
}

/// Write `error` to `out_error` when the assembler gave us a slot, else drop it.
///
/// # Safety
/// `out_error` must be null or writable.
unsafe fn report(out_error: *mut *mut TileIRLPError, error: *mut TileIRLPError) {
    if out_error.is_null() {
        // SAFETY: we made the error; nobody else will free it.
        unsafe { ErrorStorage::free(error) };
    } else {
        // SAFETY: writable per the contract.
        unsafe { out_error.write(error) };
    }
}

/// A heap buffer the assembler frees through `free_impls`. The length sits in
/// front of the bytes so the free callback, which receives only the data
/// pointer, can rebuild the allocation layout.
fn leak_prefixed(bytes: &[u8]) -> *const c_void {
    const PREFIX: usize = std::mem::size_of::<usize>();
    let layout = prefixed_layout(bytes.len());
    // SAFETY: the layout has non-zero size.
    let base = unsafe { alloc(layout) };
    if base.is_null() {
        handle_alloc_error(layout);
    }
    // SAFETY: the allocation holds the prefix and the bytes.
    unsafe {
        base.cast::<usize>().write(bytes.len());
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), base.add(PREFIX), bytes.len());
        base.add(PREFIX).cast()
    }
}

fn prefixed_layout(len: usize) -> Layout {
    Layout::from_size_align(
        std::mem::size_of::<usize>() + len,
        std::mem::align_of::<usize>(),
    )
    .expect("a catalog fits in memory")
}

unsafe extern "C" fn provide_impls(
    handle: TileIRLinkageProviderHandle,
    request: *const TileIRLPProvideImplsRequest,
    out_data: *mut *const c_void,
    out_size: *mut usize,
    out_error: *mut *mut TileIRLPError,
) -> TileIRLPResult {
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the assembler passes the handle `entry` returned and
        // compiler-owned records valid for this callback.
        unsafe {
            if out_data.is_null() || out_size.is_null() {
                return Err((
                    TileIRLPResult::TILEIR_LP_ERROR_INVALID_ARGUMENT,
                    "provide_impls: null output parameter".to_string(),
                ));
            }
            out_data.write(std::ptr::null());
            out_size.write(0);
            if !out_error.is_null() {
                out_error.write(std::ptr::null_mut());
            }
            let Some(provider) = provider(handle) else {
                return Err((
                    TileIRLPResult::TILEIR_LP_ERROR_INVALID_ARGUMENT,
                    "provide_impls: null provider handle".to_string(),
                ));
            };
            let Some(request) = request.as_ref() else {
                return Err((
                    TileIRLPResult::TILEIR_LP_ERROR_INVALID_ARGUMENT,
                    "provide_impls: null request".to_string(),
                ));
            };
            if request.format_version != REQUEST_FORMAT_VERSION {
                return Err((
                    TileIRLPResult::TILEIR_LP_ERROR_INVALID_ARGUMENT,
                    format!(
                        "provide_impls: request format {} is not the format {} this provider \
                         understands",
                        request.format_version, REQUEST_FORMAT_VERSION
                    ),
                ));
            }
            let mut wanted: Vec<&str> = Vec::new();
            for site in slice(request.callsites, request.num_callsites) {
                let Some(site) = callsite(site) else { continue };
                let export = provider.payload.export_for_decl(&site.decl);
                provider.log(|| {
                    format!(
                        "provide_impls: call site {} of @{}<[{}]>({}) -> ({}): {}",
                        site.id,
                        site.decl,
                        site.const_params
                            .iter()
                            .zip(&site.const_args)
                            .map(|(name, value)| format!("{name} = {value}"))
                            .collect::<Vec<_>>()
                            .join(", "),
                        site.args
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(", "),
                        site.results
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(", "),
                        match export {
                            Some(export) => format!("served by {}", export.symbol),
                            None => "not one of ours".to_string(),
                        }
                    )
                });
                if let Some(export) = export
                    && !wanted.contains(&export.decl.as_str())
                {
                    wanted.push(export.decl.as_str());
                }
            }
            if wanted.is_empty() {
                return Ok(());
            }
            let catalog = provider.payload.catalog(&wanted).map_err(|error| {
                (
                    TileIRLPResult::TILEIR_LP_ERROR,
                    format!("provide_impls: {error}"),
                )
            })?;
            provider.log(|| {
                format!(
                    "provide_impls: returning a {}-byte catalog for {}",
                    catalog.len(),
                    wanted.join(", ")
                )
            });
            out_data.write(leak_prefixed(&catalog));
            out_size.write(catalog.len());
            Ok(())
        }
    }));
    // SAFETY: `out_error` is null or writable per the contract.
    unsafe { finish(outcome, out_error, "provide_impls") }
}

unsafe extern "C" fn filter_and_rank(
    handle: TileIRLinkageProviderHandle,
    request: *const TileIRLPFilterAndRankRequest,
    out_results: *mut TileIRLPRankResult,
    num_results: usize,
    out_error: *mut *mut TileIRLPError,
) -> TileIRLPResult {
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: compiler-owned records valid for this callback.
        unsafe {
            if !out_error.is_null() {
                out_error.write(std::ptr::null_mut());
            }
            let Some(provider) = provider(handle) else {
                return Err((
                    TileIRLPResult::TILEIR_LP_ERROR_INVALID_ARGUMENT,
                    "filter_and_rank: null provider handle".to_string(),
                ));
            };
            let Some(request) = request.as_ref() else {
                return Err((
                    TileIRLPResult::TILEIR_LP_ERROR_INVALID_ARGUMENT,
                    "filter_and_rank: null request".to_string(),
                ));
            };
            if num_results != request.num_candidates || (num_results > 0 && out_results.is_null()) {
                return Err((
                    TileIRLPResult::TILEIR_LP_ERROR_INVALID_ARGUMENT,
                    format!(
                        "filter_and_rank: {num_results} result slots for {} candidates",
                        request.num_candidates
                    ),
                ));
            }
            provider.log(|| {
                let candidates = slice(request.candidates, request.num_candidates);
                let named: Vec<String> = candidates
                    .iter()
                    .filter_map(|candidate| resolved(&candidate.resolved_call))
                    .map(|call| format!("{} for @{}", call.symbol, call.decl))
                    .collect();
                format!(
                    "filter_and_rank: keeping {} candidate(s): {}",
                    candidates.len(),
                    named.join(", ")
                )
            });
            // Every hard-compatible candidate is as good as any other to this
            // provider; the assembler's stable order decides.
            for index in 0..num_results {
                out_results.add(index).write(TileIRLPRankResult {
                    disposition: TileIRLPRankDisposition::TILEIR_LP_RANK_SCORED,
                    score: 0,
                });
            }
            Ok(())
        }
    }));
    // SAFETY: `out_error` is null or writable per the contract.
    unsafe { finish(outcome, out_error, "filter_and_rank") }
}

unsafe extern "C" fn instantiate(
    handle: TileIRLinkageProviderHandle,
    calls: *const TileIRResolvedCall,
    num_calls: usize,
    out_module: *mut *mut TileIRForeignModule,
    out_error: *mut *mut TileIRLPError,
) -> TileIRLPResult {
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: compiler-owned records valid for this callback.
        unsafe {
            if out_module.is_null() {
                return Err(Failure::Generic(
                    TileIRLPResult::TILEIR_LP_ERROR_INVALID_ARGUMENT,
                    "instantiate: null output parameter".to_string(),
                ));
            }
            out_module.write(std::ptr::null_mut());
            if !out_error.is_null() {
                out_error.write(std::ptr::null_mut());
            }
            let Some(provider) = provider(handle) else {
                return Err(Failure::Generic(
                    TileIRLPResult::TILEIR_LP_ERROR_INVALID_ARGUMENT,
                    "instantiate: null provider handle".to_string(),
                ));
            };
            let calls = slice(calls, num_calls);
            if calls.is_empty() {
                provider.log(|| "instantiate: nothing selected from this provider".to_string());
                return Ok(());
            }
            let module = instantiate_calls(provider, calls)?;
            out_module.write(module);
            Ok(())
        }
    }));
    match outcome {
        Ok(Ok(())) => TileIRLPResult::TILEIR_LP_SUCCESS,
        Ok(Err(Failure::Generic(code, message))) => {
            // SAFETY: null or writable per the contract.
            unsafe { report(out_error, ErrorStorage::generic(code, message)) };
            code
        }
        Ok(Err(Failure::Unresolved(message, failures))) => {
            // SAFETY: as above.
            unsafe { report(out_error, ErrorStorage::unresolved(message, failures)) };
            TileIRLPResult::TILEIR_LP_ERROR_UNRESOLVED_CALLS
        }
        Err(_) => {
            // SAFETY: as above.
            unsafe {
                report(
                    out_error,
                    ErrorStorage::generic(
                        TileIRLPResult::TILEIR_LP_ERROR,
                        "instantiate panicked".into(),
                    ),
                )
            };
            TileIRLPResult::TILEIR_LP_ERROR
        }
    }
}

/// Why `instantiate` fails.
enum Failure {
    Generic(TileIRLPResult, String),
    Unresolved(String, Vec<(usize, String)>),
}

/// Serve the selected calls from the payload's device code.
///
/// With precompiled LTOIR every selected implementation symbol must be one
/// the code defines, and the kernel's architecture must be the one the code
/// was compiled for.
///
/// # Safety
/// `calls` must be records the assembler filled in.
unsafe fn instantiate_calls(
    provider: &Provider,
    calls: &[TileIRResolvedCall],
) -> Result<*mut TileIRForeignModule, Failure> {
    let payload = &provider.payload;
    let mut failures = Vec::new();
    let mut call_symbols = Vec::with_capacity(calls.len());
    let mut symbols: Vec<String> = Vec::new();
    let mut arch_mismatch: Option<String> = None;
    for (index, call) in calls.iter().enumerate() {
        // SAFETY: as documented.
        let Some(call) = (unsafe { resolved(call) }) else {
            failures.push((
                index,
                "the resolved call names no implementation".to_string(),
            ));
            continue;
        };
        provider.log(|| {
            format!(
                "instantiate: call {index} wants `{}` for @{}<[{}]>({}) on {} (scope {}, {} warps per CTA)",
                call.symbol,
                call.decl,
                call.const_args.join(", "),
                call.args.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
                if call.arch.is_empty() { "any arch" } else { &call.arch },
                call.scope.0,
                call.warps_per_cta
            )
        });
        if payload.export_for_symbol(&call.symbol).is_none() {
            failures.push((
                index,
                format!(
                    "no precompiled implementation is named `{}`; this provider defines {}",
                    call.symbol,
                    payload
                        .symbols()
                        .map(|s| format!("`{s}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ));
            continue;
        }
        if let DeviceCode::Ltoir {
            arch: Some(arch), ..
        } = &payload.code
            && !call.arch.is_empty()
            && !arch_matches(arch, &call.arch)
        {
            arch_mismatch = Some(call.arch.clone());
        }
        if !symbols.contains(&call.symbol) {
            symbols.push(call.symbol.clone());
        }
        call_symbols.push((index, call.symbol));
    }
    if !failures.is_empty() {
        return Err(Failure::Unresolved(
            format!(
                "{} of {} selected call(s) cannot be served",
                failures.len(),
                calls.len()
            ),
            failures,
        ));
    }
    if let (
        Some(wanted),
        DeviceCode::Ltoir {
            arch: Some(have), ..
        },
    ) = (&arch_mismatch, &payload.code)
    {
        return Err(Failure::Generic(
            TileIRLPResult::TILEIR_LP_ERROR,
            format!(
                "the kernel targets {wanted} but this library's device code was compiled for {have}; \
                 rebuild the library with --arch {wanted}"
            ),
        ));
    }
    let DeviceCode::Ltoir { bytes, .. } = &payload.code;
    provider.log(|| {
        format!(
            "instantiate: returning {} bytes of LTOIR exporting {}",
            bytes.len(),
            symbols.join(", ")
        )
    });
    Ok(ModuleStorage::build(
        vec![CodeBufferSpec {
            name: format!("{}.ltoir", payload.name),
            data: bytes.clone(),
            output_type: TileIROutputType::TILEIR_OUTPUT_LTO_IR,
            symbols,
        }],
        call_symbols,
    ))
}

/// `sm_120` serves `sm_120` and `sm_120a`; nothing else.
fn arch_matches(compiled: &str, wanted: &str) -> bool {
    let base = |arch: &str| {
        arch.trim_end_matches(|c: char| c.is_ascii_alphabetic() && c != 'm')
            .to_string()
    };
    base(compiled) == base(wanted)
}

/// Turn a callback outcome into its result code, reporting errors.
///
/// # Safety
/// `out_error` must be null or writable.
unsafe fn finish(
    outcome: Result<Result<(), (TileIRLPResult, String)>, Box<dyn std::any::Any + Send>>,
    out_error: *mut *mut TileIRLPError,
    what: &str,
) -> TileIRLPResult {
    match outcome {
        Ok(Ok(())) => TileIRLPResult::TILEIR_LP_SUCCESS,
        Ok(Err((code, message))) => {
            // SAFETY: as documented.
            unsafe { report(out_error, ErrorStorage::generic(code, message)) };
            code
        }
        Err(_) => {
            // SAFETY: as documented.
            unsafe {
                report(
                    out_error,
                    ErrorStorage::generic(
                        TileIRLPResult::TILEIR_LP_ERROR,
                        format!("{what} panicked"),
                    ),
                )
            };
            TileIRLPResult::TILEIR_LP_ERROR
        }
    }
}

unsafe extern "C" fn free_impls(data: *const c_void) {
    if data.is_null() {
        return;
    }
    const PREFIX: usize = std::mem::size_of::<usize>();
    // SAFETY: `leak_prefixed` produced the pointer, with the length in front.
    unsafe {
        let base = data.cast::<u8>().cast_mut().sub(PREFIX);
        let len = base.cast::<usize>().read();
        dealloc(base, prefixed_layout(len));
    }
}

unsafe extern "C" fn free_module(module: *mut TileIRForeignModule) {
    // SAFETY: `instantiate` produced the pointer through `ModuleStorage::build`.
    unsafe { ModuleStorage::free(module) }
}

unsafe extern "C" fn free_error(error: *mut TileIRLPError) {
    // SAFETY: every error we report comes from `ErrorStorage`.
    unsafe { ErrorStorage::free(error) }
}

unsafe extern "C" fn destroy(handle: TileIRLinkageProviderHandle) {
    if handle.is_null() {
        return;
    }
    // SAFETY: `entry` produced the handle with `Box::into_raw`.
    let provider = unsafe { Box::from_raw(handle.cast::<Provider>()) };
    provider.log(|| "destroyed".to_string());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payload::tests::payload;
    use crate::sys::{
        TileIRDataType, TileIRDataTypeKind, TileIRDim, TileIRDim__bindgen_ty_1, TileIRDimKind,
        TileIRExecutionContext, TileIRExecutionScope, TileIRForeignDecl, TileIRForeignImpl,
        TileIRForeignType, TileIRForeignTypeKind, TileIRLPCallsite, TileIRLPCompilationContext,
    };
    use crate::view::text;

    fn instance() -> TileIRLinkageProviderInstance {
        let mut instance = TileIRLinkageProviderInstance::default();
        // SAFETY: `instance` is writable.
        let result = unsafe { entry(ABI_VERSION, &mut instance, || Ok(payload())) };
        assert_eq!(result, TileIRLPResult::TILEIR_LP_SUCCESS);
        instance
    }

    fn tile(dims: &[TileIRDim]) -> TileIRForeignType {
        TileIRForeignType {
            kind: TileIRForeignTypeKind::TILEIR_TYPE_TILE,
            shape: dims.as_ptr(),
            num_shape: dims.len(),
            dtype: TileIRDataType {
                kind: TileIRDataTypeKind::TILEIR_DTYPE_F32,
            },
        }
    }

    fn concrete(value: i64) -> TileIRDim {
        TileIRDim {
            kind: TileIRDimKind::TILEIR_DIM_CONCRETE,
            __bindgen_anon_1: TileIRDim__bindgen_ty_1 { value },
        }
    }

    #[test]
    fn the_version_gate_and_the_handle_work() {
        let mut slot = TileIRLinkageProviderInstance::default();
        let older = TileIRLinkageProviderAPIVersion {
            major: 0,
            minor: 0,
            patch: 0,
        };
        // SAFETY: `slot` is writable.
        let result = unsafe { entry(older, &mut slot, || Ok(payload())) };
        assert_eq!(result, TileIRLPResult::TILEIR_LP_ERROR_UNSUPPORTED_VERSION);
        let instance = instance();
        // SAFETY: the instance's name points into the live provider.
        assert_eq!(unsafe { text(&instance.name) }, "relu_lib");
        // SAFETY: the handle is live.
        unsafe { instance.vtable.destroy.unwrap()(instance.handle) };
    }

    #[test]
    fn provide_impls_answers_only_for_our_declarations() {
        let instance = instance();
        let dims = [concrete(256)];
        let types = [tile(&dims), tile(&dims)];
        let decl_name = "relu";
        let decl = TileIRForeignDecl {
            symbol: view(decl_name),
            const_params: std::ptr::null(),
            num_const_params: 0,
            params: types.as_ptr(),
            num_params: 2,
            results: types[1..].as_ptr(),
            num_results: 1,
            param_access: std::ptr::null(),
            num_param_access: 0,
        };
        let site = TileIRLPCallsite {
            id: 7,
            ir_preorder: 0,
            decl: &decl,
            const_arguments: std::ptr::null(),
            num_const_arguments: 0,
            arguments: types.as_ptr(),
            num_arguments: 2,
            results: types[1..].as_ptr(),
            num_results: 1,
        };
        let request = TileIRLPProvideImplsRequest {
            format_version: REQUEST_FORMAT_VERSION,
            callsites: &site,
            num_callsites: 1,
            compilation_context: TileIRLPCompilationContext::default(),
            hints: std::ptr::null(),
            num_hints: 0,
        };
        let mut data: *const c_void = std::ptr::null();
        let mut size = 0usize;
        let mut error: *mut TileIRLPError = std::ptr::null_mut();
        // SAFETY: all records are live locals.
        unsafe {
            let result = instance.vtable.provide_impls.unwrap()(
                instance.handle,
                &request,
                &mut data,
                &mut size,
                &mut error,
            );
            assert_eq!(result, TileIRLPResult::TILEIR_LP_SUCCESS);
            assert!(error.is_null());
            assert!(!data.is_null() && size > 12);
            let bytes = std::slice::from_raw_parts(data.cast::<u8>(), size);
            assert_eq!(&bytes[..8], b"\x7fTileIR\0");
            assert!(String::from_utf8_lossy(bytes).contains("relu_impl"));
            instance.vtable.free_impls.unwrap()(data);

            let other = TileIRForeignDecl {
                symbol: view("gelu"),
                ..decl
            };
            let site = TileIRLPCallsite {
                decl: &other,
                ..site
            };
            let request = TileIRLPProvideImplsRequest {
                callsites: &site,
                ..request
            };
            let result = instance.vtable.provide_impls.unwrap()(
                instance.handle,
                &request,
                &mut data,
                &mut size,
                &mut error,
            );
            assert_eq!(result, TileIRLPResult::TILEIR_LP_SUCCESS);
            assert!(data.is_null() && size == 0 && error.is_null());

            let request = TileIRLPProvideImplsRequest {
                format_version: 2,
                ..request
            };
            let result = instance.vtable.provide_impls.unwrap()(
                instance.handle,
                &request,
                &mut data,
                &mut size,
                &mut error,
            );
            assert_eq!(result, TileIRLPResult::TILEIR_LP_ERROR_INVALID_ARGUMENT);
            assert!(text(&(*error).message).contains("request format 2"));
            instance.vtable.free_error.unwrap()(error);
            instance.vtable.destroy.unwrap()(instance.handle);
        }
    }

    #[test]
    fn instantiate_serves_known_symbols_and_names_unknown_ones() {
        let instance = instance();
        let dims = [concrete(128)];
        let types = [tile(&dims)];
        let context = |arch: &'static str| TileIRExecutionContext {
            scope: TileIRExecutionScope::TILEIR_EXEC_SCOPE_THREAD,
            num_warps_per_cta: 4,
            num_ctas_per_cga: -1,
            architecture: view(arch),
        };
        let make_impl = |symbol: &'static str| TileIRForeignImpl {
            symbol: view(symbol),
            decl_symbol: view("relu"),
            execution_context_constraints: context(""),
        };
        let known = make_impl("relu_impl");
        let unknown = make_impl("relu_impl__N256");
        let call = |selected: &TileIRForeignImpl, arch: &'static str| TileIRResolvedCall {
            decl: std::ptr::null(),
            impl_: selected,
            resolved_execution_context: context(arch),
            const_arguments: std::ptr::null(),
            num_const_arguments: 0,
            arguments: types.as_ptr(),
            num_arguments: 1,
        };
        let mut module: *mut TileIRForeignModule = std::ptr::null_mut();
        let mut error: *mut TileIRLPError = std::ptr::null_mut();
        // SAFETY: all records are live locals.
        unsafe {
            let calls = [call(&known, "sm_120"), call(&known, "sm_120a")];
            let result = instance.vtable.instantiate.unwrap()(
                instance.handle,
                calls.as_ptr(),
                calls.len(),
                &mut module,
                &mut error,
            );
            assert_eq!(
                result,
                TileIRLPResult::TILEIR_LP_SUCCESS,
                "{}",
                text(&(*error).message)
            );
            let record = &*module;
            assert_eq!(record.num_code_buffers, 1);
            assert_eq!(record.num_call_symbols, 2);
            let buffer = &*record.code_buffers;
            assert_eq!(buffer.output_type, TileIROutputType::TILEIR_OUTPUT_LTO_IR);
            assert_eq!(buffer.num_symbols, 1);
            assert_eq!(text(&*buffer.symbols), "relu_impl");
            instance.vtable.free_module.unwrap()(module);

            let calls = [call(&known, "sm_120"), call(&unknown, "sm_120")];
            let result = instance.vtable.instantiate.unwrap()(
                instance.handle,
                calls.as_ptr(),
                calls.len(),
                &mut module,
                &mut error,
            );
            assert_eq!(result, TileIRLPResult::TILEIR_LP_ERROR_UNRESOLVED_CALLS);
            assert!(module.is_null());
            let unresolved = (*error).__bindgen_anon_1.unresolved_calls;
            assert_eq!(unresolved.num_calls, 1);
            assert_eq!((*unresolved.calls).call_index, 1);
            assert!(text(&(*unresolved.calls).reason).contains("relu_impl__N256"));
            instance.vtable.free_error.unwrap()(error);

            let calls = [call(&known, "sm_100")];
            let result = instance.vtable.instantiate.unwrap()(
                instance.handle,
                calls.as_ptr(),
                calls.len(),
                &mut module,
                &mut error,
            );
            assert_eq!(result, TileIRLPResult::TILEIR_LP_ERROR);
            assert!(text(&(*error).message).contains("compiled for sm_120"));
            instance.vtable.free_error.unwrap()(error);

            let result = instance.vtable.instantiate.unwrap()(
                instance.handle,
                std::ptr::null(),
                0,
                &mut module,
                &mut error,
            );
            assert_eq!(result, TileIRLPResult::TILEIR_LP_SUCCESS);
            assert!(module.is_null());
            instance.vtable.destroy.unwrap()(instance.handle);
        }
    }

    #[test]
    fn filter_and_rank_scores_everything_zero() {
        let instance = instance();
        let request = TileIRLPFilterAndRankRequest {
            format_version: REQUEST_FORMAT_VERSION,
            compilation_context: TileIRLPCompilationContext::default(),
            candidates: std::ptr::null(),
            num_candidates: 0,
        };
        let mut error: *mut TileIRLPError = std::ptr::null_mut();
        // SAFETY: live locals.
        unsafe {
            let result = instance.vtable.filter_and_rank.unwrap()(
                instance.handle,
                &request,
                std::ptr::null_mut(),
                0,
                &mut error,
            );
            assert_eq!(result, TileIRLPResult::TILEIR_LP_SUCCESS);
            let result = instance.vtable.filter_and_rank.unwrap()(
                instance.handle,
                &request,
                std::ptr::null_mut(),
                3,
                &mut error,
            );
            assert_eq!(result, TileIRLPResult::TILEIR_LP_ERROR_INVALID_ARGUMENT);
            instance.vtable.free_error.unwrap()(error);
            instance.vtable.destroy.unwrap()(instance.handle);
        }
    }

    #[test]
    fn architectures_match_by_base_name() {
        assert!(arch_matches("sm_120", "sm_120a"));
        assert!(arch_matches("sm_120a", "sm_120"));
        assert!(!arch_matches("sm_120", "sm_100"));
    }
}
