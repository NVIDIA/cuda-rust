// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolve typed library markers and finalize one batch per native provider.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use oxide_artifacts::KernelLaunchFacts;
use rustc_hir::def_id::{CRATE_DEF_INDEX, DefId};
use rustc_middle::mir::{Operand, RETURN_PLACE, Rvalue, StatementKind, TerminatorKind};
use rustc_middle::ty::{Instance, Ty, TyCtxt, TyKind, TypingEnv};
use rustc_span::Span;
use serde::{Deserialize, Serialize};

use crate::collector::CollectionResult;

const MAX_HEADER: usize = 1024 * 1024;
const MAX_LTO: usize = 256 * 1024 * 1024;

#[derive(Debug)]
pub(crate) struct LibraryCall<'tcx> {
    pub instance: Instance<'tcx>,
    pub span: Span,
}

/// Resolve definitions through the actual dependency module, then compare DefIds
/// at call sites. A user function with the same spelling is an ordinary function.
pub(crate) fn marker_definitions(tcx: TyCtxt<'_>) -> HashSet<DefId> {
    let mut definitions = HashSet::new();
    for &krate in tcx.crates(()) {
        if tcx.crate_name(krate).as_str() != "cuda_device" {
            continue;
        }
        let root = DefId {
            krate,
            index: CRATE_DEF_INDEX,
        };
        for module in tcx.module_children(root) {
            if module.ident.name.as_str() != "library" {
                continue;
            }
            let Some(module) = module.res.opt_def_id() else {
                continue;
            };
            for item in tcx.module_children(module) {
                if item.ident.name.as_str() == reserved_oxide_symbols::LIBRARY_CALL_V1
                    && let Some(def) = item.res.opt_def_id()
                {
                    definitions.insert(def);
                }
            }
        }
    }
    definitions
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Registration {
    version: u32,
    provider: String,
    executable: PathBuf,
    operation: String,
    algorithm: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(
    tag = "kind",
    content = "bits",
    rename_all = "snake_case",
    deny_unknown_fields
)]
enum Scalar {
    Signed(u16),
    Unsigned(u16),
    Float(u16),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(
    tag = "kind",
    content = "scalar",
    rename_all = "snake_case",
    deny_unknown_fields
)]
enum Argument {
    ConstPointer(Scalar),
    MutPointer(Scalar),
    Value(Scalar),
}

#[derive(Debug, Serialize)]
struct BatchRequest<'a> {
    version: u32,
    target: &'a str,
    cuda_root: PathBuf,
    requests: &'a [Request],
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq, Hash)]
struct Request {
    operation: String,
    algorithm: String,
    dtype: Scalar,
    block: [u32; 3],
    items_per_thread: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Layout {
    size: u64,
    alignment: u64,
}

impl Layout {
    fn checked(&self) -> Result<(u32, u32), String> {
        let size = u32::try_from(self.size).map_err(|_| "library layout exceeds u32")?;
        let align = u32::try_from(self.alignment).map_err(|_| "library alignment exceeds u32")?;
        if !align.is_power_of_two() {
            return Err("library alignment must be a nonzero power of two".into());
        }
        Ok((size, align))
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Specialization {
    request_index: u32,
    identity: [u8; 32],
    symbol: String,
    arguments: Vec<Argument>,
    cub_layout: Layout,
    scratch: Layout,
    scratch_free: bool,
    mutates_payload: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BatchResponse {
    version: u32,
    batch_identity: [u8; 32],
    specializations: Vec<Specialization>,
}

#[derive(Default)]
pub(crate) struct FinalizedLibraries<'tcx> {
    pub calls: Vec<(Instance<'tcx>, mir_importer::LibraryCallPlan)>,
    pub declarations: Vec<mir_importer::DeviceExternDecl>,
    pub inputs: Vec<(String, Vec<u8>)>,
    pub roots: BTreeMap<String, KernelLaunchFacts>,
}

fn scalar(ty: Ty<'_>) -> Result<Scalar, String> {
    use rustc_middle::ty::{FloatTy, IntTy, UintTy};
    match ty.kind() {
        TyKind::Int(kind) => match kind {
            IntTy::I8 => Ok(Scalar::Signed(8)),
            IntTy::I16 => Ok(Scalar::Signed(16)),
            IntTy::I32 => Ok(Scalar::Signed(32)),
            IntTy::I64 => Ok(Scalar::Signed(64)),
            _ => Err(format!("unsupported library scalar `{ty}`")),
        },
        TyKind::Uint(kind) => match kind {
            UintTy::U8 => Ok(Scalar::Unsigned(8)),
            UintTy::U16 => Ok(Scalar::Unsigned(16)),
            UintTy::U32 => Ok(Scalar::Unsigned(32)),
            UintTy::U64 => Ok(Scalar::Unsigned(64)),
            _ => Err(format!("unsupported library scalar `{ty}`")),
        },
        TyKind::Float(FloatTy::F16) => Ok(Scalar::Float(16)),
        TyKind::Float(FloatTy::F32) => Ok(Scalar::Float(32)),
        TyKind::Float(FloatTy::F64) => Ok(Scalar::Float(64)),
        _ => Err(format!(
            "unsupported library scalar `{ty}`; use a fixed-width numeric scalar"
        )),
    }
}

fn argument(ty: Ty<'_>) -> Result<Argument, String> {
    match ty.kind() {
        TyKind::RawPtr(pointee, mutability) => Ok(if mutability.is_mut() {
            Argument::MutPointer(scalar(*pointee)?)
        } else {
            Argument::ConstPointer(scalar(*pointee)?)
        }),
        _ => Ok(Argument::Value(scalar(ty)?)),
    }
}

fn read_registration<'tcx>(tcx: TyCtxt<'tcx>, witness: Ty<'tcx>) -> Result<Registration, String> {
    let TyKind::FnDef(def, args) = witness.kind() else {
        return Err(
            "library registration must be a function item, not a closure or function pointer"
                .into(),
        );
    };
    let args = args
        .no_bound_vars()
        .ok_or("registration has unresolved bound arguments")?;
    let instance = Instance::try_resolve(tcx, TypingEnv::fully_monomorphized(), *def, args)
        .map_err(|_| "cannot resolve library registration")?
        .ok_or("cannot resolve library registration")?;
    let body = tcx.instance_mir(instance.def);
    // Deliberately accept only the constant getter contract. Do not evaluate
    // arbitrary target code or search a more complicated body for a JSON string.
    if body.basic_blocks.len() != 1 {
        return Err("registration getter must return one constant string".into());
    }
    if !matches!(
        body.basic_blocks.iter().next().unwrap().terminator().kind,
        TerminatorKind::Return
    ) {
        return Err("registration getter must return one constant string".into());
    }
    let mut text = None;
    for statement in &body.basic_blocks.iter().next().unwrap().statements {
        if let StatementKind::Assign(assign) = &statement.kind {
            let (place, value) = &**assign;
            if place.as_local() != Some(RETURN_PLACE) {
                return Err("registration getter must return one constant string".into());
            }
            let Rvalue::Use(Operand::Constant(constant), _) = value else {
                return Err("registration getter must return one constant string".into());
            };
            if text.is_some() {
                return Err("registration getter writes its result more than once".into());
            }
            // The getter body still contains its generic parameters. Resolve
            // associated constants using this concrete function-item instance.
            let constant = rustc_middle::mir::ConstOperand {
                const_: instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    TypingEnv::fully_monomorphized(),
                    rustc_middle::ty::EarlyBinder::bind(tcx, constant.const_),
                ),
                ..**constant
            };
            text = crate::collector::const_str_text(tcx, &constant);
        }
    }
    let text = text.ok_or("registration getter must return one constant UTF-8 string")?;
    let registration: Registration =
        serde_json::from_str(&text).map_err(|e| format!("invalid library registration: {e}"))?;
    if registration.version != 1
        || registration.provider.is_empty()
        || !registration.executable.is_absolute()
    {
        return Err("library registration requires version 1, a provider name, and an absolute executable path".into());
    }
    Ok(registration)
}

fn decode_response(
    bytes: &[u8],
    expected_count: usize,
) -> Result<(BatchResponse, Vec<u8>), String> {
    if bytes.len() < 24 || &bytes[..8] != b"OXDLIB01" {
        return Err("invalid or truncated library response frame".into());
    }
    let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let lto_len = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    if header_len > MAX_HEADER as u64 || lto_len == 0 || lto_len > MAX_LTO as u64 {
        return Err("library response length is out of bounds".into());
    }
    let header_end = 24 + header_len as usize;
    if bytes.len() != header_end + lto_len as usize {
        return Err("library response is truncated or has trailing bytes".into());
    }
    let response: BatchResponse = serde_json::from_slice(&bytes[24..header_end])
        .map_err(|e| format!("invalid library response JSON: {e}"))?;
    if response.version != 1 || response.specializations.len() != expected_count {
        return Err(
            "library response version or specialization count does not match request".into(),
        );
    }
    let mut symbols = HashSet::new();
    let mut identities = HashSet::new();
    for (index, spec) in response.specializations.iter().enumerate() {
        if usize::try_from(spec.request_index).ok() != Some(index)
            || !symbols.insert(&spec.symbol)
            || !identities.insert(spec.identity)
            || spec.symbol.is_empty()
            || !spec
                .symbol
                .bytes()
                .enumerate()
                .all(|(i, b)| b == b'_' || b.is_ascii_alphabetic() || (i > 0 && b.is_ascii_digit()))
        {
            return Err(
                "library response has an invalid index, symbol, or duplicate identity".into(),
            );
        }
        spec.cub_layout.checked()?;
        let (size, _) = spec.scratch.checked()?;
        if spec.scratch_free != (size == 0) {
            return Err("library scratch-free flag disagrees with its layout".into());
        }
        if !matches!(
            spec.arguments.first(),
            Some(Argument::ConstPointer(_) | Argument::MutPointer(_))
        ) || !matches!(spec.arguments.get(1), Some(Argument::MutPointer(_)))
        {
            return Err(
                "library ABI requires memory and mutable payload pointers before scratch slot 2"
                    .into(),
            );
        }
        for argument in &spec.arguments {
            let (Argument::ConstPointer(scalar)
            | Argument::MutPointer(scalar)
            | Argument::Value(scalar)) = argument;
            if !matches!(
                scalar,
                Scalar::Signed(8 | 16 | 32 | 64)
                    | Scalar::Unsigned(8 | 16 | 32 | 64)
                    | Scalar::Float(16 | 32 | 64)
            ) {
                return Err("library response contains an unsupported scalar width".into());
            }
        }
    }
    Ok((response, bytes[header_end..].to_vec()))
}

fn invoke(
    executable: &PathBuf,
    request: &BatchRequest<'_>,
) -> Result<(BatchResponse, Vec<u8>), String> {
    let json = serde_json::to_vec(request).map_err(|e| e.to_string())?;
    if json.len() > MAX_HEADER {
        return Err("library request exceeds 1 MiB".into());
    }
    let mut child = Command::new(executable)
        .arg("--oxide-library-request-v1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| {
            format!(
                "cannot execute native library provider {}: {e}",
                executable.display()
            )
        })?;
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&json)
        .map_err(|e| format!("cannot send library request: {e}"))?;
    let mut bytes = Vec::new();
    child
        .stdout
        .take()
        .unwrap()
        .take((24 + MAX_HEADER + MAX_LTO + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read library response: {e}"))?;
    if bytes.len() > 24 + MAX_HEADER + MAX_LTO {
        let _ = child.kill();
    }
    let status = child
        .wait()
        .map_err(|e| format!("cannot wait for library provider: {e}"))?;
    if !status.success() {
        return Err(format!(
            "native library provider failed with {status}; see its stderr diagnostics"
        ));
    }
    decode_response(&bytes, request.requests.len())
}

fn extern_scalar(scalar: &Scalar) -> mir_importer::DeviceExternType {
    use mir_importer::DeviceExternType as E;
    match *scalar {
        Scalar::Signed(bits) | Scalar::Unsigned(bits) => E::Integer(u32::from(bits)),
        Scalar::Float(16) => E::Float16,
        Scalar::Float(32) => E::Float32,
        Scalar::Float(64) => E::Float64,
        _ => unreachable!("validated scalar type"),
    }
}

fn extern_argument(argument: &Argument) -> mir_importer::DeviceExternType {
    use mir_importer::DeviceExternType as E;
    match argument {
        Argument::Value(Scalar::Signed(bits)) if *bits < 32 => E::SignExtInteger(u32::from(*bits)),
        Argument::Value(Scalar::Unsigned(bits)) if *bits < 32 => {
            E::ZeroExtInteger(u32::from(*bits))
        }
        Argument::Value(s) => extern_scalar(s),
        Argument::ConstPointer(s) | Argument::MutPointer(s) => E::pointer_to(extern_scalar(s), 0),
    }
}

/// Finalize all call identities once, while retaining every root's reachability.
pub(crate) fn finalize<'tcx>(
    tcx: TyCtxt<'tcx>,
    collection: &CollectionResult<'tcx>,
) -> Result<FinalizedLibraries<'tcx>, String> {
    let mut result = FinalizedLibraries::default();
    if collection.library_calls.is_empty() {
        return Ok(result);
    }
    let target = std::env::var("CUDA_OXIDE_TARGET")
        .map_err(|_| "library calls require an explicit CUDA target")?;
    let target = target
        .strip_prefix("sm_")
        .map(|v| format!("compute_{v}"))
        .unwrap_or(target);
    let cuda_root = std::env::var_os("CUDA_HOME")
        .or_else(|| std::env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .ok_or("library calls require CUDA_HOME or CUDA_PATH")?;
    // BTreeMap gives deterministic invocation order. Request order is first
    // encounter order, and the response explicitly carries each request index.
    type ProviderBatch<'a, 'tcx> = Vec<(Request, Vec<&'a LibraryCall<'tcx>>)>;
    let mut batches: BTreeMap<(String, PathBuf), ProviderBatch<'_, 'tcx>> = BTreeMap::new();
    for call in &collection.library_calls {
        let args = call.instance.args;
        let report = |error: String| -> ! {
            tcx.dcx()
                .span_fatal(call.span, format!("native device library: {error}"))
        };
        let registration =
            read_registration(tcx, args.type_at(1)).unwrap_or_else(|error| report(error));
        let positive = |index| {
            let value = args
                .const_at(index)
                .try_to_target_usize(tcx)
                .ok_or("library extent is not a concrete usize")?;
            u32::try_from(value)
                .ok()
                .filter(|v| *v != 0)
                .ok_or("library extent must fit nonzero u32")
        };
        let request = Request {
            operation: registration.operation,
            algorithm: registration.algorithm,
            dtype: scalar(args.type_at(0)).unwrap_or_else(|error| report(error)),
            block: [
                positive(3)
                    .map_err(str::to_owned)
                    .unwrap_or_else(|error| report(error)),
                1,
                1,
            ],
            items_per_thread: positive(4)
                .map_err(str::to_owned)
                .unwrap_or_else(|error| report(error)),
        };
        let batch = batches
            .entry((registration.provider, registration.executable))
            .or_default();
        if let Some((_, calls)) = batch.iter_mut().find(|(existing, _)| *existing == request) {
            calls.push(call);
        } else {
            batch.push((request, vec![call]));
        }
    }
    let mut finalized = HashMap::new();
    let mut all_symbols = HashSet::new();
    for ((provider, executable), batch) in batches {
        let requests: Vec<_> = batch.iter().map(|(request, _)| request.clone()).collect();
        let batch_span = batch[0].1[0].span;
        let (response, lto) = invoke(
            &executable,
            &BatchRequest {
                version: 1,
                target: &target,
                cuda_root: cuda_root.clone(),
                requests: &requests,
            },
        )
        .unwrap_or_else(|error| {
            tcx.dcx()
                .span_fatal(batch_span, format!("native device library: {error}"))
        });
        let identity: String = response
            .batch_identity
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        result
            .inputs
            .push((format!("library-{provider}-{identity}.ltoir"), lto));
        for ((request, calls), spec) in batch.into_iter().zip(response.specializations) {
            if !all_symbols.insert(spec.symbol.clone()) {
                return Err(format!(
                    "different provider batches returned the same symbol {}",
                    spec.symbol
                ));
            }
            let (scratch_bytes, scratch_alignment) = spec.scratch.checked()?;
            let mut params: Vec<_> = spec.arguments.iter().map(extern_argument).collect();
            params.insert(
                2,
                mir_importer::DeviceExternType::pointer_to(
                    mir_importer::DeviceExternType::Integer(8),
                    0,
                ),
            );
            result.declarations.push(mir_importer::DeviceExternDecl {
                export_name: spec.symbol.clone(),
                param_types: params,
                return_type: mir_importer::DeviceExternType::Void,
                attrs: mir_importer::DeviceExternAttrs {
                    is_convergent: true,
                    is_pure: false,
                    is_readonly: false,
                },
            });
            for call in calls {
                let TyKind::Tuple(fields) = call.instance.args.type_at(2).kind() else {
                    tcx.dcx().span_fatal(
                        call.span,
                        "library Args must be a tuple of scalars and raw pointers",
                    );
                };
                let actual = fields
                    .iter()
                    .map(argument)
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap_or_else(|error| tcx.dcx().span_fatal(call.span, error));
                if actual != spec.arguments {
                    tcx.dcx().span_fatal(call.span, format!(
                        "library runtime tuple for {} disagrees with provider ABI: actual {actual:?}, expected {:?}",
                        spec.symbol, spec.arguments
                    ));
                }
                let _ = spec.mutates_payload; // Metadata, not permission to read payload.
                let plan = mir_importer::LibraryCallPlan {
                    symbol: spec.symbol.clone(),
                    scratch_bytes,
                    scratch_alignment,
                };
                if finalized
                    .insert(
                        call.instance,
                        KernelLaunchFacts {
                            exact_block: Some((
                                request.block[0],
                                request.block[1],
                                request.block[2],
                            )),
                            dynamic_shared_bytes: scratch_bytes,
                            dynamic_shared_alignment: scratch_alignment,
                        },
                    )
                    .is_none()
                {
                    result.calls.push((call.instance, plan));
                }
            }
        }
    }
    for root in collection.functions.iter().filter(|f| f.is_kernel) {
        let mut visited = HashSet::new();
        let mut pending = vec![root.instance];
        let mut root_plan: Option<KernelLaunchFacts> = None;
        while let Some(instance) = pending.pop() {
            if !visited.insert(instance) {
                continue;
            }
            if let Some(plan) = finalized.get(&instance) {
                if let Some(current) = &mut root_plan {
                    if current.exact_block != plan.exact_block {
                        tcx.dcx().span_fatal(
                            tcx.def_span(root.instance.def_id()),
                            format!(
                                "kernel {} has conflicting exact library block shapes",
                                root.export_name
                            ),
                        );
                    }
                    current.dynamic_shared_bytes =
                        current.dynamic_shared_bytes.max(plan.dynamic_shared_bytes);
                    current.dynamic_shared_alignment = current
                        .dynamic_shared_alignment
                        .max(plan.dynamic_shared_alignment);
                } else {
                    root_plan = Some(*plan);
                }
            }
            pending.extend(
                collection
                    .call_edges
                    .iter()
                    .filter(|(caller, _)| *caller == instance)
                    .map(|(_, callee)| *callee),
            );
        }
        if let Some(plan) = root_plan {
            for instance in visited {
                let path = tcx.def_path_str(instance.def_id());
                if tcx.crate_name(instance.def_id().krate).as_str() == "cuda_device"
                    && path.contains("DynamicSharedArray")
                {
                    return Err(format!(
                        "kernel {} mixes library scratch with a native dynamic shared view",
                        root.export_name
                    ));
                }
            }
            result.roots.insert(root.export_name.clone(), plan);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response() -> serde_json::Value {
        let scalar = serde_json::json!({"kind": "signed", "bits": 32});
        serde_json::json!({
            "version": 1, "batch_identity": vec![0; 32],
            "specializations": [{
                "request_index": 0, "identity": vec![1; 32], "symbol": "library_i32",
                "arguments": [
                    {"kind": "const_pointer", "scalar": scalar},
                    {"kind": "mut_pointer", "scalar": scalar}
                ],
                "cub_layout": {"size": 1, "alignment": 1},
                "scratch": {"size": 0, "alignment": 1},
                "scratch_free": true, "mutates_payload": true
            }]
        })
    }

    fn frame(response: &serde_json::Value) -> Vec<u8> {
        let json = serde_json::to_vec(response).unwrap();
        let mut bytes = b"OXDLIB01".to_vec();
        bytes.extend((json.len() as u64).to_le_bytes());
        bytes.extend(3_u64.to_le_bytes());
        bytes.extend(json);
        bytes.extend([1, 2, 3]);
        bytes
    }

    #[test]
    fn response_frame_retains_bytes_and_rejects_truncation_or_trailing_data() {
        let bytes = frame(&response());
        assert_eq!(decode_response(&bytes, 1).unwrap().1, [1, 2, 3]);
        for end in [0, 8, 16, 23, bytes.len() - 1] {
            assert!(decode_response(&bytes[..end], 1).is_err());
        }
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(decode_response(&trailing, 1).is_err());
        let mut empty_lto = bytes;
        empty_lto[16..24].copy_from_slice(&0_u64.to_le_bytes());
        assert!(decode_response(&empty_lto, 1).is_err());
    }

    #[test]
    fn response_rejects_inconsistent_identity_layout_and_abi() {
        for (field, value) in [
            ("request_index", serde_json::json!(1)),
            ("symbol", serde_json::json!("not a symbol")),
            ("scratch_free", serde_json::json!(false)),
            ("scratch", serde_json::json!({"size": 0, "alignment": 3})),
            (
                "cub_layout",
                serde_json::json!({"size": 4294967296_u64, "alignment": 1}),
            ),
            ("arguments", serde_json::json!([])),
        ] {
            let mut value_response = response();
            value_response["specializations"][0][field] = value;
            assert!(
                decode_response(&frame(&value_response), 1).is_err(),
                "{field}"
            );
        }
        let mut duplicate = response();
        let mut second = duplicate["specializations"][0].clone();
        second["request_index"] = 1.into();
        duplicate["specializations"]
            .as_array_mut()
            .unwrap()
            .push(second);
        assert!(decode_response(&frame(&duplicate), 2).is_err());
        assert!(decode_response(&frame(&response()), 2).is_err());
        let mut version = response();
        version["version"] = 2.into();
        assert!(decode_response(&frame(&version), 1).is_err());
    }
}
