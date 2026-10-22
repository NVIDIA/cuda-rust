// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolve typed library markers and finalize one batch per native provider.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use oxide_artifacts::KernelLaunchFacts;
use rustc_hir::def_id::{CRATE_DEF_INDEX, DefId};
use rustc_middle::mir::{BasicBlock, Operand, RETURN_PLACE, Rvalue, StatementKind, TerminatorKind};
use rustc_middle::ty::consts::ConstExt;
use rustc_middle::ty::{Instance, Ty, TyCtxt, TyKind, TypingEnv};
use rustc_span::Span;
use serde::{Deserialize, Serialize};

use crate::collector::CollectionResult;

const MAX_HEADER: usize = 1024 * 1024;
const MAX_LTO: usize = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MarkerKind {
    Raw,
    CheckedBlock,
    CheckedBlockResource,
    CheckedBlockStorage,
    StorageNew,
    StorageReconfigure,
    StorageReserve,
    StorageRelease,
}

impl MarkerKind {
    pub(crate) fn checked(self) -> bool {
        matches!(
            self,
            Self::CheckedBlock
                | Self::CheckedBlockResource
                | Self::CheckedBlockStorage
                | Self::StorageNew
                | Self::StorageReconfigure
                | Self::StorageReserve
                | Self::StorageRelease
        )
    }

    pub(crate) fn native(self) -> bool {
        matches!(
            self,
            Self::Raw | Self::CheckedBlock | Self::CheckedBlockResource | Self::CheckedBlockStorage
        )
    }

    pub(crate) fn storage(self) -> bool {
        matches!(
            self,
            Self::CheckedBlockStorage
                | Self::StorageNew
                | Self::StorageReconfigure
                | Self::StorageReserve
                | Self::StorageRelease
        )
    }

    pub(crate) fn resource(self) -> bool {
        matches!(self, Self::CheckedBlockResource | Self::CheckedBlockStorage)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EdgeKind {
    Direct,
    Callable,
    Drop,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CallSite<'tcx> {
    pub caller: Instance<'tcx>,
    pub block: BasicBlock,
    pub callee: Instance<'tcx>,
    pub kind: EdgeKind,
}

#[derive(Debug)]
pub(crate) struct LibraryCall<'tcx> {
    pub instance: Instance<'tcx>,
    pub site: CallSite<'tcx>,
    pub kind: MarkerKind,
    pub span: Span,
}

/// Resolve definitions through the actual dependency module, then compare DefIds
/// at call sites. A user function with the same spelling is an ordinary function.
pub(crate) fn marker_definitions(tcx: TyCtxt<'_>) -> HashMap<DefId, MarkerKind> {
    let mut definitions = HashMap::new();
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
                let kind = match item.ident.name.as_str() {
                    reserved_oxide_symbols::LIBRARY_CALL_V1 => MarkerKind::Raw,
                    reserved_oxide_symbols::LIBRARY_CHECKED_BLOCK_CALL_V1 => {
                        MarkerKind::CheckedBlock
                    }
                    reserved_oxide_symbols::LIBRARY_CHECKED_BLOCK_RESOURCE_CALL_V1 => {
                        MarkerKind::CheckedBlockResource
                    }
                    reserved_oxide_symbols::LIBRARY_CHECKED_BLOCK_STORAGE_CALL_V1 => {
                        MarkerKind::CheckedBlockStorage
                    }
                    reserved_oxide_symbols::LIBRARY_STORAGE_NEW_V1 => MarkerKind::StorageNew,
                    reserved_oxide_symbols::LIBRARY_STORAGE_RECONFIGURE_V1 => {
                        MarkerKind::StorageReconfigure
                    }
                    reserved_oxide_symbols::LIBRARY_STORAGE_RESERVE_V1 => {
                        MarkerKind::StorageReserve
                    }
                    reserved_oxide_symbols::LIBRARY_STORAGE_RELEASE_V1 => {
                        MarkerKind::StorageRelease
                    }
                    _ => continue,
                };
                if let Some(def) = item.res.opt_def_id() {
                    definitions.insert(def, kind);
                }
            }
        }
    }
    definitions
}

fn device_type(tcx: TyCtxt<'_>, module_name: &str, name: &str) -> Option<DefId> {
    for &krate in tcx.crates(()) {
        if tcx.crate_name(krate).as_str() != "cuda_device" {
            continue;
        }
        let root = DefId {
            krate,
            index: CRATE_DEF_INDEX,
        };
        let module = tcx
            .module_children(root)
            .iter()
            .find(|item| item.ident.name.as_str() == module_name)?
            .res
            .opt_def_id()?;
        return tcx
            .module_children(module)
            .iter()
            .find(|item| item.ident.name.as_str() == name)
            .and_then(|item| item.res.opt_def_id());
    }
    None
}

/// Genuine collective carrier, independent of aliases or source spelling.
pub(crate) fn collective_carrier<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Option<(Ty<'tcx>, u64)> {
    let TyKind::Adt(def, args) = ty.kind() else {
        return None;
    };
    if Some(def.did()) != device_type(tcx, "disjoint", "DisjointSlice") {
        return None;
    }
    let mut types = args.types();
    let element = types.next()?;
    let TyKind::Adt(space, args) = types.next()?.kind() else {
        return None;
    };
    if Some(space.did()) != device_type(tcx, "disjoint", "CollectiveTiles") {
        return None;
    }
    Some((element, args.const_at(0).try_to_target_usize(tcx)?))
}

pub(crate) fn resource_parameter<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    collective_carrier(tcx, ty).is_some()
        || matches!(ty.kind(), TyKind::Ref(_, inner, rustc_hir::Mutability::Not)
            if matches!(inner.kind(), TyKind::Slice(_)))
}

pub(crate) fn resource_write(call: &LibraryCall<'_>) -> bool {
    call.instance
        .args
        .const_at(if call.kind == MarkerKind::CheckedBlockStorage {
            7
        } else {
            6
        })
        .try_to_leaf()
        .and_then(|value| value.try_to_bool().ok())
        .expect("validated resource access")
}

fn validate_resource<'tcx>(tcx: TyCtxt<'tcx>, call: &LibraryCall<'tcx>) -> Result<(), String> {
    let args = call.instance.args;
    let write = args
        .const_at(if call.kind == MarkerKind::CheckedBlockStorage {
            7
        } else {
            6
        })
        .try_to_leaf()
        .and_then(|value| value.try_to_bool().ok())
        .ok_or("resource access mode is not constant")?;
    let TyKind::Ref(_, inner, mutability) = args.type_at(2).kind() else {
        return Err(
            "resource witness must borrow a genuine collective carrier or read-only slice".into(),
        );
    };
    let element = if let TyKind::Slice(element) = inner.kind()
        && !write
        && *mutability == rustc_hir::Mutability::Not
    {
        *element
    } else {
        let (element, capacity) = collective_carrier(tcx, *inner)
            .ok_or("resource witness requires DisjointSlice<T, CollectiveTiles<C>>")?;
        if capacity == 0 {
            return Err("resource carriers require positive capacity".into());
        }
        if write && *mutability != rustc_hir::Mutability::Mut {
            return Err("resource writes require a mutable carrier witness".into());
        }
        element
    };
    if element != args.type_at(0) {
        return Err("resource witness element type disagrees with the library element type".into());
    }
    Ok(())
}

fn storage_extent<'tcx>(
    tcx: TyCtxt<'tcx>,
    args: rustc_middle::ty::GenericArgsRef<'tcx>,
    index: usize,
) -> Result<u32, String> {
    args.const_at(index)
        .try_to_target_usize(tcx)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| "storage extent must be a concrete usize fitting u32".into())
}
fn storage_boolean(
    args: rustc_middle::ty::GenericArgsRef<'_>,
    index: usize,
) -> Result<bool, String> {
    args.const_at(index)
        .try_to_leaf()
        .and_then(|n| n.try_to_bool().ok())
        .ok_or_else(|| "storage policy must be a concrete bool".into())
}
fn storage_alignment(value: u32) -> Result<u32, String> {
    if value != 0 && !value.is_power_of_two() {
        return Err("storage alignment must be zero or a power of two".into());
    }
    Ok(value)
}

fn storage_policy<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Result<mir_importer::LibraryStoragePolicy, String> {
    let TyKind::Adt(def, args) = ty.kind() else {
        return Err("expected a genuine StorageDescriptor".into());
    };
    if Some(def.did()) != device_type(tcx, "library", "StorageDescriptor") {
        return Err("expected a genuine StorageDescriptor".into());
    }
    Ok(mir_importer::LibraryStoragePolicy {
        capacity: storage_extent(tcx, args, 0)?,
        alignment: storage_alignment(storage_extent(tcx, args, 1)?)?,
        auto_sync: storage_boolean(args, 2)?,
        exclusive: storage_boolean(args, 3)?,
    })
}

pub(crate) fn storage_handle_type(tcx: TyCtxt<'_>, ty: Ty<'_>) -> bool {
    matches!(ty.kind(), TyKind::Adt(def, _) if ["StorageDescriptor","StorageReservation"].iter().any(|name|Some(def.did()) == device_type(tcx, "library", name)))
}

fn storage_borrow_policy<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
) -> Result<mir_importer::LibraryStoragePolicy, String> {
    let TyKind::Ref(_, inner, rustc_hir::Mutability::Mut) = ty.kind() else {
        return Err("storage must mutably borrow a genuine descriptor".into());
    };
    storage_policy(tcx, *inner)
}

fn storage_new_policy<'tcx>(
    tcx: TyCtxt<'tcx>,
    call: &LibraryCall<'tcx>,
) -> Result<mir_importer::LibraryStoragePolicy, String> {
    let args = call.instance.args;
    Ok(mir_importer::LibraryStoragePolicy {
        capacity: storage_extent(tcx, args, 0)?,
        alignment: storage_alignment(storage_extent(tcx, args, 1)?)?,
        auto_sync: storage_boolean(args, 2)?,
        exclusive: storage_boolean(args, 3)?,
    })
}

fn storage_reserve_layout<'tcx>(
    tcx: TyCtxt<'tcx>,
    call: &LibraryCall<'tcx>,
) -> Result<(u32, u32), String> {
    let args = call.instance.args;
    let element = args.type_at(0);
    scalar(element)?;
    // f16 is not in the frozen ten-type reservation set.
    if matches!(
        element.kind(),
        TyKind::Float(rustc_middle::ty::FloatTy::F16)
    ) {
        return Err("storage reservations require one of the ten fixed-width numeric types".into());
    }
    let count = storage_extent(tcx, args, 2)?;
    if count == 0 {
        return Err("storage reservation extent must be positive".into());
    }
    let array = Ty::new_array(tcx, element, u64::from(count));
    let layout = tcx
        .layout_of(TypingEnv::fully_monomorphized().as_query_input(array))
        .map_err(|_| "cannot compute storage reservation layout")?;
    let bytes =
        u32::try_from(layout.size.bytes()).map_err(|_| "storage reservation exceeds u32")?;
    let align = u32::try_from(layout.align.abi.bytes())
        .map_err(|_| "storage reservation alignment exceeds u32")?;
    Ok((
        bytes,
        align.max(storage_alignment(storage_extent(tcx, args, 3)?)?),
    ))
}

fn validate_storage<'tcx>(tcx: TyCtxt<'tcx>, call: &LibraryCall<'tcx>) -> Result<(), String> {
    let args = call.instance.args;
    match call.kind {
        MarkerKind::StorageNew => {
            storage_new_policy(tcx, call)?;
        }
        MarkerKind::StorageReconfigure => {
            storage_policy(tcx, args.type_at(0))?;
            storage_policy(tcx, args.type_at(1))?;
        }
        MarkerKind::StorageReserve => {
            storage_policy(tcx, args.type_at(1))?;
            storage_reserve_layout(tcx, call)?;
        }
        MarkerKind::StorageRelease => {
            scalar(args.type_at(0))?;
        }
        MarkerKind::CheckedBlockStorage => {
            let ty = args.type_at(3);
            if !matches!(ty.kind(), TyKind::Tuple(fields) if fields.is_empty()) {
                storage_borrow_policy(tcx, ty)?;
            }
        }
        _ => {}
    }
    Ok(())
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

/// Owned provider results, independent of Rust body and proof state.
#[derive(Default)]
pub(crate) struct ProviderLibraries {
    pub native_plans: mir_importer::RootedNativePlans,
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
        ) || !matches!(
            spec.arguments.get(1),
            Some(Argument::ConstPointer(_) | Argument::MutPointer(_))
        ) {
            return Err(
                "library ABI requires memory and payload pointers before scratch slot 2".into(),
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

fn call_arguments(call: &LibraryCall<'_>) -> Result<Vec<Argument>, String> {
    let args = call.instance.args;
    let first = if call.kind == MarkerKind::CheckedBlockStorage {
        4
    } else if call.kind.resource() {
        3
    } else {
        2
    };
    let TyKind::Tuple(uniform) = args.type_at(first).kind() else {
        return Err("library arguments must be a tuple".into());
    };
    if call.kind == MarkerKind::Raw {
        return uniform.iter().map(argument).collect();
    }
    let TyKind::Tuple(payload) = args.type_at(first + 1).kind() else {
        return Err("checked block payload must be a tuple".into());
    };
    let dtype = scalar(args.type_at(0))?;
    let uniform = uniform
        .iter()
        .map(argument)
        .collect::<Result<Vec<_>, _>>()?;
    let payload = payload
        .iter()
        .map(argument)
        .collect::<Result<Vec<_>, _>>()?;
    if !matches!(uniform.first(), Some(Argument::ConstPointer(t) | Argument::MutPointer(t)) if *t == dtype)
        || !uniform[1..].iter().all(|a| matches!(a, Argument::Value(_)))
        || !matches!(payload.first(), Some(Argument::ConstPointer(t) | Argument::MutPointer(t)) if *t == dtype)
        || !payload
            .iter()
            .all(|a| matches!(a, Argument::ConstPointer(_) | Argument::MutPointer(_)))
    {
        return Err("checked block call requires (memory pointer, scalar controls...) and (payload pointers...) with matching element types".into());
    }
    Ok(uniform[..1]
        .iter()
        .chain(&payload)
        .chain(&uniform[1..])
        .cloned()
        .collect())
}

/// Source semantics authenticated without provider or participation facts.
fn authenticated_metadata<'tcx>(
    tcx: TyCtxt<'tcx>,
    call: &LibraryCall<'tcx>,
) -> Result<mir_importer::LibraryCallMetadata, String> {
    use mir_importer::{
        LibraryAccess, LibraryNativeMetadata as Native, LibraryReserveLayout,
        LibraryRoleAttr as Role, LibraryScalar, LibraryStorageArg, LibraryWitness, NativeSpec,
    };
    let args = call.instance.args;
    let positive = |index| {
        let value = args
            .const_at(index)
            .try_to_target_usize(tcx)
            .ok_or("library extent is not a concrete usize")?;
        u32::try_from(value)
            .ok()
            .filter(|value| *value != 0)
            .ok_or("library extent must fit nonzero u32")
    };
    let scalar_attr = |ty| {
        scalar(ty).map(|value| match value {
            Scalar::Signed(bits) => LibraryScalar::Signed(bits),
            Scalar::Unsigned(bits) => LibraryScalar::Unsigned(bits),
            Scalar::Float(bits) => LibraryScalar::Float(bits),
        })
    };
    let role = if call.kind.native() {
        let spec = NativeSpec {
            scalar: scalar_attr(args.type_at(0))?,
            items: positive(if call.kind == MarkerKind::CheckedBlockStorage {
                6
            } else if call.kind.resource() {
                5
            } else {
                4
            })?,
        };
        let access_witness = || -> Result<_, String> {
            let TyKind::Ref(_, inner, _) = args.type_at(2).kind() else {
                return Err("resource witness must be a reference".into());
            };
            let witness = if matches!(inner.kind(), TyKind::Slice(_)) {
                LibraryWitness::ReadOnlySlice
            } else {
                let (_, capacity) = collective_carrier(tcx, *inner)
                    .ok_or("resource witness requires DisjointSlice<T, CollectiveTiles<C>>")?;
                LibraryWitness::CollectiveCarrier(capacity)
            };
            let access = if resource_write(call) {
                LibraryAccess::Write
            } else {
                LibraryAccess::Read
            };
            Ok((access, witness))
        };
        Role::Native(match call.kind {
            MarkerKind::Raw => Native::Raw {
                spec,
                block_threads: positive(3)?,
            },
            MarkerKind::CheckedBlock => Native::CheckedBlock { spec },
            MarkerKind::CheckedBlockResource => {
                let (access, witness) = access_witness()?;
                Native::CheckedBlockResource {
                    spec,
                    access,
                    witness,
                }
            }
            MarkerKind::CheckedBlockStorage => {
                let (access, witness) = access_witness()?;
                let ty = args.type_at(3);
                let storage = if matches!(ty.kind(), TyKind::Tuple(fields) if fields.is_empty()) {
                    LibraryStorageArg::Implicit
                } else {
                    LibraryStorageArg::Explicit(storage_borrow_policy(tcx, ty)?)
                };
                Native::CheckedBlockStorage {
                    spec,
                    access,
                    witness,
                    storage,
                }
            }
            _ => unreachable!("native kind checked above"),
        })
    } else {
        match call.kind {
            MarkerKind::StorageNew => Role::StorageNew(storage_new_policy(tcx, call)?),
            MarkerKind::StorageReconfigure => Role::StorageReconfigure {
                old: storage_policy(tcx, args.type_at(0))?,
                new: storage_policy(tcx, args.type_at(1))?,
            },
            MarkerKind::StorageReserve => {
                let (bytes, alignment) = storage_reserve_layout(tcx, call)?;
                Role::StorageReserve {
                    policy: storage_policy(tcx, args.type_at(1))?,
                    layout: LibraryReserveLayout {
                        scalar: scalar_attr(args.type_at(0))?,
                        count: positive(2)?,
                        bytes,
                        alignment,
                    },
                }
            }
            MarkerKind::StorageRelease => Role::StorageRelease,
            _ => unreachable!("all marker kinds covered"),
        }
    };
    Ok(mir_importer::LibraryCallMetadata {
        marker: tcx.symbol_name(call.instance).name.to_string(),
        role,
    })
}

/// Rust identities stay at the import boundary; prepared MIR owns the proof.
#[derive(Default)]
pub(crate) struct AuthenticatedLibraries<'tcx> {
    pub semantics: BTreeMap<mir_importer::LibraryCallSite, mir_importer::LibraryCallMetadata>,
    pub deferred: HashSet<Instance<'tcx>>,
    pub checked_ancestors: HashSet<Instance<'tcx>>,
    pub selected_call_targets:
        BTreeMap<mir_importer::LibraryCallSite, mir_importer::LibraryCallTargetAttr>,
    pub rooted_functions: BTreeMap<String, BTreeSet<String>>,
    pub call_targets: BTreeMap<mir_importer::LibraryCallSite, BTreeSet<String>>,
}

pub(crate) fn authenticate<'tcx>(
    tcx: TyCtxt<'tcx>,
    collection: &CollectionResult<'tcx>,
) -> Result<AuthenticatedLibraries<'tcx>, String> {
    let mut result = AuthenticatedLibraries::default();
    let definitions = marker_definitions(tcx);
    let mut sites = HashSet::new();
    for call in &collection.library_calls {
        if definitions.get(&call.instance.def_id()) != Some(&call.kind) {
            return Err("library marker is not a genuine registered definition".into());
        }
        validate_storage(tcx, call).unwrap_or_else(|error| tcx.dcx().span_fatal(call.span, error));
        if call.kind.native() {
            call_arguments(call).unwrap_or_else(|error| tcx.dcx().span_fatal(call.span, error));
        }
        if call.kind.resource() {
            validate_resource(tcx, call)
                .unwrap_or_else(|error| tcx.dcx().span_fatal(call.span, error));
        }
        if !sites.insert((call.site.caller, call.site.block)) {
            return Err("duplicate authenticated library source site".into());
        }
    }
    let names: HashMap<_, _> = collection
        .functions
        .iter()
        .map(|function| (function.instance, function.export_name.clone()))
        .collect();
    let mut rooted = HashSet::new();
    for root in collection
        .functions
        .iter()
        .filter(|function| function.is_kernel)
    {
        let reached = reachable_instances(collection, root.instance);
        result.rooted_functions.insert(
            root.export_name.clone(),
            reached
                .iter()
                .filter_map(|instance| names.get(instance).cloned())
                .collect(),
        );
        rooted.extend(reached);
    }
    result.checked_ancestors = checked_helper_ancestors(collection);
    result.deferred = result.checked_ancestors.clone();
    result
        .deferred
        .retain(|instance| !rooted.contains(instance));
    // A standalone checked helper may still have unevaluated generic extents.
    // Its consumer supplies the concrete root before typed roles are required.
    for call in &collection.library_calls {
        if !rooted.contains(&call.site.caller) {
            continue;
        }
        let caller = names
            .get(&call.site.caller)
            .ok_or("authenticated library caller was not collected")?
            .clone();
        let metadata = authenticated_metadata(tcx, call)
            .unwrap_or_else(|error| tcx.dcx().span_fatal(call.span, error));
        result.semantics.insert(
            mir_importer::LibraryCallSite {
                caller,
                block: call.site.block.index(),
            },
            metadata,
        );
    }
    let mut selected_sites = HashSet::new();
    for edge in &collection.call_edges {
        if result.deferred.contains(&edge.caller) {
            continue;
        }
        let first = selected_sites.insert((edge.caller, edge.block));
        let target = names.get(&edge.callee).or_else(|| {
            collection
                .device_externs
                .iter()
                .find(|declaration| declaration.def_id == edge.callee.def_id())
                .map(|declaration| &declaration.export_name)
        });
        if let (Some(caller), Some(target)) = (names.get(&edge.caller), target) {
            let site = mir_importer::LibraryCallSite {
                caller: caller.clone(),
                block: edge.block.index(),
            };
            if first {
                result.selected_call_targets.insert(
                    site.clone(),
                    mir_importer::LibraryCallTargetAttr {
                        kind: match edge.kind {
                            EdgeKind::Direct => mir_importer::LibraryCallKind::Direct,
                            EdgeKind::Callable => mir_importer::LibraryCallKind::Callable,
                            EdgeKind::Drop => mir_importer::LibraryCallKind::Drop,
                        },
                        export: target.clone(),
                        marker: tcx.symbol_name(edge.callee).name.to_string(),
                    },
                );
            }
            result
                .call_targets
                .entry(site)
                .or_default()
                .insert(target.clone());
        }
    }
    for helper in &result.deferred {
        let reached = reachable_instances(collection, *helper);
        if let Some(call) = collection
            .library_calls
            .iter()
            .find(|call| call.kind == MarkerKind::Raw && reached.contains(&call.site.caller))
        {
            tcx.dcx().span_fatal(
                call.span,
                "raw and checked library calls cannot be mixed in one checked helper path",
            );
        }
    }
    Ok(result)
}

/// Resolve original-source requests only after the prepared MIR proof.
pub(crate) fn finalize_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    collection: &CollectionResult<'tcx>,
    checked_shapes: &BTreeMap<String, [u32; 3]>,
) -> Result<ProviderLibraries, String> {
    let mut result = ProviderLibraries::default();
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
    type RootedCall<'a, 'tcx> = (&'a LibraryCall<'tcx>, &'a str);
    type ProviderBatch<'a, 'tcx> = Vec<(Request, Vec<RootedCall<'a, 'tcx>>)>;
    let mut batches: BTreeMap<(String, PathBuf), ProviderBatch<'_, 'tcx>> = BTreeMap::new();
    let affected = checked_helper_ancestors(collection);
    for root in collection.functions.iter().filter(|f| f.is_kernel) {
        let reached = reachable_instances(collection, root.instance);
        let calls: Vec<_> = collection
            .library_calls
            .iter()
            .filter(|c| reached.contains(&c.site.caller))
            .collect();
        if calls.is_empty() {
            continue;
        }
        let storage_root = calls.iter().any(|call| call.kind.storage());
        for instance in &reached {
            if !storage_root
                && tcx.crate_name(instance.def_id().krate).as_str() == "cuda_device"
                && tcx
                    .def_path_str(instance.def_id())
                    .contains("DynamicSharedArray")
            {
                return Err(format!(
                    "kernel {} mixes library scratch with a native dynamic shared view",
                    root.export_name
                ));
            }
        }
        if checked_shapes.contains_key(&root.export_name) {
            for function in &collection.functions {
                if !function.is_kernel
                    && reached.contains(&function.instance)
                    && affected.contains(&function.instance)
                    && tcx
                        .codegen_fn_attrs(function.instance.def_id())
                        .contains_extern_indicator()
                {
                    tcx.dcx().struct_span_fatal(tcx.def_span(function.instance.def_id()),
                        "fixed external symbols are not supported on checked collective helper paths")
                        .with_note(format!("helper is reachable from kernel {}", root.export_name))
                        .with_help("remove the helper's #[no_mangle] or #[export_name] attribute").emit();
                }
            }
        }
        for call in calls.into_iter().filter(|call| call.kind.native()) {
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
            let block = if call.kind.checked() {
                checked_shapes
                    .get(&root.export_name)
                    .copied()
                    .ok_or_else(|| {
                        format!(
                            "checked library root {} has no admitted block shape",
                            root.export_name
                        )
                    })?
            } else {
                [
                    positive(3)
                        .map_err(str::to_owned)
                        .unwrap_or_else(|error| report(error)),
                    1,
                    1,
                ]
            };
            let request = Request {
                operation: registration.operation,
                algorithm: registration.algorithm,
                dtype: scalar(args.type_at(0)).unwrap_or_else(|error| report(error)),
                block,
                items_per_thread: positive(if call.kind == MarkerKind::CheckedBlockStorage {
                    6
                } else if call.kind.resource() {
                    5
                } else {
                    4
                })
                .map_err(str::to_owned)
                .unwrap_or_else(|error| report(error)),
            };
            let batch = batches
                .entry((registration.provider, registration.executable))
                .or_default();
            if let Some((_, calls)) = batch.iter_mut().find(|(existing, _)| *existing == request) {
                calls.push((call, &root.export_name));
            } else {
                batch.push((request, vec![(call, &root.export_name)]));
            }
        }
    }
    let mut all_symbols = HashSet::new();
    for ((provider, executable), batch) in batches {
        let requests: Vec<_> = batch.iter().map(|(request, _)| request.clone()).collect();
        let batch_span = batch[0].1[0].0.span;
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
            for (call, root) in calls {
                if call.kind == MarkerKind::CheckedBlockResource && !spec.scratch_free {
                    tcx.dcx().span_fatal(
                        call.span,
                        "resource-checked block calls currently require a scratch-free provider",
                    );
                }
                let actual = call_arguments(call)
                    .unwrap_or_else(|error| tcx.dcx().span_fatal(call.span, error));
                if actual != spec.arguments {
                    tcx.dcx().span_fatal(call.span, format!(
                        "library runtime tuple for {} disagrees with provider ABI: actual {actual:?}, expected {:?}",
                        spec.symbol, spec.arguments
                    ));
                }
                let _ = spec.mutates_payload;
                let plan = mir_importer::LibraryCallPlan {
                    operation: mir_importer::LibraryOperation::Native,
                    storage_witness: call.kind == MarkerKind::CheckedBlockStorage,
                    scratch_offset: 0,
                    planned_root: None,
                    barrier_before: false,
                    barrier_after: false,
                    symbol: spec.symbol.clone(),
                    marker: tcx.symbol_name(call.instance).name.to_string(),
                    checked_block: call.kind.checked(),
                    resource_witness: call.kind.resource(),
                    scratch_bytes,
                    scratch_alignment,
                };
                let caller = collection
                    .functions
                    .iter()
                    .find(|function| function.instance == call.site.caller)
                    .ok_or("collected library caller is missing")?
                    .export_name
                    .clone();
                let site = mir_importer::LibraryCallSite {
                    caller,
                    block: call.site.block.index(),
                };
                let plans = result.native_plans.entry(root.to_string()).or_default();
                if let Some(previous) = plans.insert(site, plan.clone())
                    && previous != plan
                {
                    return Err("library call site has conflicting provider specializations".into());
                }
                let facts = KernelLaunchFacts {
                    exact_block: Some((request.block[0], request.block[1], request.block[2])),
                    dynamic_shared_bytes: scratch_bytes,
                    dynamic_shared_alignment: scratch_alignment,
                };
                if let Some(current) = result.roots.get_mut(root) {
                    if current.exact_block != facts.exact_block {
                        tcx.dcx().span_fatal(
                            call.span,
                            format!("kernel {root} has conflicting exact library block shapes"),
                        );
                    }
                    current.dynamic_shared_bytes = current.dynamic_shared_bytes.max(scratch_bytes);
                    current.dynamic_shared_alignment =
                        current.dynamic_shared_alignment.max(scratch_alignment);
                } else {
                    result.roots.insert(root.to_string(), facts);
                }
            }
        }
    }
    Ok(result)
}

/// Only bodies that can reach a checked marker need root-specific emission.
fn checked_helper_ancestors<'tcx>(collection: &CollectionResult<'tcx>) -> HashSet<Instance<'tcx>> {
    let mut affected = HashSet::new();
    let mut pending: Vec<_> = collection
        .library_calls
        .iter()
        .filter(|call| call.kind.checked())
        .map(|call| call.site.caller)
        .collect();
    while let Some(instance) = pending.pop() {
        if affected.insert(instance) {
            pending.extend(
                collection
                    .call_edges
                    .iter()
                    .filter(|edge| edge.callee == instance)
                    .map(|edge| edge.caller),
            );
        }
    }
    affected
}

pub(crate) fn reachable_instances<'tcx>(
    collection: &CollectionResult<'tcx>,
    root: Instance<'tcx>,
) -> HashSet<Instance<'tcx>> {
    let mut reached = HashSet::new();
    let mut pending = vec![root];
    while let Some(instance) = pending.pop() {
        if reached.insert(instance) {
            pending.extend(
                collection
                    .call_edges
                    .iter()
                    .filter(|site| site.caller == instance)
                    .map(|site| site.callee),
            );
        }
    }
    reached
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
            (
                "arguments",
                serde_json::json!([
                    {"kind": "const_pointer", "scalar": {"kind": "signed", "bits": 32}},
                    {"kind": "value", "scalar": {"kind": "signed", "bits": 32}}
                ]),
            ),
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

    #[test]
    fn response_preserves_const_payload_pointer_mutability() {
        let mut value = response();
        value["specializations"][0]["arguments"][1]["kind"] = "const_pointer".into();
        value["specializations"][0]["mutates_payload"] = false.into();
        let (decoded, _) = decode_response(&frame(&value), 1).unwrap();
        assert_eq!(
            decoded.specializations[0].arguments[1],
            Argument::ConstPointer(Scalar::Signed(32))
        );
        assert_ne!(
            decoded.specializations[0].arguments,
            decode_response(&frame(&response()), 1)
                .unwrap()
                .0
                .specializations[0]
                .arguments
        );
    }
}
