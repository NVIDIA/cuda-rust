/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Static identity and initializer plumbing.

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::types::is_cuda_device_adt;
use crate::translator::values::is_constant_wrapper_type;
use pliron::location::Location;
use pliron::{input_err, input_error_noloc};
use rustc_public::CrateDef;
use rustc_public::CrateDefType;
use rustc_public::mir;
use rustc_public::ty::ConstantKind;

pub(super) struct SharedStaticSourceIdentity {
    pub(super) name: String,
    pub(super) key: String,
}

/// Resolve a whole-static shared allocation to both its display path and its
/// injective compiler identity.
///
/// `StaticDef::name()` deliberately omits DefPath disambiguators, so distinct
/// same-leaf nested statics can have the same display path. The mangled static
/// instance remains unique and is used for both deduplication and the full-debug
/// side-table join. Interior pointers are rejected: a `SharedArray` or
/// `Barrier` allocation always represents the entire static.
pub(super) fn shared_static_source_identity(
    constant: &mir::ConstOperand,
) -> Option<SharedStaticSourceIdentity> {
    let ConstantKind::Allocated(allocation) = constant.const_.kind() else {
        return None;
    };
    if allocation.provenance.ptrs.len() != 1 {
        return None;
    }
    let &(relocation_offset, _) = allocation.provenance.ptrs.first()?;
    let target = static_target_from_allocation_at(allocation, relocation_offset)
        .ok()
        .flatten()?;
    if target.byte_offset != 0 {
        return None;
    }
    let name = target.static_def.name();
    let key = crate::device_static_global_key(&target.static_def);
    Some(SharedStaticSourceIdentity { name, key })
}

/// Historical best-effort source label used by non-Full builds.
///
/// This intentionally performs no strict validation: the comment label must
/// never turn an otherwise valid translation into an error.
pub(super) fn shared_static_source_name(constant: &mir::ConstOperand) -> Option<String> {
    use rustc_public::mir::alloc::GlobalAlloc;

    let ConstantKind::Allocated(allocation) = constant.const_.kind() else {
        return None;
    };
    let &(_, provenance) = allocation.provenance.ptrs.first()?;
    match GlobalAlloc::from(provenance.0) {
        GlobalAlloc::Static(static_def) => Some(static_def.name()),
        _ => None,
    }
}

/// Resolve a constant pointer/reference to the Rust static it points at, if any.
///
/// The source allocation stores the pointer's byte addend at the relocation
/// offset, while the provenance entry identifies the target allocation. Keeping
/// both pieces together prevents interior pointers from silently degrading to
/// the static's base address. Anonymous allocations return `None`.
pub(super) struct StaticPointerTarget {
    pub(super) static_def: rustc_public::mir::mono::StaticDef,
    pub(super) byte_offset: u64,
}

pub(super) fn static_target_from_constant(
    constant: &mir::ConstOperand,
    loc: Location,
) -> TranslationResult<Option<StaticPointerTarget>> {
    let ConstantKind::Allocated(allocation) = constant.const_.kind() else {
        return Ok(None);
    };

    if allocation.is_null().unwrap_or(false) {
        return Ok(None);
    }

    let Some(&(relocation_offset, _)) = allocation.provenance.ptrs.first() else {
        return Ok(None);
    };

    if allocation.provenance.ptrs.len() != 1 {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "constant pointer contains {} provenance entries; expected one static target",
                allocation.provenance.ptrs.len()
            ))
        );
    }
    static_target_from_allocation_at(allocation, relocation_offset)
}

/// Resolve the pointer relocation beginning at `relocation_offset` to a static.
///
/// Unlike `static_target_from_constant`, this operates on an aggregate's own
/// allocation and therefore does not require that the allocation contain only
/// one relocation.
pub(super) fn static_target_from_allocation_at(
    allocation: &rustc_public::ty::Allocation,
    relocation_offset: usize,
) -> TranslationResult<Option<StaticPointerTarget>> {
    use rustc_public::mir::alloc::GlobalAlloc;

    let Some(&(provenance_offset, provenance)) = allocation
        .provenance
        .ptrs
        .iter()
        .find(|(offset, _)| *offset == relocation_offset)
    else {
        return Ok(None);
    };

    let pointer_width = rustc_public::target::MachineInfo::target_pointer_width().bytes();
    let pointer_end = provenance_offset
        .checked_add(pointer_width)
        .ok_or_else(|| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "constant static-pointer relocation at byte {provenance_offset} overflowed"
            )))
        })?;
    let byte_offset = allocation
        .read_partial_uint(provenance_offset..pointer_end)
        .map_err(|error| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "Failed to read constant static-pointer addend at byte {provenance_offset}: \
                 {error:?}"
            )))
        })? as u64;

    match GlobalAlloc::from(provenance.0) {
        GlobalAlloc::Static(static_def) => Ok(Some(StaticPointerTarget {
            static_def,
            byte_offset,
        })),
        _ => Ok(None),
    }
}

/// One pointer-width relocation inside an evaluated Rust static initializer.
///
/// The source and target offsets are byte offsets. `target_key` is the same
/// domain-tagged rustc identity stored on the target `MirGlobalAllocOp`.
pub(super) struct GlobalInitializerRelocation {
    pub(super) source_offset: u64,
    pub(super) width_bytes: u32,
    target_address_space: u32,
    pub(super) target_addend: u64,
    target_key: String,
    pub(super) target: RelocationTarget,
}

/// The rustc allocation a relocated pointer slot points into.
///
/// Both kinds become module-level device globals: a Rust static keeps its own
/// global, and an anonymous allocation (a promoted constant, or memory a
/// constant's pointer field refers to) gets one immutable global per
/// allocation, keyed by its contents. Functions, vtables, and type ids have no
/// device storage and stay rejected.
#[derive(Clone, Copy)]
pub(super) enum RelocationTarget {
    /// A Rust `static`, whose global is typed as the static and keeps the
    /// static's own mutability and address space.
    Static(rustc_public::mir::mono::StaticDef),
    /// An anonymous rustc allocation, whose global is its read-only byte image
    /// (writable only if rustc interned the allocation as mutable).
    Memory(rustc_public::mir::alloc::AllocId),
}

impl RelocationTarget {
    /// Human-readable target name for diagnostics.
    pub(super) fn describe(&self) -> String {
        match self {
            RelocationTarget::Static(static_def) => static_def.name(),
            RelocationTarget::Memory(alloc_id) => format!("promoted constant {alloc_id:?}"),
        }
    }
}

/// The byte image, ABI alignment, and pointer relocations of a global initializer.
///
/// Literal bytes remain byte-exact. Pointer slots are carried separately so
/// lowering can replace their placeholder addend bytes with LLVM relocation
/// expressions without changing padding, NaN payloads, or field offsets.
pub(crate) struct GlobalInitializerData {
    pub(super) bytes: Vec<u8>,
    pub(super) alignment: u64,
    pub(super) relocations: Vec<GlobalInitializerRelocation>,
}

pub(super) fn static_global_key(static_def: &rustc_public::mir::mono::StaticDef) -> String {
    crate::device_static_global_key(static_def)
}

/// The device storage a Rust static lives in.
///
/// Ordinary and `ConstantMemory` statics are initialized device globals, so
/// another allocation can hold a relocation to them. `SharedArray` and
/// `Barrier` statics are per-block shared memory with no initializer: their
/// storage comes from `mir.shared_alloc`, which only a direct `&raw const` or
/// `&raw mut` operand reaches. Classifying them here keeps every other route,
/// such as a pointer stored inside a constant, from giving them a second,
/// ordinary global in the wrong address space.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum StaticStorage {
    /// An ordinary `static` / `static mut` in CUDA global memory.
    Global,
    /// A `ConstantMemory<T>` static in CUDA constant memory.
    Constant,
    /// A `SharedArray` or `Barrier` static in per-block shared memory.
    Shared,
}

impl StaticStorage {
    /// Classify `static_def` by its declared type.
    pub(super) fn of(static_def: &rustc_public::mir::mono::StaticDef) -> Self {
        use rustc_public::ty::{RigidTy, TyKind};

        let ty = static_def.ty();
        if is_constant_wrapper_type(&ty) {
            return Self::Constant;
        }
        match ty.kind() {
            TyKind::RigidTy(RigidTy::Adt(adt_def, _)) if is_shared_memory_adt(&adt_def) => {
                Self::Shared
            }
            _ => Self::Global,
        }
    }

    /// LLVM address space of the static's initialized device global, or
    /// `None` for shared memory, which has no initialized global to point at.
    pub(super) fn initialized_address_space(self) -> Option<u32> {
        match self {
            Self::Global => Some(dialect_mir::types::address_space::GLOBAL),
            Self::Constant => Some(dialect_mir::types::address_space::CONSTANT),
            Self::Shared => None,
        }
    }
}

/// Refuse a static with no initialized device global as a pointer target.
///
/// `context` names the route that reached the static, for the diagnostic.
pub(super) fn require_initialized_static_storage(
    static_def: &rustc_public::mir::mono::StaticDef,
    context: &str,
    loc: Location,
) -> TranslationResult<u32> {
    match StaticStorage::of(static_def).initialized_address_space() {
        Some(address_space) => Ok(address_space),
        None => input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "{context} points at shared-memory static {}; a pointer to a SharedArray or \
                 Barrier is supported only as a direct `&raw const` or `&raw mut` of the static, \
                 not stored inside a constant or another static",
                static_def.name()
            ))
        ),
    }
}

/// `true` for the cuda-device types whose pointers the type translator places
/// in the shared address space
fn is_shared_memory_adt(adt_def: &rustc_public::ty::AdtDef) -> bool {
    is_cuda_device_adt(adt_def, "SharedArray") || is_cuda_device_adt(adt_def, "Barrier")
}

/// Refuse a relocated value whose type stores a pointer to shared memory.
///
/// A `*const SharedArray` or `&Barrier` translates to an `addrspace(3)`
/// pointer, but every relocation target an initializer can name is an
/// initialized global in the global or constant address space: anonymous
/// memory, an ordinary static, or an ordinary static that merely contains a
/// `Barrier` field. Loading such a stored pointer would reinterpret a global
/// address as a shared one, so refuse the type before any global is emitted.
/// The walk follows fields, closure captures, elements, and pointees,
/// because a nested allocation such as the inner `&Barrier` of `&&Barrier`
/// is typed only through the outer constant's type. Callers run it only
/// when the value carries a relocation; it does not match slots to
/// relocations, so a null shared-memory pointer next to an unrelated
/// relocated pointer is also refused. `context` names the value for the
/// diagnostic.
pub(super) fn refuse_stored_shared_memory_pointer(
    ty: &rustc_public::ty::Ty,
    context: &str,
    loc: Location,
) -> TranslationResult<()> {
    use rustc_public::ty::{RigidTy, TyKind};

    let mut pending = vec![*ty];
    // recursive types such as `struct Node { next: Option<&'static Node> }`
    // revisit themselves through a pointee
    let mut visited = std::collections::HashSet::new();
    while let Some(current) = pending.pop() {
        if !visited.insert(current) {
            continue;
        }

        let TyKind::RigidTy(rigid) = current.kind() else {
            continue;
        };
        match rigid {
            RigidTy::Ref(_, pointee, _) | RigidTy::RawPtr(pointee, _) => {
                if let TyKind::RigidTy(RigidTy::Adt(adt_def, _)) = pointee.kind()
                    && is_shared_memory_adt(&adt_def)
                {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(format!(
                            "{context} stores a pointer to shared memory (`{current}`); a pointer \
                             to a SharedArray or Barrier is supported only as a direct `&raw \
                             const` or `&raw mut` of the static, not stored inside a constant or \
                             another static"
                        ))
                    );
                }
                pending.push(pointee);
            }
            RigidTy::Array(element, _) | RigidTy::Slice(element) => pending.push(element),
            RigidTy::Tuple(fields) => pending.extend(fields),
            // a closure's captures are laid out as its fields
            RigidTy::Closure(_, args) => pending
                .extend(crate::translator::types::closure_upvar_tys(&args).unwrap_or_default()),
            RigidTy::Adt(adt_def, args) => pending.extend(
                adt_def
                    .variants_iter()
                    .flat_map(|variant| variant.fields())
                    .map(|field| field.ty_with_args(&args)),
            ),
            // scalars, `str`, function pointers, and trait objects hold no
            // typed shared-memory pointer
            _ => {}
        }
    }

    Ok(())
}

/// Encode initializer relocations using the versioned, length-prefixed format
/// consumed by `mir-lower` and `llvm-export`.
pub(super) fn encode_global_initializer_relocations(
    relocations: &[GlobalInitializerRelocation],
) -> String {
    fn put_u64(out: &mut String, value: u64) {
        out.push_str(&value.to_string());
        out.push(' ');
    }

    fn put_str(out: &mut String, value: &str) {
        put_u64(out, value.len() as u64);
        out.push_str(value);
        out.push(' ');
    }

    let mut encoded = String::from("v1 ");
    put_u64(&mut encoded, relocations.len() as u64);
    for relocation in relocations {
        put_u64(&mut encoded, relocation.source_offset);
        put_u64(&mut encoded, u64::from(relocation.width_bytes));
        put_u64(&mut encoded, u64::from(relocation.target_address_space));
        put_u64(&mut encoded, relocation.target_addend);
        put_str(&mut encoded, &relocation.target_key);
    }
    encoded
}

/// Copy one evaluated allocation into a byte-exact global initializer.
///
/// Undefined bytes are Rust padding and become deterministic zeros. Each
/// provenance entry is preserved as a pointer relocation, either to a Rust
/// static's global or to the immutable global of an anonymous rustc
/// allocation ([`RelocationTarget`]). Functions, vtables, and malformed source
/// ranges remain diagnosed rather than being flattened to integer bytes.
pub(super) fn allocation_initializer_data(
    alloc: &rustc_public::ty::Allocation,
    description: &str,
    loc: Location,
) -> TranslationResult<GlobalInitializerData> {
    use rustc_public::mir::alloc::GlobalAlloc;

    let pointer_width = rustc_public::target::MachineInfo::target_pointer_width().bytes();
    let width_bytes = u32::try_from(pointer_width).map_err(|_| {
        input_error_noloc!(TranslationErr::unsupported(format!(
            "{description} uses a target pointer width that does not fit u32"
        )))
    })?;

    if !alloc.provenance.ptrs.is_empty() && pointer_width != 8 {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "{description} contains pointer relocations, but cuda-oxide currently supports only 8-byte NVPTX pointers"
            ))
        );
    }

    let mut entries: Vec<_> = alloc.provenance.ptrs.to_vec();
    entries.sort_by_key(|(source_offset, _)| *source_offset);

    let mut relocations = Vec::with_capacity(entries.len());
    let mut previous_end = 0usize;

    for (index, (source_offset, provenance)) in entries.into_iter().enumerate() {
        if source_offset < previous_end {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "{description} pointer relocation {index} overlaps the previous relocation"
                ))
            );
        }

        let end = source_offset.checked_add(pointer_width).ok_or_else(|| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "{description} pointer relocation {index} source range overflows"
            )))
        })?;
        if end > alloc.bytes.len() {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "{description} pointer relocation {index} occupies bytes {source_offset}..{end}, but the allocation is only {} bytes",
                    alloc.bytes.len()
                ))
            );
        }

        let target_addend = alloc
            .read_partial_uint(source_offset..end)
            .map_err(|error| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "Failed to read {description} pointer relocation {index} addend: {error:?}"
                )))
            })? as u64;

        let (target, target_address_space, target_key) = match GlobalAlloc::from(provenance.0) {
            GlobalAlloc::Static(static_def) => (
                RelocationTarget::Static(static_def),
                require_initialized_static_storage(
                    &static_def,
                    &format!("{description} pointer relocation {index}"),
                    loc.clone(),
                )?,
                static_global_key(&static_def),
            ),
            // anonymous memory is emitted as an immutable addrspace(1) global
            GlobalAlloc::Memory(_) => (
                RelocationTarget::Memory(provenance.0),
                1,
                super::promoted_alloc::promoted_allocation_global_key(provenance.0, loc.clone())?,
            ),
            other => {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "{description} pointer relocation {index} targets unsupported allocation {other:?}; only Rust statics and constant memory allocations have device storage"
                    ))
                );
            }
        };

        relocations.push(GlobalInitializerRelocation {
            source_offset: u64::try_from(source_offset).map_err(|_| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "{description} pointer relocation {index} source offset does not fit u64"
                )))
            })?,
            width_bytes,
            target_address_space,
            target_addend,
            target_key,
            target,
        });
        previous_end = end;
    }

    Ok(GlobalInitializerData {
        bytes: alloc.bytes.iter().map(|byte| byte.unwrap_or(0)).collect(),
        alignment: alloc.align,
        relocations,
    })
}

/// Build the byte-exact, relocation-aware initializer used by array promotion.
///
/// Bare `[T; N]` constants own their allocation directly, so provenance entries
/// inside that allocation belong to pointer fields in the table. Pointer-to-array
/// constants (`&[T; N]` / `*const [T; N]`) instead contain one outer relocation
/// to the backing allocation; that relocation is followed once and any
/// relocations inside the selected table range are preserved and rebased.
pub(super) fn promoted_array_initializer(
    constant: &mir::ConstOperand,
    expected_size: usize,
    kind_name: &str,
    loc: Location,
) -> TranslationResult<GlobalInitializerData> {
    use rustc_public::mir::alloc::GlobalAlloc;
    use rustc_public::ty::{RigidTy, TyConstKind, TyKind};

    fn direct_initializer(
        alloc: &rustc_public::ty::Allocation,
        expected_size: usize,
        kind_name: &str,
        loc: Location,
    ) -> TranslationResult<GlobalInitializerData> {
        let data = allocation_initializer_data(
            alloc,
            &format!("promoted {kind_name} initializer"),
            loc.clone(),
        )?;
        if data.bytes.len() != expected_size {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "promoted {kind_name} initializer is {} bytes, expected {expected_size}",
                    data.bytes.len()
                ))
            );
        }
        Ok(data)
    }

    fn project_initializer_range(
        data: GlobalInitializerData,
        target_offset: usize,
        expected_size: usize,
        kind_name: &str,
        loc: Location,
    ) -> TranslationResult<GlobalInitializerData> {
        let end = target_offset.checked_add(expected_size).ok_or_else(|| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "promoted {kind_name} initializer offset overflows its allocation"
            )))
        })?;
        let bytes = data
            .bytes
            .get(target_offset..end)
            .ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "promoted {kind_name} initializer needs bytes {target_offset}..{end}, but its backing allocation is only {} bytes",
                    data.bytes.len()
                )))
            })?
            .to_vec();
        let alignment = data.alignment;

        let mut relocations = Vec::new();
        for relocation in data.relocations {
            let source_start = usize::try_from(relocation.source_offset).map_err(|_| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "promoted {kind_name} relocation source offset {} does not fit usize",
                    relocation.source_offset
                )))
            })?;
            let width = usize::try_from(relocation.width_bytes).map_err(|_| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "promoted {kind_name} relocation width {} does not fit usize",
                    relocation.width_bytes
                )))
            })?;
            let source_end = source_start.checked_add(width).ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "promoted {kind_name} relocation at byte {source_start} overflows"
                )))
            })?;
            let overlaps = source_start < end && source_end > target_offset;
            if !overlaps {
                continue;
            }
            if source_start < target_offset || source_end > end {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "promoted {kind_name} relocation bytes {source_start}..{source_end} cross selected initializer range {target_offset}..{end}"
                    ))
                );
            }

            relocations.push(GlobalInitializerRelocation {
                source_offset: u64::try_from(source_start - target_offset).map_err(|_| {
                    input_error_noloc!(TranslationErr::unsupported(format!(
                        "promoted {kind_name} rebased relocation offset does not fit u64"
                    )))
                })?,
                width_bytes: relocation.width_bytes,
                target_address_space: relocation.target_address_space,
                target_addend: relocation.target_addend,
                target_key: relocation.target_key,
                target: relocation.target,
            });
        }

        Ok(GlobalInitializerData {
            bytes,
            alignment,
            relocations,
        })
    }

    fn pointer_initializer(
        alloc: &rustc_public::ty::Allocation,
        expected_size: usize,
        kind_name: &str,
        loc: Location,
    ) -> TranslationResult<GlobalInitializerData> {
        if alloc.provenance.ptrs.len() != 1 {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "promoted {kind_name} pointer contains {} provenance entries; expected exactly one backing allocation",
                    alloc.provenance.ptrs.len()
                ))
            );
        }
        let &(provenance_offset, provenance) =
            alloc.provenance.ptrs.first().expect("length checked above");

        let pointer_width = rustc_public::target::MachineInfo::target_pointer_width().bytes();
        let pointer_end = provenance_offset
            .checked_add(pointer_width)
            .ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "promoted {kind_name} outer pointer range overflows"
                )))
            })?;
        let target_offset = alloc
            .read_partial_uint(provenance_offset..pointer_end)
            .map_err(|e| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "Failed to read promoted {kind_name} pointer offset: {e:?}"
                )))
            })? as usize;

        let target_alloc = match GlobalAlloc::from(provenance.0) {
            GlobalAlloc::Memory(target_alloc) => target_alloc,
            GlobalAlloc::Static(static_def) => static_def.eval_initializer().map_err(|e| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "Failed to evaluate promoted {kind_name} backing static {}: {e:?}",
                    static_def.name()
                )))
            })?,
            other => {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "promoted {kind_name} provenance points to unsupported allocation {other:?}"
                    ))
                );
            }
        };

        let data = allocation_initializer_data(
            &target_alloc,
            &format!("promoted {kind_name} backing allocation"),
            loc.clone(),
        )?;
        project_initializer_range(data, target_offset, expected_size, kind_name, loc)
    }

    let pointer_form = matches!(
        constant.const_.ty().kind(),
        TyKind::RigidTy(RigidTy::RawPtr(_, _)) | TyKind::RigidTy(RigidTy::Ref(_, _, _))
    );

    match constant.const_.kind() {
        ConstantKind::Allocated(alloc) => {
            if pointer_form {
                pointer_initializer(alloc, expected_size, kind_name, loc)
            } else {
                direct_initializer(alloc, expected_size, kind_name, loc)
            }
        }
        ConstantKind::Ty(ty_const) => match ty_const.kind() {
            TyConstKind::Value(_, alloc) => {
                if pointer_form {
                    pointer_initializer(alloc, expected_size, kind_name, loc)
                } else {
                    direct_initializer(alloc, expected_size, kind_name, loc)
                }
            }
            TyConstKind::ZSTValue(_) if expected_size == 0 => Ok(GlobalInitializerData {
                bytes: Vec::new(),
                alignment: 1,
                relocations: Vec::new(),
            }),
            other => input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "promoted {kind_name} initializer must be backed by bytes, found TyConstKind::{other:?}"
                ))
            ),
        },
        ConstantKind::ZeroSized if expected_size == 0 => Ok(GlobalInitializerData {
            bytes: Vec::new(),
            alignment: 1,
            relocations: Vec::new(),
        }),
        other => input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "promoted {kind_name} initializer must be allocated, found {other:?}"
            ))
        ),
    }
}

/// Return rustc's evaluated static initializer bytes, alignment, and relocations.
pub(super) fn static_initializer_data(
    static_def: &rustc_public::mir::mono::StaticDef,
    loc: Location,
) -> TranslationResult<GlobalInitializerData> {
    let alloc = static_def.eval_initializer().map_err(|e| {
        input_error_noloc!(TranslationErr::unsupported(format!(
            "Failed to evaluate initializer for device static {}: {:?}",
            static_def.name(),
            e
        )))
    })?;
    let description = format!("device static {}", static_def.name());
    if !alloc.provenance.ptrs.is_empty() {
        refuse_stored_shared_memory_pointer(&static_def.ty(), &description, loc.clone())?;
    }
    allocation_initializer_data(&alloc, &description, loc)
}

pub(super) fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut hex, "{byte:02x}").expect("writing to String cannot fail");
    }
    hex
}
