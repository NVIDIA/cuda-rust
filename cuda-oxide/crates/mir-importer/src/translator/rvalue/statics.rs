/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Static identity and initializer plumbing.

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::types::monomorphized_ty_is_inhabited;
use crate::translator::values::is_constant_wrapper_type;
use pliron::location::Location;
use pliron::{input_err, input_error, input_error_noloc};
use rustc_public::CrateDef;
use rustc_public::mir;
use rustc_public::ty::ConstantKind;

pub(super) struct SharedStaticSourceIdentity {
    pub(super) name: String,
    pub(super) key: String,
    /// Whether the pointer names the whole static rather than one of its
    /// fields; only a whole static has a full-debug variable to join
    pub(super) whole_static: bool,
}

/// Resolve a shared allocation's constant pointer to the Rust static, or the
/// field of one, that it names, giving both a display path and an injective
/// compiler identity.
///
/// `StaticDef::name()` deliberately omits DefPath disambiguators, so distinct
/// same-leaf nested statics can have the same display path. The mangled static
/// instance remains unique and is used, in every build profile, for
/// deduplication, for the hash that names the lowered `__shared_mem_*` global,
/// and for the full-debug side-table join.
///
/// `SharedArray` is zero-sized, so a pointer to such a field of a tuple,
/// struct, union, enum or array static can carry the same byte offset as its
/// siblings, often zero, and only its type tells them apart; union fields and
/// enum variants overlap whatever their size, `Barrier` included. The static's
/// layout is walked to the one field of the pointee's type at that offset and
/// the key gains that field's path. The pointee type, not the pointer, is
/// compared, so `*mut` and `*const` pointers to one field share one global.
/// Two or more such fields cannot be told apart and would alias, so they are
/// rejected. `Ok(None)` means the constant names no static, or no field of the
/// pointee's type, and the caller falls back to the constant's own spelling.
pub(super) fn shared_static_source_identity(
    constant: &mir::ConstOperand,
    loc: &Location,
) -> TranslationResult<Option<SharedStaticSourceIdentity>> {
    let ConstantKind::Allocated(allocation) = constant.const_.kind() else {
        return Ok(None);
    };
    let [(relocation_offset, _)] = allocation.provenance.ptrs.as_slice() else {
        return Ok(None);
    };
    let Ok(Some(target)) = static_target_from_allocation_at(allocation, *relocation_offset) else {
        return Ok(None);
    };
    let Some(pointee) = constant
        .ty()
        .kind()
        .builtin_deref(true)
        .map(|deref| deref.ty)
    else {
        return Ok(None);
    };

    // the instance type is normalized like the constant's, unlike the raw
    // declared type, whose const generics can still be unevaluated paths
    let static_ty = rustc_public::mir::mono::Instance::from(target.static_def).ty();
    let mut walk = FieldWalk {
        pointee: &pointee,
        offset: target.byte_offset,
        path: Vec::new(),
        found: Vec::new(),
        loc,
    };
    walk.visit(&static_ty, 0)?;

    let name = target.static_def.name();
    let static_key = crate::device_static_global_key(&target.static_def);
    match walk.found.as_slice() {
        [] => Ok(None),
        [path] if path.is_empty() => Ok(Some(SharedStaticSourceIdentity {
            name,
            key: static_key,
            whole_static: true,
        })),
        [path] => {
            let path = field_path(path);
            Ok(Some(SharedStaticSourceIdentity {
                name: format!("{name}.{path}"),
                // the static key is length-prefixed, so the suffix cannot
                // collide with another static's key
                key: format!("{static_key}:field:{path}"),
                whole_static: false,
            }))
        }
        [first, second, ..] => input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "static `{name}` has fields `.{}` and `.{}` of the same shared-memory \
                 type at byte offset {}; a pointer to one cannot be told \
                 apart from a pointer to the other, so they would share storage. Give \
                 each `SharedArray` or `Barrier` its own static",
                field_path(first),
                field_path(second),
                target.byte_offset,
            ))
        ),
    }
}

fn field_path(path: &[usize]) -> String {
    path.iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(".")
}

/// Finds the fields of a static's type that have the pointee's type and start
/// at the pointer's byte offset.
struct FieldWalk<'a> {
    pointee: &'a rustc_public::ty::Ty,
    offset: u64,
    path: Vec<usize>,
    found: Vec<Vec<usize>>,
    loc: &'a Location,
}

impl FieldWalk<'_> {
    fn visit(&mut self, ty: &rustc_public::ty::Ty, base: u64) -> TranslationResult<()> {
        use rustc_public::CrateDefType;
        use rustc_public::abi::{FieldsShape, VariantsShape};
        use rustc_public::ty::{AdtKind, RigidTy, TyKind};
        use rustc_public_bridge::IndexedVal;

        if ty == self.pointee {
            if base == self.offset {
                self.found.push(self.path.clone());
            }
            return Ok(());
        }

        let (def, args) = match ty.kind() {
            TyKind::RigidTy(RigidTy::Tuple(fields)) => {
                let FieldsShape::Arbitrary { offsets } = self.layout(ty)?.fields else {
                    return Ok(());
                };
                let offsets = offsets.iter().map(|offset| offset.bytes() as u64);
                return self.visit_fields(fields.into_iter().zip(offsets), base);
            }
            TyKind::RigidTy(RigidTy::Array(element, _)) => {
                return self.visit_array(ty, &element, base);
            }
            TyKind::RigidTy(RigidTy::Adt(def, args)) => (def, args),
            _ => return Ok(()),
        };

        let variant_fields = |variant: &rustc_public::ty::VariantDef| -> Vec<rustc_public::ty::Ty> {
            variant
                .fields()
                .iter()
                .map(|field| field.ty_with_args(&args))
                .collect()
        };
        let variants = def.variants();
        let layout = self.layout(ty)?;
        match def.kind() {
            AdtKind::Struct => {
                let (Some(variant), FieldsShape::Arbitrary { offsets }) =
                    (variants.first(), &layout.fields)
                else {
                    return Ok(());
                };
                let offsets = offsets.iter().map(|offset| offset.bytes() as u64);
                self.visit_fields(variant_fields(variant).into_iter().zip(offsets), base)
            }
            // every union field starts at the union's base, so a field of the
            // pointee's type there competes with the siblings outside it
            AdtKind::Union => {
                let Some(variant) = variants.first() else {
                    return Ok(());
                };
                let fields = variant_fields(variant);
                self.visit_fields(fields.into_iter().map(|field| (field, 0)), base)
            }
            // each variant's fields overlap the others', so every variant that
            // can hold a value is walked; its index, rustc's own, joins the
            // path to keep keys distinct
            AdtKind::Enum => {
                let offsets: Vec<(usize, Vec<u64>)> = match &layout.variants {
                    VariantsShape::Single { index } => match &layout.fields {
                        FieldsShape::Arbitrary { offsets } => vec![(
                            index.to_index(),
                            offsets.iter().map(|offset| offset.bytes() as u64).collect(),
                        )],
                        _ => Vec::new(),
                    },
                    VariantsShape::Multiple { variants, .. } => variants
                        .iter()
                        .enumerate()
                        .map(|(index, variant)| {
                            let offsets = variant.offsets.iter();
                            (index, offsets.map(|offset| offset.bytes() as u64).collect())
                        })
                        .collect(),
                    VariantsShape::Empty => Vec::new(),
                };
                'variants: for (index, offsets) in offsets {
                    let Some(variant) = variants.get(index) else {
                        continue;
                    };
                    // rustc keeps field offsets for a variant that can never
                    // hold a value, and its fields would otherwise read as
                    // false rivals to the live variant's
                    let fields = variant_fields(variant);
                    for field in &fields {
                        if !monomorphized_ty_is_inhabited(field)? {
                            continue 'variants;
                        }
                    }

                    self.path.push(index);
                    self.visit_fields(fields.into_iter().zip(offsets), base)?;
                    self.path.pop();
                }
                Ok(())
            }
        }
    }

    /// Visits each field, given with its offset in the parent, whose byte
    /// range can hold the pointer's offset; the end is included because a
    /// zero-sized field can sit there.
    fn visit_fields(
        &mut self,
        fields: impl IntoIterator<Item = (rustc_public::ty::Ty, u64)>,
        base: u64,
    ) -> TranslationResult<()> {
        for (index, (field_ty, field_offset)) in fields.into_iter().enumerate() {
            let start = base + field_offset;
            if self.offset < start || self.offset > start + self.size(&field_ty)? {
                continue;
            }

            self.path.push(index);
            self.visit(&field_ty, start)?;
            self.path.pop();
        }
        Ok(())
    }

    fn visit_array(
        &mut self,
        ty: &rustc_public::ty::Ty,
        element: &rustc_public::ty::Ty,
        base: u64,
    ) -> TranslationResult<()> {
        use rustc_public::abi::FieldsShape;

        let FieldsShape::Array { stride, count } = self.layout(ty)?.fields else {
            return Ok(());
        };
        let Some(relative) = self.offset.checked_sub(base) else {
            return Ok(());
        };

        // a zero-sized element puts every index at the base, and two are
        // enough to report the ambiguity; otherwise the offset falls in at
        // most this element and, at its start, the end of the previous one
        let stride = stride.bytes() as u64;
        let candidates = match relative.checked_div(stride) {
            None => 0..count.min(2),
            Some(last) => last.saturating_sub(1)..(last + 1).min(count),
        };
        let element_size = self.size(element)?;
        for index in candidates {
            let start = base + index * stride;
            if self.offset < start || self.offset > start + element_size {
                continue;
            }

            self.path.push(index as usize);
            self.visit(element, start)?;
            self.path.pop();
        }
        Ok(())
    }

    fn layout(
        &self,
        ty: &rustc_public::ty::Ty,
    ) -> TranslationResult<rustc_public::abi::LayoutShape> {
        let layout = ty.layout().map_err(|error| {
            input_error!(
                self.loc.clone(),
                TranslationErr::unsupported(format!(
                    "Failed to query the layout of a shared static's type: {error:?}"
                ))
            )
        })?;
        Ok(layout.shape())
    }

    fn size(&self, ty: &rustc_public::ty::Ty) -> TranslationResult<u64> {
        Ok(self.layout(ty)?.size.bytes() as u64)
    }
}

/// Best-effort source label for a shared allocation whose constant does not
/// resolve to a static or a field of one, so [`shared_static_source_identity`]
/// gives up.
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
    pub(super) target_static: rustc_public::mir::mono::StaticDef,
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

fn static_global_address_space(static_def: &rustc_public::mir::mono::StaticDef) -> u32 {
    if is_constant_wrapper_type(&static_def.ty()) {
        4
    } else {
        1
    }
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
/// provenance entry is preserved as a static-to-static pointer relocation.
/// Anonymous memory, functions, vtables, and malformed source ranges remain
/// diagnosed rather than being flattened to integer bytes.
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

        let target_static = match GlobalAlloc::from(provenance.0) {
            GlobalAlloc::Static(static_def) => static_def,
            other => {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "{description} pointer relocation {index} targets unsupported allocation {other:?}; only Rust statics in CUDA global or constant memory are supported"
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
            target_address_space: static_global_address_space(&target_static),
            target_addend,
            target_key: static_global_key(&target_static),
            target_static,
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
                target_static: relocation.target_static,
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
    allocation_initializer_data(&alloc, &format!("device static {}", static_def.name()), loc)
}

pub(super) fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut hex, "{byte:02x}").expect("writing to String cannot fail");
    }
    hex
}
