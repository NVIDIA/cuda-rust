/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Static shared declarations, deduplicated inside the executable GPU module.
use crate::{
    MlirAttribute, MlirBlock, MlirOperation, MlirRegion, MlirResult, MlirType, MlirValueUse,
    OperationInput, OperationTranslation, TranslationError, TranslationRegistry,
    TranslationSession,
};
use dialect_mir::ops::MirSharedAllocOp;
use pliron::{
    context::{Context, Ptr},
    operation::Operation,
    r#type::Typed,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Register source static shared allocations for later module-level hoisting.
pub(crate) fn register(registry: &mut TranslationRegistry) -> Result<(), TranslationError> {
    registry.register_operation::<MirSharedAllocOp>(SharedAllocation)
}
/// Emit a named shared global and an address while preserving element identity and alignment.
struct SharedAllocation;
impl OperationTranslation for SharedAllocation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        s: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        if !input.operands.is_empty()
            || input.results.len() != 1
            || !input.regions.is_empty()
            || !input.successors.is_empty()
        {
            return Err("invalid shared allocation shape".into());
        }
        for key in input.attributes.keys() {
            if !matches!(
                key.as_str(),
                "elem_type" | "size" | "alloc_key" | "source_key" | "source_name" | "mir_alignment"
            ) {
                return Err(format!("unsupported shared allocation attribute {key}"));
            }
        }
        let allocation = MirSharedAllocOp::new(source);
        let element = allocation
            .get_attr_elem_type(ctx)
            .ok_or("missing shared element type")?
            .get_type(ctx);
        let size = allocation
            .get_attr_size(ctx)
            .ok_or("missing shared size")?
            .value()
            .to_u64();
        if size == 0 {
            return Err("empty static shared allocation".into());
        }
        let alignment = allocation
            .get_alignment_value(ctx)
            .filter(|align| *align != 0)
            .or_else(|| crate::mir::memory::source_natural_alignment(ctx, element))
            .ok_or("unknown shared alignment")?;
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err("invalid shared alignment".into());
        }
        let key = allocation
            .get_attr_alloc_key(ctx)
            .map(|key| String::from((*key).clone()))
            .unwrap_or_else(|| format!("unkeyed-{}", input.results[0].id.0));
        let symbol = format!("__cuda_oxide_shared_{:x}", Sha256::digest(key.as_bytes()));
        let ty = s.translate_type(element).map_err(|e| e.to_string())?;
        let global_type = MlirType::dialect(format!(
            "!llvm.array<{size} x {}>",
            crate::mir::memory::type_spelling(&ty)
        ))?;
        let mut global = MlirOperation::new("llvm.mlir.global")?;
        // Preserve source identity while deduplicating: signedness and enum
        // identity may disappear from the lowered storage type. Removed below.
        global.attributes.insert(
            "__source_element_identity".into(),
            MlirAttribute::String(format!("{element:?}")),
        );
        global
            .properties
            .insert("sym_name".into(), MlirAttribute::String(symbol.clone()));
        global.properties.insert(
            "global_type".into(),
            MlirAttribute::Type(global_type.clone()),
        );
        global.properties.insert(
            "linkage".into(),
            MlirAttribute::dialect("#llvm.linkage<internal>")?,
        );
        global.properties.insert(
            "addr_space".into(),
            MlirAttribute::Integer {
                value: 3,
                ty: MlirType::Integer(32),
            },
        );
        global.properties.insert(
            "alignment".into(),
            MlirAttribute::Integer {
                value: alignment.into(),
                ty: MlirType::Integer(64),
            },
        );
        let id = s.fresh_value();
        let mut undef = MlirOperation::new("llvm.mlir.undef")?;
        undef.results.push(MlirResult {
            id,
            ty: global_type.clone(),
        });
        let mut ret = MlirOperation::new("llvm.return")?;
        ret.operands.push(MlirValueUse {
            id,
            ty: global_type,
        });
        global.regions.push(MlirRegion {
            blocks: vec![MlirBlock {
                id: s.fresh_block(),
                arguments: vec![],
                operations: vec![undef, ret],
            }],
        });
        global.location = input.location.clone();
        let mut address = MlirOperation::new("llvm.mlir.addressof")?;
        address.results = input.results;
        address
            .properties
            .insert("global_name".into(), MlirAttribute::SymbolRef(symbol));
        address.location = input.location;
        // The profile hoists these declarations after translation; source IR
        // is immutable and its local allocation operation still yields an SSA address.
        Ok(vec![global, address])
    }
}

/// Move shared declarations to module scope and merge only storage-compatible duplicates.
pub(crate) fn hoist_shared_globals(operations: &mut Vec<MlirOperation>) -> Result<(), String> {
    /// Collect nested shared globals, strengthening dynamic-storage alignment and rejecting conflicting declarations.
    fn gather(
        operations: &mut Vec<MlirOperation>,
        globals: &mut BTreeMap<String, MlirOperation>,
    ) -> Result<(), String> {
        let mut remaining = vec![];
        for mut operation in std::mem::take(operations) {
            if operation.name == "llvm.mlir.global" {
                let Some(MlirAttribute::String(symbol)) =
                    operation.properties.get("sym_name").cloned()
                else {
                    return Err("global without symbol".into());
                };
                if let Some(previous) = globals.get_mut(&symbol) {
                    // All extern shared declarations refer to the same dynamic
                    // allocation. Its base must meet the strongest requested alignment.
                    if symbol == "__cuda_oxide_dynamic_shared" {
                        let alignment = match (
                            previous.properties.get("alignment"),
                            operation.properties.get("alignment"),
                        ) {
                            (
                                Some(MlirAttribute::Integer { value: a, .. }),
                                Some(MlirAttribute::Integer { value: b, .. }),
                            ) => (*a).max(*b),
                            _ => return Err("dynamic shared declaration lacks alignment".into()),
                        };
                        let value = MlirAttribute::Integer {
                            value: alignment,
                            ty: MlirType::Integer(64),
                        };
                        previous
                            .properties
                            .insert("alignment".into(), value.clone());
                        operation.properties.insert("alignment".into(), value);
                    }
                    if previous.properties != operation.properties
                        || previous.attributes != operation.attributes
                    {
                        return Err(format!(
                            "incompatible declarations for shared allocation {symbol}"
                        ));
                    }
                } else {
                    globals.insert(symbol.clone(), operation);
                }
            } else {
                for region in &mut operation.regions {
                    for block in &mut region.blocks {
                        gather(&mut block.operations, globals)?;
                    }
                }
                remaining.push(operation);
            }
        }
        *operations = remaining;
        Ok(())
    }
    let mut globals = BTreeMap::new();
    gather(operations, &mut globals)?;
    operations.splice(
        0..0,
        globals.into_values().map(|mut global| {
            global.attributes.remove("__source_element_identity");
            global
        }),
    );
    Ok(())
}

/// Dynamic shared storage for the instruction-level profile. The CuTe profile
/// retains its compiler-native dynamic-storage operation.
pub(crate) fn register_dynamic_shared(
    registry: &mut TranslationRegistry,
) -> Result<(), TranslationError> {
    registry.register_operation::<dialect_mir::ops::MirExternSharedOp>(ExternShared)
}
/// Address the one external dynamic shared allocation at the requested byte offset.
struct ExternShared;
impl OperationTranslation for ExternShared {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        use pliron::common_traits::Verify;
        let source = dialect_mir::ops::MirExternSharedOp::new(source);
        source.verify(ctx).map_err(|e| e.to_string())?;
        input.attributes.remove("extern_byte_offset");
        input.attributes.remove("extern_alignment");
        if !input.attributes.is_empty()
            || !input.operands.is_empty()
            || input.results.len() != 1
            || !input.regions.is_empty()
            || !input.successors.is_empty()
        {
            return Err("dynamic shared storage expects one shared pointer result".into());
        }
        let alignment = source.get_alignment_value(ctx);
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err("dynamic shared alignment must be a nonzero power of two".into());
        }
        let offset = source.get_byte_offset_value(ctx);
        if offset > i64::MAX as u64 {
            return Err("dynamic shared byte offset exceeds i64".into());
        }
        let symbol = "__cuda_oxide_dynamic_shared";
        let mut global = MlirOperation::new("llvm.mlir.global")?;
        global
            .properties
            .insert("sym_name".into(), MlirAttribute::String(symbol.into()));
        global.properties.insert(
            "global_type".into(),
            MlirAttribute::Type(MlirType::dialect("!llvm.array<0 x i8>")?),
        );
        global.properties.insert(
            "linkage".into(),
            MlirAttribute::dialect("#llvm.linkage<external>")?,
        );
        global.properties.insert(
            "addr_space".into(),
            MlirAttribute::Integer {
                value: 3,
                ty: MlirType::Integer(32),
            },
        );
        global.properties.insert(
            "alignment".into(),
            MlirAttribute::Integer {
                value: alignment.into(),
                ty: MlirType::Integer(64),
            },
        );
        global.regions.push(MlirRegion { blocks: vec![] });
        global.location = input.location.clone();
        let mut address = MlirOperation::new("llvm.mlir.addressof")?;
        address.properties.insert(
            "global_name".into(),
            MlirAttribute::SymbolRef(symbol.into()),
        );
        address.location = input.location.clone();
        if offset == 0 {
            address.results = input.results;
            return Ok(vec![global, address]);
        }
        let base = MlirResult {
            id: session.fresh_value(),
            ty: input.results[0].ty.clone(),
        };
        address.results.push(base.clone());
        let index = MlirResult {
            id: session.fresh_value(),
            ty: MlirType::Integer(64),
        };
        let mut constant = MlirOperation::new("arith.constant")?;
        constant.results.push(index.clone());
        constant.properties.insert(
            "value".into(),
            MlirAttribute::Integer {
                value: offset.into(),
                ty: index.ty.clone(),
            },
        );
        constant.location = input.location.clone();
        let mut gep = MlirOperation::new("llvm.getelementptr")?;
        gep.operands = vec![
            MlirValueUse {
                id: base.id,
                ty: base.ty,
            },
            MlirValueUse {
                id: index.id,
                ty: index.ty,
            },
        ];
        gep.results = input.results;
        gep.properties.insert(
            "elem_type".into(),
            MlirAttribute::Type(MlirType::Integer(8)),
        );
        gep.properties.insert(
            "rawConstantIndices".into(),
            MlirAttribute::DenseI32Array(vec![i32::MIN]),
        );
        gep.location = input.location;
        Ok(vec![global, address, constant, gep])
    }
}
