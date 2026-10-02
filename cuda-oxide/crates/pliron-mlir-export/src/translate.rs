/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::{BTreeMap, HashMap, HashSet};

use pliron::{
    attribute::AttrObj,
    basic_block::BasicBlock,
    builtin::ops::ModuleOp,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    location::{Located, Location, Source},
    op::Op,
    operation::{Operation, verify_operation},
    region::Region,
    r#type::{TypeHandle, Typed},
    uniqued_any,
    value::Value,
};

use crate::{
    MissingMappings, MlirAttribute, MlirBlock, MlirBlockArgument, MlirBlockId, MlirLocation,
    MlirModule, MlirOperation, MlirRegion, MlirResult, MlirType, MlirValueId, MlirValueUse,
    TranslationError, TranslationRegistry,
};

/// How much source location data to keep.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LocationMode {
    Off,
    FileLineCol,
    #[default]
    Full,
}

/// Stable identity and rendering choices for one export.
#[derive(Clone, Debug)]
pub struct TranslationConfig {
    pub profile: String,
    pub locations: LocationMode,
}

impl TranslationConfig {
    pub fn new(profile: impl Into<String>) -> Self {
        Self {
            profile: profile.into(),
            locations: LocationMode::Full,
        }
    }
}

/// Fully translated inputs supplied to one operation recipe.
#[derive(Clone, Debug)]
pub struct OperationInput {
    pub results: Vec<MlirResult>,
    pub operands: Vec<MlirValueUse>,
    pub successors: Vec<MlirBlockId>,
    pub regions: Vec<MlirRegion>,
    pub attributes: BTreeMap<String, MlirAttribute>,
    pub location: MlirLocation,
}

/// Mutable state shared by operation recipes during one export.
pub struct TranslationSession<'a> {
    ctx: &'a Context,
    registry: &'a TranslationRegistry,
    config: &'a TranslationConfig,
    values: HashMap<Value, MlirValueId>,
    blocks: HashMap<Ptr<BasicBlock>, MlirBlockId>,
    next_value: u64,
    next_block: u64,
}

impl<'a> TranslationSession<'a> {
    pub fn profile(&self) -> &str {
        &self.config.profile
    }

    pub fn fresh_value(&mut self) -> MlirValueId {
        let value = MlirValueId(self.next_value);
        self.next_value += 1;
        value
    }

    /// Allocate a deterministic block ID for a region created by a mapping.
    ///
    /// Source blocks are indexed before translation. Mapping-created nested
    /// regions therefore receive IDs strictly after every source block, which
    /// prevents collisions without coupling a dialect recipe to traversal
    /// order or renderer internals.
    pub fn fresh_block(&mut self) -> MlirBlockId {
        let block = MlirBlockId(self.next_block);
        self.next_block += 1;
        block
    }

    pub fn translate_type(&self, source: TypeHandle) -> Result<MlirType, TranslationError> {
        translate_type(self.ctx, self.registry, self.config, source)
    }

    pub fn translate_attribute(
        &self,
        source: &AttrObj,
    ) -> Result<Option<MlirAttribute>, TranslationError> {
        translate_attribute(self.ctx, self.registry, self.config, source)
    }

    /// Return the target SSA use assigned to any indexed source value.
    ///
    /// Most recipes consume only their operation's direct operands, which are
    /// already present in [`OperationInput`]. A semantic recipe may also need
    /// a value from a module-validated producer story—for example the original
    /// tensor base, length, or tile coordinate. This lookup keeps that need
    /// typed and deterministic instead of making dialect packs reconstruct
    /// target names or reach into exporter internals.
    pub fn target_value_use(&self, source: Value) -> Result<MlirValueUse, TranslationError> {
        let id = self.values.get(&source).copied().ok_or_else(|| {
            TranslationError::MissingIndexedValue {
                kind: "value",
                id: format!("{source:?}"),
            }
        })?;
        Ok(MlirValueUse {
            id,
            ty: self.translate_type(source.get_type(self.ctx))?,
        })
    }
}

/// Verify, preflight, translate, and build one target module.
pub fn translate_module(
    ctx: &Context,
    module: &ModuleOp,
    registry: &TranslationRegistry,
    config: &TranslationConfig,
) -> Result<MlirModule, TranslationError> {
    if config.profile.trim().is_empty() {
        return Err(TranslationError::InvalidTarget(
            "translation profile name must not be empty".into(),
        ));
    }
    verify_operation(module.get_operation(), ctx).map_err(|error| TranslationError::Mapping {
        source_item: "builtin.module".into(),
        profile: config.profile.clone(),
        message: format!("source verification failed: {error}"),
    })?;

    let missing = collect_missing(ctx, module.get_operation(), registry);
    if !missing.is_empty() {
        return Err(TranslationError::MissingMappings(missing));
    }

    let mut session = TranslationSession {
        ctx,
        registry,
        config,
        values: HashMap::new(),
        blocks: HashMap::new(),
        next_value: 0,
        next_block: 0,
    };
    let mut next_block = 0;
    index_operation(module.get_operation(), ctx, &mut session, &mut next_block);
    session.next_block = next_block;
    let mut translated = translate_operation(module.get_operation(), &mut session)?;
    if translated.len() != 1 {
        return Err(TranslationError::InvalidTarget(format!(
            "builtin.module mapping must emit exactly one root operation, got {}",
            translated.len()
        )));
    }
    let module = MlirModule {
        root: translated.remove(0),
        profile: config.profile.clone(),
    };
    validate_target(&module)?;
    Ok(module)
}

fn collect_missing(
    ctx: &Context,
    root: Ptr<Operation>,
    registry: &TranslationRegistry,
) -> MissingMappings {
    let mut missing = MissingMappings::default();
    scan_operation(ctx, root, registry, &mut missing);
    missing
}

fn scan_operation(
    ctx: &Context,
    source: Ptr<Operation>,
    registry: &TranslationRegistry,
    missing: &mut MissingMappings,
) {
    let id = Operation::get_opid(source, ctx);
    if !registry.has_operation(&id) {
        missing.operations.insert(id.to_string());
    }
    let operation = source.deref(ctx);
    for value in operation.operands().chain(operation.results()) {
        scan_type(ctx, value.get_type(ctx), registry, missing);
    }
    scan_attributes(&operation.attributes.0, registry, missing);
    scan_location(&operation.loc(), registry, missing);
    for region in operation.regions() {
        for block in region.deref(ctx).iter(ctx) {
            let block_ref = block.deref(ctx);
            for argument in block_ref.arguments() {
                scan_type(ctx, argument.get_type(ctx), registry, missing);
            }
            scan_attributes(&block_ref.attributes.0, registry, missing);
            scan_location(&block_ref.loc(), registry, missing);
            for child in block_ref.iter(ctx) {
                scan_operation(ctx, child, registry, missing);
            }
        }
    }
}

fn scan_location(
    location: &Location,
    registry: &TranslationRegistry,
    missing: &mut MissingMappings,
) {
    match location {
        Location::Fused {
            metadata,
            locations,
        } => {
            if let Some(metadata) = metadata {
                let id = metadata.get_attr_id();
                if !registry.has_attribute(&id) {
                    missing.attributes.insert(id.to_string());
                }
            }
            for location in locations {
                scan_location(location, registry, missing);
            }
        }
        Location::Named { child_loc, .. } => scan_location(child_loc, registry, missing),
        Location::CallSite { callee, caller } => {
            scan_location(callee, registry, missing);
            scan_location(caller, registry, missing);
        }
        Location::SrcPos { .. } | Location::Unknown => {}
    }
}

fn scan_type(
    ctx: &Context,
    source: TypeHandle,
    registry: &TranslationRegistry,
    missing: &mut MissingMappings,
) {
    let id = source.deref(ctx).get_type_id();
    if !registry.has_type(&id) {
        missing.types.insert(id.to_string());
    }
}

fn scan_attributes(
    attributes: &pliron::attribute::AttributeDictContainer,
    registry: &TranslationRegistry,
    missing: &mut MissingMappings,
) {
    for attribute in attributes.values() {
        let id = attribute.get_attr_id();
        if !registry.has_attribute(&id) {
            missing.attributes.insert(id.to_string());
        }
    }
}

fn index_operation(
    source: Ptr<Operation>,
    ctx: &Context,
    session: &mut TranslationSession<'_>,
    next_block: &mut u64,
) {
    for result in source.deref(ctx).results() {
        let id = session.fresh_value();
        session.values.insert(result, id);
    }
    for region in source.deref(ctx).regions() {
        let blocks = region.deref(ctx).iter(ctx).collect::<Vec<_>>();
        for block in &blocks {
            session.blocks.insert(*block, MlirBlockId(*next_block));
            *next_block += 1;
            for argument in block.deref(ctx).arguments() {
                let id = session.fresh_value();
                session.values.insert(argument, id);
            }
        }
        for block in blocks {
            for child in block.deref(ctx).iter(ctx) {
                index_operation(child, ctx, session, next_block);
            }
        }
    }
}

fn translate_operation(
    source: Ptr<Operation>,
    session: &mut TranslationSession<'_>,
) -> Result<Vec<MlirOperation>, TranslationError> {
    let ctx = session.ctx;
    let source_id = Operation::get_opid(source, ctx);
    let operation = source.deref(ctx);
    let mut regions = Vec::with_capacity(operation.num_regions());
    for region in operation.regions() {
        regions.push(translate_region(region, session)?);
    }

    let mut operands = Vec::with_capacity(operation.get_num_operands());
    for value in operation.operands() {
        let id = session.values.get(&value).copied().ok_or_else(|| {
            TranslationError::MissingIndexedValue {
                kind: "value",
                id: format!("{value:?}"),
            }
        })?;
        operands.push(MlirValueUse {
            id,
            ty: session.translate_type(value.get_type(ctx))?,
        });
    }

    let mut results = Vec::with_capacity(operation.get_num_results());
    for value in operation.results() {
        let id = session.values.get(&value).copied().ok_or_else(|| {
            TranslationError::MissingIndexedValue {
                kind: "value",
                id: format!("{value:?}"),
            }
        })?;
        results.push(MlirResult {
            id,
            ty: session.translate_type(value.get_type(ctx))?,
        });
    }

    let mut successors = Vec::with_capacity(operation.get_num_successors());
    for successor in operation.successors() {
        successors.push(session.blocks.get(&successor).copied().ok_or_else(|| {
            TranslationError::MissingIndexedValue {
                kind: "block",
                id: format!("{successor:?}"),
            }
        })?);
    }

    let attributes = translate_attributes(
        ctx,
        session.registry,
        session.config,
        &operation.attributes.0,
    )?;
    let location = translate_location(ctx, session, &operation.loc())?;
    drop(operation);

    let input = OperationInput {
        results: results.clone(),
        operands,
        successors,
        regions,
        attributes,
        location,
    };
    let translation = session.registry.operation(&source_id).ok_or_else(|| {
        let mut missing = MissingMappings::default();
        missing.operations.insert(source_id.to_string());
        TranslationError::MissingMappings(missing)
    })?;
    let translated = translation
        .translate(ctx, source, input, session)
        .map_err(|message| TranslationError::Mapping {
            source_item: source_id.to_string(),
            profile: session.config.profile.clone(),
            message,
        })?;
    check_source_results(&source_id.to_string(), &results, &translated)?;
    Ok(translated)
}

fn translate_region(
    source: Ptr<Region>,
    session: &mut TranslationSession<'_>,
) -> Result<MlirRegion, TranslationError> {
    let ctx = session.ctx;
    let blocks = source.deref(ctx).iter(ctx).collect::<Vec<_>>();
    let mut translated_blocks = Vec::with_capacity(blocks.len());
    for block in blocks {
        let block_ref = block.deref(ctx);
        if !block_ref.attributes.0.is_empty() {
            let translated = translate_attributes(
                ctx,
                session.registry,
                session.config,
                &block_ref.attributes.0,
            )?;
            if !translated.is_empty() {
                return Err(TranslationError::InvalidTarget(format!(
                    "MLIR blocks cannot carry the translated attributes on {block:?}"
                )));
            }
        }
        let id = session.blocks.get(&block).copied().ok_or_else(|| {
            TranslationError::MissingIndexedValue {
                kind: "block",
                id: format!("{block:?}"),
            }
        })?;
        let mut arguments = Vec::with_capacity(block_ref.get_num_arguments());
        for argument in block_ref.arguments() {
            arguments.push(MlirBlockArgument {
                id: session.values[&argument],
                ty: session.translate_type(argument.get_type(ctx))?,
                location: translate_location(ctx, session, &block_ref.loc())?,
            });
        }
        let operations = block_ref.iter(ctx).collect::<Vec<_>>();
        drop(block_ref);
        let mut translated_operations = vec![];
        for operation in operations {
            translated_operations.extend(translate_operation(operation, session)?);
        }
        translated_blocks.push(MlirBlock {
            id,
            arguments,
            operations: translated_operations,
        });
    }
    Ok(MlirRegion {
        blocks: translated_blocks,
    })
}

fn translate_type(
    ctx: &Context,
    registry: &TranslationRegistry,
    config: &TranslationConfig,
    source: TypeHandle,
) -> Result<MlirType, TranslationError> {
    let id = source.deref(ctx).get_type_id();
    let translation = registry.type_translation(&id).ok_or_else(|| {
        let mut missing = MissingMappings::default();
        missing.types.insert(id.to_string());
        TranslationError::MissingMappings(missing)
    })?;
    translation
        .translate(ctx, source, registry)
        .map_err(|message| TranslationError::Mapping {
            source_item: id.to_string(),
            profile: config.profile.clone(),
            message,
        })
}

fn translate_attribute(
    ctx: &Context,
    registry: &TranslationRegistry,
    config: &TranslationConfig,
    source: &AttrObj,
) -> Result<Option<MlirAttribute>, TranslationError> {
    let id = source.get_attr_id();
    let translation = registry.attribute(&id).ok_or_else(|| {
        let mut missing = MissingMappings::default();
        missing.attributes.insert(id.to_string());
        TranslationError::MissingMappings(missing)
    })?;
    translation
        .translate(ctx, source, registry)
        .map_err(|message| TranslationError::Mapping {
            source_item: id.to_string(),
            profile: config.profile.clone(),
            message,
        })
}

fn translate_attributes(
    ctx: &Context,
    registry: &TranslationRegistry,
    config: &TranslationConfig,
    source: &pliron::attribute::AttributeDictContainer,
) -> Result<BTreeMap<String, MlirAttribute>, TranslationError> {
    let mut translated = BTreeMap::new();
    for (name, attribute) in source {
        if let Some(attribute) = translate_attribute(ctx, registry, config, attribute)? {
            translated.insert(name.to_string(), attribute);
        }
    }
    Ok(translated)
}

fn translate_location(
    ctx: &Context,
    session: &TranslationSession<'_>,
    source: &Location,
) -> Result<MlirLocation, TranslationError> {
    match session.config.locations {
        LocationMode::Off => Ok(MlirLocation::Unknown),
        LocationMode::FileLineCol => Ok(first_file_location(ctx, source)),
        LocationMode::Full => full_location(ctx, session, source),
    }
}

fn first_file_location(ctx: &Context, source: &Location) -> MlirLocation {
    match source {
        Location::SrcPos { src, pos } => {
            source_position(ctx, *src, pos.line.max(0) as u32, pos.column.max(0) as u32)
        }
        Location::Fused { locations, .. } => locations
            .iter()
            .map(|location| first_file_location(ctx, location))
            .find(|location| *location != MlirLocation::Unknown)
            .unwrap_or(MlirLocation::Unknown),
        Location::Named { child_loc, .. } => first_file_location(ctx, child_loc),
        Location::CallSite { caller, callee } => {
            let caller = first_file_location(ctx, caller);
            if caller != MlirLocation::Unknown {
                caller
            } else {
                first_file_location(ctx, callee)
            }
        }
        Location::Unknown => MlirLocation::Unknown,
    }
}

fn full_location(
    ctx: &Context,
    session: &TranslationSession<'_>,
    source: &Location,
) -> Result<MlirLocation, TranslationError> {
    Ok(match source {
        Location::SrcPos { src, pos } => {
            source_position(ctx, *src, pos.line.max(0) as u32, pos.column.max(0) as u32)
        }
        Location::Fused {
            metadata,
            locations,
        } => MlirLocation::Fused {
            metadata: match metadata {
                Some(metadata) => session.translate_attribute(metadata)?.map(Box::new),
                None => None,
            },
            locations: locations
                .iter()
                .map(|location| full_location(ctx, session, location))
                .collect::<Result<_, _>>()?,
        },
        Location::Named { name, child_loc } => MlirLocation::Named {
            name: name.clone(),
            child: Box::new(full_location(ctx, session, child_loc)?),
        },
        Location::CallSite { callee, caller } => MlirLocation::CallSite {
            callee: Box::new(full_location(ctx, session, callee)?),
            caller: Box::new(full_location(ctx, session, caller)?),
        },
        Location::Unknown => MlirLocation::Unknown,
    })
}

fn source_position(ctx: &Context, source: Source, line: u32, column: u32) -> MlirLocation {
    match source {
        Source::File(path) => MlirLocation::FileLineCol {
            file: uniqued_any::get(ctx, path).display().to_string(),
            line,
            column,
        },
        Source::InMemory => MlirLocation::Unknown,
    }
}

fn check_source_results(
    source: &str,
    expected: &[MlirResult],
    translated: &[MlirOperation],
) -> Result<(), TranslationError> {
    let mut definitions = HashMap::<MlirValueId, Vec<MlirType>>::new();
    for operation in translated {
        for result in &operation.results {
            definitions
                .entry(result.id)
                .or_default()
                .push(result.ty.clone());
        }
    }
    for result in expected {
        match definitions.get(&result.id).map_or(0, Vec::len) {
            0 => {
                return Err(TranslationError::MissingResult {
                    source_item: source.into(),
                    result: result.id,
                });
            }
            1 => {
                let actual = &definitions[&result.id][0];
                if actual != &result.ty {
                    return Err(TranslationError::InvalidTarget(format!(
                        "mapping for `{source}` defines {:?} as {actual:?}, expected {:?}",
                        result.id, result.ty
                    )));
                }
            }
            _ => {
                return Err(TranslationError::DuplicateResult {
                    source_item: source.into(),
                    result: result.id,
                });
            }
        }
    }
    Ok(())
}

fn validate_target(module: &MlirModule) -> Result<(), TranslationError> {
    let mut values = HashMap::<MlirValueId, MlirType>::new();
    let mut blocks = HashSet::<MlirBlockId>::new();
    collect_target_definitions(&module.root, &mut values, &mut blocks)?;
    check_target_uses(&module.root, &values, &blocks)
}

fn collect_target_definitions(
    operation: &MlirOperation,
    values: &mut HashMap<MlirValueId, MlirType>,
    blocks: &mut HashSet<MlirBlockId>,
) -> Result<(), TranslationError> {
    for result in &operation.results {
        if values.insert(result.id, result.ty.clone()).is_some() {
            return Err(TranslationError::InvalidTarget(format!(
                "SSA value {:?} is defined more than once",
                result.id
            )));
        }
    }
    for region in &operation.regions {
        for block in &region.blocks {
            if !blocks.insert(block.id) {
                return Err(TranslationError::InvalidTarget(format!(
                    "block {:?} is defined more than once",
                    block.id
                )));
            }
            for argument in &block.arguments {
                if values.insert(argument.id, argument.ty.clone()).is_some() {
                    return Err(TranslationError::InvalidTarget(format!(
                        "SSA value {:?} is defined more than once",
                        argument.id
                    )));
                }
            }
            for child in &block.operations {
                collect_target_definitions(child, values, blocks)?;
            }
        }
    }
    Ok(())
}

fn check_target_uses(
    operation: &MlirOperation,
    values: &HashMap<MlirValueId, MlirType>,
    blocks: &HashSet<MlirBlockId>,
) -> Result<(), TranslationError> {
    for operand in &operation.operands {
        let Some(definition_type) = values.get(&operand.id) else {
            return Err(TranslationError::InvalidTarget(format!(
                "operand {:?} has no definition",
                operand.id
            )));
        };
        if definition_type != &operand.ty {
            return Err(TranslationError::InvalidTarget(format!(
                "operand {:?} uses type {:?}, but its definition has type {:?}",
                operand.id, operand.ty, definition_type
            )));
        }
    }
    for successor in &operation.successors {
        if !blocks.contains(successor) {
            return Err(TranslationError::InvalidTarget(format!(
                "successor {:?} has no block definition",
                successor
            )));
        }
    }
    for region in &operation.regions {
        for block in &region.blocks {
            for child in &block.operations {
                check_target_uses(child, values, blocks)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use pliron::{
        builtin::{
            attributes::{IdentifierAttr, StringAttr},
            ops::ModuleOp,
            types::IntegerType,
        },
        context::Context,
        identifier::Identifier,
        op::Op,
    };

    use crate::{
        OperationTranslation,
        registry::{DropAttribute, FixedType, OneToOneOperation},
    };

    use super::*;

    struct MappingCreatedRegion;

    impl OperationTranslation for MappingCreatedRegion {
        fn translate(
            &self,
            _ctx: &Context,
            _source: Ptr<Operation>,
            input: OperationInput,
            session: &mut TranslationSession<'_>,
        ) -> Result<Vec<MlirOperation>, String> {
            let mut root = MlirOperation::new("builtin.module")?;
            root.regions = input.regions;
            root.attributes = input.attributes;
            root.location = input.location.clone();
            let mut generated = MlirOperation::new("test.mapping_region")?;
            generated.regions.push(MlirRegion {
                blocks: vec![MlirBlock {
                    id: session.fresh_block(),
                    arguments: vec![],
                    operations: vec![],
                }],
            });
            generated.location = input.location;
            root.regions[0].blocks[0].operations.push(generated);
            Ok(vec![root])
        }
    }

    #[test]
    fn preflight_collects_every_missing_kind() {
        let mut ctx = Context::default();
        let module = ModuleOp::new(&mut ctx, Identifier::try_from("missing").unwrap());
        let error = translate_module(
            &ctx,
            &module,
            &TranslationRegistry::new(),
            &TranslationConfig::new("test"),
        )
        .unwrap_err();
        let TranslationError::MissingMappings(missing) = error else {
            panic!("expected missing mapping error");
        };
        assert!(missing.operations.contains("builtin.module"));
        assert!(
            missing
                .attributes
                .iter()
                .any(|id| id.contains("identifier"))
        );
    }

    #[test]
    fn module_round_trips_through_generic_ast() {
        let mut ctx = Context::default();
        let module = ModuleOp::new(&mut ctx, Identifier::try_from("empty").unwrap());
        let mut registry = TranslationRegistry::new();
        registry
            .register_operation::<ModuleOp>(OneToOneOperation::new("builtin.module"))
            .unwrap();
        registry
            .register_attribute::<IdentifierAttr>(DropAttribute)
            .unwrap();
        // Some Pliron revisions store symbol names as StringAttr instead.
        let _ = registry.register_attribute::<StringAttr>(DropAttribute);
        registry
            .register_type::<IntegerType>(FixedType(MlirType::Integer(32)))
            .unwrap();
        let translated =
            translate_module(&ctx, &module, &registry, &TranslationConfig::new("test")).unwrap();
        let text = crate::render_module(&translated);
        assert!(text.starts_with("\"builtin.module\"() ({"));
        assert!(text.ends_with(": () -> ()\n"));
    }

    #[test]
    fn mapping_created_block_ids_are_non_colliding_and_deterministic() {
        let mut ctx = Context::default();
        let module = ModuleOp::new(&mut ctx, Identifier::try_from("nested").unwrap());
        let mut registry = TranslationRegistry::new();
        registry
            .register_operation::<ModuleOp>(MappingCreatedRegion)
            .unwrap();
        registry
            .register_attribute::<IdentifierAttr>(DropAttribute)
            .unwrap();
        let _ = registry.register_attribute::<StringAttr>(DropAttribute);

        let first =
            translate_module(&ctx, &module, &registry, &TranslationConfig::new("test")).unwrap();
        let second =
            translate_module(&ctx, &module, &registry, &TranslationConfig::new("test")).unwrap();
        let source_block = first.root.regions[0].blocks[0].id;
        let generated_block = first.root.regions[0].blocks[0].operations[0].regions[0].blocks[0].id;
        assert_ne!(source_block, generated_block);
        assert_eq!(source_block, MlirBlockId(0));
        assert_eq!(generated_block, MlirBlockId(1));
        assert_eq!(crate::render_module(&first), crate::render_module(&second));
    }

    #[test]
    fn registry_rejects_duplicates_and_late_changes() {
        let mut registry = TranslationRegistry::new();
        registry
            .register_operation::<ModuleOp>(OneToOneOperation::new("builtin.module"))
            .unwrap();
        assert!(matches!(
            registry.register_operation::<ModuleOp>(OneToOneOperation::new("other.module")),
            Err(TranslationError::DuplicateMapping { .. })
        ));
        assert!(registry.has_operation(&ModuleOp::get_opid_static()));

        registry.seal();
        assert!(registry.is_sealed());
        assert!(matches!(
            registry.register_attribute::<IdentifierAttr>(DropAttribute),
            Err(TranslationError::RegistrySealed)
        ));
    }
}
