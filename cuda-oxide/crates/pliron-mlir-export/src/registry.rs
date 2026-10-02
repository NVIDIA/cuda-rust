/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::{HashMap, hash_map::Entry};

use pliron::{
    attribute::{AttrId, AttrObj, Attribute},
    context::{Context, Ptr},
    op::{Op, OpId},
    operation::Operation,
    r#type::{Type, TypeHandle, TypeId},
};

use crate::translate::TranslationSession;
use crate::{MlirAttribute, MlirOperation, MlirType, OperationInput, TranslationError};

/// Translate one source type. Implementations may recursively ask the registry
/// to translate contained types.
pub trait TypeTranslation: Send + Sync {
    fn translate(
        &self,
        ctx: &Context,
        source: TypeHandle,
        registry: &TranslationRegistry,
    ) -> Result<MlirType, String>;
}

/// Translate one source attribute. `None` is an explicit, registered drop.
pub trait AttributeTranslation: Send + Sync {
    fn translate(
        &self,
        ctx: &Context,
        source: &AttrObj,
        registry: &TranslationRegistry,
    ) -> Result<Option<MlirAttribute>, String>;
}

/// Translate one source operation into zero, one, or several target ops.
pub trait OperationTranslation: Send + Sync {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String>;
}

/// Explicit mappings for one target MLIR profile.
#[derive(Default)]
pub struct TranslationRegistry {
    operations: HashMap<OpId, Box<dyn OperationTranslation>>,
    types: HashMap<TypeId, Box<dyn TypeTranslation>>,
    attributes: HashMap<AttrId, Box<dyn AttributeTranslation>>,
    sealed: bool,
}

impl TranslationRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Prevent every later registration. Profiles normally seal their
    /// registry after all mapping packs have been composed.
    pub fn seal(&mut self) {
        self.sealed = true;
    }

    pub fn is_sealed(&self) -> bool {
        self.sealed
    }

    /// Translate a nested source type from inside another mapping recipe.
    ///
    /// Container mappings use this for their element, field, and function
    /// signature types. Missing mappings remain explicit errors; there is no
    /// name-based fallback.
    pub fn translate_type(&self, ctx: &Context, source: TypeHandle) -> Result<MlirType, String> {
        let id = source.deref(ctx).get_type_id();
        self.type_translation(&id)
            .ok_or_else(|| format!("missing nested type mapping for `{id}`"))?
            .translate(ctx, source, self)
    }

    pub fn register_operation_id(
        &mut self,
        id: OpId,
        translation: impl OperationTranslation + 'static,
    ) -> Result<(), TranslationError> {
        if self.sealed {
            return Err(TranslationError::RegistrySealed);
        }
        match self.operations.entry(id.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(Box::new(translation));
                Ok(())
            }
            Entry::Occupied(_) => Err(TranslationError::DuplicateMapping {
                kind: "operation",
                id: id.to_string(),
            }),
        }
    }

    pub fn register_operation<O: Op>(
        &mut self,
        translation: impl OperationTranslation + 'static,
    ) -> Result<(), TranslationError> {
        self.register_operation_id(O::get_opid_static(), translation)
    }

    pub fn register_type_id(
        &mut self,
        id: TypeId,
        translation: impl TypeTranslation + 'static,
    ) -> Result<(), TranslationError> {
        if self.sealed {
            return Err(TranslationError::RegistrySealed);
        }
        match self.types.entry(id.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(Box::new(translation));
                Ok(())
            }
            Entry::Occupied(_) => Err(TranslationError::DuplicateMapping {
                kind: "type",
                id: id.to_string(),
            }),
        }
    }

    pub fn register_type<T: Type>(
        &mut self,
        translation: impl TypeTranslation + 'static,
    ) -> Result<(), TranslationError> {
        self.register_type_id(T::get_type_id_static(), translation)
    }

    pub fn register_attribute_id(
        &mut self,
        id: AttrId,
        translation: impl AttributeTranslation + 'static,
    ) -> Result<(), TranslationError> {
        if self.sealed {
            return Err(TranslationError::RegistrySealed);
        }
        match self.attributes.entry(id.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(Box::new(translation));
                Ok(())
            }
            Entry::Occupied(_) => Err(TranslationError::DuplicateMapping {
                kind: "attribute",
                id: id.to_string(),
            }),
        }
    }

    pub fn register_attribute<A: Attribute>(
        &mut self,
        translation: impl AttributeTranslation + 'static,
    ) -> Result<(), TranslationError> {
        self.register_attribute_id(A::get_attr_id_static(), translation)
    }

    pub(crate) fn operation(&self, id: &OpId) -> Option<&dyn OperationTranslation> {
        self.operations.get(id).map(Box::as_ref)
    }

    pub(crate) fn type_translation(&self, id: &TypeId) -> Option<&dyn TypeTranslation> {
        self.types.get(id).map(Box::as_ref)
    }

    pub(crate) fn attribute(&self, id: &AttrId) -> Option<&dyn AttributeTranslation> {
        self.attributes.get(id).map(Box::as_ref)
    }

    pub(crate) fn has_operation(&self, id: &OpId) -> bool {
        self.operations.contains_key(id)
    }

    pub(crate) fn has_type(&self, id: &TypeId) -> bool {
        self.types.contains_key(id)
    }

    pub(crate) fn has_attribute(&self, id: &AttrId) -> bool {
        self.attributes.contains_key(id)
    }
}

/// A fixed target type, useful for builtin scalar mappings.
#[derive(Clone)]
pub struct FixedType(pub MlirType);

impl TypeTranslation for FixedType {
    fn translate(
        &self,
        _ctx: &Context,
        _source: TypeHandle,
        _registry: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        Ok(self.0.clone())
    }
}

/// An explicitly dropped source attribute.
pub struct DropAttribute;

impl AttributeTranslation for DropAttribute {
    fn translate(
        &self,
        _ctx: &Context,
        _source: &AttrObj,
        _registry: &TranslationRegistry,
    ) -> Result<Option<MlirAttribute>, String> {
        Ok(None)
    }
}

/// Rename an operation while preserving its typed operands, results, regions,
/// successors, attributes, and location.
pub struct OneToOneOperation {
    target_name: String,
}

impl OneToOneOperation {
    pub fn new(target_name: impl Into<String>) -> Self {
        Self {
            target_name: target_name.into(),
        }
    }
}

impl OperationTranslation for OneToOneOperation {
    fn translate(
        &self,
        _ctx: &Context,
        _source: Ptr<Operation>,
        input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let mut operation = MlirOperation::new(&self.target_name)?;
        operation.results = input.results;
        operation.operands = input.operands;
        operation.successors = input.successors;
        operation.regions = input.regions;
        operation.attributes = input.attributes;
        operation.location = input.location;
        Ok(vec![operation])
    }
}
