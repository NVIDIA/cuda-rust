/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Version and target requirements, independent of toolkit discovery or a GPU.

use crate::bytecode::{BytecodeVersion, Opcode};
use crate::ir::{Attribute, Location, Module, OpId, Operation, ScalarType, TileElementType, Type};
use crate::requirements::{
    attribute_feature, opcode_requirement, scalar_requirement, target_requirement, Feature,
};

/// The target selected for one compilation, not the first GPU in the process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetCapabilities {
    pub bytecode_version: BytecodeVersion,
    pub architecture: String,
}

/// An unsupported feature with the location of its use, before invoking tileiras.
#[derive(Clone, Debug)]
pub struct CapabilityError {
    pub message: String,
    pub location: Box<Location>,
}

impl std::fmt::Display for CapabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CapabilityError {}

impl TargetCapabilities {
    pub fn new(bytecode_version: BytecodeVersion, architecture: impl Into<String>) -> Self {
        Self {
            bytecode_version,
            architecture: architecture.into(),
        }
    }

    /// Accept ordinary, architecture-specific, and family-specific target names.
    pub fn sm(&self) -> Option<u32> {
        self.architecture
            .strip_prefix("sm_")?
            .trim_end_matches(['a', 'f'])
            .parse()
            .ok()
    }

    pub(crate) fn error(
        &self,
        feature: &str,
        requirement: &str,
        location: &Location,
    ) -> CapabilityError {
        CapabilityError {
            message: format!(
                "{feature} requires {requirement}; selected Tile IR {}, target {}",
                self.bytecode_version, self.architecture
            ),
            location: Box::new(location.clone()),
        }
    }

    pub fn require_version(
        &self,
        feature: &str,
        since: BytecodeVersion,
        location: &Location,
    ) -> Result<(), CapabilityError> {
        if self.bytecode_version < since {
            Err(self.error(feature, &format!("Tile IR {since} or newer"), location))
        } else {
            Ok(())
        }
    }

    pub fn require_sm(
        &self,
        feature: &str,
        minimum: u32,
        location: &Location,
    ) -> Result<(), CapabilityError> {
        if self.sm().is_none_or(|sm| sm < minimum) {
            Err(self.error(feature, &format!("sm_{minimum} or newer"), location))
        } else {
            Ok(())
        }
    }

    fn check_scalar(
        &self,
        scalar: ScalarType,
        feature: &str,
        loc: &Location,
    ) -> Result<(), CapabilityError> {
        let requirement = scalar_requirement(scalar);
        self.require_version(feature, requirement.since, loc)?;
        requirement.check_architecture(self, &format!("{feature} using {scalar:?}"), loc)
    }

    fn check_type(&self, ty: &Type, feature: &str, loc: &Location) -> Result<(), CapabilityError> {
        crate::verify_target::verify_type(ty).map_err(|message| CapabilityError {
            message: format!("{feature}: {message}"),
            location: Box::new(loc.clone()),
        })?;
        match ty {
            Type::WithPointerAttribute(base, _) => {
                Feature::PointerAttribute.check(self, loc)?;
                self.check_type(base, feature, loc)
            }
            Type::Scalar(s) => self.check_scalar(*s, feature, loc),
            Type::Pointer(p) => self.check_scalar(p.pointee, feature, loc),
            Type::Tile(t) => match &t.element_type {
                TileElementType::Scalar(s) => self.check_scalar(*s, feature, loc),
                TileElementType::Pointer(p) => self.check_scalar(p.pointee, feature, loc),
            },
            Type::TensorView(t) => self.check_scalar(t.element_type, feature, loc),
            Type::PartitionView(t) => self.check_scalar(t.tensor_view.element_type, feature, loc),
            Type::GatherScatterView(t) => {
                Feature::GatherScatterView.check(self, loc)?;
                self.check_scalar(t.tensor_view.element_type, feature, loc)
            }
            Type::StridedView(t) => {
                Feature::StridedView.check(self, loc)?;
                self.check_scalar(t.tensor_view.element_type, feature, loc)
            }
            Type::Func(f) => {
                for ty in f.inputs.iter().chain(&f.results) {
                    self.check_type(ty, feature, loc)?;
                }
                Ok(())
            }
            Type::Token => Ok(()),
        }
    }

    /// Validate reachable operations, including compiler-generated operations.
    /// Version and architecture requirements are conjunctive, never alternatives.
    pub fn validate_module(&self, module: &Module) -> Result<(), CapabilityError> {
        if module.producer.is_some() {
            Feature::Producer.check(self, &Location::Unknown)?;
        }
        Feature::Baseline.requirement().check_architecture(
            self,
            "Tile IR target",
            &Location::Unknown,
        )?;
        if let Some(requirement) = self.sm().and_then(target_requirement) {
            self.require_version(
                &format!("target {}", self.architecture),
                requirement.since,
                &Location::Unknown,
            )?;
        } else {
            return Err(self.error(
                "Tile IR target",
                "an architecture supported by the public Tile IR 13.4 dialect",
                &Location::Unknown,
            ));
        }
        for global in &module.globals {
            self.check_type(&global.value.element_type, "global", &Location::Unknown)?;
            for (enabled, feature) in [
                (global.constant, Feature::GlobalConstant),
                (
                    global.symbol_visibility != crate::ir::SymbolVisibility::Public,
                    Feature::GlobalVisibility,
                ),
            ] {
                if enabled {
                    self.require_version(
                        "global constant/visibility",
                        feature.since(),
                        &Location::Unknown,
                    )?;
                }
            }
        }
        for &entry in &module.functions {
            self.check_op(module, entry, false, false)?;
        }
        Ok(())
    }

    fn check_op(
        &self,
        module: &Module,
        id: OpId,
        in_loop: bool,
        in_for: bool,
    ) -> Result<(), CapabilityError> {
        let op = module.op(id);
        let feature = format!("cuda_tile.{}", op.opcode.name());
        let loc = &op.location;
        opcode_requirement(op.opcode).check(self, &feature, loc)?;
        for ty in op
            .result_types
            .iter()
            .chain(op.operands.iter().map(|v| module.value_type(*v)))
        {
            self.check_type(ty, &feature, loc)?;
        }
        for (name, attr) in &op.attributes {
            match attr {
                Attribute::Type(ty) => self.check_type(ty, &feature, loc)?,
                Attribute::DenseElements(d) => self.check_type(&d.element_type, &feature, loc)?,
                Attribute::OptimizationHints(hints) => {
                    for (arch, values) in &hints.entries {
                        if arch != "default" {
                            let requirement = arch
                                .strip_prefix("sm_")
                                .and_then(|sm| sm.parse().ok())
                                .filter(|sm: &u32| arch == &format!("sm_{sm}"))
                                .and_then(target_requirement)
                                .ok_or_else(|| {
                                    self.error(
                                        &format!("optimization hint target {arch}"),
                                        "a public Tile IR architecture key or default",
                                        loc,
                                    )
                                })?;
                            self.require_version(
                                &format!("optimization hint target {arch}"),
                                requirement.since,
                                loc,
                            )?;
                        }
                        for (key, value) in values {
                            if key == "num_worker_warps_per_cta" {
                                Feature::WorkerWarps.check(self, loc)?;
                            }
                            let mut hint_sm =
                                arch.strip_prefix("sm_").and_then(|s| s.parse::<u32>().ok());
                            if arch == "default"
                                && !hints.entries.iter().any(|(target, values)| {
                                    target == &self.architecture
                                        && values.iter().any(|(key, _)| key == "num_cta_in_cga")
                                })
                            {
                                hint_sm = self.sm();
                            }
                            if key == "num_cta_in_cga"
                                && matches!(value, Attribute::Integer(n, _) if *n != 1)
                            {
                                if let Some(sm) = hint_sm {
                                    let hint_target =
                                        Self::new(self.bytecode_version, format!("sm_{sm}"));
                                    Feature::ClusterCta.check(&hint_target, loc)?;
                                }
                            }
                            let valid = match (key.as_str(), value) {
                                ("num_cta_in_cga", Attribute::Integer(n, _)) => {
                                    op.opcode == Opcode::Entry && matches!(n, 1 | 2 | 4 | 8 | 16)
                                }
                                ("occupancy", Attribute::Integer(n, _)) => {
                                    op.opcode == Opcode::Entry && (1..=32).contains(n)
                                }
                                ("num_worker_warps_per_cta", Attribute::Integer(n, _)) => {
                                    op.opcode == Opcode::Entry
                                        && if self.bytecode_version
                                            >= Feature::WorkerWarpsRestricted.since()
                                        {
                                            matches!(n, 4 | 8)
                                        } else {
                                            matches!(n, 1 | 2 | 4 | 8 | 16 | 32)
                                        }
                                }
                                ("allow_tma", Attribute::Bool(_)) => {
                                    matches!(op.opcode, Opcode::LoadViewTko | Opcode::StoreViewTko)
                                }
                                ("latency", Attribute::Integer(n, _)) => {
                                    matches!(
                                        op.opcode,
                                        Opcode::LoadViewTko
                                            | Opcode::StoreViewTko
                                            | Opcode::LoadPtrTko
                                            | Opcode::StorePtrTko
                                    ) && (1..=10).contains(n)
                                }
                                _ => false,
                            };
                            if !valid {
                                return Err(CapabilityError {
                                    message: format!(
                                        "{feature}: invalid optimization hint {key}={value:?}"
                                    ),
                                    location: Box::new(loc.clone()),
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
            if let Some(requirement) = attribute_feature(op.opcode, name, attr) {
                requirement
                    .requirement()
                    .check(self, &format!("{feature}.{name}"), loc)?;
            }
        }
        if op.opcode == Opcode::MmaFScaled {
            let new_scale = op.operands.iter().skip(3).any(|v| {
                matches!(module.value_type(*v), Type::Tile(t) if t.element_type == TileElementType::Scalar(ScalarType::F8E5M3FNU))
            });
            // This is a configuration/family restriction, not sm >= 107:
            // SM120/121 support NVFP4 but not this new scale format.
            if new_scale {
                Feature::NewScaleMma.check(self, loc)?;
            }
        }
        if matches!(op.opcode, Opcode::AtomicRMW | Opcode::AtomicRedViewTko)
            && integer_attr(op, "mode") == Some(4)
            && op.operands.iter().any(|v| matches!(module.value_type(*v), Type::Tile(t) if t.element_type == TileElementType::Scalar(ScalarType::BF16))) {
            Feature::Bf16AtomicAdd.requirement().check(self, &format!("{feature} with bf16 add"), loc)?;
        }
        if op.opcode == Opcode::Return && in_loop {
            Feature::LoopReturn.check(self, loc)?;
        }
        if op.opcode == Opcode::Return && in_for {
            return Err(self.error("return inside for", "return outside the for region", loc));
        }
        crate::verify_target::verify_operation(module, op, self)?;
        for &region in &op.regions {
            for &block in &module.region(region).blocks {
                for &child in &module.block(block).ops {
                    self.check_op(
                        module,
                        child,
                        in_loop || op.opcode == Opcode::Loop,
                        in_for || op.opcode == Opcode::For,
                    )?;
                }
            }
        }
        Ok(())
    }
}

fn integer_attr(op: &Operation, name: &str) -> Option<i64> {
    op.attributes.iter().find_map(|(key, value)| match value {
        Attribute::Integer(n, _) if key == name => Some(*n),
        _ => None,
    })
}
