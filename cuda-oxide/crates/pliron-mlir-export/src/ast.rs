/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::BTreeMap;

/// A deterministic SSA value number in the target module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MlirValueId(pub u64);

/// A deterministic target block number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MlirBlockId(pub u64);

/// Builtin floating-point types understood by the generic renderer.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MlirFloatType {
    F16,
    BF16,
    F32,
    F64,
    /// A target-defined builtin spelling such as `f8E4M3FN`.
    Other(String),
}

/// A target MLIR type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MlirType {
    Index,
    Integer(u32),
    Float(MlirFloatType),
    Vector {
        shape: Vec<u64>,
        element: Box<MlirType>,
    },
    Function {
        inputs: Vec<MlirType>,
        results: Vec<MlirType>,
    },
    Tuple(Vec<MlirType>),
    /// A dialect type, including the leading `!`, held as one checked token.
    Dialect(String),
}

impl MlirType {
    /// Create a dialect type while rejecting text that could escape its token.
    pub fn dialect(spelling: impl Into<String>) -> Result<Self, String> {
        let spelling = spelling.into();
        if !spelling.starts_with('!') || has_forbidden_text(&spelling) {
            return Err(format!(
                "dialect type must start with '!' and stay on one line: {spelling:?}"
            ));
        }
        Ok(Self::Dialect(spelling))
    }
}

/// A target MLIR attribute.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MlirAttribute {
    Unit,
    Bool(bool),
    Integer {
        value: i128,
        ty: MlirType,
    },
    Float {
        value: String,
        ty: MlirType,
    },
    String(String),
    Type(MlirType),
    Array(Vec<MlirAttribute>),
    /// MLIR's compact `array<i32: ...>` form used by operation properties
    /// such as `operandSegmentSizes`.
    DenseI32Array(Vec<i32>),
    /// MLIR's compact `array<i64: ...>` form used by aggregate index
    /// properties such as LLVM dialect `position`.
    DenseI64Array(Vec<i64>),
    Dictionary(BTreeMap<String, MlirAttribute>),
    SymbolRef(String),
    /// A dialect attribute, including the leading `#`, held as one checked token.
    Dialect(String),
}

impl MlirAttribute {
    /// Create a dialect attribute while rejecting text that could escape it.
    pub fn dialect(spelling: impl Into<String>) -> Result<Self, String> {
        let spelling = spelling.into();
        if !spelling.starts_with('#') || has_forbidden_text(&spelling) {
            return Err(format!(
                "dialect attribute must start with '#' and stay on one line: {spelling:?}"
            ));
        }
        Ok(Self::Dialect(spelling))
    }
}

/// A source location carried into MLIR.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MlirLocation {
    Unknown,
    FileLineCol {
        file: String,
        line: u32,
        column: u32,
    },
    Fused {
        metadata: Option<Box<MlirAttribute>>,
        locations: Vec<MlirLocation>,
    },
    Named {
        name: String,
        child: Box<MlirLocation>,
    },
    CallSite {
        callee: Box<MlirLocation>,
        caller: Box<MlirLocation>,
    },
}

/// One typed use of an SSA value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlirValueUse {
    pub id: MlirValueId,
    pub ty: MlirType,
}

/// One typed SSA result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlirResult {
    pub id: MlirValueId,
    pub ty: MlirType,
}

/// One operation in generic MLIR form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlirOperation {
    pub name: String,
    pub results: Vec<MlirResult>,
    pub operands: Vec<MlirValueUse>,
    pub successors: Vec<MlirBlockId>,
    /// Inherent operation properties rendered as `<{...}>`.
    ///
    /// Properties are not ordinary discardable attributes in modern MLIR.
    /// Keeping them separate prevents a mapping from accidentally emitting a
    /// structurally valid-looking operation whose verifier sees defaults.
    pub properties: BTreeMap<String, MlirAttribute>,
    pub regions: Vec<MlirRegion>,
    pub attributes: BTreeMap<String, MlirAttribute>,
    pub location: MlirLocation,
}

impl MlirOperation {
    /// Start an operation with no operands, results, regions, or attributes.
    pub fn new(name: impl Into<String>) -> Result<Self, String> {
        let name = name.into();
        if name.is_empty() || has_forbidden_text(&name) || name.contains('"') {
            return Err(format!("invalid MLIR operation name: {name:?}"));
        }
        Ok(Self {
            name,
            results: vec![],
            operands: vec![],
            successors: vec![],
            properties: BTreeMap::new(),
            regions: vec![],
            attributes: BTreeMap::new(),
            location: MlirLocation::Unknown,
        })
    }
}

/// A typed argument at the start of an MLIR block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlirBlockArgument {
    pub id: MlirValueId,
    pub ty: MlirType,
    pub location: MlirLocation,
}

/// A target MLIR block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlirBlock {
    pub id: MlirBlockId,
    pub arguments: Vec<MlirBlockArgument>,
    pub operations: Vec<MlirOperation>,
}

/// A target MLIR region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlirRegion {
    pub blocks: Vec<MlirBlock>,
}

/// A translated module and the profile that defined its mappings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlirModule {
    pub root: MlirOperation,
    pub profile: String,
}

fn has_forbidden_text(value: &str) -> bool {
    value.chars().any(|ch| ch.is_control())
}
