/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Attributes carried by structured PTX operations.

use pliron::attribute::Attribute;
use pliron::builtin::attributes::StringAttr;
use pliron::common_traits::Verify;
use pliron::context::Context;
use pliron::derive::pliron_attr;
use pliron::result::Result;
use pliron::verify_err_noloc;

/// The two callable forms defined by PTX.
#[pliron_attr(name = "ptx.callable_kind", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CallableKindAttr {
    Entry,
    Function,
}

impl From<ptx_parse::CallableKind> for CallableKindAttr {
    fn from(kind: ptx_parse::CallableKind) -> Self {
        match kind {
            ptx_parse::CallableKind::Entry => Self::Entry,
            ptx_parse::CallableKind::Function => Self::Function,
        }
    }
}

/// The guard predicate of one PTX instruction: `@%p` or `@!%p` before the
/// opcode. Predication is the only instruction prefix PTX defines, so this
/// typed attribute fully replaces free-form prefix text; the emitter derives
/// the `@`/`!` spelling from it.
#[pliron_attr(name = "ptx.predicate", format)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PredicateAttr {
    register: StringAttr,
    negated: bool,
}

impl PredicateAttr {
    pub fn new(register: &str, negated: bool) -> Self {
        Self {
            register: StringAttr::new(register.to_string()),
            negated,
        }
    }

    pub fn register(&self) -> &str {
        self.register.as_str()
    }

    pub fn is_negated(&self) -> bool {
        self.negated
    }

    /// The textual guard this predicate prints as, e.g. `@%p1` or `@!%p1`.
    pub fn guard_text(&self) -> String {
        format!(
            "@{}{}",
            if self.negated { "!" } else { "" },
            self.register.as_str()
        )
    }
}

impl From<ptx_parse::Predicate<'_>> for PredicateAttr {
    fn from(predicate: ptx_parse::Predicate<'_>) -> Self {
        Self::new(predicate.register(), predicate.is_negated())
    }
}

impl Verify for PredicateAttr {
    fn verify(&self, _ctx: &Context) -> Result<()> {
        if !self.register.as_str().starts_with('%') || self.register.as_str().len() < 2 {
            return verify_err_noloc!(
                "PTX predicate register {:?} must be a %-prefixed register name",
                self.register.as_str()
            );
        }
        Ok(())
    }
}

/// The control-flow role of a native PTX block terminator.
#[pliron_attr(name = "ptx.terminator_kind", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TerminatorKindAttr {
    Fallthrough,
    Branch,
    IndexedBranch,
    Return,
    ThreadExit,
    Trap,
}

pub fn register(ctx: &mut Context) {
    CallableKindAttr::register(ctx);
    // Qualified: the inherent `PredicateAttr::register` accessor returns the
    // guarded register name.
    <PredicateAttr as Attribute>::register(ctx);
    TerminatorKindAttr::register(ctx);
}
