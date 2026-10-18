// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-mlir-interop contributors

//! MLIR translation for LLVM dialect attributes.

use std::fmt;

use pliron::{
    builtin::attributes::IntegerAttr, context::Context, derive::attr_interface_impl,
    input_err_noloc, printable::State, result::Result,
};
use pliron_llvm::attributes::{
    AddressSpaceAttr, AggregateAttr, AlignmentAttr, AtomicOrderingAttr, AtomicRmwKindAttr,
    BytesAttr, CaseValuesAttr, FCmpPredicateAttr, FastmathFlags, FastmathFlagsAttr, GepIndexAttr,
    GepIndicesAttr, GepNoWrapFlagsAttr, ICmpPredicateAttr, InsertExtractValueIndicesAttr,
    IntegerOverflowFlagsAttr, LinkageAttr, PoisonAttr, ShuffleVectorMaskAttr, SplatAttr, UndefAttr,
    ZeroAttr,
};

use crate::{
    Error, ToMlirAttr,
    llvm::constants::print_dense,
    printers::{mlir_dense_array, print_type},
};

/// The `rawConstantIndices` entry MLIR's `llvm.getelementptr` uses to say
/// "this index comes from the next `dynamicIndices` operand".
///
/// See `LLVM::GEPOp::kDynamicIndex`.
const GEP_DYNAMIC_INDEX: i32 = i32::MIN;

/// MLIR's `#llvm.linkage<...>` keyword for `linkage`.
///
/// LLVM-IR has more linkage kinds than MLIR's LLVM dialect models; those have
/// no keyword and yield an error.
fn linkage_keyword(linkage: &LinkageAttr) -> Result<&'static str> {
    let keyword = match linkage {
        LinkageAttr::ExternalLinkage => "external",
        LinkageAttr::AvailableExternallyLinkage => "available_externally",
        LinkageAttr::LinkOnceAnyLinkage => "linkonce",
        LinkageAttr::LinkOnceODRLinkage => "linkonce_odr",
        LinkageAttr::WeakAnyLinkage => "weak",
        LinkageAttr::WeakODRLinkage => "weak_odr",
        LinkageAttr::AppendingLinkage => "appending",
        LinkageAttr::InternalLinkage => "internal",
        LinkageAttr::PrivateLinkage => "private",
        LinkageAttr::ExternalWeakLinkage => "extern_weak",
        LinkageAttr::CommonLinkage => "common",
        unsupported => {
            return input_err_noloc!(Error::Untranslatable(format!(
                "{unsupported:?} has no MLIR equivalent"
            )));
        }
    };
    Ok(keyword)
}

/// `#llvm.linkage<external>` and friends.
#[attr_interface_impl]
impl ToMlirAttr for LinkageAttr {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "#llvm.linkage<{}>", linkage_keyword(self)?)?;
        Ok(())
    }
}

/// MLIR keeps these enums as plain `i64` *properties*.
/// See `mlir/include/mlir/Dialect/LLVMIR/LLVMEnums.td`.
macro_rules! int_enum_attrs {
    ($( $mlir_enum:literal $attr:ident { $($variant:ident = $value:literal),* $(,)? } )*) => { $(
        #[doc = concat!("The integer value of MLIR's `", $mlir_enum, "` enum.")]
        #[attr_interface_impl]
        impl ToMlirAttr for $attr {
            fn to_mlir(
                &self,
                _ctx: &Context,
                _state: &State,
                f: &mut fmt::Formatter<'_>,
            ) -> Result<()> {
                let value: u64 = match self {
                    $( $attr::$variant => $value, )*
                };
                write!(f, "{value} : i64")?;
                Ok(())
            }
        }
    )* };
}

int_enum_attrs! {
    "LLVM::ICmpPredicate" ICmpPredicateAttr {
        EQ = 0, NE = 1, SLT = 2, SLE = 3, SGT = 4,
        SGE = 5, ULT = 6, ULE = 7, UGT = 8, UGE = 9,
    }
    "LLVM::FCmpPredicate" FCmpPredicateAttr {
        False = 0, OEQ = 1, OGT = 2, OGE = 3, OLT = 4, OLE = 5, ONE = 6, ORD = 7,
        UEQ = 8, UGT = 9, UGE = 10, ULT = 11, ULE = 12, UNE = 13, UNO = 14, True = 15,
    }
    "LLVM::AtomicOrdering" AtomicOrderingAttr {
        Monotonic = 2, Acquire = 4, Release = 5, AcqRel = 6, SeqCst = 7,
    }
    "LLVM::AtomicBinOp" AtomicRmwKindAttr {
        Xchg = 0, Add = 1, Sub = 2, And = 3, Nand = 4, Or = 5, Xor = 6, Max = 7,
        Min = 8, UMax = 9, UMin = 10, FAdd = 11, FSub = 12, FMax = 13, FMin = 14,
    }
}

/// `#llvm.overflow<none>`, `<nsw>`, `<nuw>` or `<nsw, nuw>`.
#[attr_interface_impl]
impl ToMlirAttr for IntegerOverflowFlagsAttr {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let flags: Vec<&str> = [(self.nsw, "nsw"), (self.nuw, "nuw")]
            .into_iter()
            .filter_map(|(set, name)| set.then_some(name))
            .collect();
        if flags.is_empty() {
            write!(f, "#llvm.overflow<none>")?;
        } else {
            write!(f, "#llvm.overflow<{}>", flags.join(", "))?;
        }
        Ok(())
    }
}

/// The integer value of MLIR's `LLVM::GEPNoWrapFlags` bit enum.
///
/// pliron's normalized flags are the same as MLIR's.
#[attr_interface_impl]
impl ToMlirAttr for GepNoWrapFlagsAttr {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "{} : i32", self.0.normalized().bits())?;
        Ok(())
    }
}

/// `#llvm.fastmath<none>` or `#llvm.fastmath<nnan, ninf, ...>`.
#[attr_interface_impl]
impl ToMlirAttr for FastmathFlagsAttr {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        // MLIR's flag order, which is also the order it prints them in.
        let flags: Vec<&str> = [
            (FastmathFlags::NNAN, "nnan"),
            (FastmathFlags::NINF, "ninf"),
            (FastmathFlags::NSZ, "nsz"),
            (FastmathFlags::ARCP, "arcp"),
            (FastmathFlags::CONTRACT, "contract"),
            (FastmathFlags::AFN, "afn"),
            (FastmathFlags::REASSOC, "reassoc"),
        ]
        .into_iter()
        .filter_map(|(flag, name)| self.0.contains(flag).then_some(name))
        .collect();
        if flags.is_empty() {
            write!(f, "#llvm.fastmath<none>")?;
        } else {
            write!(f, "#llvm.fastmath<{}>", flags.join(", "))?;
        }
        Ok(())
    }
}

/// `N : i64`, matching MLIR's `alignment` property.
#[attr_interface_impl]
impl ToMlirAttr for AlignmentAttr {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "{} : i64", self.0)?;
        Ok(())
    }
}

/// `N : i32`, matching MLIR's `addr_space` property.
#[attr_interface_impl]
impl ToMlirAttr for AddressSpaceAttr {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "{} : i32", self.0)?;
        Ok(())
    }
}

/// `#llvm.zero`.
#[attr_interface_impl]
impl ToMlirAttr for ZeroAttr {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "#llvm.zero")?;
        Ok(())
    }
}

/// `#llvm.undef`.
#[attr_interface_impl]
impl ToMlirAttr for UndefAttr {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "#llvm.undef")?;
        Ok(())
    }
}

/// `#llvm.poison`.
#[attr_interface_impl]
impl ToMlirAttr for PoisonAttr {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "#llvm.poison")?;
        Ok(())
    }
}

/// `array<i64: ...>`, matching MLIR's `position` property on
/// `llvm.insertvalue` / `llvm.extractvalue`.
#[attr_interface_impl]
impl ToMlirAttr for InsertExtractValueIndicesAttr {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "{}", mlir_dense_array("i64", &self.0))?;
        Ok(())
    }
}

/// `array<i32: ...>`, matching MLIR's `mask` property on
/// `llvm.shufflevector`.
#[attr_interface_impl]
impl ToMlirAttr for ShuffleVectorMaskAttr {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "{}", mlir_dense_array("i32", &self.0))?;
        Ok(())
    }
}

/// MLIR's `rawConstantIndices` for `indices`.
///
/// pliron's method of representing dynamic indices differs from that of MLIR's.
/// This function only works when pliron's dynamic indices refer to operands in
/// an ascending order (which will happen if pliron's construction APIs are used).
fn gep_raw_constant_indices(indices: &GepIndicesAttr) -> Result<String> {
    let mut next_operand = 1;
    let mut raw = Vec::with_capacity(indices.0.len());
    for index in &indices.0 {
        match index {
            GepIndexAttr::Constant(c) => {
                // pliron stores a GEP constant index as the raw bits of LLVM's
                // (signed) i32 index, so reinterpret rather than widen.
                let c = *c as i32;
                if c == GEP_DYNAMIC_INDEX {
                    return input_err_noloc!(Error::Untranslatable(format!(
                        "llvm.gep constant index {c} is indistinguishable from \
                         MLIR's marker for a dynamic index"
                    )));
                }
                raw.push(c);
            }
            GepIndexAttr::OperandIdx(idx) => {
                if *idx != next_operand {
                    return input_err_noloc!(Error::Untranslatable(format!(
                        "llvm.gep dynamic index refers to operand {idx}, \
                         but MLIR consumes dynamic indices in operand order \
                         (expected operand {next_operand})"
                    )));
                }
                next_operand += 1;
                raw.push(GEP_DYNAMIC_INDEX);
            }
        }
    }
    Ok(mlir_dense_array("i32", raw))
}

/// `array<i32: ...>`, matching MLIR's `rawConstantIndices` property on
/// `llvm.getelementptr`.
#[attr_interface_impl]
impl ToMlirAttr for GepIndicesAttr {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(f, "{}", gep_raw_constant_indices(self)?)?;
        Ok(())
    }
}

/// `dense<[...]> : vector<Nxi32>`, matching MLIR's `case_values` property on
/// `llvm.switch`.
#[attr_interface_impl]
impl ToMlirAttr for CaseValuesAttr {
    fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        let Some(first) = self.0.first() else {
            return input_err_noloc!(Error::Untranslatable(
                "an empty llvm.switch case value list has no MLIR element type".to_string()
            ));
        };
        let ty = IntegerAttr::get_type(first);
        let signed = ty.deref(ctx).is_signed();
        let values = self
            .0
            .iter()
            .map(|v| v.value().to_string_decimal(signed))
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "dense<[{values}]> : vector<{}x", self.0.len())?;
        print_type(ctx, ty.into(), state, f)?;
        write!(f, ">")?;
        Ok(())
    }
}

/// `dense<...> : tensor<...>` or `dense<...> : vector<...>`, for the
/// constants that MLIR holds in a dense elements attribute.
///
/// Fails for other constants (see [crate::llvm::constants]).
macro_rules! dense_attrs {
    ($($attr:ty),* $(,)?) => { $(
        #[attr_interface_impl]
        impl ToMlirAttr for $attr {
            fn to_mlir(
                &self,
                ctx: &Context,
                state: &State,
                f: &mut fmt::Formatter<'_>,
            ) -> Result<()> {
                if !print_dense(ctx, self, state, f)? {
                    return input_err_noloc!(Error::Untranslatable(format!(
                        "constant `{}` has no MLIR attribute; \
                         only `llvm.mlir.constant` and `llvm.mlir.global` can build it",
                        pliron::attribute::Attribute::get_attr_id(self)
                    )));
                }
                Ok(())
            }
        }
    )* };
}

dense_attrs!(BytesAttr, AggregateAttr, SplatAttr);
