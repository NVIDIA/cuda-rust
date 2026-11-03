/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device extern declaration types for FFI with external LTOIR.

use std::fmt::Write;

/// A device-extern type that preserves pointer pointees and address spaces.
///
/// The lowered module uses opaque pointers, so this separate type description
/// is needed to emit declarations such as `declare void @f(float*)` for
/// pre-Blackwell libNVVM. Unsupported extern signatures are rejected.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DeviceExternType {
    /// Only valid as a function result.
    Void,
    /// A signless LLVM integer. Signedness is not part of an LLVM integer type.
    Integer(u32),
    /// A sub-32-bit signed integer passed or returned with `signext`.
    ///
    /// The width stored here is the ORIGINAL source width (8 or 16), never a
    /// promoted width. The declared IR type stays narrow (`i8`/`i16`) with the
    /// `signext` attribute attached, matching clang's NVPTXABIInfo and rustc's
    /// nvptx64 callconv; the NVPTX backend performs the `.param.b32` widening.
    /// Keeping the type narrow means cuda-oxide's own i8/i16 SSA values match
    /// the declaration with no inserted conversions.
    SignExtInteger(u32),
    /// A sub-32-bit unsigned integer (or `bool`, width 1) passed or returned
    /// with `zeroext`.
    ///
    /// The width stored here is the ORIGINAL source width (1, 8, or 16),
    /// never a promoted width. See [`Self::SignExtInteger`] for the rationale.
    ZeroExtInteger(u32),
    Float16,
    Float32,
    Float64,
    /// A pointer with an exact pointee and NVVM address space.
    Pointer {
        pointee: Box<DeviceExternType>,
        address_space: u32,
    },
    /// A fixed-size array. Arrays are supported as pointer pointees; passing an
    /// array by value is not yet supported.
    Array {
        element: Box<DeviceExternType>,
        len: u64,
    },
}

impl DeviceExternType {
    pub fn pointer_to(pointee: DeviceExternType, address_space: u32) -> Self {
        Self::Pointer {
            pointee: Box::new(pointee),
            address_space,
        }
    }

    pub fn pointer_parts(&self) -> Option<(&DeviceExternType, u32)> {
        match self {
            Self::Pointer {
                pointee,
                address_space,
            } => Some((pointee, *address_space)),
            _ => None,
        }
    }

    /// Return the integer width for any integer variant (plain, signext, zeroext).
    pub fn integer_width(&self) -> Option<u32> {
        match self {
            Self::Integer(bits) | Self::SignExtInteger(bits) | Self::ZeroExtInteger(bits) => {
                Some(*bits)
            }
            _ => None,
        }
    }

    /// The `signext`/`zeroext` ABI extension attribute for this type, if any.
    ///
    /// In parameter position LLVM IR places the attribute AFTER the type
    /// (`i8 signext`; see [`Self::write_llvm_with_attr`]); in return position
    /// it must come BEFORE the type (`declare signext i8 @f()`), so return
    /// emitters print this attribute first and then call [`Self::write_llvm`].
    pub(crate) fn ext_attr(&self) -> Option<&'static str> {
        match self {
            Self::SignExtInteger(_) => Some("signext"),
            Self::ZeroExtInteger(_) => Some("zeroext"),
            _ => None,
        }
    }

    /// True when this legacy pointer is already the internal byte-pointer type.
    pub(crate) fn is_canonical_byte_pointer(&self) -> bool {
        matches!(
            self,
            Self::Pointer { pointee, .. }
                if matches!(pointee.as_ref(), Self::Integer(8))
        )
    }

    pub(crate) fn contains_float16(&self) -> bool {
        match self {
            Self::Float16 => true,
            Self::Pointer { pointee, .. } => pointee.contains_float16(),
            Self::Array { element, .. } => element.contains_float16(),
            _ => false,
        }
    }

    /// Write the LLVM IR type string for this type (without ABI attributes).
    ///
    /// For `SignExtInteger` and `ZeroExtInteger`, this writes just the `iN`
    /// type. The `signext`/`zeroext` attributes are emitted separately by the
    /// declaration emitter.
    ///
    /// With modern LLVM syntax, pointer pointees are omitted while address
    /// spaces are retained.
    pub(crate) fn write_llvm(
        &self,
        output: &mut String,
        legacy_typed_pointers: bool,
    ) -> Result<(), String> {
        match self {
            Self::Void => write!(output, "void").unwrap(),
            Self::Integer(bits) | Self::SignExtInteger(bits) | Self::ZeroExtInteger(bits)
                if *bits > 0 =>
            {
                write!(output, "i{bits}").unwrap()
            }
            Self::Integer(_) | Self::SignExtInteger(_) | Self::ZeroExtInteger(_) => {
                return Err("device-extern integer width must be non-zero".to_string());
            }
            Self::Float16 => write!(output, "half").unwrap(),
            Self::Float32 => write!(output, "float").unwrap(),
            Self::Float64 => write!(output, "double").unwrap(),
            Self::Pointer {
                pointee,
                address_space,
            } => {
                if matches!(pointee.as_ref(), Self::Void) {
                    return Err(
                        "device-extern pointer cannot have LLVM `void` as its pointee; use i8"
                            .to_string(),
                    );
                }
                if legacy_typed_pointers {
                    pointee.write_llvm(output, true)?;
                    if *address_space != 0 {
                        write!(output, " addrspace({address_space})").unwrap();
                    }
                    write!(output, "*").unwrap();
                } else if *address_space == 0 {
                    write!(output, "ptr").unwrap();
                } else {
                    write!(output, "ptr addrspace({address_space})").unwrap();
                }
            }
            Self::Array { element, len } => {
                if matches!(element.as_ref(), Self::Void) {
                    return Err("device-extern array element cannot be `void`".to_string());
                }
                write!(output, "[{len} x ").unwrap();
                element.write_llvm(output, legacy_typed_pointers)?;
                write!(output, "]").unwrap();
            }
        }
        Ok(())
    }

    /// Write the LLVM IR type string with any required ABI attribute suffix,
    /// for PARAMETER position only.
    ///
    /// For `SignExtInteger(8)` this writes `i8 signext`, for
    /// `ZeroExtInteger(16)` this writes `i16 zeroext`. Plain types write
    /// just the type.
    ///
    /// LLVM's grammar places parameter attributes *after* the type inside the
    /// parameter list (`declare void @f(i8 signext %a)`) but *before* the
    /// type in return position (`declare signext i8 @f()`), so return
    /// emitters must use [`Self::ext_attr`] + [`Self::write_llvm`] instead.
    pub(crate) fn write_llvm_with_attr(
        &self,
        output: &mut String,
        legacy_typed_pointers: bool,
    ) -> Result<(), String> {
        match self {
            Self::SignExtInteger(bits) if *bits > 0 => {
                write!(output, "i{bits} signext").unwrap();
            }
            Self::ZeroExtInteger(bits) if *bits > 0 => {
                write!(output, "i{bits} zeroext").unwrap();
            }
            _ => self.write_llvm(output, legacy_typed_pointers)?,
        }
        Ok(())
    }

    pub(crate) fn llvm_string(&self, legacy_typed_pointers: bool) -> Result<String, String> {
        let mut output = String::new();
        self.write_llvm(&mut output, legacy_typed_pointers)?;
        Ok(output)
    }
}

/// An external device function declaration (for linking with external LTOIR).
///
/// These declarations are emitted as LLVM `declare` statements and resolved
/// at link time by nvJitLink when linking with external LTOIR (e.g., CCCL).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceExternDecl {
    /// The export name (e.g., "cub_block_reduce_sum").
    pub export_name: String,

    /// Function parameter types, including pointer pointees and address spaces.
    pub param_types: Vec<DeviceExternType>,

    /// Return type.
    pub return_type: DeviceExternType,

    /// NVVM attributes for this function.
    pub attrs: DeviceExternAttrs,
}

/// NVVM attributes for device extern declarations.
///
/// NOTE: These attributes are currently **not emitted** to the LLVM IR output.
/// When linking LTOIR via nvJitLink, the external library's LTOIR already contains
/// proper attributes (convergent, nounwind, memory, etc.) on the function DEFINITIONS.
/// nvJitLink uses the definition's attributes during LTO, making attributes on
/// declarations redundant.
///
/// This struct is retained for potential future use or for debugging/inspection.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct DeviceExternAttrs {
    /// Function is convergent (all threads must execute together).
    pub is_convergent: bool,

    /// Function is pure (no side effects). Maps to LLVM `readnone`.
    pub is_pure: bool,

    /// Function is read-only (only reads memory). Maps to LLVM `readonly`.
    pub is_readonly: bool,
}

/// Trait for types that can be converted to [`DeviceExternDecl`].
///
/// This allows mir-importer to pass its own DeviceExternDecl type
/// without llvm-export depending on mir-importer.
pub trait AsDeviceExtern {
    fn as_device_extern(&self) -> DeviceExternDecl;
}

impl AsDeviceExtern for DeviceExternDecl {
    fn as_device_extern(&self) -> DeviceExternDecl {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signext_integer_keeps_narrow_type_with_param_suffix_attr() {
        let ty = DeviceExternType::SignExtInteger(8);
        assert_eq!(ty.llvm_string(false).unwrap(), "i8");
        assert_eq!(ty.ext_attr(), Some("signext"));

        let mut out = String::new();
        ty.write_llvm_with_attr(&mut out, false).unwrap();
        assert_eq!(out, "i8 signext");
    }

    #[test]
    fn zeroext_integer_keeps_narrow_type_with_param_suffix_attr() {
        let ty = DeviceExternType::ZeroExtInteger(16);
        assert_eq!(ty.llvm_string(false).unwrap(), "i16");
        assert_eq!(ty.ext_attr(), Some("zeroext"));

        let mut out = String::new();
        ty.write_llvm_with_attr(&mut out, false).unwrap();
        assert_eq!(out, "i16 zeroext");
    }

    #[test]
    fn bool_is_zeroext_width_one() {
        let ty = DeviceExternType::ZeroExtInteger(1);
        assert_eq!(ty.llvm_string(false).unwrap(), "i1");

        let mut out = String::new();
        ty.write_llvm_with_attr(&mut out, false).unwrap();
        assert_eq!(out, "i1 zeroext");
    }

    #[test]
    fn plain_integer_write_llvm_with_attr_has_no_attr() {
        let ty = DeviceExternType::Integer(32);
        assert_eq!(ty.ext_attr(), None);
        let mut out = String::new();
        ty.write_llvm_with_attr(&mut out, false).unwrap();
        assert_eq!(out, "i32");
    }

    #[test]
    fn float16_write_llvm_with_attr() {
        let ty = DeviceExternType::Float16;
        assert_eq!(ty.ext_attr(), None);
        let mut out = String::new();
        ty.write_llvm_with_attr(&mut out, false).unwrap();
        assert_eq!(out, "half");
    }

    #[test]
    fn integer_width_returns_bits_for_all_integer_variants() {
        assert_eq!(DeviceExternType::Integer(64).integer_width(), Some(64));
        assert_eq!(DeviceExternType::SignExtInteger(8).integer_width(), Some(8));
        assert_eq!(
            DeviceExternType::ZeroExtInteger(16).integer_width(),
            Some(16)
        );
        assert_eq!(DeviceExternType::Float32.integer_width(), None);
    }

    #[test]
    fn signext_and_zeroext_are_distinct() {
        let s = DeviceExternType::SignExtInteger(8);
        let z = DeviceExternType::ZeroExtInteger(8);
        let p = DeviceExternType::Integer(8);
        assert_ne!(s, z);
        assert_ne!(s, p);
        assert_ne!(z, p);
    }
}
