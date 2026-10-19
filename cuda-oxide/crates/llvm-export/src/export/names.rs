/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Value names, block labels, and symbol normalization.
//!
//! Device-symbol detection and base-name extraction route through
//! `reserved-oxide-symbols`, the workspace-internal source of truth for the
//! `cuda_oxide_*` namespace.
//!
//! Note on FQDN forms: MIR import converts `::` to `__`, so a fully-qualified
//! device symbol can appear as `mycrate__cuda_oxide_device_<hash>_foo`. Because
//! the helpers in `reserved-oxide-symbols` use substring matching (not
//! `starts_with`), they handle both bare and FQDN forms uniformly — no separate
//! `FQDN_DEVICE_PREFIX` constant is needed.

use reserved_oxide_symbols::{device_base_name, is_device_extern_symbol, is_device_symbol};

/// Convert a Pliron LLVM intrinsic identifier to its exact LLVM symbol.
///
/// Ordinary identifiers use underscores for dots. Identifiers starting with
/// `llvm__` use `_d` for dots and `_u` for literal underscores.
pub(super) fn decode_intrinsic_identifier(name: &str) -> String {
    debug_assert!(name.starts_with("llvm_"));
    let Some(encoded) = name.strip_prefix("llvm__") else {
        return name.replace('_', ".");
    };

    let mut output = String::with_capacity(name.len());
    output.push_str("llvm.");
    let mut chars = encoded.chars();
    while let Some(ch) = chars.next() {
        if ch != '_' {
            output.push(ch);
        } else {
            match chars.next() {
                Some('d') => output.push('.'),
                Some('u') => output.push('_'),
                Some(other) => {
                    output.push('_');
                    output.push(other);
                }
                None => output.push('_'),
            }
        }
    }
    output
}

/// Return `(floating-point width, address space)` for the six legacy NVVM
/// floating-point atomic-add intrinsics whose pointer pointee is part of the
/// LLVM 7 ABI.
pub(super) fn legacy_nvvm_atomic_add_signature(name: &str) -> Option<(u32, u32)> {
    match name {
        "llvm.nvvm.atomic.load.add.f32.p0f32" => Some((32, 0)),
        "llvm.nvvm.atomic.load.add.f32.p1f32" => Some((32, 1)),
        "llvm.nvvm.atomic.load.add.f32.p3f32" => Some((32, 3)),
        "llvm.nvvm.atomic.load.add.f64.p0f64" => Some((64, 0)),
        "llvm.nvvm.atomic.load.add.f64.p1f64" => Some((64, 1)),
        "llvm.nvvm.atomic.load.add.f64.p3f64" => Some((64, 3)),
        _ => None,
    }
}

/// Returns true if `name` is a device function (definition, not extern).
pub(super) fn has_device_prefix(name: &str) -> bool {
    is_device_symbol(name)
}

/// Strip the device-function prefix from `name` if present.
///
/// The reserved prefix is needed internally for MIR-level detection but
/// should not leak into the final LLVM IR / PTX / LTOIR output. Returns
/// `name` unchanged for non-device symbols and for device-extern declarations
/// (those keep their original-name `link_name` attribute).
pub(super) fn strip_device_prefix(name: &str) -> String {
    if is_device_extern_symbol(name) {
        return name.to_string();
    }
    device_base_name(name)
        .map(str::to_string)
        .unwrap_or_else(|| name.to_string())
}

#[cfg(test)]
mod tests {
    use super::{decode_intrinsic_identifier, legacy_nvvm_atomic_add_signature};

    #[test]
    fn intrinsic_identifier_decoding_preserves_dots_and_literal_underscores() {
        assert_eq!(
            decode_intrinsic_identifier("llvm_nvvm_read_ptx_sreg_tid_x"),
            "llvm.nvvm.read.ptx.sreg.tid.x"
        );
        assert_eq!(
            decode_intrinsic_identifier("llvm__nvvm_dwgmma_dcommit_ugroup_dsync_daligned"),
            "llvm.nvvm.wgmma.commit_group.sync.aligned"
        );
        assert_eq!(
            decode_intrinsic_identifier(
                "llvm__nvvm_dldmatrix_dsync_daligned_dm16n16_dx1_dtrans_db8x16_db4x16_up64_dp3"
            ),
            "llvm.nvvm.ldmatrix.sync.aligned.m16n16.x1.trans.b8x16.b4x16_p64.p3"
        );
    }

    #[test]
    fn legacy_atomic_add_signature_matches_only_the_documented_six() {
        for width in [32, 64] {
            for address_space in [0, 1, 3] {
                let name = format!("llvm.nvvm.atomic.load.add.f{width}.p{address_space}f{width}");
                assert_eq!(
                    legacy_nvvm_atomic_add_signature(&name),
                    Some((width, address_space))
                );
            }
        }
        for unsupported in [
            "llvm.nvvm.atomic.load.add.f16.p1f16",
            "llvm.nvvm.atomic.load.add.f32.p2f32",
            "llvm.nvvm.atomic.load.add.f32.p1f64",
            "llvm.nvvm.atomic.load.max.f32.p1f32",
        ] {
            assert_eq!(legacy_nvvm_atomic_add_signature(unsupported), None);
        }
    }
}
