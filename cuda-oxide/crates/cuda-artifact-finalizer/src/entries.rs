/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Launchable entry inventory of a finalized image.
//!
//! nvJitLink can succeed while dropping a module's kernels (an unresolved
//! device-runtime symbol, or a kernel the link-time optimizer found
//! unreachable), returning a well-formed but empty image. The finalizer
//! therefore checks that every kernel its caller expects is an entry point of
//! the output, and fails otherwise.

use crate::FinalizerError;
use crate::options::FinalizerOutput;

const SECTION_TYPE_SYMBOL_TABLE: u32 = 2;
const ELF64_SECTION_HEADER_LENGTH: usize = 64;
const ELF64_SYMBOL_LENGTH: usize = 24;
const SYMBOL_TYPE_FUNCTION: u8 = 2;
/// `st_other` flag CUDA sets on the symbol of a launchable kernel.
const STO_CUDA_ENTRY: u8 = 0x10;

/// Fail unless every name in `expected` is an entry point of `image`.
///
/// Every finalizer link applies this check. Callers that reuse a stored image
/// (a cache hit) apply it themselves. An empty `expected` checks nothing: the
/// module declares no kernels.
pub fn require_expected_kernels(
    image: &[u8],
    output: FinalizerOutput,
    expected: &[&str],
) -> Result<(), FinalizerError> {
    if expected.is_empty() {
        return Ok(());
    }
    let entries = match output {
        FinalizerOutput::Cubin => cubin_entry_names(image),
        FinalizerOutput::Ptx => ptx_entry_names(image),
    }
    .ok_or(FinalizerError::UnreadableEntryInventory { output })?;
    let missing = expected
        .iter()
        .filter(|name| !entries.iter().any(|entry| entry == *name))
        .map(|name| (*name).to_string())
        .collect::<Vec<_>>();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(FinalizerError::MissingKernels {
            output,
            missing,
            present: entries.len(),
        })
    }
}

/// First line of a `<module>.kernels` sidecar. The sidecar lists, one export
/// name per line, the kernels a module's finalized image must define; the
/// compiler writes it next to the NVVM IR so that finalizing from files checks
/// the same kernels as finalizing an embedded bundle.
pub const EXPECTED_KERNELS_SIDECAR_HEADER: &str = "cuda-oxide expected-kernels v1";

/// Text of a `<module>.kernels` sidecar listing `kernels`.
pub fn expected_kernels_sidecar_text<'a>(kernels: impl IntoIterator<Item = &'a str>) -> String {
    let mut text = format!("{EXPECTED_KERNELS_SIDECAR_HEADER}\n");
    for kernel in kernels {
        debug_assert!(!kernel.is_empty() && !kernel.contains(char::is_whitespace));
        text.push_str(kernel);
        text.push('\n');
    }
    text
}

/// Kernels listed by a `<module>.kernels` sidecar, or `None` when it has
/// another header or a malformed name.
pub fn parse_expected_kernels_sidecar(text: &str) -> Option<Vec<String>> {
    let mut lines = text.lines();
    if lines.next()? != EXPECTED_KERNELS_SIDECAR_HEADER {
        return None;
    }
    lines
        .map(|name| {
            (!name.is_empty() && !name.contains(char::is_whitespace)).then(|| name.to_string())
        })
        .collect()
}

/// Names of the entry-point function symbols of a CUDA ELF, or `None` when
/// its symbol table cannot be read. An image without a symbol table defines
/// no entries.
pub fn cubin_entry_names(bytes: &[u8]) -> Option<Vec<String>> {
    let section_offset = usize::try_from(read_u64(bytes, 40)?).ok()?;
    let section_count = usize::from(read_u16(bytes, 60)?);
    let section = |index: usize| -> Option<usize> {
        (index < section_count).then(|| section_offset + index * ELF64_SECTION_HEADER_LENGTH)
    };
    let mut names = Vec::new();
    for index in 0..section_count {
        let header = section(index)?;
        if read_u32(bytes, header + 4)? != SECTION_TYPE_SYMBOL_TABLE {
            continue;
        }
        let symbols = file_range(bytes, header)?;
        let strings = file_range(
            bytes,
            section(usize::try_from(read_u32(bytes, header + 40)?).ok()?)?,
        )?;
        let (symbols, rest) = symbols.as_chunks::<ELF64_SYMBOL_LENGTH>();
        if !rest.is_empty() {
            return None;
        }
        for symbol in symbols {
            let info = symbol[4];
            let other = symbol[5];
            if info & 0xf != SYMBOL_TYPE_FUNCTION || other & STO_CUDA_ENTRY == 0 {
                continue;
            }
            let name_offset = usize::try_from(read_u32(symbol, 0)?).ok()?;
            let name = strings.get(name_offset..)?;
            let end = name.iter().position(|&byte| byte == 0)?;
            names.push(std::str::from_utf8(&name[..end]).ok()?.to_string());
        }
    }
    Some(names)
}

/// Names of the `.entry` kernels of a PTX module, or `None` when it cannot
/// be parsed.
pub fn ptx_entry_names(bytes: &[u8]) -> Option<Vec<String>> {
    let text = std::str::from_utf8(bytes.strip_suffix(b"\0").unwrap_or(bytes)).ok()?;
    let document = ptx_parse::Document::parse(text).ok()?;
    Some(
        document
            .callables()
            .iter()
            .filter(|callable| callable.kind() == ptx_parse::CallableKind::Entry)
            .map(|callable| callable.name().to_string())
            .collect(),
    )
}

/// File bytes of the section whose header starts at `header`.
fn file_range(bytes: &[u8], header: usize) -> Option<&[u8]> {
    let offset = usize::try_from(read_u64(bytes, header + 24)?).ok()?;
    let length = usize::try_from(read_u64(bytes, header + 32)?).ok()?;
    bytes.get(offset..offset.checked_add(length)?)
}

fn read_u16(bytes: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        bytes.get(offset..offset + 2)?.try_into().ok()?,
    ))
}

fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        bytes.get(offset..offset + 8)?.try_into().ok()?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A CUDA ELF with a symbol table holding `symbols` as
    /// `(name, st_info, st_other)`, a string table, and nothing else.
    fn cubin_with_symbols(symbols: &[(&str, u8, u8)]) -> Vec<u8> {
        let mut strings = vec![0_u8];
        let mut table = vec![0_u8; ELF64_SYMBOL_LENGTH];
        for &(name, info, other) in symbols {
            let mut symbol = vec![0_u8; ELF64_SYMBOL_LENGTH];
            symbol[0..4].copy_from_slice(&(strings.len() as u32).to_le_bytes());
            symbol[4] = info;
            symbol[5] = other;
            table.extend_from_slice(&symbol);
            strings.extend_from_slice(name.as_bytes());
            strings.push(0);
        }
        let headers = 64;
        let symbols_offset = headers + 3 * ELF64_SECTION_HEADER_LENGTH;
        let strings_offset = symbols_offset + table.len();
        let mut bytes = vec![0_u8; strings_offset + strings.len()];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[6] = 1;
        bytes[16..18].copy_from_slice(&2_u16.to_le_bytes());
        bytes[18..20].copy_from_slice(&190_u16.to_le_bytes());
        bytes[20..24].copy_from_slice(&1_u32.to_le_bytes());
        bytes[40..48].copy_from_slice(&(headers as u64).to_le_bytes());
        bytes[52..54].copy_from_slice(&64_u16.to_le_bytes());
        bytes[58..60].copy_from_slice(&(ELF64_SECTION_HEADER_LENGTH as u16).to_le_bytes());
        bytes[60..62].copy_from_slice(&3_u16.to_le_bytes());
        let symtab = headers + ELF64_SECTION_HEADER_LENGTH;
        bytes[symtab + 4..symtab + 8].copy_from_slice(&SECTION_TYPE_SYMBOL_TABLE.to_le_bytes());
        bytes[symtab + 24..symtab + 32].copy_from_slice(&(symbols_offset as u64).to_le_bytes());
        bytes[symtab + 32..symtab + 40].copy_from_slice(&(table.len() as u64).to_le_bytes());
        bytes[symtab + 40..symtab + 44].copy_from_slice(&2_u32.to_le_bytes());
        let strtab = symtab + ELF64_SECTION_HEADER_LENGTH;
        bytes[strtab + 4..strtab + 8].copy_from_slice(&3_u32.to_le_bytes());
        bytes[strtab + 24..strtab + 32].copy_from_slice(&(strings_offset as u64).to_le_bytes());
        bytes[strtab + 32..strtab + 40].copy_from_slice(&(strings.len() as u64).to_le_bytes());
        bytes[symbols_offset..strings_offset].copy_from_slice(&table);
        bytes[strings_offset..].copy_from_slice(&strings);
        bytes
    }

    const GLOBAL_FUNCTION: u8 = 0x12;
    const LOCAL_FUNCTION: u8 = 0x02;

    #[test]
    fn cubin_inventory_lists_only_entry_flagged_functions() {
        let cubin = cubin_with_symbols(&[
            ("kernel", GLOBAL_FUNCTION, STO_CUDA_ENTRY),
            ("helper", LOCAL_FUNCTION, 0),
            ("device_global", GLOBAL_FUNCTION, 0),
        ]);
        assert_eq!(cubin_entry_names(&cubin).unwrap(), ["kernel"]);
    }

    #[test]
    fn a_dropped_kernel_fails_the_link_even_in_a_well_formed_image() {
        // The shape nvJitLink returns when it drops a module's kernels: a
        // valid ELF whose symbol table has no entries.
        let empty = cubin_with_symbols(&[]);
        assert!(crate::is_valid_cubin(&empty));
        assert!(matches!(
            require_expected_kernels(&empty, FinalizerOutput::Cubin, &["kernel"]),
            Err(FinalizerError::MissingKernels { ref missing, present: 0, .. })
                if missing == &["kernel"]
        ));

        let partial = cubin_with_symbols(&[("first", GLOBAL_FUNCTION, STO_CUDA_ENTRY)]);
        assert!(matches!(
            require_expected_kernels(&partial, FinalizerOutput::Cubin, &["first", "second"]),
            Err(FinalizerError::MissingKernels { ref missing, present: 1, .. })
                if missing == &["second"]
        ));
        assert!(require_expected_kernels(&partial, FinalizerOutput::Cubin, &["first"]).is_ok());
    }

    #[test]
    fn ptx_inventory_requires_entries_not_device_functions() {
        let ptx = b".version 8.0\n.target sm_80\n.address_size 64\n\
            .func helper()\n{\n\tret;\n}\n\
            .visible .entry kernel()\n{\n\tret;\n}\n";
        assert_eq!(ptx_entry_names(ptx).unwrap(), ["kernel"]);
        assert!(matches!(
            require_expected_kernels(ptx, FinalizerOutput::Ptx, &["helper"]),
            Err(FinalizerError::MissingKernels { ref missing, .. }) if missing == &["helper"]
        ));
    }

    #[test]
    fn the_kernels_sidecar_round_trips_and_rejects_other_text() {
        let text = expected_kernels_sidecar_text(["first", "second_TID_0123"]);
        assert_eq!(
            parse_expected_kernels_sidecar(&text).unwrap(),
            ["first", "second_TID_0123"]
        );
        assert_eq!(
            parse_expected_kernels_sidecar(&expected_kernels_sidecar_text([])).unwrap(),
            Vec::<String>::new()
        );
        assert!(parse_expected_kernels_sidecar("first\n").is_none());
        assert!(parse_expected_kernels_sidecar("").is_none());
        let blank = format!("{EXPECTED_KERNELS_SIDECAR_HEADER}\nfirst\n\n");
        assert!(parse_expected_kernels_sidecar(&blank).is_none());
    }

    #[test]
    fn an_unreadable_inventory_fails_closed_and_no_expectation_checks_nothing() {
        let not_text = b"\xff\xfe";
        assert!(matches!(
            require_expected_kernels(not_text, FinalizerOutput::Ptx, &["kernel"]),
            Err(FinalizerError::UnreadableEntryInventory { .. })
        ));
        assert!(matches!(
            require_expected_kernels(b"\x7fELF", FinalizerOutput::Cubin, &["kernel"]),
            Err(FinalizerError::UnreadableEntryInventory { .. })
        ));
        assert!(require_expected_kernels(not_text, FinalizerOutput::Ptx, &[]).is_ok());
    }
}
