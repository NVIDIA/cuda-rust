/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Parsing of non-semantic CUDA compiler resource diagnostics.
//!
//! Standalone `ptxas` emits resource usage directly, while nvJitLink can
//! forward the same output through its InfoLog. This module converts that
//! human-readable output into a stable internal representation without making
//! the raw text part of artifact provenance.

/// Per-kernel resource usage reported by ptxas.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelResourceUsage {
    /// Kernel entry name as emitted in PTX/cubin.
    pub kernel: String,
    /// Register count reported by ptxas, when present.
    pub registers: Option<u32>,
    /// Per-thread local-memory stack frame in bytes. Nonzero when locals are
    /// demoted to local memory; distinct from register-spill traffic.
    pub stack_frame_bytes: u64,
    /// Bytes written to local memory because of register spills.
    pub spill_store_bytes: u64,
    /// Bytes read from local memory because of register spills.
    pub spill_load_bytes: u64,
}

impl KernelResourceUsage {
    /// Whether ptxas reported any register spill traffic for this kernel.
    pub fn has_spills(&self) -> bool {
        self.spill_store_bytes != 0 || self.spill_load_bytes != 0
    }
}

/// Parse ptxas resource statistics from direct or nvJitLink-forwarded output.
///
/// The parser intentionally keys off stable ptxas phrases rather than exact
/// whitespace or the complete line prefix. CUDA versions vary the amount of
/// padding around `ptxas info`, while the resource phrases themselves remain
/// stable.
pub(crate) fn parse_ptxas_resource_usage(log: &str) -> Vec<KernelResourceUsage> {
    let mut usage = Vec::new();
    let mut current_kernel: Option<String> = None;

    for line in log.lines() {
        if let Some(kernel) = kernel_name_from_line(line) {
            ensure_kernel(&mut usage, &kernel);
            current_kernel = Some(kernel);
        }

        let Some(kernel) = current_kernel.as_deref() else {
            continue;
        };
        let entry = ensure_kernel(&mut usage, kernel);

        if let Some(registers) =
            number_before(line, " registers").and_then(|value| u32::try_from(value).ok())
            && line.contains("Used ")
        {
            entry.registers = Some(entry.registers.map_or(registers, |old| old.max(registers)));
        }

        if let Some(bytes) = number_before(line, " bytes stack frame") {
            entry.stack_frame_bytes = entry.stack_frame_bytes.max(bytes);
        }
        if let Some(bytes) = number_before(line, " bytes spill stores") {
            entry.spill_store_bytes = entry.spill_store_bytes.max(bytes);
        }
        if let Some(bytes) = number_before(line, " bytes spill loads") {
            entry.spill_load_bytes = entry.spill_load_bytes.max(bytes);
        }
    }

    usage
}

fn kernel_name_from_line(line: &str) -> Option<String> {
    if let Some(rest) = line
        .split_once("Compiling entry function '")
        .map(|(_, rest)| rest)
        && let Some((kernel, _)) = rest.split_once('\'')
        && !kernel.is_empty()
    {
        return Some(kernel.to_string());
    }

    if let Some((_, rest)) = line.split_once("Function properties for ") {
        // Some log shapes terminate the line with a colon after the (possibly
        // quoted) name: "Function properties for '_Z3fooPi':".
        let kernel = rest.trim().trim_end_matches(':').trim_matches('\'');
        if !kernel.is_empty() {
            return Some(kernel.to_string());
        }
    }

    None
}

fn ensure_kernel<'a>(
    usage: &'a mut Vec<KernelResourceUsage>,
    kernel: &str,
) -> &'a mut KernelResourceUsage {
    if let Some(index) = usage.iter().position(|entry| entry.kernel == kernel) {
        return &mut usage[index];
    }

    usage.push(KernelResourceUsage {
        kernel: kernel.to_string(),
        registers: None,
        stack_frame_bytes: 0,
        spill_store_bytes: 0,
        spill_load_bytes: 0,
    });
    usage.last_mut().expect("entry was just pushed")
}

fn number_before(line: &str, marker: &str) -> Option<u64> {
    let marker_start = line.find(marker)?;
    line[..marker_start].split_whitespace().last()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_zero_and_nonzero_spills_for_multiple_kernels() {
        let log = r#"
ptxas info    : Compiling entry function 'kernel_a' for 'sm_90a'
ptxas info    : Function properties for kernel_a
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 128 registers, used 0 barriers, 392 bytes cmem[0]
ptxas info    : Compiling entry function 'kernel_b' for 'sm_90a'
ptxas info    : Function properties for kernel_b
    32 bytes stack frame, 24 bytes spill stores, 16 bytes spill loads
ptxas info    : Used 127 registers, used 0 barriers, 392 bytes cmem[0]
"#;

        assert_eq!(
            parse_ptxas_resource_usage(log),
            vec![
                KernelResourceUsage {
                    kernel: "kernel_a".to_string(),
                    registers: Some(128),
                    stack_frame_bytes: 0,
                    spill_store_bytes: 0,
                    spill_load_bytes: 0,
                },
                KernelResourceUsage {
                    kernel: "kernel_b".to_string(),
                    registers: Some(127),
                    stack_frame_bytes: 32,
                    spill_store_bytes: 24,
                    spill_load_bytes: 16,
                },
            ]
        );
    }

    #[test]
    fn repeated_kernel_sections_merge_conservatively() {
        let log = r#"
ptxas info : Compiling entry function 'kernel' for 'sm_90'
ptxas info : 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 64 registers
ptxas info : Function properties for kernel
ptxas info : 8 bytes spill stores, 4 bytes spill loads
ptxas info : Used 72 registers
"#;

        assert_eq!(
            parse_ptxas_resource_usage(log),
            vec![KernelResourceUsage {
                kernel: "kernel".to_string(),
                registers: Some(72),
                stack_frame_bytes: 0,
                spill_store_bytes: 8,
                spill_load_bytes: 4,
            }]
        );
    }

    #[test]
    fn quoted_function_properties_name_with_trailing_colon_is_unwrapped() {
        let log = r#"
ptxas info    : Function properties for '_Z8kernel_bPf':
    16 bytes stack frame, 8 bytes spill stores, 4 bytes spill loads
"#;

        assert_eq!(
            parse_ptxas_resource_usage(log),
            vec![KernelResourceUsage {
                kernel: "_Z8kernel_bPf".to_string(),
                registers: None,
                stack_frame_bytes: 16,
                spill_store_bytes: 8,
                spill_load_bytes: 4,
            }]
        );
    }

    #[test]
    fn unrelated_info_log_text_is_ignored() {
        assert!(parse_ptxas_resource_usage("nvJitLink: timing only\n").is_empty());
    }

    #[test]
    fn has_spills_checks_both_directions_and_ignores_stack_frame() {
        let mut usage = KernelResourceUsage {
            kernel: "kernel".to_string(),
            registers: None,
            stack_frame_bytes: 0,
            spill_store_bytes: 0,
            spill_load_bytes: 0,
        };
        assert!(!usage.has_spills());
        usage.stack_frame_bytes = 32;
        assert!(!usage.has_spills());
        usage.spill_load_bytes = 4;
        assert!(usage.has_spills());
    }
}
