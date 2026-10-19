/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared CUDA target parsing and recorded target-to-PTX policy.
//!
//! This crate is the one owner of the target vocabulary: parse a target
//! string once at the boundary, then pass the typed value around.
//!
//! ```text
//! before:  "sm_90a" --> ptx path parses --> nvvm path parses --> probe parses
//! after:   "sm_90a" --> CudaArch { capability: 90, suffix: Some('a') }
//!                       (one parse; every later check takes &CudaArch)
//! ```
//!
//! The same idea covers PTX ISA spellings: [`PtxSpelling`] can only be
//! built from the supported set, so holding one is the membership proof.
//!
//! The floors in [`RECORDED_PTX_FLOORS`] describe the defaults emitted by the
//! pinned LLVM 23 NVPTX backend. They are not backend-independent CUDA facts;
//! in particular, LLVM 21 does not accept every target recorded here.

use std::fmt;
use std::str::FromStr;

/// A validated CUDA compute capability, independent of its textual prefix.
///
/// libNVVM takes `compute_XX`, while cubin-producing tools take `sm_XX`.
/// Keeping one parsed value prevents those consumers from accidentally
/// targeting different devices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CudaArch {
    capability: u32,
    suffix: Option<char>,
}

impl CudaArch {
    /// Construct a CUDA architecture from its numeric and suffix components.
    ///
    /// This enforces the same capability-width and suffix grammar as
    /// [`FromStr`]: capabilities have at least two digits, and the only
    /// architecture-family suffixes are `a` and `f`.
    pub fn new(capability: u32, suffix: Option<char>) -> Result<Self, CudaArchParseError> {
        let target = render_parts("sm_", capability, suffix);
        if capability < 10 {
            return Err(CudaArchParseError::new(
                &target,
                "compute capability must contain at least two digits",
            ));
        }
        if !matches!(suffix, None | Some('a' | 'f')) {
            return Err(CudaArchParseError::new(
                &target,
                "the only supported architecture suffixes are `a` and `f`",
            ));
        }
        Ok(Self { capability, suffix })
    }

    /// Numeric CUDA capability (`86`, `90`, `100`, `120`, ...).
    pub fn capability(&self) -> u32 {
        self.capability
    }

    /// Optional architecture-family suffix (`a` or `f`).
    ///
    /// Targets such as `sm_90a` enable architecture-specific instructions and
    /// cannot be forwarded to a different compute capability.
    pub fn suffix(&self) -> Option<char> {
        self.suffix
    }

    /// Whether libNVVM selects its legacy LLVM 7 input dialect.
    pub fn uses_legacy_llvm(&self) -> bool {
        self.capability < 100
    }

    /// Render the target for cubin-producing tools such as nvJitLink.
    pub fn sm(&self) -> String {
        self.render("sm_")
    }

    /// Render the target for libNVVM.
    pub fn compute(&self) -> String {
        self.render("compute_")
    }
    fn render(&self, prefix: &str) -> String {
        render_parts(prefix, self.capability, self.suffix)
    }
}

/// A device architecture spelled `sm_<capability>` with an optional `a` suffix.
///
/// Unlike [`CudaArch`], this type cannot be constructed from a virtual
/// `compute_` architecture or an `f` family target, which names a compilation
/// compatibility set rather than the detected GPU. Grammar validation stays
/// in `CudaArch`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeviceArch(CudaArch);

impl FromStr for DeviceArch {
    type Err = CudaArchParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if !value.starts_with("sm_") {
            return Err(CudaArchParseError::new(
                value,
                "device architecture must use the `sm_` prefix",
            ));
        }
        let arch: CudaArch = value.parse()?;
        if arch.suffix() == Some('f') {
            return Err(CudaArchParseError::new(
                value,
                "a device architecture cannot name an `f` compilation family",
            ));
        }
        Ok(Self(arch))
    }
}

impl From<DeviceArch> for CudaArch {
    fn from(value: DeviceArch) -> Self {
        value.0
    }
}

impl AsRef<CudaArch> for DeviceArch {
    fn as_ref(&self) -> &CudaArch {
        &self.0
    }
}

fn render_parts(prefix: &str, capability: u32, suffix: Option<char>) -> String {
    match suffix {
        Some(suffix) => format!("{prefix}{capability}{suffix}"),
        None => format!("{prefix}{capability}"),
    }
}

impl FromStr for CudaArch {
    type Err = CudaArchParseError;
    fn from_str(target: &str) -> Result<Self, Self::Err> {
        let rest = target
            .strip_prefix("sm_")
            .or_else(|| target.strip_prefix("compute_"))
            .ok_or_else(|| CudaArchParseError::new(target, "expected `sm_XX` or `compute_XX`"))?;
        let digit_count = rest.chars().take_while(|c| c.is_ascii_digit()).count();
        let (digits, suffix_text) = rest.split_at(digit_count);
        let suffix = match suffix_text {
            "" => None,
            "a" => Some('a'),
            "f" => Some('f'),
            _ => {
                return Err(CudaArchParseError::new(
                    target,
                    "the only supported architecture suffixes are `a` and `f`",
                ));
            }
        };
        let capability = digits.parse::<u32>().map_err(|_| {
            CudaArchParseError::new(target, "compute capability is not a valid integer")
        })?;
        Self::new(capability, suffix)
    }
}

impl fmt::Display for CudaArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.sm())
    }
}

/// A malformed CUDA architecture string.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaArchParseError {
    target: String,
    reason: &'static str,
}
impl CudaArchParseError {
    fn new(target: &str, reason: &'static str) -> Self {
        Self {
            target: target.to_string(),
            reason,
        }
    }
}
impl fmt::Display for CudaArchParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid CUDA target `{}`: {}", self.target, self.reason)
    }
}
impl std::error::Error for CudaArchParseError {}

/// One exact CUDA target and a recorded PTX ISA floor.
///
/// The suffix is part of the key; consumers must not infer fallback entries
/// for other suffixes. The containing table defines whether the floor is an
/// ISA introduction or a pinned LLVM backend default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetPtxFloor {
    /// Numeric CUDA compute capability.
    pub capability: u32,
    /// Exact optional architecture-family suffix.
    pub suffix: Option<char>,
    /// PTX ISA encoded as `major * 10 + minor`.
    pub floor: u16,
}

/// PTX ISA introductions for the hardware floors used by the intrinsic catalog.
///
/// This is a lower bound on an authored instruction requirement, not an LLVM
/// default or a derived instruction floor. Only targets used by top-level
/// `minimum_sm` or native hardware floors are recorded; backend evidence
/// targets and architecture/family target matrices are outside this check.
///
/// Entries before `sm_75` are pinned from the [PTX ISA release notes]. Entries
/// from `sm_75` onward were measured with CUDA 13.2 `ptxas` by assembling an
/// empty kernel at successive `.version` spellings.
///
/// [PTX ISA release notes]: https://docs.nvidia.com/cuda/parallel-thread-execution/#release-notes
pub const ARCH_INTRODUCTION_PTX: &[TargetPtxFloor] = &[
    TargetPtxFloor {
        capability: 11,
        suffix: None,
        floor: 10,
    },
    TargetPtxFloor {
        capability: 20,
        suffix: None,
        floor: 20,
    },
    TargetPtxFloor {
        capability: 30,
        suffix: None,
        floor: 30,
    },
    TargetPtxFloor {
        capability: 53,
        suffix: None,
        floor: 42,
    },
    TargetPtxFloor {
        capability: 60,
        suffix: None,
        floor: 50,
    },
    TargetPtxFloor {
        capability: 61,
        suffix: None,
        floor: 50,
    },
    TargetPtxFloor {
        capability: 70,
        suffix: None,
        floor: 60,
    },
    TargetPtxFloor {
        capability: 75,
        suffix: None,
        floor: 63,
    },
    TargetPtxFloor {
        capability: 80,
        suffix: None,
        floor: 70,
    },
    TargetPtxFloor {
        capability: 86,
        suffix: None,
        floor: 71,
    },
    TargetPtxFloor {
        capability: 89,
        suffix: None,
        floor: 78,
    },
    TargetPtxFloor {
        capability: 90,
        suffix: None,
        floor: 78,
    },
    TargetPtxFloor {
        capability: 100,
        suffix: None,
        floor: 86,
    },
];

/// Return the recorded introduction version for an exact hardware-floor target.
///
/// An unrecorded spelling is an error; no nearby target or backend default is
/// substituted.
pub fn arch_introduced_ptx(arch: &CudaArch) -> Result<u16, UnsupportedTargetError> {
    ARCH_INTRODUCTION_PTX
        .iter()
        .find(|entry| entry.capability == arch.capability && entry.suffix == arch.suffix)
        .map(|entry| entry.floor)
        .ok_or_else(|| UnsupportedTargetError {
            target: arch.to_string(),
        })
}

/// Exact target floors recorded from the pinned LLVM 23 NVPTX backend.
///
/// These entries describe backend defaults, not backend-independent CUDA
/// facts. There are no wildcard or suffix-fallback entries. Entry order is the
/// target-selection preference: base targets first, then `a` and `f` families.
pub const RECORDED_PTX_FLOORS: &[TargetPtxFloor] = &[
    TargetPtxFloor {
        capability: 70,
        suffix: None,
        floor: 60,
    },
    TargetPtxFloor {
        capability: 72,
        suffix: None,
        floor: 61,
    },
    TargetPtxFloor {
        capability: 75,
        suffix: None,
        floor: 63,
    },
    TargetPtxFloor {
        capability: 80,
        suffix: None,
        floor: 70,
    },
    TargetPtxFloor {
        capability: 86,
        suffix: None,
        floor: 71,
    },
    TargetPtxFloor {
        capability: 87,
        suffix: None,
        floor: 74,
    },
    TargetPtxFloor {
        capability: 88,
        suffix: None,
        floor: 90,
    },
    TargetPtxFloor {
        capability: 89,
        suffix: None,
        floor: 78,
    },
    TargetPtxFloor {
        capability: 90,
        suffix: None,
        floor: 78,
    },
    TargetPtxFloor {
        capability: 100,
        suffix: None,
        floor: 86,
    },
    TargetPtxFloor {
        capability: 101,
        suffix: None,
        floor: 86,
    },
    TargetPtxFloor {
        capability: 103,
        suffix: None,
        floor: 88,
    },
    TargetPtxFloor {
        capability: 110,
        suffix: None,
        floor: 90,
    },
    TargetPtxFloor {
        capability: 120,
        suffix: None,
        floor: 87,
    },
    TargetPtxFloor {
        capability: 121,
        suffix: None,
        floor: 88,
    },
    TargetPtxFloor {
        capability: 90,
        suffix: Some('a'),
        floor: 80,
    },
    TargetPtxFloor {
        capability: 100,
        suffix: Some('a'),
        floor: 86,
    },
    TargetPtxFloor {
        capability: 101,
        suffix: Some('a'),
        floor: 86,
    },
    TargetPtxFloor {
        capability: 103,
        suffix: Some('a'),
        floor: 88,
    },
    TargetPtxFloor {
        capability: 110,
        suffix: Some('a'),
        floor: 90,
    },
    TargetPtxFloor {
        capability: 120,
        suffix: Some('a'),
        floor: 87,
    },
    TargetPtxFloor {
        capability: 121,
        suffix: Some('a'),
        floor: 88,
    },
    TargetPtxFloor {
        capability: 100,
        suffix: Some('f'),
        floor: 88,
    },
    TargetPtxFloor {
        capability: 101,
        suffix: Some('f'),
        floor: 88,
    },
    TargetPtxFloor {
        capability: 103,
        suffix: Some('f'),
        floor: 88,
    },
    TargetPtxFloor {
        capability: 110,
        suffix: Some('f'),
        floor: 90,
    },
    TargetPtxFloor {
        capability: 120,
        suffix: Some('f'),
        floor: 88,
    },
    TargetPtxFloor {
        capability: 121,
        suffix: Some('f'),
        floor: 88,
    },
];

/// A CUDA target without an exact recorded LLVM 23 PTX floor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnsupportedTargetError {
    target: String,
}
impl fmt::Display for UnsupportedTargetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CUDA target `{}` has no recorded PTX ISA floor",
            self.target
        )
    }
}
impl std::error::Error for UnsupportedTargetError {}

/// Return the pinned LLVM 23 default PTX floor for an exact target.
///
/// Suffixed targets require their own entry; this lookup never falls back to
/// the unsuffixed capability or another architecture-family suffix.
pub fn recorded_ptx_floor(arch: &CudaArch) -> Result<u16, UnsupportedTargetError> {
    RECORDED_PTX_FLOORS
        .iter()
        .find(|entry| entry.capability == arch.capability && entry.suffix == arch.suffix)
        .map(|entry| entry.floor)
        .ok_or_else(|| UnsupportedTargetError {
            target: arch.to_string(),
        })
}

/// Discrete PTX ISA feature spellings supported by the pinned LLVM backend.
pub const PTX_ISA_SPELLINGS: &[u16] = &[62, 65, 70, 71, 73, 78, 80, 86, 87, 88, 90];

/// A proof-carrying member of the supported PTX ISA spelling vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PtxSpelling(u16);

impl PtxSpelling {
    /// Construct a spelling exactly when it belongs to the supported vocabulary.
    pub const fn from_spelling(spelling: u16) -> Option<Self> {
        let mut index = 0;
        while index < PTX_ISA_SPELLINGS.len() {
            if PTX_ISA_SPELLINGS[index] == spelling {
                return Some(Self(spelling));
            }
            index += 1;
        }
        None
    }

    /// Return the smallest supported spelling at least `floor`.
    pub fn round_up(floor: u16) -> Option<Self> {
        PTX_ISA_SPELLINGS
            .iter()
            .copied()
            .find(|spelling| *spelling >= floor)
            .and_then(Self::from_spelling)
    }

    /// Return the encoded `major * 10 + minor` spelling.
    pub const fn get(self) -> u16 {
        self.0
    }

    /// Render this supported spelling as an LLVM `-mattr` feature.
    pub const fn feature(self) -> &'static str {
        match self.0 {
            62 => "+ptx62",
            65 => "+ptx65",
            70 => "+ptx70",
            71 => "+ptx71",
            73 => "+ptx73",
            78 => "+ptx78",
            80 => "+ptx80",
            86 => "+ptx86",
            87 => "+ptx87",
            88 => "+ptx88",
            90 => "+ptx90",
            _ => unreachable!(),
        }
    }

    /// Render this feature only when it is newer than a recorded target floor.
    pub fn feature_beyond_floor(self, recorded_floor: u16) -> Option<&'static str> {
        if self.get() <= recorded_floor {
            None
        } else {
            Some(self.feature())
        }
    }
}

/// Render one supported PTX ISA spelling as an LLVM `-mattr` feature.
pub fn spelling_feature(spelling: u16) -> Option<&'static str> {
    PtxSpelling::from_spelling(spelling).map(PtxSpelling::feature)
}

/// Return the smallest supported PTX feature spelling at least `floor`.
///
/// Returns `None` when the requested floor is newer than every supported
/// spelling.
pub fn spelling_at_least(floor: u16) -> Option<u16> {
    PtxSpelling::round_up(floor).map(PtxSpelling::get)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn device_arch_requires_sm_spelling_and_reuses_cuda_grammar() {
        for entry in RECORDED_PTX_FLOORS {
            let arch = CudaArch::new(entry.capability, entry.suffix).unwrap();
            let device = arch.sm().parse::<DeviceArch>();
            if entry.suffix == Some('f') {
                assert!(device.is_err());
            } else {
                assert_eq!(CudaArch::from(device.unwrap()), arch);
            }
            assert!(arch.compute().parse::<DeviceArch>().is_err());
        }
        for raw in [
            "compute_120",
            "sm_",
            "foo",
            "sm_05",
            "sm_9",
            "sm_90x",
            "sm_90aa",
            "sm_4294967296",
        ] {
            assert!(raw.parse::<DeviceArch>().is_err(), "{raw}");
        }
    }

    #[test]
    fn cuda_arch_parses_and_renders_api_specific_spellings() {
        for (input, capability, suffix, sm, compute, legacy) in [
            ("sm_75", 75, None, "sm_75", "compute_75", true),
            ("compute_90a", 90, Some('a'), "sm_90a", "compute_90a", true),
            ("sm_100f", 100, Some('f'), "sm_100f", "compute_100f", false),
            ("compute_120", 120, None, "sm_120", "compute_120", false),
        ] {
            let arch: CudaArch = input.parse().unwrap();
            assert_eq!((arch.capability(), arch.suffix()), (capability, suffix));
            assert_eq!(
                (arch.sm(), arch.compute()),
                (sm.to_string(), compute.to_string())
            );
            assert_eq!(arch.uses_legacy_llvm(), legacy);
        }
    }
    #[test]
    fn cuda_arch_rejects_ambiguous_or_malformed_targets() {
        for input in [
            "", "86", "sm_", "sm_9", "sm_90x", "sm_90aa", "SM_90", "gfx90a",
        ] {
            assert!(input.parse::<CudaArch>().is_err(), "{input}");
        }
    }

    /// Entry order in [`RECORDED_PTX_FLOORS`] is load-bearing: target
    /// selection walks the list front to back and takes the first
    /// satisfying candidate. So the list must stay in this order:
    ///
    /// ```text
    /// [ base targets, ascending ]  [ 'a' family, ascending ]  [ 'f' family, ascending ]
    ///   sm_70 .. sm_121             sm_90a .. sm_121a          sm_100f .. sm_121f
    ///   preferred first  ------------------------------------------>  last resort
    /// ```
    ///
    /// An entry inserted out of section order would silently reshuffle
    /// which target wins selection; this test turns that into a red build.
    #[test]
    fn recorded_floors_keep_selection_preference_order() {
        let section = |suffix: Option<char>| match suffix {
            None => 0,
            Some('a') => 1,
            Some('f') => 2,
            other => panic!("unknown suffix section {other:?}"),
        };
        for pair in RECORDED_PTX_FLOORS.windows(2) {
            let (prev, next) = (&pair[0], &pair[1]);
            let ordered =
                (section(prev.suffix), prev.capability) < (section(next.suffix), next.capability);
            assert!(
                ordered,
                "sm_{}{} must come before sm_{}{}",
                prev.capability,
                prev.suffix.map(String::from).unwrap_or_default(),
                next.capability,
                next.suffix.map(String::from).unwrap_or_default(),
            );
        }
    }

    #[test]
    fn construction_from_parts_and_text_agree() {
        for entry in RECORDED_PTX_FLOORS {
            let from_parts = CudaArch::new(entry.capability, entry.suffix).unwrap();
            let reparsed = from_parts.sm().parse::<CudaArch>();
            assert_eq!(reparsed, Ok(from_parts));
        }
        assert!(CudaArch::new(5, None).is_err());
        assert!("sm_5".parse::<CudaArch>().is_err());
        assert!(CudaArch::new(90, Some('x')).is_err());
        assert!("sm_90x".parse::<CudaArch>().is_err());
    }

    #[test]
    fn every_ptx_spelling_has_one_canonical_feature() {
        for raw in PTX_ISA_SPELLINGS {
            let expected = format!("+ptx{raw}");
            assert_eq!(spelling_feature(*raw), Some(expected.as_str()));
            let spelling = PtxSpelling::from_spelling(*raw).unwrap();
            assert_eq!(spelling.get(), *raw);
            assert_eq!(spelling.feature(), expected);
        }
        assert_eq!(spelling_feature(74), None);
        assert_eq!(PtxSpelling::from_spelling(74), None);
        assert_eq!(PtxSpelling::round_up(74).map(PtxSpelling::get), Some(78));
        assert_eq!(PtxSpelling::round_up(91), None);
    }

    #[test]
    fn feature_beyond_floor_only_renders_supported_newer_spellings() {
        for (spelling, floor, expected) in
            [(78, 73, Some("+ptx78")), (73, 73, None), (70, 73, None)]
        {
            assert_eq!(
                PtxSpelling::from_spelling(spelling)
                    .unwrap()
                    .feature_beyond_floor(floor),
                expected
            );
        }
    }

    #[cfg(unix)]
    mod backend {
        use super::*;
        use std::fs;
        use std::path::{Path, PathBuf};
        use std::process::{Command, Output};

        struct TestDir(PathBuf);
        impl TestDir {
            fn new() -> Self {
                let path = std::env::temp_dir().join(format!(
                    "cuda-target-spec-{}-{:?}",
                    std::process::id(),
                    std::thread::current().id()
                ));
                fs::create_dir_all(&path).unwrap();
                Self(path)
            }
        }
        impl Drop for TestDir {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }

        fn rust_toolchain_llc() -> PathBuf {
            let sysroot = Command::new("rustc")
                .args(["--print", "sysroot"])
                .output()
                .unwrap();
            assert!(sysroot.status.success(), "rustc --print sysroot failed");
            let verbose = Command::new("rustc").arg("-vV").output().unwrap();
            assert!(verbose.status.success(), "rustc -vV failed");
            let host = String::from_utf8_lossy(&verbose.stdout)
                .lines()
                .find_map(|line| line.strip_prefix("host: "))
                .expect("rustc -vV did not report a host")
                .to_owned();
            let path = PathBuf::from(String::from_utf8_lossy(&sysroot.stdout).trim())
                .join("lib/rustlib")
                .join(host)
                .join("bin/llc");
            assert!(
                path.is_file(),
                "rust toolchain has no llc at {}",
                path.display()
            );
            path
        }

        fn llvm_23() -> Option<PathBuf> {
            let llc = rust_toolchain_llc();
            let output = Command::new(&llc).arg("--version").output().unwrap();
            assert!(output.status.success(), "llc --version failed");
            let version = String::from_utf8_lossy(&output.stdout);
            let major = version
                .lines()
                .find_map(|line| {
                    line.trim()
                        .strip_prefix("LLVM version ")?
                        .split('.')
                        .next()?
                        .parse::<u32>()
                        .ok()
                })
                .expect("llc --version did not report an LLVM version");
            if major != 23 {
                eprintln!("skipping LLVM-derived PTX-floor test: expected LLVM 23, found {major}");
                return None;
            }
            Some(llc)
        }

        fn module(directory: &Path) -> PathBuf {
            let module = directory.join("probe.ll");
            fs::write(&module, "target triple = \"nvptx64-nvidia-cuda\"\n\ndefine void @probe() {\nentry:\n  ret void\n}\n").unwrap();
            module
        }

        fn lower(
            llc: &Path,
            module: &Path,
            target: &str,
            feature: Option<&str>,
            output: &Path,
        ) -> Output {
            let mut command = Command::new(llc);
            command
                .arg("-mtriple=nvptx64-nvidia-cuda")
                .arg(format!("-mcpu={target}"));
            if let Some(feature) = feature {
                command.arg(format!("-mattr={feature}"));
            }
            command
                .arg("-filetype=asm")
                .arg(module)
                .arg("-o")
                .arg(output)
                .output()
                .unwrap()
        }

        fn emitted_ptx_isa(path: &Path) -> u16 {
            let ptx = fs::read_to_string(path).unwrap();
            let version = ptx
                .lines()
                .find_map(|line| line.trim().strip_prefix(".version "))
                .expect("emitted PTX carries no .version");
            let (major, minor) = version.split_once('.').unwrap();
            major.parse::<u16>().unwrap() * 10 + minor.parse::<u16>().unwrap()
        }

        #[test]
        fn recorded_floors_match_llvm_23_defaults() {
            let Some(llc) = llvm_23() else { return };
            let directory = TestDir::new();
            let module = module(&directory.0);
            for entry in RECORDED_PTX_FLOORS {
                let target = match entry.suffix {
                    Some(s) => format!("sm_{}{s}", entry.capability),
                    None => format!("sm_{}", entry.capability),
                };
                let output = directory.0.join(format!("{target}.ptx"));
                let result = lower(&llc, &module, &target, None, &output);
                assert!(
                    result.status.success(),
                    "{target}: {}",
                    String::from_utf8_lossy(&result.stderr)
                );
                assert_eq!(emitted_ptx_isa(&output), entry.floor, "{target}");
            }
        }

        #[test]
        fn sm_90a_requires_ptx_80() {
            let Some(llc) = llvm_23() else { return };
            let directory = TestDir::new();
            let module = module(&directory.0);
            let output = directory.0.join("sm_90a.ptx");
            let pass = lower(&llc, &module, "sm_90a", Some("+ptx80"), &output);
            assert!(
                pass.status.success(),
                "{}",
                String::from_utf8_lossy(&pass.stderr)
            );
            assert_eq!(emitted_ptx_isa(&output), 80);
            let reject = lower(&llc, &module, "sm_90a", Some("+ptx78"), &output);
            assert!(!reject.status.success());
            assert!(
                String::from_utf8_lossy(&reject.stderr).contains("Minimum required PTX version")
            );
        }

        #[test]
        fn sm_103a_rejects_ptx_86_near_miss() {
            let Some(llc) = llvm_23() else { return };
            let directory = TestDir::new();
            let module = module(&directory.0);
            let output = directory.0.join("sm_103a.ptx");
            let reject = lower(&llc, &module, "sm_103a", Some("+ptx86"), &output);
            assert!(!reject.status.success());
            assert!(
                String::from_utf8_lossy(&reject.stderr).contains("Minimum required PTX version")
            );
            let pass = lower(&llc, &module, "sm_103a", Some("+ptx88"), &output);
            assert!(
                pass.status.success(),
                "{}",
                String::from_utf8_lossy(&pass.stderr)
            );
            assert_eq!(emitted_ptx_isa(&output), 88);
        }
    }
}
