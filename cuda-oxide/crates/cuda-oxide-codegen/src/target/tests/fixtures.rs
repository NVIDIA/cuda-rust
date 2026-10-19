/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::generated_intrinsic_targets::{
    GeneratedBackendRequirement, GeneratedHardwareAlternative, GeneratedHardwareTarget,
    GeneratedIntrinsicBackend, GeneratedIntrinsicTarget, GeneratedIntrinsicVariant,
    GeneratedPtxVersion, GeneratedTargetAlternative, GeneratedTargetContract,
    GeneratedTargetRequirement, GeneratedTargetSelectorBinding,
};

static EXACT_SM120A: &[GeneratedHardwareAlternative] =
    &[GeneratedHardwareAlternative::ExactArchitecture(120)];
pub(super) static PTX87_EXACT_SM120A: GeneratedIntrinsicTarget = GeneratedIntrinsicTarget {
    marker: "test:ptx87",
    id: "ptx87_exact_sm120a",
    abi_id: "test",
    dialect_op: "test.ptx87",
    variant: GeneratedIntrinsicVariant::Scalar,
    requirement: GeneratedTargetRequirement {
        minimum_ptx: GeneratedPtxVersion::from_encoded(87),
        hardware: GeneratedHardwareTarget::AnyOf(EXACT_SM120A),
    },
    backend_requirements: &[],
    selections: &[],
    llvm: None,
};
pub(super) static PTX88: GeneratedIntrinsicTarget = GeneratedIntrinsicTarget {
    marker: "test:ptx88",
    id: "ptx88",
    abi_id: "test",
    dialect_op: "test.ptx88",
    variant: GeneratedIntrinsicVariant::Scalar,
    requirement: GeneratedTargetRequirement {
        minimum_ptx: GeneratedPtxVersion::from_encoded(88),
        hardware: GeneratedHardwareTarget::All,
    },
    backend_requirements: &[],
    selections: &[],
    llvm: None,
};
pub(super) static PTX90: GeneratedIntrinsicTarget = GeneratedIntrinsicTarget {
    marker: "test:ptx90",
    id: "ptx90",
    abi_id: "test",
    dialect_op: "test.ptx90",
    variant: GeneratedIntrinsicVariant::Scalar,
    requirement: GeneratedTargetRequirement {
        minimum_ptx: GeneratedPtxVersion::from_encoded(90),
        hardware: GeneratedHardwareTarget::All,
    },
    backend_requirements: &[],
    selections: &[],
    llvm: None,
};
pub(super) static PTX91_FUTURE: GeneratedIntrinsicTarget = GeneratedIntrinsicTarget {
    marker: "test:ptx91",
    id: "ptx91_future",
    abi_id: "test",
    dialect_op: "test.ptx91",
    variant: GeneratedIntrinsicVariant::Scalar,
    requirement: GeneratedTargetRequirement {
        minimum_ptx: GeneratedPtxVersion::from_encoded(91),
        hardware: GeneratedHardwareTarget::All,
    },
    backend_requirements: &[],
    selections: &[],
    llvm: None,
};
static TCGEN_F16_SELECTORS: &[GeneratedTargetSelectorBinding] = &[GeneratedTargetSelectorBinding {
    name: "kind",
    value: "f16",
}];
static TCGEN_F16_TARGETS: &[GeneratedTargetAlternative] = &[
    GeneratedTargetAlternative {
        minimum_ptx: GeneratedPtxVersion::from_encoded(86),
        hardware: GeneratedHardwareAlternative::ExactArchitecture(100),
    },
    GeneratedTargetAlternative {
        minimum_ptx: GeneratedPtxVersion::from_encoded(86),
        hardware: GeneratedHardwareAlternative::ExactArchitecture(101),
    },
    GeneratedTargetAlternative {
        minimum_ptx: GeneratedPtxVersion::from_encoded(88),
        hardware: GeneratedHardwareAlternative::FamilyTarget(100),
    },
    GeneratedTargetAlternative {
        minimum_ptx: GeneratedPtxVersion::from_encoded(88),
        hardware: GeneratedHardwareAlternative::FamilyTarget(101),
    },
    GeneratedTargetAlternative {
        minimum_ptx: GeneratedPtxVersion::from_encoded(88),
        hardware: GeneratedHardwareAlternative::ExactArchitecture(103),
    },
    GeneratedTargetAlternative {
        minimum_ptx: GeneratedPtxVersion::from_encoded(88),
        hardware: GeneratedHardwareAlternative::FamilyTarget(103),
    },
    GeneratedTargetAlternative {
        minimum_ptx: GeneratedPtxVersion::from_encoded(90),
        hardware: GeneratedHardwareAlternative::ExactArchitecture(110),
    },
    GeneratedTargetAlternative {
        minimum_ptx: GeneratedPtxVersion::from_encoded(90),
        hardware: GeneratedHardwareAlternative::FamilyTarget(110),
    },
];
static TCGEN_F16_CONTRACTS: &[GeneratedTargetContract] = &[GeneratedTargetContract {
    selectors: TCGEN_F16_SELECTORS,
    alternatives: TCGEN_F16_TARGETS,
}];
pub(super) static TCGEN_F16: GeneratedIntrinsicTarget = GeneratedIntrinsicTarget {
    marker: "test:tcgen_f16",
    id: "tcgen_f16",
    abi_id: "test",
    dialect_op: "test.tcgen_f16",
    variant: GeneratedIntrinsicVariant::Scalar,
    requirement: GeneratedTargetRequirement {
        minimum_ptx: GeneratedPtxVersion::from_encoded(86),
        hardware: GeneratedHardwareTarget::TargetMatrix {
            contracts: TCGEN_F16_CONTRACTS,
        },
    },
    backend_requirements: &[],
    selections: &[],
    llvm: None,
};
static TCGEN_I8_SELECTORS: &[GeneratedTargetSelectorBinding] = &[GeneratedTargetSelectorBinding {
    name: "kind",
    value: "i8",
}];
static TCGEN_I8_TARGETS: &[GeneratedTargetAlternative] = &[
    GeneratedTargetAlternative {
        minimum_ptx: GeneratedPtxVersion::from_encoded(86),
        hardware: GeneratedHardwareAlternative::ExactArchitecture(100),
    },
    GeneratedTargetAlternative {
        minimum_ptx: GeneratedPtxVersion::from_encoded(86),
        hardware: GeneratedHardwareAlternative::ExactArchitecture(101),
    },
    GeneratedTargetAlternative {
        minimum_ptx: GeneratedPtxVersion::from_encoded(90),
        hardware: GeneratedHardwareAlternative::ExactArchitecture(110),
    },
];
static TCGEN_I8_LIBNVVM_TARGETS: &[GeneratedTargetAlternative] = &[
    GeneratedTargetAlternative {
        minimum_ptx: GeneratedPtxVersion::from_encoded(86),
        hardware: GeneratedHardwareAlternative::ExactArchitecture(100),
    },
    GeneratedTargetAlternative {
        minimum_ptx: GeneratedPtxVersion::from_encoded(90),
        hardware: GeneratedHardwareAlternative::ExactArchitecture(110),
    },
];
static TCGEN_I8_CONTRACTS: &[GeneratedTargetContract] = &[GeneratedTargetContract {
    selectors: TCGEN_I8_SELECTORS,
    alternatives: TCGEN_I8_TARGETS,
}];
static TCGEN_I8_LIBNVVM_CONTRACTS: &[GeneratedTargetContract] = &[GeneratedTargetContract {
    selectors: TCGEN_I8_SELECTORS,
    alternatives: TCGEN_I8_LIBNVVM_TARGETS,
}];
static TCGEN_I8_BACKENDS: &[GeneratedBackendRequirement] = &[GeneratedBackendRequirement {
    backend: GeneratedIntrinsicBackend::LibNvvm,
    requirement: GeneratedTargetRequirement {
        minimum_ptx: GeneratedPtxVersion::from_encoded(86),
        hardware: GeneratedHardwareTarget::TargetMatrix {
            contracts: TCGEN_I8_LIBNVVM_CONTRACTS,
        },
    },
}];
pub(super) static TCGEN_I8: GeneratedIntrinsicTarget = GeneratedIntrinsicTarget {
    marker: "test:tcgen_i8",
    id: "tcgen_i8",
    abi_id: "test",
    dialect_op: "test.tcgen_i8",
    variant: GeneratedIntrinsicVariant::Scalar,
    requirement: GeneratedTargetRequirement {
        minimum_ptx: GeneratedPtxVersion::from_encoded(86),
        hardware: GeneratedHardwareTarget::TargetMatrix {
            contracts: TCGEN_I8_CONTRACTS,
        },
    },
    backend_requirements: TCGEN_I8_BACKENDS,
    selections: &[],
    llvm: None,
};
