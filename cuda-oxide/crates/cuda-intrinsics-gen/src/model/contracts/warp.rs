/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusterBarrierMode {
    Arrive,
    ArriveAligned,
    ArriveRelaxed,
    ArriveRelaxedAligned,
    Wait,
    WaitAligned,
}

/// Closed semantics for one cluster-barrier spelling.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterBarrier {
    pub mode: ClusterBarrierMode,
    pub ordering: ClusterBarrierOrdering,
    pub aligned: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusterBarrierOrdering {
    Release,
    Relaxed,
    Acquire,
}

/// Closed semantic and lowering contract for the generated `redux.sync`
/// family.
///
/// The Rust and NVVM dialect APIs intentionally put the participation mask
/// first, while LLVM's NVVM intrinsic puts the lane value first. Keeping that
/// adapter typed prevents a generic direct-call renderer from silently
/// swapping the collective's source and member mask.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Redux {
    pub operation: ReduxOperation,
    pub participation: ReduxParticipation,
    pub adapter: ReduxAdapter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReduxOperation {
    Add,
    Umin,
    Min,
    Umax,
    Max,
    And,
    Or,
    Xor,
    Fmin,
    FminNan,
    FminAbs,
    FminAbsNan,
    Fmax,
    FmaxNan,
    FmaxAbs,
    FmaxAbsNan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReduxParticipation {
    ExecutingLaneNamedAllNamedLanesSameInstructionAndMask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReduxAdapter {
    MaskValueToSourceMemberMask,
}

/// Closed semantic and lowering contract for the generated `vote.sync`
/// family.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Vote {
    pub mode: VoteMode,
    pub participation: VoteParticipation,
    pub legacy_pre_sm70: PreSm70MemberMaskRule,
    pub adapter: VoteAdapter,
    pub mask_encoding: MaskEncoding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoteMode {
    All,
    Any,
    Ballot,
    Uni,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoteParticipation {
    ExecutingLaneNamedAllNamedLanesSameInstructionAndMask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoteAdapter {
    DirectMaskPredicate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskEncoding {
    RegisterOrImmediate,
}

/// Closed semantic and lowering contract for `activemask`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActiveMask {
    pub observation: ActiveMaskObservation,
    pub adapter: ActiveMaskAdapter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActiveMaskObservation {
    ExecutingLanesAtInstruction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActiveMaskAdapter {
    DirectZeroOperandMask,
}

/// Closed semantic and lowering contract for `match.sync`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WarpMatch {
    pub mode: WarpMatchMode,
    pub value_width: WarpMatchValueWidth,
    pub participation: WarpMatchParticipation,
    pub adapter: WarpMatchAdapter,
    pub value_encoding: MatchOperandEncoding,
    pub mask_encoding: MatchOperandEncoding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpMatchMode {
    Any,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpMatchValueWidth {
    B32,
    B64,
}

impl WarpMatchValueWidth {
    pub const fn bits(self) -> u32 {
        match self {
            Self::B32 => 32,
            Self::B64 => 64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpMatchParticipation {
    ExecutingLaneNamedAllNamedLanesSameInstructionAndMask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpMatchAdapter {
    DirectMask,
    ProjectMaskDiscardPredicate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchOperandEncoding {
    RegisterOrImmediate,
}

/// Closed semantic and lowering contract for `bar.warp.sync`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WarpBarrier {
    pub participation: WarpBarrierParticipation,
    pub legacy_pre_sm70: PreSm70MemberMaskRule,
    pub adapter: WarpBarrierAdapter,
    pub mask_encoding: WarpBarrierMaskEncoding,
    pub memory_ordering: WarpBarrierMemoryOrdering,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpBarrierParticipation {
    ExecutingLaneNamedAllNamedLanesSameInstructionAndMask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreSm70MemberMaskRule {
    AllNamedLanesConvergedAndOnlyNamedLanesActive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpBarrierAdapter {
    DirectMemberMask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpBarrierMaskEncoding {
    RegisterOrImmediate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpBarrierMemoryOrdering {
    ParticipatingLanes,
}

/// Closed semantic and lowering contract for `shfl.sync`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WarpShuffle {
    pub mode: WarpShuffleMode,
    pub value_kind: WarpShuffleValueKind,
    pub participation: WarpShuffleParticipation,
    pub legacy_pre_sm70: PreSm70MemberMaskRule,
    pub source_lane: WarpShuffleSourceLane,
    pub adapter: WarpShuffleAdapter,
    pub clamp: u32,
    pub lane_encoding: WarpShuffleOperandEncoding,
    pub mask_encoding: WarpShuffleOperandEncoding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpShuffleMode {
    Idx,
    Bfly,
    Down,
    Up,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpShuffleValueKind {
    I32,
    F32,
    I64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpShuffleParticipation {
    ExecutingLaneNamedAllNamedLanesSameInstructionAndMask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpShuffleSourceLane {
    InRangeSourceActiveAndNamedOutOfRangeCopiesSelf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpShuffleAdapter {
    MaskValueLaneOrDeltaInsertClamp,
    /// Split i64 into low/high b32 halves, shuffle both in one convergent
    /// side-effecting block, then reassemble the original bit layout.
    MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpShuffleOperandEncoding {
    RegisterOrImmediate,
    RegisterOnly,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redux_contract_rejects_unknown_operand_adapter() {
        let input = r#"
operation = "add"
participation = "executing_lane_named_all_named_lanes_same_instruction_and_mask"
adapter = "mask_value_direct"
"#;
        let error = toml::from_str::<Redux>(input).unwrap_err();
        assert!(error.to_string().contains("mask_value_direct"));
    }

    #[test]
    fn vote_contract_rejects_unknown_modes_and_mask_encodings() {
        let valid = r#"
mode = "all"
participation = "executing_lane_named_all_named_lanes_same_instruction_and_mask"
legacy_pre_sm70 = "all_named_lanes_converged_and_only_named_lanes_active"
adapter = "direct_mask_predicate"
mask_encoding = "register_or_immediate"
"#;
        toml::from_str::<Vote>(valid).unwrap();

        for invalid in [
            valid.replace("mode = \"all\"", "mode = \"match\""),
            valid.replace(
                "mask_encoding = \"register_or_immediate\"",
                "mask_encoding = \"any_operand\"",
            ),
            valid.replace(
                "legacy_pre_sm70 = \"all_named_lanes_converged_and_only_named_lanes_active\"",
                "legacy_pre_sm70 = \"independent_threads\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(toml::from_str::<Vote>(&invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn warp_shuffle_contract_rejects_open_ended_policy() {
        let valid = r#"
mode = "idx"
value_kind = "i32"
participation = "executing_lane_named_all_named_lanes_same_instruction_and_mask"
legacy_pre_sm70 = "all_named_lanes_converged_and_only_named_lanes_active"
source_lane = "in_range_source_active_and_named_out_of_range_copies_self"
adapter = "mask_value_lane_or_delta_insert_clamp"
clamp = 31
lane_encoding = "register_or_immediate"
mask_encoding = "register_or_immediate"
"#;
        toml::from_str::<WarpShuffle>(valid).unwrap();

        for invalid in [
            valid.replace("mode = \"idx\"", "mode = \"rotate\""),
            valid.replace("value_kind = \"i32\"", "value_kind = \"b32\""),
            valid.replace(
                "source_lane = \"in_range_source_active_and_named_out_of_range_copies_self\"",
                "source_lane = \"unchecked\"",
            ),
            valid.replace(
                "lane_encoding = \"register_or_immediate\"",
                "lane_encoding = \"anything\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(
                toml::from_str::<WarpShuffle>(&invalid).is_err(),
                "{invalid}"
            );
        }

        let i64 = r#"
mode = "down"
value_kind = "i64"
participation = "executing_lane_named_all_named_lanes_same_instruction_and_mask"
legacy_pre_sm70 = "all_named_lanes_converged_and_only_named_lanes_active"
source_lane = "in_range_source_active_and_named_out_of_range_copies_self"
adapter = "mask_value_lane_or_delta_split_i64_low_high_b32_insert_clamp_reassemble"
clamp = 31
lane_encoding = "register_only"
mask_encoding = "register_only"
"#;
        let parsed = toml::from_str::<WarpShuffle>(i64).unwrap();
        assert_eq!(parsed.value_kind, WarpShuffleValueKind::I64);
        assert_eq!(
            parsed.adapter,
            WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble
        );
        assert_eq!(
            parsed.lane_encoding,
            WarpShuffleOperandEncoding::RegisterOnly
        );

        for invalid in [
            i64.replace("value_kind = \"i64\"", "value_kind = \"u64\""),
            i64.replace(
                "adapter = \"mask_value_lane_or_delta_split_i64_low_high_b32_insert_clamp_reassemble\"",
                "adapter = \"split_any_width\"",
            ),
            i64.replace(
                "mask_encoding = \"register_only\"",
                "mask_encoding = \"any_operand\"",
            ),
        ] {
            assert!(
                toml::from_str::<WarpShuffle>(&invalid).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn warp_match_contract_rejects_open_ended_adapters_and_encodings() {
        let valid = r#"
mode = "all"
value_width = "b64"
participation = "executing_lane_named_all_named_lanes_same_instruction_and_mask"
adapter = "project_mask_discard_predicate"
value_encoding = "register_or_immediate"
mask_encoding = "register_or_immediate"
"#;
        toml::from_str::<WarpMatch>(valid).unwrap();

        for invalid in [
            valid.replace("mode = \"all\"", "mode = \"equal\""),
            valid.replace(
                "adapter = \"project_mask_discard_predicate\"",
                "adapter = \"first_result\"",
            ),
            valid.replace(
                "value_encoding = \"register_or_immediate\"",
                "value_encoding = \"anything\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(toml::from_str::<WarpMatch>(&invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn warp_barrier_contract_rejects_open_ended_policy() {
        let valid = r#"
participation = "executing_lane_named_all_named_lanes_same_instruction_and_mask"
legacy_pre_sm70 = "all_named_lanes_converged_and_only_named_lanes_active"
adapter = "direct_member_mask"
mask_encoding = "register_or_immediate"
memory_ordering = "participating_lanes"
"#;
        toml::from_str::<WarpBarrier>(valid).unwrap();

        for invalid in [
            valid.replace("adapter = \"direct_member_mask\"", "adapter = \"direct\""),
            valid.replace(
                "legacy_pre_sm70 = \"all_named_lanes_converged_and_only_named_lanes_active\"",
                "legacy_pre_sm70 = \"independent_threads\"",
            ),
            valid.replace(
                "mask_encoding = \"register_or_immediate\"",
                "mask_encoding = \"any_operand\"",
            ),
            valid.replace(
                "memory_ordering = \"participating_lanes\"",
                "memory_ordering = \"none\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(
                toml::from_str::<WarpBarrier>(&invalid).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn cluster_barrier_contract_rejects_open_ended_semantics() {
        let valid = r#"
mode = "arrive_relaxed_aligned"
ordering = "relaxed"
aligned = true
"#;
        let parsed = toml::from_str::<ClusterBarrier>(valid).unwrap();
        assert_eq!(parsed.mode, ClusterBarrierMode::ArriveRelaxedAligned);
        assert_eq!(parsed.ordering, ClusterBarrierOrdering::Relaxed);
        assert!(parsed.aligned);

        for invalid in [
            valid.replace("ordering = \"relaxed\"", "ordering = \"unordered\""),
            valid.replace("aligned = true", "aligned = \"sometimes\""),
            valid.replace(
                "mode = \"arrive_relaxed_aligned\"",
                "mode = \"arrive_release_aligned\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(
                toml::from_str::<ClusterBarrier>(&invalid).is_err(),
                "{invalid}"
            );
        }
    }
}
