/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Reusable analyses over the `dialect-mir` IR.
//!
//! An analysis is read-only: it inspects the code and reports facts about it,
//! but never changes the code. Passes that *do* change the code (such as the
//! loop unroller) call these to decide what to do, and future loop passes can
//! reuse them too.
//!
//! Naming follows pliron's convention (`analyses/liveness.rs`,
//! `graph/dominance.rs`): each file is named for the concept it computes, with
//! no `-analysis` suffix; sitting in the `analyses/` directory is what marks it
//! as an analysis.

pub mod induction;
pub mod loop_info;
