/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Proofs.ControlFlow.Lattice
import CudaOxide.Proofs.ControlFlow.Soundness

/-!
# Certified Control-Flow Verification (`CudaOxide.Proofs.ControlFlow`)

- `ControlFlow/Lattice.lean`: `CtaState.subsumes` lattice and `stepOp_mono` / `stepEdge_mono`.
- `ControlFlow/Soundness.lean`: `StateMap` inference, `ctaReachable_subsumed`, and contract bridge theorems.
-/
