/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Umbrella header for bindgen. Scope: enough MLIR C-API surface to register
 * and build cuda-tile dialect ops/types via the MLIR builder API. Upstream
 * MLIR dialects (arith, func, scf, linalg, etc.) are intentionally not
 * exposed — add their headers here if/when needed.
 *
 * Bindings are filtered via allowlist_file in build.rs.
 */

/* --- cuda-tile C-API --------------------------------------------------- */
#include "cuda_tile-c/Registration.h"
#include "cuda_tile-c/Dialect/CudaTileDialect.h"
#include "cuda_tile-c/Dialect/CudaTileOptimizer.h"

/* --- MLIR C-API (core) ------------------------------------------------- */
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/AffineExpr.h"
#include "mlir-c/AffineMap.h"
#include "mlir-c/IntegerSet.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/Debug.h"
#include "mlir-c/Pass.h"
