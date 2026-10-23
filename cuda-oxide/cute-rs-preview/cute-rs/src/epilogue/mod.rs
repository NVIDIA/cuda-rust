/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Output staging operations grouped by architecture.
//!
//! The SM120 staging types and helpers are also exported here. The compiler
//! hooks have one path, `sm120::__compiler`.

pub mod sm120;

pub use sm120::{
    SM120_EPILOGUE_BYTES, SM120_EPILOGUE_COLS, SM120_EPILOGUE_ELEMENTS, SM120_EPILOGUE_HALF_BYTES,
    SM120_EPILOGUE_HALF_COLS, SM120_EPILOGUE_HALF_ELEMENTS, SM120_EPILOGUE_ROWS,
    Sm120Epilogue128x128, Sm120EpilogueHalfLayout, Sm120EpilogueWarp128x128,
    sm120_epilogue_atom_origin, sm120_epilogue_physical_offset,
};
