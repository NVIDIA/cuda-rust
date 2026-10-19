// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Two host mappings, one for each half of the launch-uniformity story.
//!
//! A `Uniform<T>` kernel parameter is marshalled as a bare `T`, because the
//! host is what makes the value uniform: one marshalled value reaches every
//! thread of the launch. A slice whose index space carries a runtime row
//! width is marshalled as `RowWidth<T>`, which binds the width to that slice for the
//! launch, so `tile_2d32_rt` needs neither a stride argument nor `unsafe`.

use cuda_device::{
    DisjointSlice, RuntimeRowMajorTiles, Uniform, cuda_module, kernel, launch_contract, thread,
};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel(launch_context = lc)]
    #[launch_contract(domain = 2, coordinates = u32, block = (8, 8, 1))]
    pub fn write_cells(
        rows: Uniform<u32>,
        mut out: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>,
    ) {
        let coord = thread::coord_2d_u32(lc);
        if coord.row() >= rows.get() {
            return;
        }
        if let Some(mut cell) = out.tile_2d32_rt(coord) {
            cell.at_const::<0, 0>().write(1.0);
        }
    }

    /// Uniformity is closed under arithmetic whose operands are all uniform,
    /// so a derived bound keeps the witness.
    #[kernel(launch_context = lc)]
    #[launch_contract(domain = 2, coordinates = u32, block = (8, 8, 1))]
    pub fn write_doubled_bound(
        rows: Uniform<u32>,
        mut out: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>,
    ) {
        let coord = thread::coord_2d_u32(lc);
        let doubled = rows.wrapping_mul_const::<2>();
        if coord.row() >= doubled.get() {
            return;
        }
        if let Some(mut cell) = out.tile_2d32_rt(coord) {
            cell.at_const::<0, 0>().write(2.0);
        }
    }
}

/// The generated host method takes a plain `u32` for the witness, and a
/// `RowWidth` for the slice that carries the row width.
fn host_signature_takes_the_bare_scalar_and_a_row_width(
    module: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
    out: &mut cuda_core::DeviceBuffer<f32>,
) -> Result<(), cuda_core::LaunchContractError> {
    let prepared = module.prepare_write_cells(cuda_core::LaunchConfig2D::new((1, 1), (8, 8), 0))?;
    module.write_cells(stream, &prepared, 8u32, cuda_host::RowWidth::new(out, 64))
}

fn main() {
    let _ = host_signature_takes_the_bare_scalar_and_a_row_width;
}
