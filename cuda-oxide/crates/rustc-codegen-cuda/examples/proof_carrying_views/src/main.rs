/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Safe proof-carrying views beside equivalent raw-pointer kernels.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D, LaunchConfig2D};
use cuda_device::{
    DisjointSlice, LinearTiles, RowMajorTiles, cuda_module, kernel, launch_bounds, launch_contract,
    thread,
};

const EPILOGUE_ROWS: u32 = 64;
const EPILOGUE_STRIDE: u32 = 64;
const EPILOGUE_COLS_PER_THREAD: u32 = 2;

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(always)]
    fn checked_raw_tile_start(thread: u32, len: usize) -> u64 {
        const WIDTH: u32 = 4;
        const LAST_OFFSET: u32 = WIDTH - 1;
        if thread > (u32::MAX - LAST_OFFSET) / WIDTH {
            return u64::MAX;
        }
        let base = thread * WIDTH;
        let last = base + LAST_OFFSET;
        if (last as usize) < len {
            u64::from(base)
        } else {
            u64::MAX
        }
    }

    #[inline(always)]
    fn checked_raw_epilogue_start(row: u32, tile_col: u32, len: usize) -> u64 {
        let last_col_offset = EPILOGUE_COLS_PER_THREAD - 1;
        if tile_col > (u32::MAX - last_col_offset) / EPILOGUE_COLS_PER_THREAD {
            return u64::MAX;
        }
        let col = tile_col * EPILOGUE_COLS_PER_THREAD;
        let last_col = col + last_col_offset;
        if last_col >= EPILOGUE_STRIDE || row > (u32::MAX - last_col) / EPILOGUE_STRIDE {
            return u64::MAX;
        }
        let start = row * EPILOGUE_STRIDE + col;
        let last = start + last_col_offset;
        if (last as usize) < len {
            u64::from(start)
        } else {
            u64::MAX
        }
    }

    /// One bounds proof gives this thread read/write access to one element.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, coordinates = u32, block = (64, 1, 1))]
    pub fn safe_element(mut values: DisjointSlice<u32>) {
        let index = thread::index_1d_u32(launch_context);
        if let Some(mut element) = values.element_thread32(index) {
            let value = element.read();
            element.write(value.wrapping_mul(3).wrapping_add(1));
        }
    }

    /// Legacy kernels now fail closed if a caller supplies non-1D Y/Z axes.
    #[kernel]
    pub fn legacy_rank_guard(mut values: DisjointSlice<u32>) {
        let index = thread::index_1d();
        if let Some(value) = values.get_mut(index) {
            *value = 1;
        }
    }

    /// The same operation with a manually checked raw pointer.
    ///
    /// # Safety
    ///
    /// `values` must reference `len` readable and writable device `u32`
    /// elements for the duration of the launch.
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, coordinates = u32, block = (64, 1, 1))]
    pub unsafe fn raw_element(values: *mut u32, len: usize) {
        let index = thread::blockIdx_x()
            .wrapping_mul(thread::blockDim_x())
            .wrapping_add(thread::threadIdx_x());
        if (index as usize) < len {
            // SAFETY: the branch proves `index < len`, the host allocation has
            // `len` elements, and each 1-D thread computes a distinct index.
            unsafe {
                let element = values.add(index as usize);
                let value = element.read();
                element.write(value.wrapping_mul(3).wrapping_add(1));
            }
        }
    }

    /// One range proof gives this thread a four-element static view.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, coordinates = u32, block = (64, 1, 1))]
    pub fn safe_tile(mut values: DisjointSlice<u32, LinearTiles<4>>) {
        let index = thread::index_1d_u32(launch_context);
        if let Some(mut tile) = values.tile_thread32(index) {
            let v0 = tile.at_const::<0>().read();
            let v1 = tile.at_const::<1>().read();
            let v2 = tile.at_const::<2>().read();
            let v3 = tile.at_const::<3>().read();
            tile.at_const::<0>().write(v0.wrapping_add(1));
            tile.at_const::<1>().write(v1.wrapping_add(2));
            tile.at_const::<2>().write(v2.wrapping_add(3));
            tile.at_const::<3>().write(v3.wrapping_add(4));
        }
    }

    /// The same operation with explicit overflow and range checks.
    ///
    /// # Safety
    ///
    /// `values` must reference `len` readable and writable device `u32`
    /// elements for the duration of the launch.
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, coordinates = u32, block = (64, 1, 1))]
    pub unsafe fn raw_tile(values: *mut u32, len: usize) {
        let thread = thread::blockIdx_x()
            .wrapping_mul(thread::blockDim_x())
            .wrapping_add(thread::threadIdx_x());
        let base = checked_raw_tile_start(thread, len);
        if base == u64::MAX {
            return;
        }
        let base = base as u32;
        // SAFETY: the combined guard proves all four offsets are in the
        // allocation; distinct 1-D threads own disjoint four-element ranges.
        unsafe {
            let tile = values.add(base as usize);
            let v0 = tile.add(0).read();
            let v1 = tile.add(1).read();
            let v2 = tile.add(2).read();
            let v3 = tile.add(3).read();
            tile.add(0).write(v0.wrapping_add(1));
            tile.add(1).write(v1.wrapping_add(2));
            tile.add(2).write(v2.wrapping_add(3));
            tile.add(3).write(v3.wrapping_add(4));
        }
    }

    /// A two-column GEMM-style epilogue tile with static row-major layout.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 2, coordinates = u32, block = (8, 8, 1))]
    pub fn safe_epilogue(mut values: DisjointSlice<f32, RowMajorTiles<1, 2, 64>>) {
        let coord = thread::coord_2d_u32(launch_context);
        if let Some(mut tile) = values.tile_2d32(coord) {
            let left = tile.at_const::<0, 0>().read();
            let right = tile.at_const::<0, 1>().read();
            tile.at_const::<0, 0>().write(left + 1.0);
            tile.at_const::<0, 1>().write(right + 1.0);
        }
    }

    /// Equivalent manually proved raw-pointer epilogue.
    ///
    /// # Safety
    ///
    /// `values` must reference `len` readable and writable device `f32`
    /// elements for the duration of the launch.
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(domain = 2, coordinates = u32, block = (8, 8, 1))]
    pub unsafe fn raw_epilogue(values: *mut f32, len: usize) {
        let row = thread::blockIdx_y()
            .wrapping_mul(thread::blockDim_y())
            .wrapping_add(thread::threadIdx_y());
        let tile_col = thread::blockIdx_x()
            .wrapping_mul(thread::blockDim_x())
            .wrapping_add(thread::threadIdx_x());
        let start = checked_raw_epilogue_start(row, tile_col, len);
        if start == u64::MAX {
            return;
        }
        // SAFETY: the scalar proof above covers both columns, and distinct 2D
        // threads own disjoint row/column tiles.
        unsafe {
            let pair = values.add(start as usize);
            let left = pair.read();
            let right = pair.add(1).read();
            pair.write(left + 1.0);
            pair.add(1).write(right + 1.0);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().any(|arg| arg == "--verify-ptx") {
        return verify_ptx();
    }

    const N: usize = 4096;
    const BLOCK: u32 = 64;

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // SAFETY: this standalone example owns the package-named device bundle,
    // whose four entry definitions are generated by the module above.
    let module = unsafe { kernels::load(&context)? };
    let initial: Vec<u32> = (0..N as u32).collect();

    let mut safe_elements = DeviceBuffer::from_host(&stream, &initial)?;
    let raw_elements = DeviceBuffer::from_host(&stream, &initial)?;
    let element_grid = (N as u32).div_ceil(BLOCK);
    let safe_element_launch =
        module.prepare_safe_element(LaunchConfig1D::new(element_grid, BLOCK, 0))?;
    let raw_element_launch =
        module.prepare_raw_element(LaunchConfig1D::new(element_grid, BLOCK, 0))?;
    module.safe_element(&stream, &safe_element_launch, &mut safe_elements)?;
    // SAFETY: `raw_elements` owns exactly N live device u32 elements and
    // remains alive until after stream synchronization in `to_host_vec`.
    unsafe {
        module.raw_element(
            &stream,
            &raw_element_launch,
            raw_elements.cu_deviceptr() as *mut u32,
            N,
        )?;
    }

    let safe_elements = safe_elements.to_host_vec(&stream)?;
    let raw_elements = raw_elements.to_host_vec(&stream)?;
    assert_eq!(safe_elements, raw_elements);
    assert!(
        safe_elements
            .iter()
            .enumerate()
            .all(|(i, &value)| value == (i as u32).wrapping_mul(3).wrapping_add(1))
    );

    let mut safe_tiles = DeviceBuffer::from_host(&stream, &initial)?;
    let raw_tiles = DeviceBuffer::from_host(&stream, &initial)?;
    let tile_threads = (N as u32).div_ceil(4);
    let tile_grid = tile_threads.div_ceil(BLOCK);
    let safe_tile_launch = module.prepare_safe_tile(LaunchConfig1D::new(tile_grid, BLOCK, 0))?;
    let raw_tile_launch = module.prepare_raw_tile(LaunchConfig1D::new(tile_grid, BLOCK, 0))?;
    module.safe_tile(&stream, &safe_tile_launch, &mut safe_tiles)?;
    // SAFETY: `raw_tiles` owns exactly N live device u32 elements and remains
    // alive until after stream synchronization in `to_host_vec`.
    unsafe {
        module.raw_tile(
            &stream,
            &raw_tile_launch,
            raw_tiles.cu_deviceptr() as *mut u32,
            N,
        )?;
    }

    let safe_tiles = safe_tiles.to_host_vec(&stream)?;
    let raw_tiles = raw_tiles.to_host_vec(&stream)?;
    assert_eq!(safe_tiles, raw_tiles);
    assert!(safe_tiles.iter().enumerate().all(|(i, &value)| {
        let lane = (i % 4) as u32;
        value == i as u32 + lane + 1
    }));

    let epilogue_input: Vec<f32> = (0..N)
        .map(|index| index as f32 - (N as f32 / 2.0))
        .collect();
    let mut safe_epilogue = DeviceBuffer::from_host(&stream, &epilogue_input)?;
    let raw_epilogue = DeviceBuffer::from_host(&stream, &epilogue_input)?;
    let epilogue_config = LaunchConfig2D::new(
        (
            EPILOGUE_STRIDE / EPILOGUE_COLS_PER_THREAD / 8,
            EPILOGUE_ROWS / 8,
        ),
        (8, 8),
        0,
    );
    let safe_epilogue_launch = module.prepare_safe_epilogue(epilogue_config)?;
    let raw_epilogue_launch = module.prepare_raw_epilogue(epilogue_config)?;
    module.safe_epilogue(&stream, &safe_epilogue_launch, &mut safe_epilogue)?;
    // SAFETY: `raw_epilogue` owns exactly N live device f32 elements and
    // remains alive until after stream synchronization in `to_host_vec`.
    unsafe {
        module.raw_epilogue(
            &stream,
            &raw_epilogue_launch,
            raw_epilogue.cu_deviceptr() as *mut f32,
            N,
        )?;
    }
    let safe_epilogue = safe_epilogue.to_host_vec(&stream)?;
    let raw_epilogue = raw_epilogue.to_host_vec(&stream)?;
    assert_eq!(safe_epilogue, raw_epilogue);
    assert!(
        safe_epilogue
            .iter()
            .enumerate()
            .all(|(index, &value)| { value == epilogue_input[index] + 1.0 })
    );

    println!("SUCCESS: safe proof-carrying views matched raw kernels");
    Ok(())
}

fn verify_ptx() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("proof_carrying_views.ptx");
    let ptx = std::fs::read_to_string(&path)?;
    let document = ptx_parse::Document::parse(&ptx)?;

    for marker in [
        "__launch_contract_config",
        "__launch_contract_block_config",
        "__launch_bounds_config",
        "make_kernel_scope",
    ] {
        if ptx.contains(marker) {
            return Err(
                format!("compile-time marker `{marker}` leaked into the PTX module").into(),
            );
        }
    }

    let safe_element = entry(&document, "safe_element")?;
    let raw_element = entry(&document, "raw_element")?;
    let safe_tile = entry(&document, "safe_tile")?;
    let raw_tile = entry(&document, "raw_tile")?;
    let legacy_rank_guard = entry(&document, "legacy_rank_guard")?;
    let safe_epilogue = entry(&document, "safe_epilogue")?;
    let raw_epilogue = entry(&document, "raw_epilogue")?;

    for (name, definition) in [
        ("safe_element", safe_element),
        ("raw_element", raw_element),
        ("safe_tile", safe_tile),
        ("raw_tile", raw_tile),
    ] {
        let body = definition.text();
        verify_exact_block(name, body, ".reqntid 64, 1, 1")?;
        verify_u32_coordinates(name, body)?;
        verify_no_calls(name, definition)?;
        let branches = conditional_branches(definition);
        let expected_branches = 1;
        if branches != expected_branches {
            return Err(format!(
                "{name} has {branches} guard branches; expected {expected_branches}"
            )
            .into());
        }
        verify_no_interior_branches(name, definition)?;
    }

    compare_memory_widths("element", safe_element, raw_element)?;
    compare_memory_widths("tile", safe_tile, raw_tile)?;
    compare_memory_operations("element", safe_element, raw_element)?;
    compare_memory_operations("tile", safe_tile, raw_tile)?;
    verify_legacy_rank_guard(legacy_rank_guard)?;

    for (name, definition) in [
        ("safe_epilogue", safe_epilogue),
        ("raw_epilogue", raw_epilogue),
    ] {
        let body = definition.text();
        // A 2-D contract records its real shape. A thread maximum bounds the
        // product alone, so its per-axis fields would claim one thread on Y
        // for a block that has eight.
        verify_exact_block(name, body, ".reqntid 8, 8, 1")?;
        verify_u32_coordinates(name, body)?;
        verify_no_calls(name, definition)?;
        for register in ["%ctaid.y", "%ntid.y", "%tid.y"] {
            if !body.contains(register) {
                return Err(format!("{name} does not read {register}").into());
            }
        }
        verify_no_interior_branches(name, definition)?;
    }
    let safe_epilogue_branches = conditional_branches(safe_epilogue);
    let raw_epilogue_branches = conditional_branches(raw_epilogue);
    if safe_epilogue_branches != raw_epilogue_branches {
        return Err(format!(
            "epilogue guard branches differ: safe={safe_epilogue_branches}, raw={raw_epilogue_branches}"
        )
        .into());
    }
    compare_memory_widths("epilogue", safe_epilogue, raw_epilogue)?;
    compare_memory_operations("epilogue", safe_epilogue, raw_epilogue)?;

    println!("SUCCESS: proof-carrying views match raw PTX structure");
    Ok(())
}

fn verify_legacy_rank_guard(
    definition: ptx_parse::CallableDefinition<'_, '_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let body = definition.text();
    for register in ["%ntid.y", "%nctaid.y", "%ntid.z", "%nctaid.z"] {
        if !body.contains(register) {
            return Err(format!("legacy 1D witness does not validate {register}").into());
        }
    }
    let instructions = definition.instructions().collect::<Vec<_>>();
    let first_store = instructions
        .iter()
        .position(|instruction| data_memory_width(instruction, "st").is_some())
        .ok_or("legacy rank-guard entry has no data store")?;
    if !instructions[..first_store]
        .iter()
        .any(|instruction| instruction.base_opcode() == "bra" && instruction.predicate().is_some())
    {
        return Err("legacy 1D store is not dominated by a rank guard".into());
    }
    Ok(())
}

fn compare_memory_operations(
    pair: &str,
    safe: ptx_parse::CallableDefinition<'_, '_>,
    raw: ptx_parse::CallableDefinition<'_, '_>,
) -> Result<(), Box<dyn std::error::Error>> {
    for operation in ["ld", "st"] {
        let safe_ops = data_memory_operations(safe, operation);
        let raw_ops = data_memory_operations(raw, operation);
        if safe_ops != raw_ops {
            return Err(format!(
                "{pair} {operation} operations differ: safe={safe_ops:?}, raw={raw_ops:?}"
            )
            .into());
        }
    }
    Ok(())
}

fn data_memory_operations(
    definition: ptx_parse::CallableDefinition<'_, '_>,
    operation: &str,
) -> Vec<String> {
    let mut operations: Vec<String> = definition
        .instructions()
        .filter_map(|instruction| data_memory_operation(instruction, operation))
        .collect();
    operations.sort();
    operations
}

fn data_memory_operation(
    instruction: &ptx_parse::Instruction<'_>,
    operation: &str,
) -> Option<String> {
    if instruction.base_opcode() != operation {
        return None;
    }
    let mnemonic = instruction.head();
    if mnemonic.contains(".param.") || mnemonic.contains(".shared.") || mnemonic.contains(".local.")
    {
        return None;
    }
    Some(mnemonic.to_owned())
}

fn entry<'document, 'source>(
    document: &'document ptx_parse::Document<'source>,
    name: &str,
) -> Result<ptx_parse::CallableDefinition<'document, 'source>, Box<dyn std::error::Error>> {
    document
        .definitions_named(name)
        .find(|definition| definition.callable().kind() == ptx_parse::CallableKind::Entry)
        .ok_or_else(|| format!("missing or incomplete PTX entry `{name}`").into())
}

/// A kernel declaring an exact `block` in its launch contract carries that
/// shape into PTX as `.reqntid`, which the driver enforces on every axis.
///
/// `.reqntid` displaces `.maxntid`: ptxas rejects an entry declaring both, and
/// an exact shape already bounds the thread count.
fn verify_exact_block(
    name: &str,
    body: &str,
    expected: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if !body.contains(expected) {
        return Err(format!("{name} lost its exact block shape `{expected}`").into());
    }
    if body.contains(".maxntid") {
        return Err(
            format!("{name} declares both .maxntid and .reqntid, which ptxas rejects").into(),
        );
    }
    Ok(())
}

fn verify_u32_coordinates(name: &str, body: &str) -> Result<(), Box<dyn std::error::Error>> {
    for register in ["%ctaid.x", "%ntid.x", "%tid.x"] {
        if !body.contains(register) {
            return Err(format!("{name} does not read {register}").into());
        }
    }
    for forbidden in ["mul.lo.s64", "mul.lo.u64", "mad.lo.s64", "mad.lo.u64"] {
        if body.contains(forbidden) {
            return Err(
                format!("{name} widened coordinate arithmetic through `{forbidden}`").into(),
            );
        }
    }
    if !body.contains("mul.wide.u32") && !body.contains("cvt.u64.u32") {
        return Err(format!("{name} has no final u32-to-address widening operation").into());
    }
    Ok(())
}

fn conditional_branches(definition: ptx_parse::CallableDefinition<'_, '_>) -> usize {
    definition
        .instructions()
        .filter(|instruction| {
            instruction.base_opcode() == "bra" && instruction.predicate().is_some()
        })
        .count()
}

fn verify_no_calls(
    name: &str,
    definition: ptx_parse::CallableDefinition<'_, '_>,
) -> Result<(), Box<dyn std::error::Error>> {
    if definition
        .instructions()
        .any(|instruction| instruction.base_opcode() == "call")
    {
        return Err(format!("{name} contains an out-of-line device call").into());
    }
    Ok(())
}

fn verify_no_interior_branches(
    name: &str,
    definition: ptx_parse::CallableDefinition<'_, '_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let instructions = definition.instructions().collect::<Vec<_>>();
    let first_load = instructions
        .iter()
        .position(|instruction| data_memory_width(instruction, "ld").is_some())
        .ok_or_else(|| format!("{name} has no data load"))?;
    if instructions[first_load..]
        .iter()
        .any(|instruction| instruction.base_opcode() == "bra")
    {
        return Err(format!("{name} repeats a bounds branch inside its proven data range").into());
    }
    Ok(())
}

fn compare_memory_widths(
    pair: &str,
    safe: ptx_parse::CallableDefinition<'_, '_>,
    raw: ptx_parse::CallableDefinition<'_, '_>,
) -> Result<(), Box<dyn std::error::Error>> {
    for operation in ["ld", "st"] {
        let safe_widths = data_memory_widths(safe, operation);
        let raw_widths = data_memory_widths(raw, operation);
        if safe_widths.is_empty() {
            return Err(format!("safe {pair} entry has no `{operation}` data operation").into());
        }
        if safe_widths != raw_widths {
            return Err(format!(
                "{pair} {operation} widths differ: safe={safe_widths:?}, raw={raw_widths:?}"
            )
            .into());
        }
    }
    Ok(())
}

fn data_memory_widths(
    definition: ptx_parse::CallableDefinition<'_, '_>,
    operation: &str,
) -> Vec<String> {
    let mut widths: Vec<String> = definition
        .instructions()
        .filter_map(|instruction| data_memory_width(instruction, operation))
        .collect();
    widths.sort();
    widths
}

fn data_memory_width(instruction: &ptx_parse::Instruction<'_>, operation: &str) -> Option<String> {
    if instruction.base_opcode() != operation {
        return None;
    }
    let mnemonic = instruction.head();
    if mnemonic.contains(".param.") || mnemonic.contains(".shared.") || mnemonic.contains(".local.")
    {
        return None;
    }

    let parts: Vec<&str> = mnemonic.split('.').collect();
    let value_type = parts.last()?;
    let vector_width = parts
        .iter()
        .find(|part| part.starts_with('v') && part[1..].chars().all(|ch| ch.is_ascii_digit()));
    Some(match vector_width {
        Some(width) => format!("{width}.{value_type}"),
        None => (*value_type).to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_control_flow_parser_handles_suffixes_and_inverted_predicates() {
        let ptx = ".visible .entry test() {\n\
                   @!%p1 bra L1;\n@%p2 bra.uni L2;\nbra.uni L3;\ncall.uni helper;\n}";
        let document = ptx_parse::Document::parse(ptx).unwrap();
        let definition = entry(&document, "test").unwrap();
        assert_eq!(conditional_branches(definition), 2);
        assert!(
            definition
                .instructions()
                .any(|instruction| instruction.head() == "bra.uni")
        );
        assert!(
            definition
                .instructions()
                .any(|instruction| instruction.head() == "call.uni")
        );
    }
}
