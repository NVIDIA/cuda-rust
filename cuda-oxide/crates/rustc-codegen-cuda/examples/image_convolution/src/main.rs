/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! 2D image convolution example.
//!
//! Compares a naive 3x3 Gaussian convolution using global-memory reads
//! with a shared-memory tiled implementation. Image edges use zero padding.
//!
//! Run with:
//!   cargo oxide run image_convolution

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig2D};
use cuda_device::{
    DisjointSlice, RuntimeRowMajorTiles, SharedArray, cuda_module, kernel, launch_bounds,
    launch_contract, thread,
};

const BLOCK: u32 = 16;
const IMAGE_DEMO_PASSES: usize = 128;

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(always)]
    fn sample_zero(input: &[f32], width: u32, height: u32, x: i64, y: i64) -> f32 {
        if x < 0 || y < 0 || x >= width as i64 || y >= height as i64 {
            return 0.0;
        }

        input[y as usize * width as usize + x as usize]
    }

    /// Naive 3x3 Gaussian convolution.
    ///
    /// Each output thread reads its complete 3x3 neighborhood directly from
    /// global memory. A later tiled kernel will reuse those neighboring pixels
    /// through shared memory.
    #[kernel(launch_context = lc)]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 2,
        coordinates = u32,
        block = (16, 16, 1),
        requires = (
            input.len() >= width * height,
            output.len() >= width * height
        )
    )]
    pub fn convolution_naive(
        width: u32,
        height: u32,
        input: &[f32],
        mut output: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>,
    ) {
        let coord = thread::coord_2d_u32(lc);
        let y = coord.row();
        let x = coord.col();

        // This kernel has no block-wide synchronization, so threads outside
        // the image may return independently.
        if x >= width || y >= height {
            return;
        }

        let x = x as i64;
        let y = y as i64;

        // 3x3 Gaussian kernel:
        //
        //  1  2  1
        //  2  4  2   / 16
        //  1  2  1
        let mut sum = 0.0f32;

        sum += sample_zero(input, width, height, x - 1, y - 1);
        sum += 2.0 * sample_zero(input, width, height, x, y - 1);
        sum += sample_zero(input, width, height, x + 1, y - 1);

        sum += 2.0 * sample_zero(input, width, height, x - 1, y);
        sum += 4.0 * sample_zero(input, width, height, x, y);
        sum += 2.0 * sample_zero(input, width, height, x + 1, y);

        sum += sample_zero(input, width, height, x - 1, y + 1);
        sum += 2.0 * sample_zero(input, width, height, x, y + 1);
        sum += sample_zero(input, width, height, x + 1, y + 1);

        let value = sum * (1.0 / 16.0);

        if let Some(mut cell) = output.tile_2d32_rt(coord) {
            cell.at_const::<0, 0>().write(value);
        }
    }

    /// Shared-memory tiled 3x3 Gaussian convolution.
    ///
    /// A 16x16 block cooperatively loads an 18x18 tile: the 16x16 output
    /// region plus a one-pixel halo on every side. Neighboring output threads
    /// then reuse those staged pixels instead of re-reading them from global
    /// memory.
    #[kernel(launch_context = lc)]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 2,
        coordinates = u32,
        block = (16, 16, 1),
        requires = (
            input.len() >= width * height,
            output.len() >= width * height
        )
    )]
    pub fn convolution_tiled(
        width: u32,
        height: u32,
        input: &[f32],
        mut output: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>,
    ) {
        const OUTPUT_TILE: usize = 16;
        const SHARED_TILE: usize = OUTPUT_TILE + 2;
        const SHARED_ELEMENTS: usize = SHARED_TILE * SHARED_TILE;
        const THREADS_PER_BLOCK: usize = OUTPUT_TILE * OUTPUT_TILE;

        static mut TILE: SharedArray<f32, SHARED_ELEMENTS> = SharedArray::UNINIT;

        // Derive raw element pointers so cooperative writers do not borrow
        // the complete shared allocation mutably in every thread.
        let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) };

        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let local_tid = ty * OUTPUT_TILE + tx;

        let block_x = thread::blockIdx_x() as i64 * OUTPUT_TILE as i64;
        let block_y = thread::blockIdx_y() as i64 * OUTPUT_TILE as i64;

        // 256 threads cooperatively load 324 shared-memory elements.
        // The first 68 threads perform a second load.
        let mut shared_idx = local_tid;
        while shared_idx < SHARED_ELEMENTS {
            let sx = shared_idx % SHARED_TILE;
            let sy = shared_idx / SHARED_TILE;

            // Subtract one to account for the one-pixel halo.
            let gx = block_x + sx as i64 - 1;
            let gy = block_y + sy as i64 - 1;

            let value = if gx >= 0 && gy >= 0 && gx < width as i64 && gy < height as i64 {
                input[gy as usize * width as usize + gx as usize]
            } else {
                0.0
            };

            unsafe {
                tile.add(shared_idx).write(value);
            }

            shared_idx += THREADS_PER_BLOCK;
        }

        // Every thread in the block must reach this barrier, including threads
        // whose final output coordinate lies outside a partial edge tile.
        thread::sync_threads();

        let coord = thread::coord_2d_u32(lc);
        let y = coord.row();
        let x = coord.col();

        if x < width && y < height {
            // Each output thread is offset by one inside the shared tile
            // because row/column zero hold the top/left halo.
            let sx = tx + 1;
            let sy = ty + 1;

            let mut sum = 0.0f32;

            unsafe {
                sum += tile.add((sy - 1) * SHARED_TILE + (sx - 1)).read();
                sum += 2.0 * tile.add((sy - 1) * SHARED_TILE + sx).read();
                sum += tile.add((sy - 1) * SHARED_TILE + (sx + 1)).read();

                sum += 2.0 * tile.add(sy * SHARED_TILE + (sx - 1)).read();
                sum += 4.0 * tile.add(sy * SHARED_TILE + sx).read();
                sum += 2.0 * tile.add(sy * SHARED_TILE + (sx + 1)).read();

                sum += tile.add((sy + 1) * SHARED_TILE + (sx - 1)).read();
                sum += 2.0 * tile.add((sy + 1) * SHARED_TILE + sx).read();
                sum += tile.add((sy + 1) * SHARED_TILE + (sx + 1)).read();
            }

            if let Some(mut cell) = output.tile_2d32_rt(coord) {
                cell.at_const::<0, 0>().write(sum * (1.0 / 16.0));
            }
        }
    }
}

fn sample_zero_cpu(input: &[f32], width: usize, height: usize, x: isize, y: isize) -> f32 {
    if x < 0 || y < 0 || x >= width as isize || y >= height as isize {
        return 0.0;
    }

    input[y as usize * width + x as usize]
}

fn convolution_cpu(input: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; width * height];

    for y in 0..height {
        for x in 0..width {
            let x = x as isize;
            let y = y as isize;

            let mut sum = 0.0f32;

            sum += sample_zero_cpu(input, width, height, x - 1, y - 1);
            sum += 2.0 * sample_zero_cpu(input, width, height, x, y - 1);
            sum += sample_zero_cpu(input, width, height, x + 1, y - 1);

            sum += 2.0 * sample_zero_cpu(input, width, height, x - 1, y);
            sum += 4.0 * sample_zero_cpu(input, width, height, x, y);
            sum += 2.0 * sample_zero_cpu(input, width, height, x + 1, y);

            sum += sample_zero_cpu(input, width, height, x - 1, y + 1);
            sum += 2.0 * sample_zero_cpu(input, width, height, x, y + 1);
            sum += sample_zero_cpu(input, width, height, x + 1, y + 1);

            output[y as usize * width + x as usize] = sum * (1.0 / 16.0);
        }
    }

    output
}

fn validate_shape(width: usize, height: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Image Convolution Example ===");
    println!("Image: {width}x{height}");
    println!("Kernel: 3x3 Gaussian, zero-padded edges");
    println!("Block: {BLOCK}x{BLOCK}");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // Non-trivial deterministic input makes incorrect indexing easier to
    // detect than a constant or simple linear image.
    let input_host: Vec<f32> = (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                let value = 1 + (x * 17 + y * 13 + (x * y) % 29) % 250;
                value as f32 / 251.0
            })
        })
        .collect();

    let expected = convolution_cpu(&input_host, width, height);

    let input_dev = DeviceBuffer::from_host(&stream, &input_host)?;
    // A missing output write must fail even when the CPU result is zero.
    let sentinel = vec![f32::NAN; width * height];
    let mut naive_output_dev = DeviceBuffer::from_host(&stream, &sentinel)?;
    let mut tiled_output_dev = DeviceBuffer::from_host(&stream, &sentinel)?;

    let module = unsafe { kernels::load(&ctx)? };

    let grid = (
        (width as u32).div_ceil(BLOCK),
        (height as u32).div_ceil(BLOCK),
    );
    let config = LaunchConfig2D::new(grid, (BLOCK, BLOCK), 0);

    let naive_launch = module.prepare_convolution_naive(config)?;
    let tiled_launch = module.prepare_convolution_tiled(config)?;

    module.convolution_naive(
        &stream,
        &naive_launch,
        width as u32,
        height as u32,
        &input_dev,
        cuda_host::RowWidth::new(&mut naive_output_dev, width as u32),
    )?;

    module.convolution_tiled(
        &stream,
        &tiled_launch,
        width as u32,
        height as u32,
        &input_dev,
        cuda_host::RowWidth::new(&mut tiled_output_dev, width as u32),
    )?;

    let naive_actual = naive_output_dev.to_host_vec(&stream)?;
    let tiled_actual = tiled_output_dev.to_host_vec(&stream)?;

    for i in 0..expected.len() {
        let naive_error = (naive_actual[i] - expected[i]).abs();
        assert!(
            naive_error <= 1e-6,
            "naive mismatch at pixel {} ({}, {}): expected {}, got {}, error {}",
            i,
            i % width,
            i / width,
            expected[i],
            naive_actual[i],
            naive_error
        );

        let tiled_error = (tiled_actual[i] - expected[i]).abs();
        assert!(
            tiled_error <= 1e-6,
            "tiled mismatch at pixel {} ({}, {}): expected {}, got {}, error {}",
            i,
            i % width,
            i / width,
            expected[i],
            tiled_actual[i],
            tiled_error
        );
    }

    println!("✓ naive convolution matches CPU reference");
    println!("✓ tiled convolution matches CPU reference");

    Ok(())
}

fn run_synthetic_validation() -> Result<(), Box<dyn std::error::Error>> {
    // Tiny images stress halos without any interior pixels; 16/17 cover exact
    // tiles and a one-pixel tail. The rectangular case leaves tails on both axes.
    for (width, height) in [
        (1, 1),
        (1, 17),
        (17, 1),
        (2, 3),
        (16, 16),
        (17, 17),
        (37, 23),
    ] {
        validate_shape(width, height)?;
    }
    println!("SUCCESS: image convolution verified.");
    Ok(())
}

fn run_image_demo(input_path: &str, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!();
    println!("=== Real Image Demo ===");
    println!("Input: {input_path}");

    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(input_path)?));
    let mut reader = decoder.read_info()?;
    let mut buffer = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buffer)?;
    let bytes = &buffer[..info.buffer_size()];

    if info.color_type != png::ColorType::Rgba || info.bit_depth != png::BitDepth::Eight {
        return Err(format!(
            "image demo currently expects an 8-bit RGBA PNG, got {:?} {:?}",
            info.color_type, info.bit_depth
        )
        .into());
    }

    let width = info.width;
    let height = info.height;
    let pixel_count = width as usize * height as usize;

    println!("Image: {width}x{height} RGBA8");
    println!("Applying {IMAGE_DEMO_PASSES} tiled 3x3 Gaussian passes on RGB channels...");

    let mut red = Vec::with_capacity(pixel_count);
    let mut green = Vec::with_capacity(pixel_count);
    let mut blue = Vec::with_capacity(pixel_count);
    let mut alpha = Vec::with_capacity(pixel_count);

    let (pixels, remainder) = bytes.as_chunks::<4>();
    debug_assert!(remainder.is_empty());

    for pixel in pixels {
        red.push(pixel[0] as f32 / 255.0);
        green.push(pixel[1] as f32 / 255.0);
        blue.push(pixel[2] as f32 / 255.0);
        alpha.push(pixel[3]);
    }

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = unsafe { kernels::load(&ctx)? };

    let grid = (width.div_ceil(BLOCK), height.div_ceil(BLOCK));
    let config = LaunchConfig2D::new(grid, (BLOCK, BLOCK), 0);
    let tiled_launch = module.prepare_convolution_tiled(config)?;

    let mut blurred_channels = Vec::with_capacity(3);

    for channel in [&red, &green, &blue] {
        let mut input_dev = DeviceBuffer::from_host(&stream, channel)?;
        let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, pixel_count)?;

        for _ in 0..IMAGE_DEMO_PASSES {
            module.convolution_tiled(
                &stream,
                &tiled_launch,
                width,
                height,
                &input_dev,
                cuda_host::RowWidth::new(&mut output_dev, width),
            )?;

            std::mem::swap(&mut input_dev, &mut output_dev);
        }

        blurred_channels.push(input_dev.to_host_vec(&stream)?);
    }

    let mut output_rgba = Vec::with_capacity(pixel_count * 4);

    for i in 0..pixel_count {
        let r = (blurred_channels[0][i].clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = (blurred_channels[1][i].clamp(0.0, 1.0) * 255.0).round() as u8;
        let b = (blurred_channels[2][i].clamp(0.0, 1.0) * 255.0).round() as u8;

        output_rgba.extend_from_slice(&[r, g, b, alpha[i]]);
    }

    let output_file = std::io::BufWriter::new(std::fs::File::create(output_path)?);
    let mut encoder = png::Encoder::new(output_file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);

    let mut writer = encoder.write_header()?;
    writer.write_image_data(&output_rgba)?;

    println!("Output: {output_path}");
    println!("SUCCESS: real image convolution completed.");

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);

    let mode = args.next();

    match mode.as_deref() {
        None => run_synthetic_validation(),
        Some("--image-demo") => {
            let input = args
                .next()
                .unwrap_or_else(|| "input/banner-dark.png".to_string());
            let output = args
                .next()
                .unwrap_or_else(|| "output/banner-dark-blurred.png".to_string());

            run_synthetic_validation()?;
            run_image_demo(&input, &output)
        }
        Some(other) => Err(format!(
            "unknown argument `{other}`; use no arguments or `--image-demo [input.png] [output.png]`"
        )
        .into()),
    }
}
