# Image Convolution

This example demonstrates 2D image convolution with two CUDA kernels:

- a naive implementation that reads the 3x3 neighborhood directly from global memory;
- a tiled implementation that cooperatively loads pixels and halo regions into shared memory.

Both kernels apply the following 3x3 Gaussian filter with zero-padded image boundaries:

```text
1 2 1
2 4 2   / 16
1 2 1
```

The tiled kernel uses 16x16 output tiles backed by an 18x18 shared-memory tile so that each block can reuse neighboring pixels while computing the convolution.

## Correctness validation

Running the example without arguments performs a deterministic correctness test:

```bash
cargo oxide run image_convolution
```

The test covers 1x1, 1x17, 17x1, 2x3, 16x16, 17x17, and 37x23 images. These include tiny images with no interior pixels, exact tiles, and partial blocks on both axes. Shared-memory staging uses raw element pointers, with a block barrier before neighboring threads read the tile.

Both GPU implementations are compared against a CPU reference implementation.

Expected output includes:

```text
✓ naive convolution matches CPU reference
✓ tiled convolution matches CPU reference
SUCCESS: image convolution verified.
```

## Real image demo

A sample RGBA PNG is included at:

```text
input/banner-dark.png
```

Run the real-image demo with:

```bash
cargo oxide run image_convolution -- --image-demo
```

This applies 128 tiled 3x3 Gaussian passes to the RGB channels while preserving the original alpha channel.

The generated image is written to:

```text
output/banner-dark-blurred.png
```

The repeated passes are used only to make the visual blur clearly noticeable on the large bundled image. The underlying CUDA operation remains the same tiled 3x3 convolution kernel.

Generated files under `output/` are ignored by Git.

## Using your own image

Place an 8-bit RGBA PNG in the example's `input/` directory, for example:

```text
input/photo.png
```

Then run:

```bash
cargo oxide run image_convolution -- --image-demo input/photo.png output/photo-blurred.png
```

You can also pass absolute input and output paths.

User-provided files under `input/` are ignored by Git; the bundled `banner-dark.png` is the only tracked sample image.

The image demo currently expects an 8-bit RGBA PNG.

## Concepts demonstrated

- 2D CUDA launch geometry
- runtime image dimensions
- row-major output views
- zero-padded boundary handling
- global-memory neighborhood reads
- shared-memory tiling
- halo loading
- cooperative block loading
- `sync_threads()` placement for partial blocks
- repeated GPU-only convolution passes using ping-pong device buffers
- CPU reference validation
- PNG input and output on the host
