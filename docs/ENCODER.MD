# Fractal Video Codec: Encoder and Decoder Technical Specification

## 1. Encoder Technical Specification

### Core Encoding Pipeline

The Fractal Video Codec (FVC) Encoder is a stateful, GPU-accelerated
encoding system that transforms raw video frames into highly compressed
binary payloads.

The encoder operates through three primary stages:

1.  **Domain Pool Preparation**
2.  **Cooperative Quadtree Evaluation**
3.  **Bitstream Optimization and Packing**

------------------------------------------------------------------------

### 1.1 Domain Pool Preparation

Before the primary encoding kernel is launched, the encoder constructs a
reference library of spatial patterns used during fractal searches.

#### Downsampling

The current frame is partitioned into `8×8` pixel blocks. A dedicated
CUDA kernel, `build8x8DownsampledPoolKernel`, averages each `2×2`
quadrant within these blocks to generate a downsampled pool of domain
blocks.

This domain pool provides the reference regions used by the fractal
search performed during the base-level encoding stage.

#### CUDA Texture Binding

The current frame, previous frame, and downsampled domain pool are bound
to CUDA Texture Objects (`cudaTextureObject_t`).

This allows the codec to take advantage of hardware-accelerated 2D
spatial caching and efficient texture fetches during the computationally
intensive domain-search and temporal-comparison stages.

------------------------------------------------------------------------

### 1.2 Cooperative Quadtree Evaluation

The primary CUDA kernel, `hybridTemporalEncodeCooperativeQuadtree`, uses
cooperative thread-block execution to dynamically select an efficient
representation for each region of the frame.

The evaluation hierarchy progresses from `32×32` macroblocks down to
`2×2` microblocks.

#### 32×32 to 8×8: Temporal Skips and Motion Vectors

Threads within each CUDA block cooperatively calculate and share Sum of
Absolute Differences (SAD) data using CUDA `__shared__` memory.

For each `32×32`, `16×16`, and `8×8` region:

-   If the region has not changed significantly relative to the previous
    frame, the leader thread emits a **Temporal Skip** code (`0xFFFF`).
-   If the region exhibits limited translational movement, the encoder
    searches for and stores a rigid **Motion Vector** (`0xFFFE`).
-   If neither temporal representation satisfies the configured error
    threshold, evaluation proceeds to smaller blocks.

This hierarchical approach avoids performing expensive fractal searches
on regions that can be efficiently represented through temporal reuse.

#### 4×4 Base Level: Fractal vs. Residual Representation

When larger regions cannot satisfy the error threshold, individual
threads evaluate `4×4` blocks using two competing representations.

**Fractal Search**

Each thread compares the `4×4` range block against the `8×8` domain pool
across eight isometric transformations, including rotations and
reflections.

For each candidate domain, the encoder calculates the optimal contrast
and brightness transformation and evaluates the resulting approximation
error.

**Residual Search (`0xFFFD`)**

In parallel, the encoder evaluates a `4×4` motion-compensated residual
representation.

The process calculates:

-   A `4×4` motion vector.
-   The motion-compensated reference pixels.
-   The exact pixel-by-pixel difference between the current frame and
    the reference.

The encoder selects the representation that satisfies the configured
`BAILOUT_42DB_ERROR` threshold.

#### 2×2 Fallback: Edge Preservation

If a `4×4` residual contains pixel differences outside the representable
range of an 8-bit signed integer (`<-128` or `>127`), the residual
representation is considered unsuitable.

Such a condition can occur around sharp contrast transitions, geometric
silhouettes, or other high-frequency regions.

Instead of introducing clipping or visible artifacts, the encoder:

1.  Aborts the `4×4` residual representation.
2.  Splits the region into four `2×2` blocks.
3.  Stores the exact raw pixel values using 32-bit floating-point
    representation.

This fallback preserves high-contrast edges that cannot be accurately
represented by the compact residual format.

------------------------------------------------------------------------

### 1.3 Bitstream Optimization and Packing

After the GPU encoding kernel produces the raw `HybridCodeData`
structures, the C++ `FractalBitstreamProcessor` performs a second-stage
representation optimization.

#### Geometric Sorting

Encoded blocks are sorted in strict raster order:

-   Top-to-bottom.
-   Left-to-right.

This ordering improves spatial locality and enables more compact
coordinate representation.

#### 1D Delta Coordinates

Absolute X/Y coordinates are removed from the final representation.

Instead, the processor flattens the two-dimensional frame into a
one-dimensional coordinate space and stores the delta between the
current block and the previous block.

Small spatial jumps can therefore be represented using a compact
single-byte representation where applicable.

#### Binary Packing

A custom `BitWriter` packs fields with heterogeneous bit widths directly
into a contiguous bitstream.

The format supports fields such as:

-   4-bit quadtree depths.
-   8-bit isometry identifiers.
-   Truncated FP16 floating-point values.
-   Motion vectors.
-   Residual data.
-   Raw pixel fallback data.

Fields are packed continuously across byte boundaries to minimize
alignment overhead and maximize bitstream density.

#### Zstandard Compression

The resulting packed byte array is compressed using **Zstandard at
compression level 15**.

The final compressed payload is suitable for storage or network
transmission.
