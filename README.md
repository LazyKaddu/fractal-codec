# FractalCodec

> **A modern, GPU-accelerated video codec built from first principles using CUDA and C++17, replacing Discrete Cosine Transforms with Partitioned Iterated Function Systems (PIFS).**

[![CI Build](https://github.com/your-username/fractal-codec/actions/workflows/cmake-build-test.yml/badge.svg)](https://github.com/your-username/fractal-codec/actions)
[![CUDA Version](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Modern video compression standards such as **H.264, HEVC, and AV1** rely on Discrete Cosine Transforms (DCT) to discard imperceptible high-frequency details. While effective for everyday streaming, DCT-based codecs can suffer from severe structural degradation—known as **macroblocking**—under ultra-low bitrates and are fundamentally bound to fixed pixel grids.

**FractalCodec** revives and modernizes **Fractal Image and Video Compression**, an approach largely abandoned in the late 1990s due to CPU computational limitations.

Instead of storing discrete pixel values or frequency coefficients, FractalCodec encodes video frames as **Iterated Function Systems (IFS)**—sets of contractive affine transformations that mathematically generate the target imagery.

By offloading the exhaustive, high-dimensional block-matching search to thousands of parallel CUDA threads and utilizing on-chip shared memory hierarchies, FractalCodec aims to achieve near-real-time encoding of mathematically self-similar video streams.

---

## Key Features

### Resolution Independence — Infinite Zoom

Because the compressed stream is composed of mathematical functions rather than raster pixels, frames can be decoded at resolutions higher than the source input without conventional bilinear blur.

Edge fidelity is synthesized natively through fractal reconstruction.

### Zero Macroblocking Artifacts

Under aggressive compression ratios such as **500:1+**, the codec is designed to avoid conventional square macroblock boundaries and instead degrade toward smooth, continuous contours.

### Massively Parallel CUDA Acceleration

The historically expensive **O(N²)** encoding bottleneck is mapped onto thousands of CUDA threads.

Domain-range block matching is parallelized across GPU thread blocks with parallel reduction strategies used to identify the best candidate transformations.

### Memory-Safe Architecture

The host architecture is written in **C++17** and follows RAII principles.

A custom:

```cpp
CudaBuffer<T>
```

abstraction manages GPU allocations and releases resources automatically, reducing the risk of GPU memory leaks and double frees.

### Asymmetric Decoding

Decoding is designed to be significantly cheaper than encoding.

A compressed frame can be reconstructed iteratively, starting from an initial image and applying the stored transformations for several passes.

---

## How It Works: Partitioned Iterated Function Systems (PIFS)

The codec treats every video frame as the attractor of an **Iterated Function System**, based on the contraction principle described by the Banach Fixed-Point Theorem.

### 1. Partitioning

The target frame is partitioned into small, non-overlapping **Range Blocks**:

```text
Rᵢ = 4 × 4 pixels
```

Each range block represents a region that needs to be reconstructed.

### 2. Domain Pooling

A larger set of overlapping **Domain Blocks** is extracted from the same frame or, depending on the encoding mode, from preceding keyframes.

Example:

```text
Dⱼ = 8 × 8 pixels
```

Domain blocks provide candidate source regions that can be transformed into the corresponding range blocks.

### 3. Parallel Search — CUDA

For every Range Block, CUDA threads concurrently evaluate candidate Domain Blocks.

Each candidate can be evaluated under multiple geometric isometries, such as:

* Identity
* Horizontal reflection
* Vertical reflection
* 180° rotation
* 90° rotations
* Reflected rotations

This allows the encoder to search for self-similar structures throughout the image.

### 4. Least-Squares Optimization

For each candidate domain block, the encoder solves for the optimal **contrast** and **brightness** parameters:

* `s` — contrast/scaling coefficient
* `o` — brightness offset

The objective is to minimize the reconstruction error:

$$
\min_{s,o}
\left\|
R_i - \left(s \cdot \phi(D_j) + o\right)
\right\|^2
$$

where:

* $R_i$ is the target Range Block.
* $D_j$ is a candidate Domain Block.
* $\phi(D_j)$ represents a geometric transformation of the Domain Block.
* $s$ controls contrast.
* $o$ controls brightness.

### 5. Serialization

After the best candidate has been identified, the encoder stores only the parameters required to reconstruct the Range Block.

A `.frac` stream therefore contains information such as:

```text
Domain X/Y Offset
Isometry / Transformation Flag
Contrast (s)
Brightness (o)
Range Block Position
```

Conceptually:

```text
Range Block
     │
     ▼
Best Domain Block
     │
     ├── Spatial Offset
     ├── Isometry
     ├── Contrast
     └── Brightness
```

---

## Repository Architecture

```text
fractal-codec/
├── include/
│   └── fractal/
│       ├── core/
│       │   └── # Encoder, Decoder, and Fractal math data structures
│       │
│       ├── cuda/
│       │   └── # RAII VRAM wrappers and CUDA error checkers
│       │
│       └── io/
│           └── # Video container demuxers and frame extraction
│
├── src/
│   ├── core/
│   │   └── # Host-side orchestration logic
│   │
│   ├── cuda/
│   │   └── # Parallel search and reduction kernels (.cu)
│   │
│   └── main.cpp
│       └── # CLI entry point
│
├── tests/
│   └── # GoogleTest test suite with GPU auto-detection fixtures
│
├── CMakeLists.txt
│   └── # Target-based CMake configuration for NVCC and CXX
│
└── CONTRIBUTING.md
    └── # Contribution workflow, clang-format, and testing rules
```

---

## Prerequisites

### Hardware

* NVIDIA GPU
* Compute Capability **7.5+**
* Supported architectures include:

  * Turing
  * Ampere
  * Ada Lovelace
  * Hopper
  * Blackwell

### Software

| Dependency          | Requirement |
| ------------------- | ----------- |
| NVIDIA CUDA Toolkit | 11.8+       |
| Recommended CUDA    | 12.x        |
| C++ Standard        | C++17       |
| GCC                 | 9+          |
| Clang               | 11+         |
| MSVC                | 2019+       |
| CMake               | 3.21+       |
| OpenCV              | Optional    |

OpenCV is used for direct ingestion of video formats such as:

```text
.mp4
.mkv
```

---

# Quickstart & Build Instructions

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/fractal-codec.git
cd fractal-codec
```

## 2. Configure with CMake

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

## 3. Build the Project

### Linux

```bash
cmake --build build --config Release -j$(nproc)
```

### Windows

```powershell
cmake --build build --config Release
```

## 4. Run the Test Suite

The automated test suite uses **GoogleTest**.

GPU-dependent tests can detect the availability of NVIDIA hardware and safely skip GPU-specific assertions when an appropriate GPU is unavailable.

```bash
ctest --test-dir build --output-on-failure
```

---

# Usage

## Encode a Video

Compress an input video into a FractalCodec bitstream:

```bash
./build/fractal_cli \
    --encode \
    --input sample.mp4 \
    --output encoded.frac \
    --range-size 4 \
    --domain-size 8
```

### Parameters

| Parameter       | Description             |
| --------------- | ----------------------- |
| `--encode`      | Enable encoding mode    |
| `--input`       | Input video             |
| `--output`      | Output `.frac` stream   |
| `--range-size`  | Range block dimensions  |
| `--domain-size` | Domain block dimensions |

---

## Decode with Custom Scaling

Decode an encoded bitstream at double the original resolution:

```bash
./build/fractal_cli \
    --decode \
    --input encoded.frac \
    --output reconstructed.mp4 \
    --scale 2.0 \
    --iterations 8
```

### Parameters

| Parameter      | Description                                 |
| -------------- | ------------------------------------------- |
| `--decode`     | Enable decoding mode                        |
| `--input`      | Input `.frac` stream                        |
| `--output`     | Reconstructed video                         |
| `--scale`      | Output resolution scale                     |
| `--iterations` | Number of fractal reconstruction iterations |

---

# Target Use Cases

## Low-Bandwidth Telemetry

Potential applications include:

* Remote sensing
* Deep-sea communications
* Space communications
* Other bandwidth-constrained environments

where maintaining structural information under extreme compression is important.

## Medical & Topographical Imaging

Potential applications include imagery such as:

* MRI
* CT
* Satellite imagery
* Topographical data

where smooth gradients and high-resolution inspection are important.

## Procedural Engine Textures

Fractal representations may be useful for extremely compact procedural assets in real-time 3D engines, potentially reducing the storage requirements of texture and video assets.

---

# Contributing

Contributions are welcome!

Please review [`CONTRIBUTING.md`](CONTRIBUTING.md) for information about:

* Coding standards
* `clang-format` configuration
* Branch naming conventions
* Commit conventions
* Testing requirements
* Pull request workflow

---

# License

FractalCodec is distributed under the **MIT License**.

See [`LICENSE`](LICENSE) for the complete license text.

---

## Project Status

FractalCodec is an experimental research-oriented codec exploring the use of **Partitioned Iterated Function Systems (PIFS)** and GPU-accelerated fractal block matching as an alternative approach to conventional transform-based video compression.

The project is intended to investigate the practical trade-offs between:

```text
Compression Ratio
        │
        ├── Encoding Complexity
        │
        ├── Decoding Complexity
        │
        ├── Reconstruction Quality
        │
        └── GPU Acceleration
```

rather than serving as a drop-in replacement for established codecs such as H.264, HEVC, or AV1.
