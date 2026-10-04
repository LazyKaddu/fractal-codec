# Fractal Codec Architecture

## High-Level Overview
The Fractal Video Codec is designed to compress raw video frames using a highly parallelized GPU fractal compression algorithm, heavily utilizing CUDA, OpenCV, and Zstd. 

The system is separated into four main layers:
1. **CLI Layer (`src/main.cpp`)**: Uses `CLI11` to parse user arguments and direct traffic to either the Encoder or Decoder.
2. **Pipeline Layer (`VideoEncoder.cpp` / `VideoDecoder.cpp`)**: Written in standard C++, orchestrating file I/O (via `VideoReader`), color-space conversion (YUV), 4:2:0 chroma subsampling, and writing the custom `.frc` container format to disk.
3. **GPU Layer (`EncoderState.cu` / `DecoderState.cu`)**: Executes the massive parallel fractal math. Uses quad-tree partitioning, domain block downsampling, and motion compensation. Exposed to the C++ pipeline via a clean C-style API (`FractalCodec.h`).
4. **Data Packing Layer (`DataPacker.cpp`)**: Shrinks the raw GPU output structs down to half-precision floats, delta-encodes them, and heavily compresses them using Facebook's `Zstd` entropy encoder.

## The Data Flow (Encoding)
1. `.mp4` file is streamed via OpenCV without loading it all into RAM (`VideoReader`).
2. Frames are converted to YUV and split.
3. Cb and Cr channels are downscaled to 50% (4:2:0 Subsampling).
4. The 3 channels are pushed to the GPU `EncoderState`.
5. GPU outputs raw `HybridCodeData` structures.
6. `FractalBitstreamProcessor` bit-packs and `Zstd` compresses the structures.
7. `VideoEncoder` writes the channel sizes and compressed bytes into the `.frc` binary file.
