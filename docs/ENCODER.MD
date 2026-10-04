# Encoder Pipeline

## VideoEncoder (C++ Orchestrator)
The `VideoEncoder` class is responsible for the top-level orchestration of the encoding process.
1. It reads the global metadata from the video (Width, Height, FPS, Frames) using the `VideoReader`.
2. It writes the `FRAC` magic byte header and metadata to the output binary stream.
3. It performs **4:2:0 Chroma Subsampling** to save 50% of the data by downscaling the Cb and Cr channels.
4. It normalizes the pixel data into floats and calls `EncodeFractalFrame()` on the 3 respective channels.
5. It writes the exact sizes of the returned payloads and the payloads themselves to disk.

## EncoderState (CUDA Kernel)
The actual heavy lifting is done in `EncoderState.cu` on the GPU.
1. **Domain Pools**: It dynamically scales down the previous frame to create 8x8 and 4x4 domain block pools.
2. **Motion Search**: It checks if a block can be predicted entirely from the previous frame. If so, it issues an extremely cheap "Temporal Skip" code.
3. **Cooperative Quadtree**: If the block changed, threads cooperate to find the best fractal transform (contrast/brightness adjustment against a domain block).
4. **Residual Fallback**: If the contrast is too extreme to model, it falls back to capturing the raw residual differences.
