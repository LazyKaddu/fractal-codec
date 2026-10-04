# Decoder Pipeline

## VideoDecoder (C++ Orchestrator)
The `VideoDecoder` class mirrors the Encoder precisely:
1. It validates the `.frc` file via the `FRAC` magic bytes.
2. It detects if the file was compressed using 4:2:0 subsampling. If so, it boots up the Cb and Cr CUDA decoders at half-resolution.
3. It streams the chunk sizes, extracts the `Zstd` buffers, and sends them to the GPU.
4. Once decoded, it uses `cv::INTER_CUBIC` to upscale the Cb and Cr channels back to 4:4:4.
5. It merges them back into `BGR` and writes the final output frame to an `.mp4` using `cv::VideoWriter`.

## DecoderState (CUDA Kernel)
The Decoder is highly parallelized and uses an iterative "Ping-Pong" buffer approach.
Because fractals are recursive, the decoder runs multiple iterations, ping-ponging the image data between `buffer_A` and `buffer_B` until the image mathematically converges on the original visual approximation.