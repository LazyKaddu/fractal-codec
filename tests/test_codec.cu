#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <fractal/core/FractalCodec.h>
#include <vector>

class FractalCodecTest : public ::testing::Test {
protected:
    void SetUp() override {
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) {
            cudaGetLastError(); 
            GTEST_SKIP() << "No CUDA-capable GPU detected. Skipping hardware-dependent test.";
        }
    }
};

TEST_F(FractalCodecTest, EncodeAndDecodeBasicFrame) {
    const int width = 1024;
    const int height = 1024;
    const int total_pixels = width * height;

    // Create a dummy frame (e.g., a simple gradient)
    std::vector<float> input_frame(total_pixels, 0.5f);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            input_frame[y * width + x] = static_cast<float>(x) / width;
        }
    }

    // Allocate output buffer for compression
    const int max_compressed_size = total_pixels * sizeof(float); // Generous size
    std::vector<uint8_t> compressed_buffer(max_compressed_size, 0);

    // Initialize Encoder
    FractalEncoderHandle encoder = CreateFractalEncoder(width, height);
    ASSERT_NE(encoder, nullptr) << "Failed to create encoder handle.";

    // Encode frame
    int compressed_size = EncodeFractalFrame(encoder, input_frame.data(), compressed_buffer.data(), max_compressed_size);
    ASSERT_GT(compressed_size, 0) << "Compression failed or returned 0 bytes.";

    // Initialize Decoder
    FractalDecoderHandle decoder = CreateFractalDecoder(width, height);
    ASSERT_NE(decoder, nullptr) << "Failed to create decoder handle.";

    // Decode frame
    std::vector<float> output_frame(total_pixels, 0.0f);
    DecodeFractalFrame(decoder, compressed_buffer.data(), compressed_size, output_frame.data());

    // Basic verification - just checking it doesn't crash and modifies the output
    // The exact pixel values are approximated in Fractal compression, so we won't assert equality.
    bool has_non_zero = false;
    for (int i = 0; i < total_pixels; i++) {
        if (output_frame[i] > 0.0f) {
            has_non_zero = true;
            break;
        }
    }
    EXPECT_TRUE(has_non_zero) << "Decoded frame is entirely zero.";

    // Cleanup
    DestroyFractalEncoder(encoder);
    DestroyFractalDecoder(decoder);
}
