#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>

// ============================================================
// Configuration
// ============================================================
#define IMAGE_WIDTH 1024
#define IMAGE_HEIGHT 1024
#define RANGE_SIZE 4
#define DOMAIN_SIZE 8

// CRITICAL: Set this to match your CPU domain extractor!
// If you stepped by 8 pixels, STRIDE is 8. 
// If your domains overlapped by 4 pixels, STRIDE is 4.
#define DOMAIN_STRIDE 8 

const int RANGES_PER_ROW = IMAGE_WIDTH / RANGE_SIZE;
const int DOMAINS_PER_ROW = (IMAGE_WIDTH - DOMAIN_SIZE) / DOMAIN_STRIDE + 1;
const int TOTAL_RANGES = (IMAGE_WIDTH * IMAGE_HEIGHT) / (RANGE_SIZE * RANGE_SIZE);
const int TOTAL_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;

struct FractalCode {
    uint16_t domain_idx;
    uint8_t isometry_id;
    float contrast;
    float brightness;
};

__constant__ uint8_t d_iso_lut_4x4[8][16];

// ============================================================
// The Fast Decoder Kernel
// ============================================================
__global__ void decodeFractalKernel(
    const FractalCode* d_codes,
    const float* d_image_in,
    float* d_image_out,
    int total_ranges
) {
    int range_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (range_idx >= total_ranges) return;

    // Read the instruction for this specific 4x4 block
    FractalCode code = d_codes[range_idx];

    // Map the 1D Range Index back to 2D Image Coordinates
    int range_x = (range_idx % RANGES_PER_ROW) * RANGE_SIZE;
    int range_y = (range_idx / RANGES_PER_ROW) * RANGE_SIZE;

    // Map the 1D Domain Index back to 2D Image Coordinates
    int domain_x = (code.domain_idx % DOMAINS_PER_ROW) * DOMAIN_STRIDE;
    int domain_y = (code.domain_idx / DOMAINS_PER_ROW) * DOMAIN_STRIDE;

    // Read the 8x8 domain from the source image, downsample, and apply transform
    float downsampled_domain[16];

    #pragma unroll
    for (int py = 0; py < 4; py++) {
        #pragma unroll
        for (int px = 0; px < 4; px++) {
            int dx = domain_x + (px * 2);
            int dy = domain_y + (py * 2);

            // Average the 2x2 pixels safely
            float avg = 0.25f * (
                d_image_in[(dy * IMAGE_WIDTH) + dx] +
                d_image_in[(dy * IMAGE_WIDTH) + dx + 1] +
                d_image_in[((dy + 1) * IMAGE_WIDTH) + dx] +
                d_image_in[((dy + 1) * IMAGE_WIDTH) + dx + 1]
            );

            // Route it through the LUT instantly
            uint8_t target_idx = d_iso_lut_4x4[code.isometry_id][py * 4 + px];
            downsampled_domain[target_idx] = avg;
        }
    }

    // Apply the Least-Squares Math and write to the new image buffer
    #pragma unroll
    for (int p = 0; p < 16; p++) {
        int out_x = range_x + (p % 4);
        int out_y = range_y + (p / 4);

        float pixel_val = (code.contrast * downsampled_domain[p]) + code.brightness;
        
        // Clamp to valid 0.0 - 1.0 range
        pixel_val = fmaxf(0.0f, fminf(1.0f, pixel_val));

        d_image_out[(out_y * IMAGE_WIDTH) + out_x] = pixel_val;
    }
}

// ============================================================
// Simple Image Writer (Portable Gray Map)
// ============================================================
bool writePGM(const char* filename, const std::vector<float>& image_data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;

    // PGM Header: Magic Number, Width, Height, Max Value
    file << "P5\n" << IMAGE_WIDTH << " " << IMAGE_HEIGHT << "\n255\n";

    // Convert 0.0-1.0 floats back to 0-255 bytes
    std::vector<uint8_t> byte_data(TOTAL_PIXELS);
    for (int i = 0; i < TOTAL_PIXELS; i++) {
        byte_data[i] = static_cast<uint8_t>(image_data[i] * 255.0f);
    }

    file.write(reinterpret_cast<char*>(byte_data.data()), TOTAL_PIXELS);
    return true;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./decoder <encoded_codes.bin>\n";
        return 1;
    }
    const char* filename = argv[1];

    // 1. Read the encoded binary file
    std::vector<FractalCode> h_codes(TOTAL_RANGES);
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open encoded file: " << filename << "\n";
        return 1;
    }
    
    // Read the exact number of bytes for our array of structs
    file.read(reinterpret_cast<char*>(h_codes.data()), TOTAL_RANGES * sizeof(FractalCode));
    if (!file) {
        std::cerr << "Warning: Could not read full 65,536 structs. File might be too small.\n";
    }
    std::cout << "Loaded " << TOTAL_RANGES << " fractal codes from " << filename << "\n";

    // 2. Initialize the Constant Memory LUT
    uint8_t h_iso_lut_4x4[8][16];
    for (int iso = 0; iso < 8; iso++) {
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
                int new_x = x, new_y = y;
                switch (iso) {
                    case 0: break;
                    case 1: new_x = y; new_y = 3 - x; break;
                    case 2: new_x = 3 - x; new_y = 3 - y; break;
                    case 3: new_x = 3 - y; new_y = x; break;
                    case 4: new_x = 3 - x; new_y = y; break;
                    case 5: new_x = y; new_y = x; break;
                    case 6: new_x = x; new_y = 3 - y; break;
                    case 7: new_x = 3 - y; new_y = 3 - x; break;
                }
                h_iso_lut_4x4[iso][y * 4 + x] = static_cast<uint8_t>(new_y * 4 + new_x);
            }
        }
    }
    cudaMemcpyToSymbol(d_iso_lut_4x4, h_iso_lut_4x4, sizeof(h_iso_lut_4x4));

    // 3. GPU Memory Allocation
    FractalCode* d_codes;
    float *d_buffer_A, *d_buffer_B;
    
    cudaMalloc(&d_codes, TOTAL_RANGES * sizeof(FractalCode));
    cudaMalloc(&d_buffer_A, TOTAL_PIXELS * sizeof(float));
    cudaMalloc(&d_buffer_B, TOTAL_PIXELS * sizeof(float));

    cudaMemcpy(d_codes, h_codes.data(), TOTAL_RANGES * sizeof(FractalCode), cudaMemcpyHostToDevice);

    // Initialize Buffer A with mid-gray (0.5f) to start the fractal generation
    std::vector<float> initial_noise(TOTAL_PIXELS, 0.5f);
    cudaMemcpy(d_buffer_A, initial_noise.data(), TOTAL_PIXELS * sizeof(float), cudaMemcpyHostToDevice);

    // 4. The Decoding Ping-Pong Loop
    int threads = 256;
    int blocks = (TOTAL_RANGES + threads - 1) / threads;

    std::cout << "Starting 8-iteration GPU decoding sequence...\n";
    for (int iter = 0; iter < 16; iter++) {
        if (iter % 2 == 0) {
            decodeFractalKernel<<<blocks, threads>>>(d_codes, d_buffer_A, d_buffer_B, TOTAL_RANGES);
        } else {
            decodeFractalKernel<<<blocks, threads>>>(d_codes, d_buffer_B, d_buffer_A, TOTAL_RANGES);
        }
        cudaDeviceSynchronize();
        std::cout << "  Iteration " << iter + 1 << " complete.\n";
    }

    // 5. Download the final image
    // Because we run 8 iterations (an even number), the final result lands back in d_buffer_A
    std::vector<float> h_final_image(TOTAL_PIXELS);
    cudaMemcpy(h_final_image.data(), d_buffer_A, TOTAL_PIXELS * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. Save to disk
    const char* out_file = "decoded_output.pgm";
    if (writePGM(out_file, h_final_image)) {
        std::cout << "Success! Decoded image saved as: " << out_file << "\n";
    } else {
        std::cerr << "Failed to write image file.\n";
    }

    // Cleanup
    cudaFree(d_codes);
    cudaFree(d_buffer_A);
    cudaFree(d_buffer_B);

    return 0;
}