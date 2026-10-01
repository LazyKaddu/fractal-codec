#include "common.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>

// ============================================================
// Helper: Map Isometry
// ============================================================
__device__ __forceinline__ int getIsoPixel(int px, int py, int dim, int iso) {
    int nx = px, ny = py;
    switch(iso) {
        case 0: nx = px; ny = py; break;
        case 1: nx = py; ny = (dim-1)-px; break;
        case 2: nx = (dim-1)-px; ny = (dim-1)-py; break;
        case 3: nx = (dim-1)-py; ny = px; break;
        case 4: nx = (dim-1)-px; ny = py; break;
        case 5: nx = py; ny = px; break;
        case 6: nx = px; ny = (dim-1)-py; break;
        case 7: nx = (dim-1)-py; ny = (dim-1)-px; break;
    }
    return ny * dim + nx;
}

// ============================================================
// The Quadtree Temporal Decode Kernel
// ============================================================
__global__ void hybridTemporalDecodeQuadtreeKernel(
    const HybridCode* d_codes,
    const float* d_in,
    float* d_out,
    int total_codes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_codes) return;

    HybridCode code = d_codes[idx];
    int rx = code.x;
    int ry = code.y;

    // Calculate block size dynamically based on depth: 
    // 0=32, 1=16, 2=8, 3=4, 4=2
    int size = 32 >> code.depth; 

    // 1. The Dynamic Temporal Skip (Can be 32x32, 16x16, 8x8, or 4x4)
    if (code.domain_idx == 0xFFFF) {
        for (int py = 0; py < size; py++) {
            for (int px = 0; px < size; px++) {
                int p_idx = (ry + py) * IMAGE_WIDTH + (rx + px);
                d_out[p_idx] = d_in[p_idx];
            }
        }
        return;
    }

    // 2. Fractal Decode (Only executes at depth == 3, size == 4x4)
    if (code.depth == 3) {
        int dom_x = (code.domain_idx % (IMAGE_WIDTH / 8)) * 8;
        int dom_y = (code.domain_idx / (IMAGE_WIDTH / 8)) * 8;

        for (int p = 0; p < 16; p++) {
            int px = p % 4;
            int py = p / 4;

            int iso_p = getIsoPixel(px, py, 4, code.isometry_id);
            int iso_x = iso_p % 4;
            int iso_y = iso_p / 4;

            // Downsample the 8x8 domain to a single pixel (average 2x2 area)
            float avg = 0.0f;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    avg += d_in[(dom_y + iso_y * 2 + dy) * IMAGE_WIDTH + (dom_x + iso_x * 2 + dx)];
                }
            }
            avg *= 0.25f;

            float val = (code.contrast * avg) + code.brightness;
            d_out[(ry + py) * IMAGE_WIDTH + (rx + px)] = val;
        }
    }
    // 3. Raw Pixel Fallback (Executes at depth == 4, size == 2x2)
    else if (code.depth == 4) {
        d_out[(ry + 0) * IMAGE_WIDTH + (rx + 0)] = code.raw_pixels[0];
        d_out[(ry + 0) * IMAGE_WIDTH + (rx + 1)] = code.raw_pixels[1];
        d_out[(ry + 1) * IMAGE_WIDTH + (rx + 0)] = code.raw_pixels[2];
        d_out[(ry + 1) * IMAGE_WIDTH + (rx + 1)] = code.raw_pixels[3];
    }
}

// ============================================================
// Helpers: Load and Save
// ============================================================
bool loadRawImage(const char* filename, std::vector<float>& image_data) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) return false;
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    image_data.resize(IMAGE_WIDTH * IMAGE_HEIGHT);
    file.read(reinterpret_cast<char*>(image_data.data()), size);
    return true;
}

bool writePGM(const char* filename, const std::vector<float>& image_data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;
    file << "P5\n" << IMAGE_WIDTH << " " << IMAGE_HEIGHT << "\n255\n";
    for (float val : image_data) {
        float clamped = fminf(fmaxf(val, 0.0f), 1.0f);
        uint8_t pixel = static_cast<uint8_t>(clamped * 255.0f);
        file.write(reinterpret_cast<char*>(&pixel), 1);
    }
    return true;
}

// ============================================================
// Main Execution
// ============================================================
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./decoder <prev_frame.bin> <encoded_temporal.bin>\n";
        return 1;
    }

    const char* prev_file = argv[1];
    const char* codes_file = argv[2];
    int total_pixels = IMAGE_WIDTH * IMAGE_HEIGHT;

    // 1. Load the Previous Frame
    std::vector<float> h_prev_frame;
    if (!loadRawImage(prev_file, h_prev_frame)) {
        std::cerr << "Error loading previous frame.\n";
        return 1;
    }

    // 2. Load the Fractal Codes
    std::ifstream file(codes_file, std::ios::binary | std::ios::ate);
    if (!file) return 1;
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    int total_codes = size / sizeof(HybridCode);
    std::vector<HybridCode> h_codes(total_codes);
    file.read(reinterpret_cast<char*>(h_codes.data()), size);

    // 3. GPU Allocation
    HybridCode* d_codes;
    float *d_buffer_A, *d_buffer_B;

    cudaMalloc(&d_codes, total_codes * sizeof(HybridCode));
    cudaMalloc(&d_buffer_A, total_pixels * sizeof(float));
    cudaMalloc(&d_buffer_B, total_pixels * sizeof(float));

    cudaMemcpy(d_codes, h_codes.data(), total_codes * sizeof(HybridCode), cudaMemcpyHostToDevice);
    cudaMemcpy(d_buffer_A, h_prev_frame.data(), total_pixels * sizeof(float), cudaMemcpyHostToDevice);

    // 4. The Decoding Ping-Pong Loop
    int threads = 256;
    int blocks = (total_codes + threads - 1) / threads;
    
    // 3 iterations are plenty because the seed frame acts as a highly accurate baseline
    int iterations = 3; 

    std::cout << "Seeding decoder with " << prev_file << "...\n";
    std::cout << "Running " << iterations << " iterations...\n";

    for (int iter = 0; iter < iterations; iter++) {
        if (iter % 2 == 0) {
            hybridTemporalDecodeQuadtreeKernel<<<blocks, threads>>>(d_codes, d_buffer_A, d_buffer_B, total_codes);
        } else {
            hybridTemporalDecodeQuadtreeKernel<<<blocks, threads>>>(d_codes, d_buffer_B, d_buffer_A, total_codes);
        }
        cudaDeviceSynchronize();
    }

    // 5. Download the Final Image
    std::vector<float> h_final_image(total_pixels);
    if (iterations % 2 == 0) {
        cudaMemcpy(h_final_image.data(), d_buffer_A, total_pixels * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(h_final_image.data(), d_buffer_B, total_pixels * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Save as raw float array for the NEXT frame's input
    std::ofstream out_bin("decoded_current.bin", std::ios::binary);
    out_bin.write(reinterpret_cast<const char*>(h_final_image.data()), total_pixels * sizeof(float));

    // Save as PGM for your visual inspection
    if (writePGM("decoded_current.pgm", h_final_image)) {
        std::cout << "Success! Saved decoded_current.bin (for next frame) and decoded_current.pgm (for viewing).\n";
    }

    cudaFree(d_codes); cudaFree(d_buffer_A); cudaFree(d_buffer_B);
    return 0;
}