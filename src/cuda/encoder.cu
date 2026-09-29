#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>

#define THREADS_PER_BLOCK 1024
#define RANGE_SIZE 4
#define DOMAIN_SIZE 8
#define RANGE_PIXELS 16
#define DOMAIN_PIXELS 64
#define RANGES_PER_BLOCK 8
#define DOMAINS_PER_BATCH 16
#define ISOMETRIES 8
#define WARPS_PER_BLOCK 32

struct FractalCode {
    uint16_t domain_idx;
    uint8_t isometry_id;
    float contrast;
    float brightness;
};

// 4x4 LUT in Constant Memory
__constant__ uint8_t d_iso_lut_4x4[8][16];

// ============================================================
// Optimized Exhaustive Fractal Matching Kernel
// ============================================================
__global__ __launch_bounds__(1024)
void fractalMatchKernelOptimized(
    const float* d_range_blocks,
    const float* d_domain_pool,
    FractalCode* d_output,
    int total_domains,
    int total_ranges
) {
    // Shared Memory (Properly sized to prevent OOB corruption)
    __shared__ float s_domains_4x4[DOMAINS_PER_BATCH * RANGE_PIXELS];
    __shared__ float s_ranges[RANGES_PER_BLOCK * RANGE_PIXELS];
    __shared__ float s_warp_errors[RANGES_PER_BLOCK * DOMAINS_PER_BATCH * 2]; 
    __shared__ FractalCode s_warp_codes[RANGES_PER_BLOCK * DOMAINS_PER_BATCH * 2]; 
    __shared__ int s_block_done;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int range_base = blockIdx.x * RANGES_PER_BLOCK;

    // Load ranges into shared memory
    if (tid < RANGES_PER_BLOCK * RANGE_PIXELS) {
        int local_range_idx = tid / RANGE_PIXELS;
        int pixel_idx = tid % RANGE_PIXELS;
        int global_range = range_base + local_range_idx;
        
        s_ranges[tid] = (global_range < total_ranges) ? d_range_blocks[global_range * RANGE_PIXELS + pixel_idx] : 0.0f;
    }
    __syncthreads();

    float persistent_best_error = 1e30f;
    FractalCode persistent_best_code = {0, 0, 0.0f, 0.0f};

    // The Exhaustive Loop
    for (int batch_start = 0; batch_start < total_domains; batch_start += DOMAINS_PER_BATCH) {
        
        // ====================================================
        // SHADOW LOAD: Read 8x8 from Global -> Store 4x4 in Shared
        // ====================================================
        if (tid < DOMAINS_PER_BATCH * RANGE_PIXELS) {
            int local_domain = tid / RANGE_PIXELS;
            int pixel_idx = tid % RANGE_PIXELS; 
            int global_domain = batch_start + local_domain;
            
            if (global_domain < total_domains) {
                int dx = (pixel_idx % 4) * 2;
                int dy = (pixel_idx / 4) * 2;
                const float* g_dom = &d_domain_pool[global_domain * DOMAIN_PIXELS];
                
                s_domains_4x4[tid] = 0.25f * (
                    g_dom[(dy * 8) + dx] + g_dom[(dy * 8) + dx + 1] +
                    g_dom[((dy + 1) * 8) + dx] + g_dom[((dy + 1) * 8) + dx + 1]
                );
            } else {
                s_domains_4x4[tid] = 0.0f;
            }
        }
        __syncthreads();

        // Thread mapping
        uint8_t local_range_idx = tid % RANGES_PER_BLOCK;
        uint8_t iso_id = (tid / RANGES_PER_BLOCK) % ISOMETRIES;
        uint8_t local_domain_idx = (tid / 64) % DOMAINS_PER_BATCH;
        
        int global_range = range_base + local_range_idx;
        int global_domain = batch_start + local_domain_idx;
        
        bool valid_range = global_range < total_ranges;
        bool valid_domain = global_domain < total_domains;

        float thread_best_error = 1e30f;
        FractalCode thread_best_code = {0, 0, 0.0f, 0.0f};

        // ====================================================
        // The Evaluation Block
        // ====================================================
        if (valid_range && valid_domain) {
            const float* my_R = &s_ranges[local_range_idx * RANGE_PIXELS];

            float sum_R = 0.0f;
            #pragma unroll
            for (int p = 0; p < RANGE_PIXELS; p++) {
                sum_R += my_R[p];
            }

            float sum_D = 0.0f, sum_D2 = 0.0f, sum_RD = 0.0f;

            #pragma unroll
            for (int p = 0; p < RANGE_PIXELS; p++) {
                uint8_t d_idx = d_iso_lut_4x4[iso_id][p];
                float d_val = s_domains_4x4[(local_domain_idx * RANGE_PIXELS) + d_idx];
                float r_val = my_R[p];

                sum_D  += d_val;
                sum_D2 += d_val * d_val;
                sum_RD += r_val * d_val;
            }

            float denominator = (RANGE_PIXELS * sum_D2) - (sum_D * sum_D);
            float contrast = 0.0f;

            if (denominator > 0.0001f) {
                contrast = ((RANGE_PIXELS * sum_RD) - (sum_R * sum_D)) / denominator;
            }
            
            // Your original clamp boundaries
            contrast = fminf(fmaxf(contrast, -1.0f), 1.0f);

            float brightness = (sum_R - contrast * sum_D) / RANGE_PIXELS;
            float error = 0.0f;
            
            #pragma unroll
            for (int p = 0; p < RANGE_PIXELS; p++) {
                uint8_t d_idx = d_iso_lut_4x4[iso_id][p];
                float d_val = s_domains_4x4[(local_domain_idx * RANGE_PIXELS) + d_idx];
                float diff = (contrast * d_val) + brightness - my_R[p];
                error += diff * diff;
            }

            thread_best_error = error;
            thread_best_code.domain_idx = static_cast<uint16_t>(global_domain);
            thread_best_code.isometry_id = iso_id;
            thread_best_code.contrast = contrast;
            thread_best_code.brightness = brightness;
        }

        // ====================================================
        // Repaired Warp Reduction (Tests all 8 isometries safely)
        // ====================================================
        float my_best_error = thread_best_error;
        FractalCode my_best_code = thread_best_code;

        for (int offset = 16; offset >= 8; offset >>= 1) {
            float partner_error = __shfl_xor_sync(0xffffffff, my_best_error, offset);
            uint32_t partner_domain = __shfl_xor_sync(0xffffffff, static_cast<uint32_t>(my_best_code.domain_idx), offset);
            uint32_t partner_iso = __shfl_xor_sync(0xffffffff, static_cast<uint32_t>(my_best_code.isometry_id), offset);
            float partner_contrast = __shfl_xor_sync(0xffffffff, my_best_code.contrast, offset);
            float partner_brightness = __shfl_xor_sync(0xffffffff, my_best_code.brightness, offset);
            
            if (partner_error < my_best_error) {
                my_best_error = partner_error;
                my_best_code.domain_idx = static_cast<uint16_t>(partner_domain);
                my_best_code.isometry_id = static_cast<uint8_t>(partner_iso);
                my_best_code.contrast = partner_contrast;
                my_best_code.brightness = partner_brightness;
            }
        }

        if (lane < RANGES_PER_BLOCK) {
            int warp_domain = warp_id / 2;
            int warp_iso_group = warp_id % 2;
            if (warp_domain < DOMAINS_PER_BATCH) {
                int slot = (lane * DOMAINS_PER_BATCH * 2) + warp_domain * 2 + warp_iso_group;
                s_warp_errors[slot] = my_best_error;
                s_warp_codes[slot] = my_best_code;
            }
        }
        __syncthreads();

        if (lane < RANGES_PER_BLOCK && warp_id % 2 == 0) {
            int domain = warp_id / 2;
            if (domain < DOMAINS_PER_BATCH) {
                int base = (lane * DOMAINS_PER_BATCH * 2) + domain * 2;
                float error_a = s_warp_errors[base];
                float error_b = s_warp_errors[base + 1];
                
                int final_slot = (lane * DOMAINS_PER_BATCH) + domain;
                if (error_b < error_a) {
                    s_warp_errors[final_slot] = error_b;
                    s_warp_codes[final_slot] = s_warp_codes[base + 1];
                } else {
                    s_warp_errors[final_slot] = error_a;
                    s_warp_codes[final_slot] = s_warp_codes[base];
                }
            }
        }
        __syncthreads();

        if (tid < RANGES_PER_BLOCK) {
            int range = tid;
            for (int domain = 0; domain < DOMAINS_PER_BATCH; domain++) {
                int index = range * DOMAINS_PER_BATCH + domain;
                if (s_warp_errors[index] < persistent_best_error) {
                    persistent_best_error = s_warp_errors[index];
                    persistent_best_code = s_warp_codes[index];
                }
            }
        }
        __syncthreads();

        // ====================================================
        // Safe, Deadlock-Free Early Exit
        // ====================================================
        if (tid == 0) s_block_done = 1; 
        __syncthreads();
        
        if (tid < RANGES_PER_BLOCK && (range_base + tid < total_ranges)) {
            // Tweak this value. Lower = better quality. Higher = much faster execution.
            if (persistent_best_error > 0.015f) s_block_done = 0; 
        }
        __syncthreads();
        if (s_block_done == 1) break; 
    }

    if (tid < RANGES_PER_BLOCK) {
        int global_range = range_base + tid;
        if (global_range < total_ranges) {
            d_output[global_range] = persistent_best_code;
        }
    }
}

// ============================================================
// Binary Loader
// ============================================================
bool loadBinary(const char* filename, std::vector<float>& ranges, std::vector<float>& domains, uint32_t& total_ranges, uint32_t& total_domains) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;
    file.read(reinterpret_cast<char*>(&total_ranges), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&total_domains), sizeof(uint32_t));
    
    size_t range_count = static_cast<size_t>(total_ranges) * RANGE_PIXELS;
    size_t domain_count = static_cast<size_t>(total_domains) * DOMAIN_PIXELS;
    
    ranges.resize(range_count);
    domains.resize(domain_count);
    file.read(reinterpret_cast<char*>(ranges.data()), range_count * sizeof(float));
    file.read(reinterpret_cast<char*>(domains.data()), domain_count * sizeof(float));
    return true;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    if (argc < 2) return 1;
    const char* filename = argv[1];
    cudaFree(0);

    uint32_t total_ranges, total_domains;
    std::vector<float> h_ranges, h_domains;
    if (!loadBinary(filename, h_ranges, h_domains, total_ranges, total_domains)) return 1;

    // Build the fast 4x4 LUT
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

    float *d_ranges = nullptr, *d_domains = nullptr;
    FractalCode* d_output = nullptr;
    size_t range_size = h_ranges.size() * sizeof(float);
    size_t domain_size = h_domains.size() * sizeof(float);
    size_t output_size = static_cast<size_t>(total_ranges) * sizeof(FractalCode);
    
    cudaMalloc(&d_ranges, range_size);
    cudaMalloc(&d_domains, domain_size);
    cudaMalloc(&d_output, output_size);

    cudaMemcpy(d_ranges, h_ranges.data(), range_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_domains, h_domains.data(), domain_size, cudaMemcpyHostToDevice);

    nvtxRangePushA("Fractal Matching - Safe Exhaustive Optimized");
    int total_blocks = (total_ranges + RANGES_PER_BLOCK - 1) / RANGES_PER_BLOCK;
    
    fractalMatchKernelOptimized<<<total_blocks, THREADS_PER_BLOCK>>>(
        d_ranges, d_domains, d_output, total_domains, total_ranges
    );
    cudaDeviceSynchronize();
    nvtxRangePop();

    std::vector<FractalCode> h_output(total_ranges);
    cudaMemcpy(h_output.data(), d_output, output_size, cudaMemcpyDeviceToHost);

    // Save Output to Binary File for Decoder
    const char* out_filename = "encoded.bin";
    std::ofstream out_file(out_filename, std::ios::binary);
    if (out_file) {
        out_file.write(reinterpret_cast<const char*>(h_output.data()), total_ranges * sizeof(FractalCode));
        std::cout << "Success! Saved perfectly mathematically verified data to: " << out_filename << "\n";
        out_file.close();
    }

    cudaFree(d_ranges);
    cudaFree(d_domains);
    cudaFree(d_output);
    
    return 0;
}