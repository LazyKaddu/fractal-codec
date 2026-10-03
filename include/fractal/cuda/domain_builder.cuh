#pragma once
#include <fractal/core/common.h>
#include <cuda_runtime.h>

// ============================================================
// Domain Pool Builder (Hybrid Downsampling)
// ============================================================

// Downsamples 8x8 regions into 4x4 blocks (16 pixels)
__global__ void build8x8DownsampledPoolKernel(const float* d_img, float* d_pool, int width) {
    int dom_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dom_y = blockIdx.y * blockDim.y + threadIdx.y;
    int domains_per_row = width / 8;

    if (dom_x < domains_per_row && dom_y < (width / 8)) {
        int dom_idx = dom_y * domains_per_row + dom_x;
        int start_x = dom_x * 8;
        int start_y = dom_y * 8;

        for (int p = 0; p < 16; p++) {
            int px = p % 4;
            int py = p / 4;
            float avg = 0.0f;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    avg += d_img[(start_y + py * 2 + dy) * width + (start_x + px * 2 + dx)];
                }
            }
            d_pool[dom_idx * 16 + p] = avg * 0.25f;
        }
    }
}

// Downsamples 4x4 regions into 2x2 micro-blocks (4 pixels)
__global__ void build4x4DownsampledPoolKernel(const float* d_img, float* d_pool, int width) {
    int dom_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dom_y = blockIdx.y * blockDim.y + threadIdx.y;
    int domains_per_row = width / 4;

    if (dom_x < domains_per_row && dom_y < (width / 4)) {
        int dom_idx = dom_y * domains_per_row + dom_x;
        int start_x = dom_x * 4;
        int start_y = dom_y * 4;

        for (int p = 0; p < 4; p++) {
            int px = p % 2;
            int py = p / 2;
            float avg = 0.0f;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    avg += d_img[(start_y + py * 2 + dy) * width + (start_x + px * 2 + dx)];
                }
            }
            d_pool[dom_idx * 4 + p] = avg * 0.25f;
        }
    }
}

// Host function to allocate memory and launch the kernels
void buildDomainPools(const float* d_curr_frame, float** d_domains_8x8, float** d_domains_4x4) {
    int d8_count = (IMAGE_WIDTH / 8) * (IMAGE_HEIGHT / 8);
    int d4_count = (IMAGE_WIDTH / 4) * (IMAGE_HEIGHT / 4);

    cudaMalloc(d_domains_8x8, d8_count * 16 * sizeof(float));
    cudaMalloc(d_domains_4x4, d4_count * 4 * sizeof(float));

    dim3 block(16, 16);
    dim3 grid8((IMAGE_WIDTH / 8 + 15) / 16, (IMAGE_HEIGHT / 8 + 15) / 16);
    dim3 grid4((IMAGE_WIDTH / 4 + 15) / 16, (IMAGE_HEIGHT / 4 + 15) / 16);

    build8x8DownsampledPoolKernel<<<grid8, block>>>(d_curr_frame, *d_domains_8x8, IMAGE_WIDTH);
    build4x4DownsampledPoolKernel<<<grid4, block>>>(d_curr_frame, *d_domains_4x4, IMAGE_WIDTH);
    cudaDeviceSynchronize();
}