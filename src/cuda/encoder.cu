#include "common.h"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>

const float BAILOUT_42DB_ERROR = 0.001009f;

// ============================================================
// Domain Pool Builder
// ============================================================
__global__ void build8x8DownsampledPoolKernel(const float* d_img, float* d_pool, int width) {
    int dom_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dom_y = blockIdx.y * blockDim.y + threadIdx.y;
    int domains_per_row = width / 8;
    if (dom_x < domains_per_row && dom_y < (width / 8)) {
        int dom_idx = dom_y * domains_per_row + dom_x;
        int start_x = dom_x * 8; int start_y = dom_y * 8;
        for (int p = 0; p < 16; p++) {
            float avg = 0.0f;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    avg += d_img[(start_y + (p/4) * 2 + dy) * width + (start_x + (p%4) * 2 + dx)];
                }
            }
            d_pool[dom_idx * 16 + p] = avg * 0.25f;
        }
    }
}

void buildDomainPools(const float* d_curr_frame, float** d_domains_8x8) {
    int d8_count = (IMAGE_WIDTH / 8) * (IMAGE_HEIGHT / 8);
    cudaMalloc(d_domains_8x8, d8_count * 16 * sizeof(float));
    dim3 block(16, 16);
    dim3 grid8((IMAGE_WIDTH / 8 + 15) / 16, (IMAGE_HEIGHT / 8 + 15) / 16);
    build8x8DownsampledPoolKernel<<<grid8, block>>>(d_curr_frame, *d_domains_8x8, IMAGE_WIDTH);
    cudaDeviceSynchronize();
}

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
// Cooperative Quadtree Kernel (64 Threads = One 32x32 Block)
// ============================================================
__global__ void hybridTemporalEncodeCooperativeQuadtree(
    cudaTextureObject_t tex_curr,
    cudaTextureObject_t tex_prev,
    cudaTextureObject_t tex_dom8,
    HybridCode* d_output_codes,
    int* d_output_counter,
    int total_domains_4x4_pass
) {
    // Shared memory to allow threads to vote on quadtree skips
    __shared__ float s_diff[64];
    __shared__ float s_flags_16[4];
    __shared__ float s_flags_8[16];

    int tx = threadIdx.x; // 0 to 7
    int ty = threadIdx.y; // 0 to 7
    int tid = ty * 8 + tx;

    // Base coordinates for the 32x32 macroblock
    int bx = blockIdx.x * 32;
    int by = blockIdx.y * 32;

    // Base coordinates for this specific thread's 4x4 block
    int rx = bx + tx * 4;
    int ry = by + ty * 4;

    if (rx >= IMAGE_WIDTH || ry >= IMAGE_HEIGHT) return;

    // 1. Calculate the 4x4 temporal difference
    float my_diff = 0.0f;
    for (int py = 0; py < 4; py++) {
        for (int px = 0; px < 4; px++) {
            int g_idx = (ry + py) * IMAGE_WIDTH + (rx + px);
            my_diff += fabsf(tex1Dfetch<float>(tex_curr, g_idx) - tex1Dfetch<float>(tex_prev, g_idx));
        }
    }
    s_diff[tid] = my_diff;
    __syncthreads();

    // ==========================================
    // 32x32 SKIP CHECK
    // ==========================================
    if (tid == 0) {
        float total_32 = 0;
        for (int i=0; i<64; i++) total_32 += s_diff[i];
        
        if (total_32 < TEMPORAL_SKIP_THRESHOLD * 64.0f) {
            int out_idx = atomicAdd(d_output_counter, 1);
            d_output_codes[out_idx] = {(uint16_t)bx, (uint16_t)by, 0xFFFF, 0, 0, 0, 0, {0}};
            s_diff[0] = -1.0f; // Flag to exit all threads
        }
    }
    __syncthreads();
    if (s_diff[0] < 0.0f) return;

    // ==========================================
    // 16x16 SKIP CHECK
    // ==========================================
    int q_idx = (ty / 4) * 2 + (tx / 4); 
    int leader_16 = (ty / 4) * 32 + (tx / 4) * 4; 
    
    if (tid == leader_16) {
        float total_16 = 0;
        for (int dy=0; dy<4; dy++) {
            for (int dx=0; dx<4; dx++) {
                total_16 += s_diff[(ty/4*4 + dy)*8 + (tx/4*4 + dx)];
            }
        }
        if (total_16 < TEMPORAL_SKIP_THRESHOLD * 16.0f) {
            int out_idx = atomicAdd(d_output_counter, 1);
            d_output_codes[out_idx] = {(uint16_t)(bx + (tx/4)*16), (uint16_t)(by + (ty/4)*16), 0xFFFF, 1, 0, 0, 0, {0}};
            s_flags_16[q_idx] = -1.0f;
        } else {
            s_flags_16[q_idx] = 1.0f;
        }
    }
    __syncthreads();
    if (s_flags_16[q_idx] < 0.0f) return;

    // ==========================================
    // 8x8 SKIP CHECK
    // ==========================================
    int e_idx = (ty / 2) * 4 + (tx / 2);
    int leader_8 = (ty / 2) * 16 + (tx / 2) * 2;
    
    if (tid == leader_8) {
        float total_8 = 0;
        for (int dy=0; dy<2; dy++) {
            for (int dx=0; dx<2; dx++) {
                total_8 += s_diff[(ty/2*2 + dy)*8 + (tx/2*2 + dx)];
            }
        }
        if (total_8 < TEMPORAL_SKIP_THRESHOLD * 4.0f) {
            int out_idx = atomicAdd(d_output_counter, 1);
            d_output_codes[out_idx] = {(uint16_t)(bx + (tx/2)*8), (uint16_t)(by + (ty/2)*8), 0xFFFF, 2, 0, 0, 0, {0}};
            s_flags_8[e_idx] = -1.0f;
        } else {
            s_flags_8[e_idx] = 1.0f;
        }
    }
    __syncthreads();
    if (s_flags_8[e_idx] < 0.0f) return;

    // ==========================================
    // 4x4 SKIP CHECK
    // ==========================================
    if (my_diff < TEMPORAL_SKIP_THRESHOLD) {
        int out_idx = atomicAdd(d_output_counter, 1);
        d_output_codes[out_idx] = {(uint16_t)rx, (uint16_t)ry, 0xFFFF, 3, 0, 0, 0, {0}};
        return;
    }

    // ==========================================
    // PARALLEL 4x4 FRACTAL SEARCH
    // ==========================================
    float best_error = 1e30f;
    HybridCode best_code = {0};
    float R[16];
    float sum_R = 0.0f;

    for (int i = 0; i < 16; i++) {
        R[i] = tex1Dfetch<float>(tex_curr, (ry + (i/4)) * IMAGE_WIDTH + (rx + (i%4)));
        sum_R += R[i];
    }

    for (int dom_idx = 0; dom_idx < total_domains_4x4_pass; dom_idx += 2) {
        for (int iso = 0; iso < 8; iso+=2) {
            float sum_D = 0.0f, sum_D2 = 0.0f, sum_RD = 0.0f;
            for (int p = 0; p < 16; p++) {
                int mapped = getIsoPixel(p%4, p/4, 4, iso);
                float d_val = tex1Dfetch<float>(tex_dom8, (dom_idx * 16) + mapped);
                sum_D += d_val; sum_D2 += d_val * d_val; sum_RD += R[p] * d_val;
            }

            float den = (16.0f * sum_D2) - (sum_D * sum_D);
            float contrast = (den > 0.0001f) ? ((16.0f * sum_RD) - (sum_R * sum_D)) / den : 0.0f;
            contrast = fminf(fmaxf(contrast, -0.99f), 0.99f);
            float brightness = (sum_R - contrast * sum_D) / 16.0f;

            float error = 0.0f;
            float max_px_error = 0.0f;
            for (int p = 0; p < 16; p++) {
                int mapped = getIsoPixel(p%4, p/4, 4, iso);
                float diff = fabsf((contrast * tex1Dfetch<float>(tex_dom8, (dom_idx * 16) + mapped)) + brightness - R[p]);
                error += diff * diff;
                max_px_error = fmaxf(max_px_error, diff);
            }

            if (max_px_error > MAX_PIXEL_ERROR) error = 1e30f;

            if (error < best_error) {
                best_error = error;
                best_code = {(uint16_t)rx, (uint16_t)ry, (uint16_t)dom_idx, 3, (uint8_t)iso, contrast, brightness, {0}};
                if (best_error < BAILOUT_42DB_ERROR) {
                    goto BAILOUT_NODE;
                }
            }
        }
    }

BAILOUT_NODE:
    if (best_error < UI_4x4_THRESHOLD) {
        int out_idx = atomicAdd(d_output_counter, 1);
        d_output_codes[out_idx] = best_code;
        return; 
    }

    // ==========================================
    // 2x2 RAW PIXEL FALLBACK
    // ==========================================
    for (int quad = 0; quad < 4; quad++) {
        int qx = rx + (quad % 2) * 2;
        int qy = ry + (quad / 2) * 2;
        
        float r0 = tex1Dfetch<float>(tex_curr, (qy + 0) * IMAGE_WIDTH + (qx + 0));
        float r1 = tex1Dfetch<float>(tex_curr, (qy + 0) * IMAGE_WIDTH + (qx + 1));
        float r2 = tex1Dfetch<float>(tex_curr, (qy + 1) * IMAGE_WIDTH + (qx + 0));
        float r3 = tex1Dfetch<float>(tex_curr, (qy + 1) * IMAGE_WIDTH + (qx + 1));
        
        int out_idx = atomicAdd(d_output_counter, 1);
        d_output_codes[out_idx] = {(uint16_t)qx, (uint16_t)qy, 0, 4, 0, 0.0f, 0.0f, {r0, r1, r2, r3}};
    }
}

// ============================================================
// Helpers & Main
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

cudaTextureObject_t createLinearTexture(float* d_ptr, int size_in_floats) {
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_ptr;
    resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
    resDesc.res.linear.sizeInBytes = size_in_floats * sizeof(float);
    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    cudaTextureObject_t tex = 0;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
    return tex;
}

int main(int argc, char** argv) {
    nvtxRangePushA("Encoder_Initialization");
    if (argc < 3) return 1;
    const char* prev_file = argv[1];
    const char* curr_file = argv[2];
    int total_pixels = IMAGE_WIDTH * IMAGE_HEIGHT;

    cudaFree(0);
    nvtxRangePop();

    nvtxRangePushA("Disk_IO_Load");
    std::vector<float> h_prev_frame, h_curr_frame;
    loadRawImage(prev_file, h_prev_frame);
    loadRawImage(curr_file, h_curr_frame);
    nvtxRangePop();

    nvtxRangePushA("MemAlloc_H2D");
    float *d_curr_frame, *d_prev_frame;
    cudaMalloc(&d_curr_frame, total_pixels * sizeof(float));
    cudaMalloc(&d_prev_frame, total_pixels * sizeof(float));
    cudaMemcpy(d_curr_frame, h_curr_frame.data(), total_pixels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_frame, h_prev_frame.data(), total_pixels * sizeof(float), cudaMemcpyHostToDevice);

    cudaTextureObject_t tex_curr = createLinearTexture(d_curr_frame, total_pixels);
    cudaTextureObject_t tex_prev = createLinearTexture(d_prev_frame, total_pixels);
    nvtxRangePop();

    nvtxRangePushA("Domain_Pool_Generation");
    float *d_domains_8x8;
    buildDomainPools(d_curr_frame, &d_domains_8x8);

    int d8_count = (IMAGE_WIDTH / 8) * (IMAGE_HEIGHT / 8);
    cudaTextureObject_t tex_dom8 = createLinearTexture(d_domains_8x8, d8_count * 16);
    nvtxRangePop();

    nvtxRangePushA("Output_Buffer_Setup");
    int max_possible_codes = (total_pixels / 4); 
    HybridCode* d_output_codes;
    int* d_output_counter;
    cudaMalloc(&d_output_codes, max_possible_codes * sizeof(HybridCode));
    cudaMalloc(&d_output_counter, sizeof(int));
    cudaMemset(d_output_counter, 0, sizeof(int));
    nvtxRangePop();

    std::cout << "Starting Cooperative Quadtree Temporal Encode...\n";
    nvtxRangePushA("Hybrid_Encode_Kernel");
    
    // IMPORTANT LAUNCH CONFIGURATION CHANGE:
    // Launch exactly one 8x8 thread block per 32x32 macroblock area.
    dim3 threads(8, 8); 
    dim3 grid(IMAGE_WIDTH / 32, IMAGE_HEIGHT / 32);

    hybridTemporalEncodeCooperativeQuadtree<<<grid, threads>>>(
        tex_curr, tex_prev, tex_dom8,
        d_output_codes, d_output_counter,
        (IMAGE_WIDTH/8)*(IMAGE_HEIGHT/8)
    );
    cudaDeviceSynchronize();
    nvtxRangePop();

    nvtxRangePushA("D2H_Save_Bitstream");
    int h_output_counter = 0;
    cudaMemcpy(&h_output_counter, d_output_counter, sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<HybridCode> h_output_codes(h_output_counter);
    cudaMemcpy(h_output_codes.data(), d_output_codes, h_output_counter * sizeof(HybridCode), cudaMemcpyDeviceToHost);

    std::ofstream outfile("encoded_temporal.bin", std::ios::binary);
    outfile.write(reinterpret_cast<const char*>(h_output_codes.data()), h_output_counter * sizeof(HybridCode));
    outfile.close();
    nvtxRangePop();

    std::cout << "Success! Wrote " << h_output_counter << " codes.\n";

    cudaDestroyTextureObject(tex_curr); cudaDestroyTextureObject(tex_prev);
    cudaDestroyTextureObject(tex_dom8); 
    cudaFree(d_curr_frame); cudaFree(d_prev_frame);
    cudaFree(d_domains_8x8); 
    cudaFree(d_output_codes); cudaFree(d_output_counter);

    return 0;
}