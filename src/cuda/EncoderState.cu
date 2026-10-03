#include <fractal/core/FractalCodec.h>
#include <fractal/core/common.h>
#include <fractal/cuda/CudaBuffer.cuh>
// The C++ Bitstream processor handles entropy coding of the output data
#include <fractal/core/FractalBitstreamProcessor.h>
#include <cuda_runtime.h>
#include <fractal/cuda/CudaError.cuh>
#include <vector>
#include <algorithm>
#include <iostream>

const float TEMPORAL_SKIP_THRESHOLD = 0.045f;
const float UI_4x4_THRESHOLD        = 0.0005f;
const float MAX_PIXEL_ERROR         = 0.05f;
const float MAX_PIXEL_DRIFT         = 0.015f; 
const float BAILOUT_42DB_ERROR      = 0.001009f;

// ==========================================
// KERNEL HELPERS
// ==========================================
cudaTextureObject_t createLinearTexture(float* d_ptr, int size_in_floats) {
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_ptr;
    resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
    resDesc.res.linear.sizeInBytes = size_in_floats * sizeof(float);
    
    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    
    cudaTextureObject_t tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));
    return tex;
}

__global__ void build8x8DownsampledPoolKernel(const float* d_img, float* d_pool, int width, int height) {
    int dom_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dom_y = blockIdx.y * blockDim.y + threadIdx.y;
    int domains_per_row = width / 8;
    
    if (dom_x < domains_per_row && dom_y < (height / 8)) {
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

void buildDomainPools(const float* d_curr_frame, float** d_domains_8x8, int width, int height) {
    int d8_count = (width / 8) * (height / 8);
    // Note: Assuming d_domains_8x8 is pre-allocated in the state class to avoid cudaMalloc per frame
    dim3 block(16, 16);
    dim3 grid8((width / 8 + 15) / 16, (height / 8 + 15) / 16);
    build8x8DownsampledPoolKernel<<<grid8, block>>>(d_curr_frame, *d_domains_8x8, width, height);
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

__device__ bool searchMotionVector(
    cudaTextureObject_t tex_curr, cudaTextureObject_t tex_prev,
    int rx, int ry, int size, float threshold_sad, float max_drift,
    int width, int height,
    int& best_dx, int& best_dy
) {
    float best_diff = threshold_sad;
    bool found = false;
    for (int dy = -8; dy <= 8; dy++) {
        for (int dx = -8; dx <= 8; dx++) {
            if (dx == 0 && dy == 0) continue;
            if (rx + dx < 0 || rx + dx + size > width ||
                ry + dy < 0 || ry + dy + size > height) continue;

            float diff = 0.0f; float current_max = 0.0f; bool invalid = false;
            for (int py = 0; py < size; py++) {
                for (int px = 0; px < size; px++) {
                    float curr = tex1Dfetch<float>(tex_curr, (ry + py) * width + (rx + px));
                    float prev = tex1Dfetch<float>(tex_prev, (ry + dy + py) * width + (rx + dx + px));
                    float px_diff = fabsf(curr - prev);
                    diff += px_diff; current_max = fmaxf(current_max, px_diff);
                    if (diff >= best_diff || current_max >= max_drift) { invalid = true; break; }
                }
                if (invalid) break;
            }
            if (!invalid && diff < best_diff) {
                best_diff = diff; best_dx = dx; best_dy = dy; found = true;
            }
        }
    }
    return found;
}

// ==========================================
// CORE ENCODER KERNEL
// ==========================================
__global__ void hybridTemporalEncodeCooperativeQuadtree(
    cudaTextureObject_t tex_curr, cudaTextureObject_t tex_prev, cudaTextureObject_t tex_dom8,
    HybridCodeData* d_output_codes, int* d_output_counter, int total_domains_4x4_pass,
    int width, int height
) {
    __shared__ float s_diff[64];
    __shared__ float s_max_diff[64];
    __shared__ float s_flags_16[4];
    __shared__ float s_flags_8[16];

    int tx = threadIdx.x; int ty = threadIdx.y; int tid = ty * 8 + tx;
    int bx = blockIdx.x * 32; int by = blockIdx.y * 32;
    int rx = bx + tx * 4; int ry = by + ty * 4;

    if (rx >= width || ry >= height) return;

    float my_diff = 0.0f; float my_max = 0.0f;
    for (int py = 0; py < 4; py++) {
        for (int px = 0; px < 4; px++) {
            int g_idx = (ry + py) * width + (rx + px);
            float curr = tex1Dfetch<float>(tex_curr, g_idx);
            float prev = tex1Dfetch<float>(tex_prev, g_idx);
            float err = fabsf(curr - prev);
            my_diff += err; my_max = fmaxf(my_max, err);
        }
    }
    s_diff[tid] = my_diff; s_max_diff[tid] = my_max;
    __syncthreads();

    if (tid == 0) {
        float total_32 = 0; float max_32 = 0;
        for (int i = 0; i < 64; i++) {
            total_32 += s_diff[i]; max_32 = fmaxf(max_32, s_max_diff[i]);
        }
        float thresh = TEMPORAL_SKIP_THRESHOLD * 64.0f;
        if (total_32 < thresh && max_32 < MAX_PIXEL_DRIFT) {
            int out_idx = atomicAdd(d_output_counter, 1);
            d_output_codes[out_idx] = {(uint16_t)bx, (uint16_t)by, 0xFFFF, 0, 0, 0, 0, {0}};
            s_diff[0] = -1.0f;
        } else {
            s_diff[0] = 1.0f;
        }
    }
    __syncthreads();
    if (s_diff[0] < 0.0f) return;

    int q_idx = (ty / 4) * 2 + (tx / 4);
    int leader_16 = (ty / 4) * 32 + (tx / 4) * 4;
    if (tid == leader_16) {
        float total_16 = 0; float max_16 = 0;
        for (int dy = 0; dy < 4; dy++) {
            for (int dx = 0; dx < 4; dx++) {
                int idx = (ty/4*4 + dy)*8 + (tx/4*4 + dx);
                total_16 += s_diff[idx]; max_16 = fmaxf(max_16, s_max_diff[idx]);
            }
        }
        float thresh = TEMPORAL_SKIP_THRESHOLD * 16.0f;
        int sub_x = bx + (tx/4)*16; int sub_y = by + (ty/4)*16;

        if (total_16 < thresh && max_16 < MAX_PIXEL_DRIFT) {
            int out_idx = atomicAdd(d_output_counter, 1);
            d_output_codes[out_idx] = {(uint16_t)sub_x, (uint16_t)sub_y, 0xFFFF, 1, 0, 0, 0, {0}};
            s_flags_16[q_idx] = -1.0f;
        } else {
            int dx, dy;
            if (searchMotionVector(tex_curr, tex_prev, sub_x, sub_y, 16, thresh, MAX_PIXEL_DRIFT, width, height, dx, dy)) {
                int out_idx = atomicAdd(d_output_counter, 1);
                d_output_codes[out_idx] = {(uint16_t)sub_x, (uint16_t)sub_y, 0xFFFE, 1, 0, (float)dx, (float)dy, {0}};
                s_flags_16[q_idx] = -1.0f;
            } else {
                s_flags_16[q_idx] = 1.0f;
            }
        }
    }
    __syncthreads();
    if (s_flags_16[q_idx] < 0.0f) return;

    int e_idx = (ty / 2) * 4 + (tx / 2);
    int leader_8 = (ty / 2) * 16 + (tx / 2) * 2;
    if (tid == leader_8) {
        float total_8 = 0; float max_8 = 0;
        for (int dy = 0; dy < 2; dy++) {
            for (int dx = 0; dx < 2; dx++) {
                int idx = (ty/2*2 + dy)*8 + (tx/2*2 + dx);
                total_8 += s_diff[idx]; max_8 = fmaxf(max_8, s_max_diff[idx]);
            }
        }
        float thresh = TEMPORAL_SKIP_THRESHOLD * 4.0f;
        int sub_x = bx + (tx/2)*8; int sub_y = by + (ty/2)*8;

        if (total_8 < thresh && max_8 < MAX_PIXEL_DRIFT) {
            int out_idx = atomicAdd(d_output_counter, 1);
            d_output_codes[out_idx] = {(uint16_t)sub_x, (uint16_t)sub_y, 0xFFFF, 2, 0, 0, 0, {0}};
            s_flags_8[e_idx] = -1.0f;
        } else {
            int dx, dy;
            if (searchMotionVector(tex_curr, tex_prev, sub_x, sub_y, 8, thresh, MAX_PIXEL_DRIFT, width, height, dx, dy)) {
                int out_idx = atomicAdd(d_output_counter, 1);
                d_output_codes[out_idx] = {(uint16_t)sub_x, (uint16_t)sub_y, 0xFFFE, 2, 0, (float)dx, (float)dy, {0}};
                s_flags_8[e_idx] = -1.0f;
            } else {
                s_flags_8[e_idx] = 1.0f;
            }
        }
    }
    __syncthreads();
    if (s_flags_8[e_idx] < 0.0f) return;

    if (my_diff < TEMPORAL_SKIP_THRESHOLD && my_max < MAX_PIXEL_DRIFT) {
        int out_idx = atomicAdd(d_output_counter, 1);
        d_output_codes[out_idx] = {(uint16_t)rx, (uint16_t)ry, 0xFFFF, 3, 0, 0, 0, {0}};
        return;
    }

    int dx, dy;
    if (searchMotionVector(tex_curr, tex_prev, rx, ry, 4, TEMPORAL_SKIP_THRESHOLD, MAX_PIXEL_DRIFT, width, height, dx, dy)) {
        int out_idx = atomicAdd(d_output_counter, 1);
        d_output_codes[out_idx] = {(uint16_t)rx, (uint16_t)ry, 0xFFFE, 3, 0, (float)dx, (float)dy, {0}};
        return;
    }

    float best_error = 1e30f; HybridCodeData best_code = {0}; float R[16]; float sum_R = 0.0f;
    for (int i = 0; i < 16; i++) {
        R[i] = tex1Dfetch<float>(tex_curr, (ry + (i/4)) * width + (rx + (i%4))); sum_R += R[i];
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
            float error = 0.0f; float max_px_error = 0.0f;

            for (int p = 0; p < 16; p++) {
                int mapped = getIsoPixel(p%4, p/4, 4, iso);
                float diff = fabsf((contrast * tex1Dfetch<float>(tex_dom8, (dom_idx * 16) + mapped)) + brightness - R[p]);
                error += diff * diff; max_px_error = fmaxf(max_px_error, diff);
            }
            if (max_px_error > MAX_PIXEL_ERROR) error = 1e30f;
            if (error < best_error) {
                best_error = error; best_code = {(uint16_t)rx, (uint16_t)ry, (uint16_t)dom_idx, 3, (uint8_t)iso, contrast, brightness, {0}};
                if (best_error < BAILOUT_42DB_ERROR) goto BAILOUT_NODE;
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
    // HYBRID EDGE-PRESERVATION LOGIC 
    // ==========================================
    int res_dx = 0, res_dy = 0;
    searchMotionVector(tex_curr, tex_prev, rx, ry, 4, 1e30f, 1e30f, width, height, res_dx, res_dy);

    HybridCodeData res_code = {(uint16_t)rx, (uint16_t)ry, 0xFFFD, 3, 0, (float)res_dx, (float)res_dy, {0}};
    bool residual_failed = false;

    for (int py = 0; py < 4; py++) {
        for (int px = 0; px < 4; px++) {
            float curr = tex1Dfetch<float>(tex_curr, (ry + py) * width + (rx + px));
            float prev = tex1Dfetch<float>(tex_prev, (ry + res_dy + py) * width + (rx + res_dx + px));

            float diff = (curr - prev) * 255.0f;

            // Abort residual block if contrast is too high to prevent jagged tearing
            if (diff < -127.0f || diff > 127.0f) {
                residual_failed = true;
                break;
            }
            res_code.raw_pixels_bytes[py * 4 + px] = (int8_t)roundf(diff);
        }
        if (residual_failed) break;
    }

    if (!residual_failed) {
        int out_idx = atomicAdd(d_output_counter, 1);
        d_output_codes[out_idx] = res_code;
        return;
    }

    // High contrast edge detected - shatter to 2x2 Raw Fallback
    for (int quad = 0; quad < 4; quad++) {
        int qx = rx + (quad % 2) * 2;
        int qy = ry + (quad / 2) * 2;

        float r0 = tex1Dfetch<float>(tex_curr, (qy + 0) * width + (qx + 0));
        float r1 = tex1Dfetch<float>(tex_curr, (qy + 0) * width + (qx + 1));
        float r2 = tex1Dfetch<float>(tex_curr, (qy + 1) * width + (qx + 0));
        float r3 = tex1Dfetch<float>(tex_curr, (qy + 1) * width + (qx + 1));

        HybridCodeData raw_code = {(uint16_t)qx, (uint16_t)qy, 0, 4, 0, 0.0f, 0.0f, {0}};
        float* raw_floats = reinterpret_cast<float*>(raw_code.raw_pixels_bytes);
        raw_floats[0] = r0; raw_floats[1] = r1; raw_floats[2] = r2; raw_floats[3] = r3;

        int out_idx = atomicAdd(d_output_counter, 1);
        d_output_codes[out_idx] = raw_code;
    }
}

// ==========================================
// STATEFUL ENCODER CLASS
// ==========================================
class FractalEncoderState {
public:
    int width, height, total_pixels;
    fractal::cuda::CudaBuffer<float> d_curr_frame;
    fractal::cuda::CudaBuffer<float> d_prev_frame;
    fractal::cuda::CudaBuffer<float> d_domains_8x8;
    fractal::cuda::CudaBuffer<HybridCodeData> d_output_codes;
    fractal::cuda::CudaBuffer<int> d_output_counter;
    
    // Pre-allocated host memory for async readback
    HybridCodeData *h_output_codes;
    
    FractalEncoderState(int w, int h) : 
        width(w), height(h), total_pixels(w * h),
        d_curr_frame(w * h),
        d_prev_frame(w * h),
        d_domains_8x8((w / 8) * (h / 8) * 16),
        d_output_codes((w * h) / 4),
        d_output_counter(1)
    {
        CUDA_CHECK(cudaMemset(d_prev_frame.get(), 0, d_prev_frame.byte_size())); // Init black
        CUDA_CHECK(cudaMallocHost(&h_output_codes, d_output_codes.byte_size()));
    }

    ~FractalEncoderState() {
        CUDA_CHECK(cudaFreeHost(h_output_codes));
    }

    int Encode(const float* raw_in, uint8_t* compressed_out, int max_out_size) {
        // 1. Swap pointers: Current becomes Previous
        std::swap(d_curr_frame, d_prev_frame);
        
        // 2. Upload new frame
        CUDA_CHECK(cudaMemcpy(d_curr_frame.get(), raw_in, total_pixels * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_output_counter.get(), 0, sizeof(int)));

        // 3. Build Domains & Textures
        float* dom_ptr = d_domains_8x8.get();
        buildDomainPools(d_curr_frame.get(), &dom_ptr, width, height);
        
        cudaTextureObject_t tex_curr = createLinearTexture(d_curr_frame.get(), total_pixels);
        cudaTextureObject_t tex_prev = createLinearTexture(d_prev_frame.get(), total_pixels);
        
        int d8_count = (width / 8) * (height / 8);
        cudaTextureObject_t tex_dom8 = createLinearTexture(d_domains_8x8.get(), d8_count * 16);
        
        // 4. Launch Kernel
        dim3 threads(8, 8); 
        dim3 grid((width + 31) / 32, (height + 31) / 32);
        
        hybridTemporalEncodeCooperativeQuadtree<<<grid, threads>>>(
            tex_curr, tex_prev, tex_dom8,
            d_output_codes.get(), d_output_counter.get(),
            d8_count, width, height
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Destroy textures to prevent leaks
        CUDA_CHECK(cudaDestroyTextureObject(tex_curr));
        CUDA_CHECK(cudaDestroyTextureObject(tex_prev));
        CUDA_CHECK(cudaDestroyTextureObject(tex_dom8));

        // 5. Readback
        int h_counter = 0;
        CUDA_CHECK(cudaMemcpy(&h_counter, d_output_counter.get(), sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_output_codes, d_output_codes.get(), h_counter * sizeof(HybridCodeData), cudaMemcpyDeviceToHost));

        // 6. Push data to C++ Bitstream Processor
        std::vector<HybridCodeData> codes_vec(h_output_codes, h_output_codes + h_counter);
        
        std::vector<uint8_t> final_payload = FractalBitstreamProcessor::compress_bitstream(codes_vec, width);
        
        // 7. Verify buffer size and return
        if (final_payload.size() > static_cast<size_t>(max_out_size)) {
            std::cerr << "Error: Output buffer too small!\n";
            return -1; 
        }
        
        std::memcpy(compressed_out, final_payload.data(), final_payload.size());
        return static_cast<int>(final_payload.size());
    }
};

// ==========================================
// C-API EXPORTS
// ==========================================
extern "C" {

FractalEncoderHandle CreateFractalEncoder(int width, int height) {
    return new FractalEncoderState(width, height);
}

int EncodeFractalFrame(FractalEncoderHandle handle, const float* raw_in, uint8_t* compressed_out, int max_out_size) {
    if (!handle) return -1;
    return static_cast<FractalEncoderState*>(handle)->Encode(raw_in, compressed_out, max_out_size);
}

void DestroyFractalEncoder(FractalEncoderHandle handle) {
    if (handle) {
        delete static_cast<FractalEncoderState*>(handle);
    }
}

}