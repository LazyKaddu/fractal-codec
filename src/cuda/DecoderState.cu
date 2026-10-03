#include <fractal/core/FractalCodec.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <iostream>

// Assuming FractalBitstreamProcessor and HybridCodeData are available via header
// #include "FractalBitstreamProcessor.h"

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

__global__ void hybridTemporalDecodeQuadtreeKernel(
    const HybridCodeData* d_codes, const float* d_prev, const float* d_in_iter, float* d_out_iter, int total_codes, int width, int height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_codes) return;

    HybridCodeData code = d_codes[idx];
    int rx = code.x; int ry = code.y;
    int size = 32 >> code.depth;

    // TEMPORAL BLOCKS: Strictly read from immutable d_prev
    if (code.dom_idx == 0xFFFF) {
        for (int py = 0; py < size; py++) {
            for (int px = 0; px < size; px++) {
                int p_idx = (ry + py) * width + (rx + px);
                d_out_iter[p_idx] = d_prev[p_idx];
            }
        }
        return;
    }
    else if (code.dom_idx == 0xFFFE) {
        int dx = static_cast<int>(code.contrast); 
        int dy = static_cast<int>(code.brightness);
        for (int py = 0; py < size; py++) {
            for (int px = 0; px < size; px++) {
                int src_idx = (ry + py + dy) * width + (rx + px + dx);
                int dst_idx = (ry + py) * width + (rx + px);
                d_out_iter[dst_idx] = d_prev[src_idx];
            }
        }
        return;
    }
    else if (code.dom_idx == 0xFFFD) {
        int dx = static_cast<int>(code.contrast); 
        int dy = static_cast<int>(code.brightness);
        const int8_t* residual_payload = code.raw_pixels_bytes;
        
        for (int py = 0; py < 4; py++) {
            for (int px = 0; px < 4; px++) {
                int src_idx = (ry + py + dy) * width + (rx + px + dx);
                int dst_idx = (ry + py) * width + (rx + px);
                
                float base_val = d_prev[src_idx]; // Read from read-only prev frame
                float residual_val = static_cast<float>(residual_payload[py * 4 + px]) / 255.0f;
                d_out_iter[dst_idx] = fminf(fmaxf(base_val + residual_val, 0.0f), 1.0f);
            }
        }
        return;
    }
    // FRACTAL BLOCKS: Read from the iterative ping-pong buffer
    else if (code.depth == 3) {
        int dom_x = (code.dom_idx % (width / 8)) * 8;
        int dom_y = (code.dom_idx / (width / 8)) * 8;
        for (int p = 0; p < 16; p++) {
            int px = p % 4; int py = p / 4;
            int iso_p = getIsoPixel(px, py, 4, code.iso);
            int iso_x = iso_p % 4; int iso_y = iso_p / 4;
            float avg = 0.0f;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    avg += d_in_iter[(dom_y + iso_y * 2 + dy) * width + (dom_x + iso_x * 2 + dx)];
                }
            }
            avg *= 0.25f;
            d_out_iter[(ry + py) * width + (rx + px)] = (code.contrast * avg) + code.brightness;
        }
    }
    else if (code.depth == 4) {
        const float* raw_floats = reinterpret_cast<const float*>(code.raw_pixels_bytes);
        d_out_iter[(ry + 0) * width + (rx + 0)] = raw_floats[0];
        d_out_iter[(ry + 0) * width + (rx + 1)] = raw_floats[1];
        d_out_iter[(ry + 1) * width + (rx + 0)] = raw_floats[2];
        d_out_iter[(ry + 1) * width + (rx + 1)] = raw_floats[3];
    }
}

// ============================================================
// Decoder State Class
// ============================================================
class FractalDecoderState {
public:
    int width, height, total_pixels;
    float *d_prev, *d_buffer_A, *d_buffer_B;
    HybridCodeData *d_codes;
    int max_codes;

    FractalDecoderState(int w, int h) : width(w), height(h), total_pixels(w * h) {
        cudaMalloc(&d_prev, total_pixels * sizeof(float));
        cudaMalloc(&d_buffer_A, total_pixels * sizeof(float));
        cudaMalloc(&d_buffer_B, total_pixels * sizeof(float));

        // Initialize background to black to start
        cudaMemset(d_prev, 0, total_pixels * sizeof(float));

        max_codes = total_pixels / 4; 
        cudaMalloc(&d_codes, max_codes * sizeof(HybridCodeData));
    }

    ~FractalDecoderState() {
        cudaFree(d_prev);
        cudaFree(d_buffer_A);
        cudaFree(d_buffer_B);
        cudaFree(d_codes);
    }

    void Decode(const uint8_t* compressed_in, int compressed_size, float* raw_out) {
        // 1. Zstd Decompress & Bit-Unpack natively in C++
        std::vector<uint8_t> comp_data(compressed_in, compressed_in + compressed_size);
        std::vector<HybridCodeData> h_codes = FractalBitstreamProcessor::decompress_bitstream(comp_data, width);
        int total_codes = h_codes.size();

        // 2. Upload codes to GPU
        if (total_codes > 0) {
            cudaMemcpy(d_codes, h_codes.data(), total_codes * sizeof(HybridCodeData), cudaMemcpyHostToDevice);
        }

        // 3. Seed ping-pong buffers with the persistent previous frame
        cudaMemcpy(d_buffer_A, d_prev, total_pixels * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_buffer_B, d_prev, total_pixels * sizeof(float), cudaMemcpyDeviceToDevice);

        // 4. Execute Ping-Pong Loop
        if (total_codes > 0) {
            int threads = 256; 
            int blocks = (total_codes + threads - 1) / threads;
            
            for (int iter = 0; iter < 3; iter++) {
                if (iter % 2 == 0) {
                    hybridTemporalDecodeQuadtreeKernel<<<blocks, threads>>>(d_codes, d_prev, d_buffer_A, d_buffer_B, total_codes, width, height);
                } else {
                    hybridTemporalDecodeQuadtreeKernel<<<blocks, threads>>>(d_codes, d_prev, d_buffer_B, d_buffer_A, total_codes, width, height);
                }
            }
            cudaDeviceSynchronize();
        }

        // 5. Readback Output (Iteration 2 outputs to d_buffer_B)
        if (raw_out != nullptr) {
            cudaMemcpy(raw_out, d_buffer_B, total_pixels * sizeof(float), cudaMemcpyDeviceToHost);
        }

        // 6. CRITICAL: Update d_prev for the next frame
        cudaMemcpy(d_prev, d_buffer_B, total_pixels * sizeof(float), cudaMemcpyDeviceToDevice);
    }
};

// ============================================================
// C-API Implementations
// ============================================================
extern "C" {

FractalDecoderHandle CreateFractalDecoder(int width, int height) {
    return new FractalDecoderState(width, height);
}

void DecodeFractalFrame(FractalDecoderHandle handle, const uint8_t* compressed_buffer_in, int compressed_size, float* raw_y_channel_out) {
    if (handle) {
        static_cast<FractalDecoderState*>(handle)->Decode(compressed_buffer_in, compressed_size, raw_y_channel_out);
    }
}

void DestroyFractalDecoder(FractalDecoderHandle handle) {
    if (handle) {
        delete static_cast<FractalDecoderState*>(handle);
    }
}

}