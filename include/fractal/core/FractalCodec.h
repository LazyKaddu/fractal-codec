#pragma once
#include <stdint.h>

#ifdef _WIN32
    #define FRACTAL_API __declspec(dllexport)
#else
    #define FRACTAL_API __attribute__((visibility("default")))
#endif

extern "C" {
    // Opaque handle to hide internal CUDA state from the host engine
    typedef void* FractalEncoderHandle;
    typedef void* FractalDecoderHandle;

    // --- ENCODER API ---
    FRACTAL_API FractalEncoderHandle CreateFractalEncoder(int width, int height);
    
    // Returns the size of the compressed payload. 
    // out_compressed_buffer must be pre-allocated by the engine.
    FRACTAL_API int EncodeFractalFrame(
        FractalEncoderHandle handle,
        const float* raw_y_channel_in, 
        uint8_t* out_compressed_buffer, 
        int max_out_size
    );
    
    FRACTAL_API void DestroyFractalEncoder(FractalEncoderHandle handle);

    // --- DECODER API ---
    FRACTAL_API FractalDecoderHandle CreateFractalDecoder(int width, int height);
    
    FRACTAL_API void DecodeFractalFrame(
        FractalDecoderHandle handle,
        const uint8_t* compressed_buffer_in,
        int compressed_size,
        float* raw_y_channel_out
    );
    
    FRACTAL_API void DestroyFractalDecoder(FractalDecoderHandle handle);
}