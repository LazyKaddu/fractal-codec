#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// A macro to wrap every CUDA API call. It checks the return code and throws on failure.
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::string error_msg = std::string("CUDA Error: ") + \
                                    cudaGetErrorString(err) + \
                                    " at " + __FILE__ + ":" + std::to_string(__LINE__); \
            throw std::runtime_error(error_msg); \
        } \
    } while (0)