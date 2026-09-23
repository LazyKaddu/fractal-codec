#pragma once
#include "CudaError.cuh"
#include <vector>

namespace fractal {
namespace cuda {

template <typename T>
class CudaBuffer {
private:
    T* d_ptr;
    size_t elements;

public:
    // Constructor: Allocates VRAM automatically
    explicit CudaBuffer(size_t num_elements) : elements(num_elements), d_ptr(nullptr) {
        if (elements > 0) {
            CUDA_CHECK(cudaMalloc(&d_ptr, elements * sizeof(T)));
        }
    }

    // Destructor: Frees VRAM automatically
    ~CudaBuffer() {
        if (d_ptr) {
            cudaFree(d_ptr);
        }
    }

    // Rule of 5: Prevent accidental copying (which causes double-frees)
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // Allow Move semantics (transferring ownership of the VRAM pointer safely)
    CudaBuffer(CudaBuffer&& other) noexcept : d_ptr(other.d_ptr), elements(other.elements) {
        other.d_ptr = nullptr;
        other.elements = 0;
    }

    // Easy data transfer: Host (CPU) to Device (GPU)
    void copyFromHost(const std::vector<T>& host_data) {
        if (host_data.size() != elements) {
            throw std::runtime_error("Size mismatch during Host-to-Device transfer.");
        }
        CUDA_CHECK(cudaMemcpy(d_ptr, host_data.data(), byte_size(), cudaMemcpyHostToDevice));
    }

    // Easy data transfer: Device (GPU) to Host (CPU)
    void copyToHost(std::vector<T>& host_data) const {
        host_data.resize(elements);
        CUDA_CHECK(cudaMemcpy(host_data.data(), d_ptr, byte_size(), cudaMemcpyDeviceToHost));
    }

    // Getters for the raw CUDA kernels
    T* get() const { return d_ptr; }
    size_t size() const { return elements; }
    size_t byte_size() const { return elements * sizeof(T); }
};

} // namespace cuda
} // namespace fractal