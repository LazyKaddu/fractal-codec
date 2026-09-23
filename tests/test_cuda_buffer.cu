#include <gtest/gtest.h>
#include <cuda_runtime.h>
// #include "cuda/CudaBuffer.cuh" // We will uncomment this once you write the class

// 1. The Test Fixture
// The SetUp() method runs automatically before every single TEST_F in this file.
class CudaHardwareTest : public ::testing::Test {
protected:
    void SetUp() override {
        int deviceCount = 0;
        
        // cudaGetDeviceCount returns an error if no Nvidia drivers are installed (like on GitHub runners),
        // or sets deviceCount to 0 if drivers exist but no physical GPU is present.
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        
        if (err != cudaSuccess || deviceCount == 0) {
            // Clear the CUDA error state so it doesn't pollute subsequent tests
            cudaGetLastError(); 
            
            // This macro halts the test immediately and reports it as skipped to CTest.
            GTEST_SKIP() << "No CUDA-capable GPU detected. Skipping hardware-dependent test.";
        }
    }
};

// 2. The Actual Test
// We use TEST_F (Test Fixture) instead of the standard TEST macro.
TEST_F(CudaHardwareTest, CanAllocateAndFreeVRAM) {
    // If the test runner makes it to this line, a GPU absolutely exists.
    
    const size_t numElements = 1024;
    const size_t byteSize = numElements * sizeof(float);
    
    /* 
     * Eventually, this will test your custom RAII class like this:
     * 
     * CudaBuffer<float> buffer(numElements);
     * EXPECT_NE(buffer.get(), nullptr);
     * EXPECT_EQ(buffer.size(), numElements);
     */
    
    // For now, let's assert that raw CUDA malloc works on the physical hardware
    float* d_ptr = nullptr;
    cudaError_t err = cudaMalloc(&d_ptr, byteSize);
    
    // ASSERT macros instantly halt the test if they fail, preventing segfaults.
    ASSERT_EQ(err, cudaSuccess) << "cudaMalloc failed to allocate VRAM.";
    ASSERT_NE(d_ptr, nullptr) << "Device pointer is still null after allocation.";
    
    err = cudaFree(d_ptr);
    ASSERT_EQ(err, cudaSuccess) << "cudaFree failed to release VRAM.";
}