# CUDA Utilities

## CudaBuffer (`include/fractal/cuda/CudaBuffer.cuh`)
Memory management on the GPU can easily lead to memory leaks (forgetting to call `cudaFree`) or double-frees. 
To solve this, we use the `CudaBuffer<T>` class, which acts as a smart pointer (following C++ RAII principles) for VRAM.

### Features
* **Automatic Allocation:** Calls `cudaMalloc` on creation.
* **Automatic Deallocation:** Calls `cudaFree` when it goes out of scope (e.g. when the `EncoderState` is destroyed).
* **Move Semantics:** Supports `std::swap` perfectly, allowing us to swap the "Current" and "Previous" frame buffers instantly without copying any bytes across the PCI-e bus.

## CudaError (`include/fractal/cuda/CudaError.cuh`)
A common issue in CUDA is that an API call (like `cudaMalloc` or `cudaMemcpy`) fails, but the program continues silently, causing catastrophic data corruption later.
We solve this using the `CUDA_CHECK(...)` macro.

### Features
* Wraps any naked CUDA API call.
* Instantly throws a `std::runtime_error` if the API returns anything other than `cudaSuccess`.
* Injects `__FILE__` and `__LINE__` into the error message, making debugging GPU crashes trivial.
