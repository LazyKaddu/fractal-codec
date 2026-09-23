cmake_minimum_required(VERSION 3.21)

# Declaring LANGUAGES CXX and CUDA tells CMake to invoke nvcc for .cu files
project(FractalCodec VERSION 0.1.0 LANGUAGES CXX CUDA)

# Enforce C++17 for both host and device code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target modern Nvidia architectures (Turing=75, Ampere=80, Ada=89)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")

# ==============================================================================
# 1. THE CORE CODEC LIBRARY
# ==============================================================================
# Combine C++ and CUDA source files into a single static library
add_library(fractal_core STATIC
    src/core/Encoder.cpp
    src/cuda/CudaBuffer.cu
    src/cuda/Kernels.cu
    src/io/VideoReader.cpp
)

# Generator expressions ensure that anything linking to fractal_core 
# automatically knows where to find the headers in the include/ directory.
target_include_directories(fractal_core PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Enable aggressive optimization for the core library in Release mode
target_compile_options(fractal_core PRIVATE 
    $<$<CONFIG:Release>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3 -use_fast_math>
)

# ==============================================================================
# 2. THE CLI EXECUTABLE
# ==============================================================================
add_executable(fractal_cli src/main.cpp)

# Link the CLI to the core library. It inherits the include directories automatically.
target_link_libraries(fractal_cli PRIVATE fractal_core)

# ==============================================================================
# 3. THE TESTING SUITE (GoogleTest)
# ==============================================================================
# Fetch GoogleTest directly from GitHub during the build process
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Compile C++ and CUDA tests into a single test executable
add_executable(fractal_tests 
    tests/test_core.cpp
    tests/test_cuda_buffer.cu
)

# Link the tests to the codec AND to gtest_main
target_link_libraries(fractal_tests PRIVATE 
    fractal_core
    gtest_main
)

# Automatically discover and register the tests with CTest
include(GoogleTest)
gtest_discover_tests(fractal_tests)