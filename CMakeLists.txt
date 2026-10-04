cmake_minimum_required(VERSION 3.18)
project(FractalCodec LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# Find dependencies
find_package(CUDAToolkit REQUIRED)
# Assuming Zstd is installed on your system or provided locally
find_package(zstd REQUIRED) 
find_package(OpenCV REQUIRED)

include(FetchContent)
FetchContent_Declare(
    cli11
    URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.3.2.zip
)
FetchContent_MakeAvailable(cli11)

# Collect source files
set(CODEC_SOURCES 
    src/cuda/EncoderState.cu
    src/cuda/DecoderState.cu
    src/core/DataPacker.cpp
)

# ---------------------------------------------
# TARGET 1: The Hookable Engine Plugin (.dll / .so)
# ---------------------------------------------
add_library(fractal_codec SHARED ${CODEC_SOURCES})
target_include_directories(fractal_codec PUBLIC include)
target_link_libraries(fractal_codec PRIVATE CUDA::cudart zstd::libzstd_shared)

# Set CUDA architecture (e.g., Ada Lovelace / RTX 40 series is sm_89)
set_target_properties(fractal_codec PROPERTIES CUDA_ARCHITECTURES "89;86;75")

# ---------------------------------------------
# TARGET 2: The Standalone CLI Executable (.exe)
# ---------------------------------------------
add_executable(fractal_cli 
    src/main.cpp 
    src/core/VideoEncoder.cpp 
    src/core/VideoDecoder.cpp 
    src/io/VideoReader.cpp 
    ${CODEC_SOURCES}
)
target_include_directories(fractal_cli PUBLIC include)
target_link_libraries(fractal_cli PRIVATE CUDA::cudart zstd::libzstd_static ${OpenCV_LIBS} CLI11::CLI11)
set_target_properties(fractal_cli PROPERTIES CUDA_ARCHITECTURES "89;86;75")

# ---------------------------------------------
# TARGET 3: Tests
# ---------------------------------------------
enable_testing()

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)

# Prevent GoogleTest from overriding compiler/linker settings
# of the parent project.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)
add_subdirectory(tests)