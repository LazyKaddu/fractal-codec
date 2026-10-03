# We use GTest for our C++ testing framework
find_package(GTest REQUIRED)

# Add the codec test executable
add_executable(test_codec test_codec.cu)
target_include_directories(test_codec PRIVATE ../include)

# Link against GTest and our main library
target_link_libraries(test_codec PRIVATE GTest::gtest_main fractal_codec)

# If CUDA is needed for tests
set_target_properties(test_codec PROPERTIES CUDA_ARCHITECTURES "89;86;75")

# Register the test with CTest
add_test(NAME CodecTests COMMAND test_codec)

# Add the cuda buffer test
add_executable(test_cuda_buffer test_cuda_buffer.cu)
target_include_directories(test_cuda_buffer PRIVATE ../include)
target_link_libraries(test_cuda_buffer PRIVATE GTest::gtest_main CUDA::cudart)
set_target_properties(test_cuda_buffer PROPERTIES CUDA_ARCHITECTURES "89;86;75")
add_test(NAME CudaBufferTests COMMAND test_cuda_buffer)
