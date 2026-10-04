#pragma once

#include <vector>
#include <cstdint>

#pragma pack(push, 1)
struct HybridCodeData {
    uint16_t x;
    uint16_t y;
    uint16_t dom_idx;
    uint8_t depth;
    uint8_t iso;
    float contrast;
    float brightness;
    int8_t raw_pixels_bytes[16];
};
#pragma pack(pop)

class FractalBitstreamProcessor {
public:
    // Takes the raw array of C++ codes (by value to allow in-place sorting) and returns the Zstd buffer
    static std::vector<uint8_t> compress_bitstream(std::vector<HybridCodeData> codes, int width);
    
    // Takes the Zstd buffer and returns the reconstructed C++ structures ready for CUDA
    static std::vector<HybridCodeData> decompress_bitstream(const std::vector<uint8_t>& compressed_data, int width);
};
