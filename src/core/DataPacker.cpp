#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <zstd.h>

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

// ============================================================
// IEEE-754 Half-Precision Float Converters
// ============================================================
uint16_t float_to_fp16_bits(float val) {
    uint32_t f;
    std::memcpy(&f, &val, sizeof(float));
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t exponent = ((f >> 23) & 0xFF) - 127;
    uint32_t mantissa = f & 0x007FFFFF;

    if (exponent <= -15) return sign;
    if (exponent > 15) return sign | 0x7C00;
    return sign | ((exponent + 15) << 10) | (mantissa >> 13);
}

float fp16_bits_to_float(uint16_t val) {
    uint32_t sign = (val & 0x8000) << 16;
    int32_t exponent = (val & 0x7C00) >> 10;
    uint32_t mantissa = val & 0x03FF;

    uint32_t f;
    if (exponent == 0) f = sign;
    else if (exponent == 31) f = sign | 0x7F800000 | (mantissa << 13);
    else f = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    
    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

// ============================================================
// Custom Bit-Level Writer & Reader
// ============================================================
class BitWriter {
    std::vector<uint8_t> buffer;
    uint8_t current_byte = 0;
    int bits_in_byte = 0;

public:
    void write_bits(uint32_t value, int num_bits) {
        value &= ((1ULL << num_bits) - 1);
        for (int i = num_bits - 1; i >= 0; i--) {
            uint8_t bit = (value >> i) & 1;
            current_byte = (current_byte << 1) | bit;
            bits_in_byte++;

            if (bits_in_byte == 8) {
                buffer.push_back(current_byte);
                current_byte = 0;
                bits_in_byte = 0;
            }
        }
    }

    std::vector<uint8_t> get_bytes() {
        if (bits_in_byte > 0) {
            uint8_t padded_byte = current_byte << (8 - bits_in_byte);
            buffer.push_back(padded_byte);
            bits_in_byte = 0;
            current_byte = 0;
        }
        return buffer;
    }
};

class BitReader {
    const std::vector<uint8_t>& data;
    size_t byte_pos = 0;
    int bit_pos = 7;

public:
    BitReader(const std::vector<uint8_t>& d) : data(d) {}

    bool read_bits(int num_bits, uint32_t& out_value) {
        uint32_t value = 0;
        for (int i = 0; i < num_bits; i++) {
            if (byte_pos >= data.size()) return false;
            
            uint8_t bit = (data[byte_pos] >> bit_pos) & 1;
            value = (value << 1) | bit;
            
            bit_pos--;
            if (bit_pos < 0) {
                bit_pos = 7;
                byte_pos++;
            }
        }
        out_value = value;
        return true;
    }
};

// ============================================================
// Core Compressor Class
// ============================================================
class FractalBitstreamProcessor {
public:
    // Takes the raw array of C++ codes (by value to allow in-place sorting) and returns the Zstd buffer
    static std::vector<uint8_t> compress_bitstream(std::vector<HybridCodeData> codes, int width) {
        size_t original_size = codes.size() * sizeof(HybridCodeData);
        int grid_width = width / 4;

        // Geometric Sort (Top-to-Bottom, Left-to-Right)
        std::sort(codes.begin(), codes.end(), [](const HybridCodeData& a, const HybridCodeData& b) {
            if (a.y != b.y) return a.y < b.y;
            return a.x < b.x;
        });

        BitWriter writer;
        uint32_t skip_count = 0;
        int prev_idx = 0;

        for (const auto& data : codes) {
            if (data.dom_idx == 0xFFFF) {
                skip_count++;
                continue;
            }

            if (skip_count > 0) {
                writer.write_bits(0xFFFF, 16);
                writer.write_bits(skip_count, 32);
                skip_count = 0;
            }

            int curr_idx = (data.y / 4) * grid_width + (data.x / 4);
            int delta_idx = curr_idx - prev_idx;
            prev_idx = curr_idx;

            auto write_coordinates = [&]() {
                if (delta_idx < 255) {
                    writer.write_bits(delta_idx, 8);
                } else {
                    writer.write_bits(255, 8);
                    writer.write_bits(delta_idx, 16);
                }
            };

            if (data.dom_idx == 0xFFFE) {
                writer.write_bits(0xFFFE, 16);
                writer.write_bits(data.depth, 8);
                write_coordinates();
                writer.write_bits(static_cast<int>(data.contrast) + 128, 8);
                writer.write_bits(static_cast<int>(data.brightness) + 128, 8);
            } 
            else if (data.dom_idx == 0xFFFD) {
                writer.write_bits(0xFFFD, 16);
                writer.write_bits(data.depth, 8);
                write_coordinates();
                writer.write_bits(static_cast<int>(data.contrast) + 128, 8);
                writer.write_bits(static_cast<int>(data.brightness) + 128, 8);

                for (int i = 0; i < 16; i++) {
                    writer.write_bits(data.raw_pixels_bytes[i] + 128, 8);
                }
            } 
            else {
                writer.write_bits(data.dom_idx, 16);
                writer.write_bits(data.depth, 8);
                write_coordinates();
                writer.write_bits(data.iso, 8);
                writer.write_bits(float_to_fp16_bits(data.contrast), 16);
                writer.write_bits(float_to_fp16_bits(data.brightness), 16);

                if (data.depth == 4) {
                    const float* raw_floats = reinterpret_cast<const float*>(data.raw_pixels_bytes);
                    for (int i = 0; i < 4; i++) {
                        writer.write_bits(float_to_fp16_bits(raw_floats[i]), 16);
                    }
                }
            }
        }

        if (skip_count > 0) {
            writer.write_bits(0xFFFF, 16);
            writer.write_bits(skip_count, 32);
        }

        std::vector<uint8_t> packed_bytes = writer.get_bytes();

        size_t max_dst_size = ZSTD_compressBound(packed_bytes.size());
        std::vector<uint8_t> compressed_data(max_dst_size);
        size_t cSize = ZSTD_compress(compressed_data.data(), max_dst_size, packed_bytes.data(), packed_bytes.size(), 15);
        
        if (ZSTD_isError(cSize)) {
            std::cerr << "ZSTD Compression failed: " << ZSTD_getErrorName(cSize) << "\n";
            return {};
        }

        compressed_data.resize(cSize);
        return compressed_data;
    }

    // Takes the Zstd buffer and returns the reconstructed C++ structures ready for CUDA
    static std::vector<HybridCodeData> decompress_bitstream(const std::vector<uint8_t>& compressed_data, int width) {
        unsigned long long const uncompressed_size = ZSTD_getFrameContentSize(compressed_data.data(), compressed_data.size());
        
        if (uncompressed_size == ZSTD_CONTENTSIZE_UNKNOWN || uncompressed_size == ZSTD_CONTENTSIZE_ERROR) {
            std::cerr << "ZSTD Decompression size unknown or error.\n";
            return {};
        }

        std::vector<uint8_t> packed_bytes(uncompressed_size);
        size_t dSize = ZSTD_decompress(packed_bytes.data(), uncompressed_size, compressed_data.data(), compressed_data.size());
        
        if (ZSTD_isError(dSize)) {
            std::cerr << "ZSTD Decompression failed: " << ZSTD_getErrorName(dSize) << "\n";
            return {};
        }

        BitReader reader(packed_bytes);
        std::vector<HybridCodeData> codes;
        
        int prev_idx = 0;
        int grid_width = width / 4;

        uint32_t dom_idx_val;
        while (reader.read_bits(16, dom_idx_val)) {
            uint16_t dom_idx = static_cast<uint16_t>(dom_idx_val);

            if (dom_idx == 0xFFFF) {
                uint32_t skip_count;
                if (!reader.read_bits(32, skip_count)) break;
                // Skips are dropped inherently here as desired by the CUDA decoder architecture
                continue;
            }

            HybridCodeData data = {};
            data.dom_idx = dom_idx;

            uint32_t depth_val;
            reader.read_bits(8, depth_val);
            data.depth = static_cast<uint8_t>(depth_val);

            uint32_t delta_idx_val;
            reader.read_bits(8, delta_idx_val);
            if (delta_idx_val == 255) {
                reader.read_bits(16, delta_idx_val);
            }

            int curr_idx = prev_idx + static_cast<int>(delta_idx_val);
            prev_idx = curr_idx;

            data.x = (curr_idx % grid_width) * 4;
            data.y = (curr_idx / grid_width) * 4;

            if (dom_idx == 0xFFFE) {
                uint32_t contrast_val, brightness_val;
                reader.read_bits(8, contrast_val);
                reader.read_bits(8, brightness_val);
                data.contrast = static_cast<float>(static_cast<int>(contrast_val) - 128);
                data.brightness = static_cast<float>(static_cast<int>(brightness_val) - 128);
            } 
            else if (dom_idx == 0xFFFD) {
                uint32_t contrast_val, brightness_val;
                reader.read_bits(8, contrast_val);
                reader.read_bits(8, brightness_val);
                data.contrast = static_cast<float>(static_cast<int>(contrast_val) - 128);
                data.brightness = static_cast<float>(static_cast<int>(brightness_val) - 128);

                for (int i = 0; i < 16; i++) {
                    uint32_t res_val;
                    reader.read_bits(8, res_val);
                    data.raw_pixels_bytes[i] = static_cast<int8_t>(static_cast<int>(res_val) - 128);
                }
            } 
            else {
                uint32_t iso_val, contrast_bits, brightness_bits;
                reader.read_bits(8, iso_val);
                reader.read_bits(16, contrast_bits);
                reader.read_bits(16, brightness_bits);
                
                data.iso = static_cast<uint8_t>(iso_val);
                data.contrast = fp16_bits_to_float(static_cast<uint16_t>(contrast_bits));
                data.brightness = fp16_bits_to_float(static_cast<uint16_t>(brightness_bits));

                if (data.depth == 4) {
                    float* raw_floats = reinterpret_cast<float*>(data.raw_pixels_bytes);
                    for (int i = 0; i < 4; i++) {
                        uint32_t p_val_bits;
                        reader.read_bits(16, p_val_bits);
                        raw_floats[i] = fp16_bits_to_float(static_cast<uint16_t>(p_val_bits));
                    }
                }
            }
            codes.push_back(data);
        }

        return codes;
    }
};