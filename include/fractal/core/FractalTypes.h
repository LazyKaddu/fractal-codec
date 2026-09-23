#pragma once
#include <cstdint>

namespace fractal {
namespace core {

// Represents the mathematical transformation required to turn a Domain block into a Range block
struct AffineTransform {
    float contrast;     // Multiplier (scale)
    float brightness;   // Offset (shift)
    uint8_t rotation;   // 0=0deg, 1=90deg, 2=180deg, 3=270deg
    bool flipHorizontal; // Symmetry operations
};

// The final "compressed" data structure. Instead of pixels, we just store this mapping.
struct FractalCode {
    uint32_t range_x;           // Position of the target block
    uint32_t range_y;
    uint32_t domain_x;          // Position of the matching source block
    uint32_t domain_y;
    AffineTransform transform;  // The math to make them match
};

} // namespace core
} // namespace fractal