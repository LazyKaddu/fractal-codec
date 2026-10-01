#pragma once
#include <cstdint>
#include <cuda_runtime.h>

#define IMAGE_WIDTH 1024
#define IMAGE_HEIGHT 1024

// Thresholds for the Hybrid Architecture
const float TEMPORAL_SKIP_THRESHOLD = 0.065f; // If difference is less than this, do nothing
const float UI_4x4_THRESHOLD        = 0.0005f; // Must be nearly perfect to stay at 4x4
const float MAX_PIXEL_ERROR         = 0.05f;   // Catch sharp text edges

struct HybridCode {
    uint16_t x;
    uint16_t y;
    uint16_t domain_idx; 
    uint8_t depth;       // 0=32x32, 1=16x16, 2=8x8, 3=4x4, 4=Raw(2x2)
    uint8_t isometry_id;
    float contrast;
    float brightness;
    float raw_pixels[4]; // Used if depth == 4
};