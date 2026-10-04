#pragma once

#include <string>

namespace fractal {
namespace core {

class VideoDecoder {
public:
    /**
     * @brief Construct a new Video Decoder to orchestrate the pipeline
     * 
     * @param input_filepath Path to the compressed binary file (e.g. .frc)
     * @param output_filepath Path to the output video file (e.g. .mp4)
     */
    VideoDecoder(const std::string& input_filepath, const std::string& output_filepath);
    ~VideoDecoder();

    /**
     * @brief Run the decoding pipeline.
     * @return true if successful, false if an error occurred.
     */
    bool run();

private:
    std::string input_filepath_;
    std::string output_filepath_;
};

} // namespace core
} // namespace fractal
