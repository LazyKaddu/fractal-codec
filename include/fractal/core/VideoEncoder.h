#pragma once

#include <string>

namespace fractal {
namespace core {

class VideoEncoder {
public:
    /**
     * @brief Construct a new Video Encoder to orchestrate the pipeline
     * 
     * @param input_filepath Path to the input video file (e.g. .mp4)
     * @param output_filepath Path to the output compressed binary (e.g. .frc)
     */
    VideoEncoder(const std::string& input_filepath, const std::string& output_filepath);
    ~VideoEncoder();

    /**
     * @brief Run the encoding pipeline.
     * @return true if successful, false if an error occurred.
     */
    bool run();

private:
    std::string input_filepath_;
    std::string output_filepath_;
};

} // namespace core
} // namespace fractal
