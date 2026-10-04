#include <fractal/core/VideoDecoder.h>
#include <fractal/core/FractalCodec.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

namespace fractal {
namespace core {

VideoDecoder::VideoDecoder(const std::string& input_filepath, const std::string& output_filepath)
    : input_filepath_(input_filepath), output_filepath_(output_filepath) {}

VideoDecoder::~VideoDecoder() = default;

bool VideoDecoder::run() {
    std::ifstream infile(input_filepath_, std::ios::binary);
    if (!infile.is_open()) {
        std::cerr << "Failed to open input file: " << input_filepath_ << std::endl;
        return false;
    }

    // 1. Read Global Header
    char magic[4];
    infile.read(magic, 4);
    if (std::string(magic, 4) != "FRAC") {
        std::cerr << "Invalid file format: Not a FRAC file." << std::endl;
        return false;
    }

    int width = 0, height = 0, total_frames = 0;
    double fps = 0.0;
    uint8_t chroma_format = 0;

    infile.read(reinterpret_cast<char*>(&width), sizeof(width));
    infile.read(reinterpret_cast<char*>(&height), sizeof(height));
    infile.read(reinterpret_cast<char*>(&fps), sizeof(fps));
    infile.read(reinterpret_cast<char*>(&total_frames), sizeof(total_frames));
    infile.read(reinterpret_cast<char*>(&chroma_format), sizeof(chroma_format));

    std::cout << "Starting Decode: " << width << "x" << height << " @ " << fps << "fps (" << total_frames << " frames)\n";
    std::cout << "Chroma Format: " << (chroma_format == 1 ? "4:2:0" : "4:4:4") << "\n";

    // 2. Setup OpenCV Video Writer
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter writer(output_filepath_, fourcc, fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Failed to open output video writer: " << output_filepath_ << std::endl;
        return false;
    }

    // 3. Setup CUDA Decoders
    int c_width = (chroma_format == 1) ? (width / 2) : width;
    int c_height = (chroma_format == 1) ? (height / 2) : height;

    FractalDecoderHandle dec_y = CreateFractalDecoder(width, height);
    FractalDecoderHandle dec_cb = CreateFractalDecoder(c_width, c_height);
    FractalDecoderHandle dec_cr = CreateFractalDecoder(c_width, c_height);

    // Buffers for reading compressed data
    std::vector<uint8_t> comp_y, comp_cb, comp_cr;

    // Buffers for raw decoded float outputs (0.0 to 1.0)
    std::vector<float> raw_y(width * height);
    std::vector<float> raw_cb(c_width * c_height);
    std::vector<float> raw_cr(c_width * c_height);

    for (int frame_idx = 0; frame_idx < total_frames; frame_idx++) {
        // 4. Read Chunk Sizes & Payloads
        int size_y = 0, size_cb = 0, size_cr = 0;

        infile.read(reinterpret_cast<char*>(&size_y), sizeof(size_y));
        if (infile.eof()) break; // Prevent reading past end of file safely
        comp_y.resize(size_y);
        infile.read(reinterpret_cast<char*>(comp_y.data()), size_y);

        infile.read(reinterpret_cast<char*>(&size_cb), sizeof(size_cb));
        comp_cb.resize(size_cb);
        infile.read(reinterpret_cast<char*>(comp_cb.data()), size_cb);

        infile.read(reinterpret_cast<char*>(&size_cr), sizeof(size_cr));
        comp_cr.resize(size_cr);
        infile.read(reinterpret_cast<char*>(comp_cr.data()), size_cr);

        // 5. Decode on GPU
        DecodeFractalFrame(dec_y, comp_y.data(), size_y, raw_y.data());
        DecodeFractalFrame(dec_cb, comp_cb.data(), size_cb, raw_cb.data());
        DecodeFractalFrame(dec_cr, comp_cr.data(), size_cr, raw_cr.data());

        // 6. Denormalize and Rescale to YUV cv::Mat
        cv::Mat mat_y(height, width, CV_8UC1);
        cv::Mat mat_cb(c_height, c_width, CV_8UC1);
        cv::Mat mat_cr(c_height, c_width, CV_8UC1);

        for (int i = 0; i < width * height; i++) {
            mat_y.data[i] = static_cast<uint8_t>(std::clamp(raw_y[i] * 255.0f, 0.0f, 255.0f));
        }
        for (int i = 0; i < c_width * c_height; i++) {
            mat_cb.data[i] = static_cast<uint8_t>(std::clamp(raw_cb[i] * 255.0f, 0.0f, 255.0f));
            mat_cr.data[i] = static_cast<uint8_t>(std::clamp(raw_cr[i] * 255.0f, 0.0f, 255.0f));
        }

        // Upscale Chroma if 4:2:0
        if (chroma_format == 1) { 
            cv::resize(mat_cb, mat_cb, cv::Size(width, height), 0, 0, cv::INTER_CUBIC);
            cv::resize(mat_cr, mat_cr, cv::Size(width, height), 0, 0, cv::INTER_CUBIC);
        }

        // 7. Merge channels & Convert back to BGR
        std::vector<cv::Mat> channels = {mat_y, mat_cb, mat_cr};
        cv::Mat yuv_frame, bgr_frame;
        cv::merge(channels, yuv_frame);
        cv::cvtColor(yuv_frame, bgr_frame, cv::COLOR_YUV2BGR);

        // 8. Write frame to MP4
        writer.write(bgr_frame);
        
        std::cout << "\rDecoded frame " << frame_idx + 1 << "/" << total_frames << std::flush;
    }
    
    std::cout << "\nDecoding complete! Output saved to: " << output_filepath_ << std::endl;

    // 9. Cleanup GPU resources
    DestroyFractalDecoder(dec_y);
    DestroyFractalDecoder(dec_cb);
    DestroyFractalDecoder(dec_cr);
    
    return true;
}

} // namespace core
} // namespace fractal
