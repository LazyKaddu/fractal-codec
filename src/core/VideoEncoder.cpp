#include <fractal/core/VideoEncoder.h>
#include <fractal/io/VideoReader.h>
#include <fractal/core/FractalCodec.h>
#include <fstream>
#include <iostream>
#include <vector>

namespace fractal {
namespace core {

VideoEncoder::VideoEncoder(const std::string& input_filepath, const std::string& output_filepath)
    : input_filepath_(input_filepath), output_filepath_(output_filepath) {}

VideoEncoder::~VideoEncoder() = default;

bool VideoEncoder::run() {
    fractal::io::VideoReader reader(input_filepath_);
    if (!reader.isOpened()) {
        std::cerr << "Failed to open input video: " << input_filepath_ << std::endl;
        return false;
    }

    int width = reader.getWidth();
    int height = reader.getHeight();
    double fps = reader.getFPS();
    int total_frames = reader.getTotalFrames();

    std::cout << "Starting Encode: " << width << "x" << height << " @ " << fps << "fps (" << total_frames << " frames)\n";

    // Create 3 separate encoder states on the GPU (Y, U, V)
    // We are using 4:2:0 chroma subsampling to save 50% data
    FractalEncoderHandle enc_y = CreateFractalEncoder(width, height);
    FractalEncoderHandle enc_cb = CreateFractalEncoder(width / 2, height / 2);
    FractalEncoderHandle enc_cr = CreateFractalEncoder(width / 2, height / 2);

    std::ofstream outfile(output_filepath_, std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file: " << output_filepath_ << std::endl;
        return false;
    }

    // 1. Write the Global Header
    outfile.write("FRAC", 4); // Magic bytes
    outfile.write(reinterpret_cast<const char*>(&width), sizeof(width));
    outfile.write(reinterpret_cast<const char*>(&height), sizeof(height));
    outfile.write(reinterpret_cast<const char*>(&fps), sizeof(fps));
    outfile.write(reinterpret_cast<const char*>(&total_frames), sizeof(total_frames));
    
    uint8_t chroma_format = 1; // 1 = 4:2:0
    outfile.write(reinterpret_cast<const char*>(&chroma_format), sizeof(chroma_format));

    cv::Mat frame, yuv_frame;
    std::vector<cv::Mat> channels(3);
    
    // Allocate float arrays to feed the CUDA encoder (values 0.0 to 1.0)
    std::vector<float> raw_y(width * height);
    std::vector<float> raw_cb((width / 2) * (height / 2));
    std::vector<float> raw_cr((width / 2) * (height / 2));
    
    // Pre-allocate safety buffers for the compressed outputs
    int max_out_size = width * height * sizeof(float);
    std::vector<uint8_t> comp_y(max_out_size);
    std::vector<uint8_t> comp_cb(max_out_size / 4);
    std::vector<uint8_t> comp_cr(max_out_size / 4);

    int frame_idx = 0;
    while (reader.readFrame(frame)) {
        // 2. Convert Color Space, Split, and Subsample
        cv::cvtColor(frame, yuv_frame, cv::COLOR_BGR2YUV);
        cv::split(yuv_frame, channels);
        
        cv::Mat cb_down, cr_down;
        cv::resize(channels[1], cb_down, cv::Size(width / 2, height / 2), 0, 0, cv::INTER_AREA);
        cv::resize(channels[2], cr_down, cv::Size(width / 2, height / 2), 0, 0, cv::INTER_AREA);
        
        // 3. Normalize channels to float (0.0f - 1.0f)
        for (int i = 0; i < width * height; i++) {
            raw_y[i] = channels[0].data[i] / 255.0f;
        }
        for (int i = 0; i < (width / 2) * (height / 2); i++) {
            raw_cb[i] = cb_down.data[i] / 255.0f;
            raw_cr[i] = cr_down.data[i] / 255.0f;
        }

        // 4. Encode each channel
        int size_y = EncodeFractalFrame(enc_y, raw_y.data(), comp_y.data(), max_out_size);
        int size_cb = EncodeFractalFrame(enc_cb, raw_cb.data(), comp_cb.data(), max_out_size / 4);
        int size_cr = EncodeFractalFrame(enc_cr, raw_cr.data(), comp_cr.data(), max_out_size / 4);

        if (size_y < 0 || size_cb < 0 || size_cr < 0) {
            std::cerr << "\nEncoding failed at frame " << frame_idx << std::endl;
            break;
        }

        // 5. Write Frame Payload to disk
        // Y Channel
        outfile.write(reinterpret_cast<const char*>(&size_y), sizeof(size_y));
        outfile.write(reinterpret_cast<const char*>(comp_y.data()), size_y);
        
        // Cb Channel
        outfile.write(reinterpret_cast<const char*>(&size_cb), sizeof(size_cb));
        outfile.write(reinterpret_cast<const char*>(comp_cb.data()), size_cb);
        
        // Cr Channel
        outfile.write(reinterpret_cast<const char*>(&size_cr), sizeof(size_cr));
        outfile.write(reinterpret_cast<const char*>(comp_cr.data()), size_cr);

        frame_idx++;
        std::cout << "\rEncoded frame " << frame_idx << "/" << total_frames << " (Payload: " 
                  << (size_y + size_cb + size_cr) / 1024 << " KB)    " << std::flush;
    }
    
    std::cout << "\nEncoding complete! Output saved to: " << output_filepath_ << std::endl;

    // 6. Cleanup GPU resources
    DestroyFractalEncoder(enc_y);
    DestroyFractalEncoder(enc_cb);
    DestroyFractalEncoder(enc_cr);
    
    return true;
}

} // namespace core
} // namespace fractal
