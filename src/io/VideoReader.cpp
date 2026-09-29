#include "fractal/io/VideoReader.h"
#include <iostream>

namespace fractal {
namespace io {

VideoReader::VideoReader(const std::string& filepath)
    : filepath_(filepath), cap_(std::make_unique<cv::VideoCapture>()) 
{
    // Open the video file. This initializes the stream without loading 
    // all frames into RAM, which is ideal for large files like .braw.
    if (!cap_->open(filepath_)) {
        std::cerr << "Error: Could not open video file: " << filepath_ << std::endl;
    }
}

VideoReader::~VideoReader() {
    if (cap_ && cap_->isOpened()) {
        cap_->release();
    }
}

bool VideoReader::isOpened() const {
    return cap_ && cap_->isOpened();
}

bool VideoReader::readFrame(cv::Mat& frame) {
    if (!isOpened()) {
        return false;
    }
    // The read() function pulls the next frame from the stream and decodes it.
    // Memory is reused/allocated for the single frame, not the entire video.
    return cap_->read(frame);
}

int VideoReader::getTotalFrames() const {
    if (!isOpened()) return 0;
    return static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_COUNT));
}

double VideoReader::getFPS() const {
    if (!isOpened()) return 0.0;
    return cap_->get(cv::CAP_PROP_FPS);
}

int VideoReader::getWidth() const {
    if (!isOpened()) return 0;
    return static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_WIDTH));
}

int VideoReader::getHeight() const {
    if (!isOpened()) return 0;
    return static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_HEIGHT));
}

} // namespace io
} // namespace fractal
