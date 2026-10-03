#pragma once

#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

namespace fractal {
namespace io {

class VideoReader {
public:
    /**
     * @brief Open a video file for streaming frames.
     * @param filepath Path to the video file (e.g., .braw, .mp4)
     */
    explicit VideoReader(const std::string& filepath);
    ~VideoReader();

    // Prevent copying
    VideoReader(const VideoReader&) = delete;
    VideoReader& operator=(const VideoReader&) = delete;

    /**
     * @brief Check if the video was successfully opened.
     */
    bool isOpened() const;

    /**
     * @brief Set the target resolution. Frames will be automatically resized to this if set.
     * @param width The target width.
     * @param height The target height.
     */
    void setTargetResolution(int width, int height);

    /**
     * @brief Reads the next frame in the stream without loading the whole video into RAM.
     * @param frame The output frame.
     * @return true if a frame was read, false if the end of the video is reached or an error occurred.
     */
    bool readFrame(cv::Mat& frame);

    /**
     * @brief Get the next frame, checking against total frames.
     * @return The frame, or an empty cv::Mat if at the end.
     */
    cv::Mat getNextFrame();

    /**
     * @brief Get the current frame index.
     */
    int getCurrentFrameIndex() const;


    /**
     * @brief Get the total number of frames in the video.
     */
    int getTotalFrames() const;

    /**
     * @brief Get the frame rate (frames per second).
     */
    double getFPS() const;

    /**
     * @brief Get the width of the video frames.
     */
    int getWidth() const;

    /**
     * @brief Get the height of the video frames.
     */
    int getHeight() const;

private:
    std::unique_ptr<cv::VideoCapture> cap_;
    std::string filepath_;
    int current_frame_index_ = 0;
    int target_width_ = -1;
    int target_height_ = -1;
};

} // namespace io
} // namespace fractal
