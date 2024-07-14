#pragma once

#include <opencv2/videoio.hpp>

#include <filesystem>

namespace pffdtd {

struct VideoWriter {
  struct Options {
    std::filesystem::path file;
    size_t width;
    size_t height;
    double fps;
  };

  explicit VideoWriter(Options const& options);

  auto write(cv::InputArray frame) -> void;

  private:
  cv::VideoWriter _writer;
  cv::Size _size;
};

} // namespace pffdtd
