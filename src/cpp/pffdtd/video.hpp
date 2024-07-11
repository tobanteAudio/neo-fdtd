#pragma once

#include "pffdtd/exception.hpp"

#include <opencv2/videoio.hpp>

#include <filesystem>
#include <span>

namespace pffdtd {

struct VideoWriter {
  VideoWriter(
      std::filesystem::path const& filename,
      double fps,
      size_t width,
      size_t height
  );

  auto write(cv::InputArray frame) -> void;

  private:
  cv::VideoWriter _writer;
  cv::Size _size;
};

} // namespace pffdtd
