#include "video.hpp"

#include <opencv2/imgproc.hpp>

namespace pffdtd {

VideoWriter::VideoWriter(
    std::filesystem::path const& filename,
    double fps,
    size_t width,
    size_t height
)
    : _size{int(width), int(height)} {
  auto const isColor = false;
  auto const file    = filename.string();
  auto const codec   = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
  _writer.open(file, codec, fps, _size, isColor);
  if (not _writer.isOpened()) {
    raisef<std::runtime_error>("could not open video writer for: {}", file);
  }
}

auto VideoWriter::write(std::span<double> buf, size_t width, size_t height)
    -> void {
  auto const input = cv::Mat{
      static_cast<int>(width),
      static_cast<int>(height),
      CV_64F,
      static_cast<void*>(buf.data()),
  };

  auto normalized = cv::Mat{};
  cv::normalize(input, normalized, 0, 255, cv::NORM_MINMAX);
  normalized.convertTo(normalized, CV_8U);

  auto resized = cv::Mat{};
  cv::resize(input, resized, _size);

  _writer.write(resized);
}

} // namespace pffdtd
