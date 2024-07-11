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

auto VideoWriter::write(cv::InputArray frame) -> void {
  if (_size != frame.size()) {
    auto resized = cv::Mat{};
    cv::resize(frame, resized, _size, 0, 0, cv::INTER_AREA);
    _writer.write(resized);
    return;
  }

  _writer.write(frame);
}

} // namespace pffdtd
