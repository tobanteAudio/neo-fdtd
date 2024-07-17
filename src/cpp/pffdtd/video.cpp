#include "video.hpp"

#include "pffdtd/exception.hpp"

#include <opencv2/imgproc.hpp>

namespace pffdtd {

VideoWriter::VideoWriter(Options const& options)
    : _size{int(options.width), int(options.height)} {
  auto const file  = options.file.string();
  auto const codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
  _writer.open(file, codec, options.fps, _size, options.withColor);
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
