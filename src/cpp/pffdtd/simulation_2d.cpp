#include "simulation_2d.hpp"

#include "pffdtd/hdf.hpp"

#include <fmt/format.h>

#include <opencv2/imgproc.hpp>

namespace pffdtd {

auto loadSimulation2D(std::filesystem::path const& dir, bool video)
    -> Simulation2D {
  // auto constants = H5FReader{(dir / "constants.h5").string().c_str()};
  auto sim = H5FReader{(dir / "sim.h5").string().c_str()};

  auto const Nx = sim.read<int64_t>("Nx");
  auto const Ny = sim.read<int64_t>("Ny");

  auto const videoRatio   = static_cast<double>(Ny) / static_cast<double>(Nx);
  auto const videoWidth   = std::min<size_t>(2000, static_cast<size_t>(Nx));
  auto const videoOptions = VideoWriter::Options{
      .file      = dir.parent_path() / "out.avi",
      .width     = videoWidth,
      .height    = static_cast<size_t>(videoWidth * videoRatio),
      .fps       = sim.read<double>("video_fps"),
      .withColor = false,
  };

  return Simulation2D{
      .dir = dir,

      .Nx = Nx,
      .Ny = Ny,
      .Nt = sim.read<int64_t>("Nt"),

      .in_mask     = sim.read<std::vector<uint8_t>>("in_mask"),
      .adj_bn      = sim.read<std::vector<int64_t>>("adj_bn"),
      .bn_ixy      = sim.read<std::vector<int64_t>>("bn_ixy"),
      .loss_factor = sim.read<double>("loss_factor"),

      .inx     = sim.read<int64_t>("inx"),
      .iny     = sim.read<int64_t>("iny"),
      .src_sig = sim.read<std::vector<double>>("src_sig"),

      .out_ixy = sim.read<std::vector<int64_t>>("out_ixy"),

      .videoOptions = video ? std::optional{videoOptions} : std::nullopt,
  };
}

auto summary(Simulation2D const& sim) -> void {
  fmt::println("Nt: {}", sim.Nt);
  fmt::println("Nx: {}", sim.Nx);
  fmt::println("Ny: {}", sim.Ny);
  fmt::println("N: {}", sim.Nx * sim.Ny);
  fmt::println("inx: {}", sim.inx);
  fmt::println("iny: {}", sim.iny);
  fmt::println("in_mask: {}", sim.in_mask.size());
  fmt::println("bn_ixy: {}", sim.bn_ixy.size());
  fmt::println("adj_bn: {}", sim.adj_bn.size());
  fmt::println("out_ixy: {}", sim.out_ixy.size());
  fmt::println("src_sig: {}", sim.src_sig.size());
  fmt::println("loss_factor: {}", sim.loss_factor);
}

BackgroundVideoWriter::BackgroundVideoWriter(VideoWriter::Options const& opt)
    : writer{opt}
    , useColor{opt.withColor} {}

auto BackgroundVideoWriter::run(Simulation2D const& sim) -> void {

  auto frame      = std::vector<double>{};
  auto normalized = cv::Mat{};
  auto colored    = cv::Mat{};
  auto rotated    = cv::Mat{};

  while (not done or not queue.empty()) {
    auto shouldSleep = false;
    {
      auto lock   = std::scoped_lock{mutex};
      shouldSleep = queue.empty();
    }

    if (shouldSleep) {
      std::this_thread::sleep_for(std::chrono::milliseconds{10});
      continue;
    }

    {
      auto lock = std::scoped_lock{mutex};
      frame     = queue.front();
      queue.pop();
    }

    if (not useColor) {
      std::ranges::transform(frame, frame.begin(), [](auto v) {
        return std::abs(v);
      });
    }

    auto input = cv::Mat{
        static_cast<int>(sim.Nx),
        static_cast<int>(sim.Ny),
        CV_64F,
        static_cast<void*>(frame.data()),
    };

    cv::normalize(input, normalized, 0, 255, cv::NORM_MINMAX);
    normalized.convertTo(normalized, CV_8U);

    if (useColor) {
      cv::applyColorMap(normalized, colored, cv::COLORMAP_VIRIDIS);
      for (auto ix{0L}; ix < sim.Nx; ++ix) {
        for (auto iy{0L}; iy < sim.Ny; ++iy) {
          if (not sim.in_mask[ix * sim.Ny + iy]) {
            colored.at<cv::Vec3b>(ix, iy) = cv::Vec3b(255, 255, 255);
          }
        }
      }
    } else {
      colored = normalized;
      for (auto ix{0L}; ix < sim.Nx; ++ix) {
        for (auto iy{0L}; iy < sim.Ny; ++iy) {
          if (not sim.in_mask[ix * sim.Ny + iy]) {
            colored.at<uint8_t>(ix, iy) = 255;
          }
        }
      }
    }

    cv::rotate(colored, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);

    writer.write(rotated);
  }
}

auto BackgroundVideoWriter::push(std::vector<double> frame) -> void {
  while (true) {
    auto const wait = [&] {
      auto lock = std::scoped_lock{mutex};
      return queue.size() > 10;
    }();

    if (wait) {
      std::this_thread::sleep_for(std::chrono::milliseconds{10});
    } else {
      break;
    }
  }

  {
    auto lock = std::scoped_lock{mutex};
    queue.push(std::move(frame));
  }
}

auto BackgroundVideoWriter::finish() -> void { done.store(true); }

} // namespace pffdtd
