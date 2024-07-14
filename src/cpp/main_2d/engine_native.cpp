#include "engine_native.hpp"

#include "pffdtd/video.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <concepts>
#include <ranges>

namespace pffdtd {

[[nodiscard]] constexpr auto to_ixy(
    std::integral auto x,
    std::integral auto y,
    std::integral auto /*Nx*/,
    std::integral auto Ny
) -> std::integral auto {
  return x * Ny + y;
}

auto EngineNative::operator()(Simulation2D const& sim) const
    -> stdex::mdarray<double, stdex::dextents<size_t, 2>> {

  auto const Nx         = sim.Nx;
  auto const Ny         = sim.Ny;
  auto const Nt         = sim.Nt;
  auto const Nb         = sim.adj_bn.size();
  auto const inx        = sim.inx;
  auto const iny        = sim.iny;
  auto const Nr         = sim.out_ixy.size();
  auto const lossFactor = sim.loss_factor;

  pffdtd::summary(sim);

  auto shouldRenderVideo = sim.videoOptions.has_value();
  auto videoWriter       = std::unique_ptr<BackgroundVideoWriter>();
  auto videoThread       = std::unique_ptr<std::thread>();

  if (shouldRenderVideo) {
    auto opt    = sim.videoOptions.value();
    auto func   = [&videoWriter, &sim] { run(*videoWriter, sim); };
    videoWriter = std::make_unique<BackgroundVideoWriter>(VideoWriter{opt});
    videoThread = std::make_unique<std::thread>(func);
  }

  auto u0_buf  = stdex::mdarray<double, stdex::dextents<size_t, 2>>(Nx, Ny);
  auto u1_buf  = stdex::mdarray<double, stdex::dextents<size_t, 2>>(Nx, Ny);
  auto u2_buf  = stdex::mdarray<double, stdex::dextents<size_t, 2>>(Nx, Ny);
  auto out_buf = stdex::mdarray<double, stdex::dextents<size_t, 2>>(Nr, Nt);

  auto u0  = u0_buf.to_mdspan();
  auto u1  = u1_buf.to_mdspan();
  auto u2  = u2_buf.to_mdspan();
  auto out = out_buf.to_mdspan();

  for (auto n{0LL}; n < Nt; ++n) {
    fmt::print(stdout, "\r\r\r\r\r\r\r\r\r");
    fmt::print(stdout, "{:04d}/{:04d}", n, Nt);
    std::fflush(stdout);

    // Air Update
    for (int64_t x = 1; x < Nx - 1; ++x) {
      for (int64_t y = 1; y < Ny - 1; ++y) {
        auto const idx    = to_ixy(x, y, 0, Ny);
        auto const left   = u1.data_handle()[idx - 1];
        auto const right  = u1.data_handle()[idx + 1];
        auto const bottom = u1.data_handle()[idx - Ny];
        auto const top    = u1.data_handle()[idx + Ny];
        auto const last   = u2.data_handle()[idx];
        auto const tmp    = 0.5 * (left + right + bottom + top) - last;

        u0.data_handle()[idx] = sim.in_mask[idx] * tmp;
      }
    }

    // Boundary Rigid
    for (size_t i = 0; i < Nb; ++i) {
      auto const ib = sim.bn_ixy[i];
      auto const K  = sim.adj_bn[i];

      auto const last1 = u1.data_handle()[ib];
      auto const last2 = u2.data_handle()[ib];

      auto const left      = u1.data_handle()[ib - 1];
      auto const right     = u1.data_handle()[ib + 1];
      auto const bottom    = u1.data_handle()[ib - Ny];
      auto const top       = u1.data_handle()[ib + Ny];
      auto const neighbors = left + right + top + bottom;

      u0.data_handle()[ib] = (2 - 0.5 * K) * last1 + 0.5 * neighbors - last2;
    }

    // Boundary Loss
    for (size_t i = 0; i < Nb; ++i) {
      auto const ib = sim.bn_ixy[i];
      auto const K  = sim.adj_bn[i];
      auto const K4 = 4 - K;
      auto const lf = lossFactor;

      auto const current = u0.data_handle()[ib];
      auto const prev    = u2.data_handle()[ib];

      u0.data_handle()[ib] = (current + lf * K4 * prev) / (1 + lf * K4);
    }

    // Copy Input
    u0(inx, iny) += sim.src_sig[n];

    // Copy Output
    for (size_t i = 0; i < Nr; ++i) {
      auto r_ixy = sim.out_ixy[i];
      out(i, n)  = u0.data_handle()[r_ixy];
    }

    if (shouldRenderVideo) {
      auto frame = u0_buf.container();
      std::ranges::transform(frame, frame.begin(), [](auto v) {
        return std::abs(v);
      });
      push(*videoWriter, frame);
    }

    auto tmp = u2;
    u2       = u1;
    u1       = u0;
    u0       = tmp;
  }

  if (shouldRenderVideo) {
    videoWriter->done.store(true);
    videoThread->join();
  }

  fmt::println("");

  return out_buf;
}

} // namespace pffdtd
