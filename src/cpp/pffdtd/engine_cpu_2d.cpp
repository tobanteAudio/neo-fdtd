// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#include "engine_cpu_2d.hpp"

#include "pffdtd/progress.hpp"
#include "pffdtd/time.hpp"

#include <fmt/format.h>

#include <omp.h>

#include <algorithm>
#include <concepts>
#include <ranges>

namespace pffdtd {

[[nodiscard]] constexpr auto
to_ixy(std::integral auto x, std::integral auto y, std::integral auto /*Nx*/, std::integral auto Ny) -> std::integral
    auto {
  return x * Ny + y;
}

auto EngineCPU2D::operator()(Simulation2D const& sim) const -> stdex::mdarray<double, stdex::dextents<size_t, 2>> {

  auto const Nx         = sim.Nx;
  auto const Ny         = sim.Ny;
  auto const Nt         = sim.Nt;
  auto const Nb         = static_cast<int64_t>(sim.adj_bn.size());
  auto const inx        = sim.inx;
  auto const iny        = sim.iny;
  auto const Nr         = static_cast<int64_t>(sim.out_ixy.size());
  auto const lossFactor = sim.loss_factor;

  summary(sim);

  auto u0_buf  = stdex::mdarray<double, stdex::dextents<size_t, 2>>(Nx, Ny);
  auto u1_buf  = stdex::mdarray<double, stdex::dextents<size_t, 2>>(Nx, Ny);
  auto u2_buf  = stdex::mdarray<double, stdex::dextents<size_t, 2>>(Nx, Ny);
  auto out_buf = stdex::mdarray<double, stdex::dextents<size_t, 2>>(Nr, Nt);

  auto u0  = u0_buf.to_mdspan();
  auto u1  = u1_buf.to_mdspan();
  auto u2  = u2_buf.to_mdspan();
  auto out = out_buf.to_mdspan();

  auto elapsedAir      = std::chrono::nanoseconds{0};
  auto elapsedBoundary = std::chrono::nanoseconds{0};

  auto const numWorkers = omp_get_max_threads();
  auto const start      = getTime();

  for (auto n{0LL}; n < Nt; ++n) {
    auto const sampleStart = getTime();

    auto const elapsedAirSample = timeit([&] {
// Air Update
#pragma omp parallel for
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
    });

    auto const elapsedBoundarySample = timeit([&] {
// Boundary Rigid
#pragma omp parallel for
      for (int64_t i = 0; i < Nb; ++i) {
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
#pragma omp parallel for
      for (int64_t i = 0; i < Nb; ++i) {
        auto const ib = sim.bn_ixy[i];
        auto const K  = sim.adj_bn[i];
        auto const K4 = 4 - K;
        auto const lf = lossFactor;

        auto const current = u0.data_handle()[ib];
        auto const prev    = u2.data_handle()[ib];

        u0.data_handle()[ib] = (current + lf * K4 * prev) / (1 + lf * K4);
      }
    });

    // Copy Input
    u0(inx, iny) += sim.src_sig[n];

    // Copy Output
    for (int64_t i = 0; i < Nr; ++i) {
      auto r_ixy = sim.out_ixy[i];
      out(i, n)  = u0.data_handle()[r_ixy];
    }

    auto tmp = u2;
    u2       = u1;
    u1       = u0;
    u0       = tmp;

    auto const now = getTime();

    auto const elapsed       = now - start;
    auto const elapsedSample = now - sampleStart;
    elapsedAir += elapsedAirSample;
    elapsedBoundary += elapsedBoundarySample;

    print(ProgressReport{
        .n                     = n,
        .Nt                    = Nt,
        .Npts                  = Nx * Ny,
        .Nb                    = static_cast<int64_t>(Nb),
        .elapsed               = elapsed,
        .elapsedSample         = elapsedSample,
        .elapsedAir            = elapsedAir,
        .elapsedSampleAir      = elapsedAirSample,
        .elapsedBoundary       = elapsedBoundary,
        .elapsedSampleBoundary = elapsedBoundarySample,
        .numWorkers            = numWorkers,
    });
  }

  fmt::println("");

  return out_buf;
}

} // namespace pffdtd
