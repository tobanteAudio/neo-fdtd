// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#include "engine_sycl_2d.hpp"

#include "pffdtd/progress.hpp"
#include "pffdtd/sycl.hpp"
#include "pffdtd/time.hpp"

#include <fmt/format.h>

#include <concepts>

namespace pffdtd {

namespace {

[[nodiscard]] constexpr auto to_ixy(auto x, auto y, auto /*Nx*/, auto Ny) { return x * Ny + y; }

template<typename Real>
auto kernelAirUpdate(sycl::id<2> id, Real* u0, Real const* u1, Real const* u2, uint8_t const* inMask, int64_t Ny)
    -> void {
  auto const x   = id.get(0) + 1;
  auto const y   = id.get(1) + 1;
  auto const idx = to_ixy(x, y, 0, Ny);

  if (inMask[idx] == 0) {
    return;
  }

  auto const left   = u1[idx - 1];
  auto const right  = u1[idx + 1];
  auto const bottom = u1[idx - Ny];
  auto const top    = u1[idx + Ny];
  auto const last   = u2[idx];

  u0[idx] = Real(0.5) * (left + right + bottom + top) - last;
}

template<typename Real>
auto kernelBoundaryRigid(
    sycl::id<1> idx,
    Real* u0,
    Real const* u1,
    Real const* u2,
    int64_t const* bn_ixy,
    int64_t const* adj_bn,
    int64_t Ny
) -> void {
  auto const ib = bn_ixy[idx];
  auto const K  = adj_bn[idx];

  auto const last1 = u1[ib];
  auto const last2 = u2[ib];

  auto const left      = u1[ib - 1];
  auto const right     = u1[ib + 1];
  auto const bottom    = u1[ib - Ny];
  auto const top       = u1[ib + Ny];
  auto const neighbors = left + right + top + bottom;

  u0[ib] = (Real(2) - Real(0.5) * K) * last1 + Real(0.5) * neighbors - last2;
}

template<typename Real>
auto kernelBoundaryLoss(
    sycl::id<1> idx,
    Real* u0,
    Real const* u2,
    int64_t const* bn_ixy,
    int64_t const* adj_bn,
    Real lossFactor
) -> void {
  auto const ib      = bn_ixy[idx];
  auto const K       = adj_bn[idx];
  auto const current = u0[ib];
  auto const prev    = u2[ib];
  auto const K4      = 4 - K;

  u0[ib] = (current + lossFactor * K4 * prev) / (Real(1) + lossFactor * K4);
}
} // namespace

auto EngineSYCL2D::operator()(Simulation2D const& sim) const -> stdex::mdarray<double, stdex::dextents<size_t, 2>> {
  summary(sim);

  auto const Nx          = sim.Nx;
  auto const Ny          = sim.Ny;
  auto const Nt          = sim.Nt;
  auto const Nb          = sim.adj_bn.size();
  auto const inx         = sim.inx;
  auto const iny         = sim.iny;
  auto const Nr          = sim.out_ixy.size();
  auto const loss_factor = sim.loss_factor;

  auto queue  = sycl::queue{sycl::property::queue::enable_profiling{}};
  auto device = queue.get_device();
  summary(device);

  auto u0_buf  = sycl::buffer<double, 2>(sycl::range<2>(Nx, Ny));
  auto u1_buf  = sycl::buffer<double, 2>(sycl::range<2>(Nx, Ny));
  auto u2_buf  = sycl::buffer<double, 2>(sycl::range<2>(Nx, Ny));
  auto out_buf = sycl::buffer<double, 2>(sycl::range<2>(Nr, Nt));

  auto in_mask_buf = sycl::buffer<uint8_t, 1>{sim.in_mask};
  auto bn_ixy_buf  = sycl::buffer<int64_t, 1>{sim.bn_ixy};
  auto adj_bn_buf  = sycl::buffer<int64_t, 1>{sim.adj_bn};
  auto out_ixy_buf = sycl::buffer<int64_t, 1>{sim.out_ixy};
  auto src_sig_buf = sycl::buffer<double, 1>{sim.src_sig};

  auto elapsedAir      = std::chrono::nanoseconds{0};
  auto elapsedBoundary = std::chrono::nanoseconds{0};
  auto const start     = getTime();

  for (auto n{0LL}; n < Nt; ++n) {
    auto const sampleStart = getTime();

    auto const airEvent = queue.submit([&](sycl::handler& cgh) {
      auto u0      = sycl::accessor{u0_buf, cgh, sycl::write_only};
      auto u1      = sycl::accessor{u1_buf, cgh, sycl::read_only};
      auto u2      = sycl::accessor{u2_buf, cgh, sycl::read_only};
      auto in_mask = sycl::accessor{in_mask_buf, cgh, sycl::read_only};

      cgh.parallel_for<struct AirUpdate>(sycl::range<2>(Nx - 2, Ny - 2), [=](sycl::id<2> id) {
        kernelAirUpdate(id, getPtr(u0), getPtr(u1), getPtr(u2), getPtr(in_mask), Ny);
      });
    });

    auto const boundaryStartEvent = queue.submit([&](sycl::handler& cgh) {
      auto u0     = sycl::accessor{u0_buf, cgh, sycl::write_only};
      auto u1     = sycl::accessor{u1_buf, cgh, sycl::read_only};
      auto u2     = sycl::accessor{u2_buf, cgh, sycl::read_only};
      auto bn_ixy = sycl::accessor{bn_ixy_buf, cgh, sycl::read_only};
      auto adj_bn = sycl::accessor{adj_bn_buf, cgh, sycl::read_only};

      cgh.parallel_for<struct BoundaryRigid>(Nb, [=](sycl::id<1> id) {
        kernelBoundaryRigid(id, getPtr(u0), getPtr(u1), getPtr(u2), getPtr(bn_ixy), getPtr(adj_bn), Ny);
      });
    });

    auto const boundaryEndEvent = queue.submit([&](sycl::handler& cgh) {
      auto u0     = sycl::accessor{u0_buf, cgh, sycl::write_only};
      auto u2     = sycl::accessor{u2_buf, cgh, sycl::read_only};
      auto bn_ixy = sycl::accessor{bn_ixy_buf, cgh, sycl::read_only};
      auto adj_bn = sycl::accessor{adj_bn_buf, cgh, sycl::read_only};

      cgh.parallel_for<struct BoundaryLoss>(Nb, [=](sycl::id<1> id) {
        kernelBoundaryLoss(id, getPtr(u0), getPtr(u2), getPtr(bn_ixy), getPtr(adj_bn), loss_factor);
      });
    });

    queue.submit([&](sycl::handler& cgh) {
      auto u0      = sycl::accessor{u0_buf, cgh, sycl::read_write};
      auto src_sig = sycl::accessor{src_sig_buf, cgh, sycl::read_only};

      cgh.parallel_for<struct CopyInput>(1, [=](sycl::id<1>) { u0[inx][iny] += src_sig[n]; });
    });

    queue.submit([&](sycl::handler& cgh) {
      auto u0      = sycl::accessor{u0_buf, cgh, sycl::read_only};
      auto out     = sycl::accessor{out_buf, cgh, sycl::write_only};
      auto out_ixy = sycl::accessor{out_ixy_buf, cgh, sycl::read_only};

      cgh.parallel_for<struct CopyOutput>(Nr, [=](sycl::id<1> id) {
        auto r         = id[0];
        auto r_ixy     = out_ixy[r];
        auto const* p0 = getPtr(u0);
        out[r][n]      = p0[r_ixy];
      });
    });

    queue.wait_and_throw();

    auto tmp = u2_buf;
    u2_buf   = u1_buf;
    u1_buf   = u0_buf;
    u0_buf   = tmp;

    auto const now = getTime();

    auto const elapsed       = now - start;
    auto const elapsedSample = now - sampleStart;

    auto const elapsedAirSample = elapsedTime(airEvent);
    elapsedAir += elapsedAirSample;

    auto const elapsedBoundarySample = elapsedTime(boundaryStartEvent, boundaryEndEvent);
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
        .numWorkers            = 1,
    });
  }

  auto outputs = stdex::mdarray<double, stdex::dextents<size_t, 2>>(Nr, Nt);
  auto host    = sycl::host_accessor{out_buf, sycl::read_only};
  for (auto it{0UL}; it < static_cast<size_t>(Nt); ++it) {
    for (auto ir{0UL}; ir < Nr; ++ir) {
      outputs(ir, it) = host[ir][it];
    }
  }

  fmt::println("");

  return outputs;
}
} // namespace pffdtd
