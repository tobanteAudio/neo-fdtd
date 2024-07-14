#include "engine.hpp"

#include "pffdtd/sycl.hpp"
#include "pffdtd/video.hpp"

#include <fmt/format.h>
#include <fmt/os.h>

#include <concepts>

namespace pffdtd {

[[nodiscard]] constexpr auto to_ixy(
    std::integral auto x,
    std::integral auto y,
    std::integral auto /*Nx*/,
    std::integral auto Ny
) -> std::integral auto {
  return x * Ny + y;
}

static auto kernelBoundaryRigid(
    sycl::id<1> idx,
    double* u0,
    double const* u1,
    double const* u2,
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

  u0[ib] = (2 - 0.5 * K) * last1 + 0.5 * neighbors - last2;
}

static auto kernelBoundaryLoss(
    sycl::id<1> idx,
    double* u0,
    double const* u2,
    int64_t const* bn_ixy,
    int64_t const* adj_bn,
    double lossFactor
) -> void {
  auto const ib      = bn_ixy[idx];
  auto const K       = adj_bn[idx];
  auto const current = u0[ib];
  auto const prev    = u2[ib];
  auto const K4      = 4 - K;

  u0[ib] = (current + lossFactor * K4 * prev) / (1 + lossFactor * K4);
}

auto run(Simulation2D const& sim)
    -> stdex::mdarray<double, stdex::dextents<size_t, 2>> {

  auto const Nx          = sim.Nx;
  auto const Ny          = sim.Ny;
  auto const Nt          = sim.Nt;
  auto const Nb          = sim.adj_bn.size();
  auto const inx         = sim.inx;
  auto const iny         = sim.iny;
  auto const N           = size_t(sim.Nx * sim.Ny);
  auto const Nr          = sim.out_ixy.size();
  auto const loss_factor = sim.loss_factor;

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

  auto prop   = sycl::property_list{sycl::property::queue::in_order()};
  auto queue  = sycl::queue{prop};
  auto device = queue.get_device();
  pffdtd::summary(device);

  auto u0  = sycl::buffer<double, 2>(sycl::range<2>(Nx, Ny));
  auto u1  = sycl::buffer<double, 2>(sycl::range<2>(Nx, Ny));
  auto u2  = sycl::buffer<double, 2>(sycl::range<2>(Nx, Ny));
  auto out = sycl::buffer<double, 2>(sycl::range<2>(Nr, Nt));

  auto in_mask = sycl::buffer<uint8_t, 1>{sim.in_mask};
  auto bn_ixy  = sycl::buffer<int64_t, 1>{sim.bn_ixy};
  auto adj_bn  = sycl::buffer<int64_t, 1>{sim.adj_bn};
  auto out_ixy = sycl::buffer<int64_t, 1>{sim.out_ixy};
  auto src_sig = sycl::buffer<double, 1>{sim.src_sig};

  auto frame = std::vector<double>(Nx * Ny);

  for (auto n{0LL}; n < Nt; ++n) {
    fmt::print(stdout, "\r\r\r\r\r\r\r\r\r");
    fmt::print(stdout, "{:04d}/{:04d}", n, Nt);
    std::fflush(stdout);

    queue.submit([&](sycl::handler& cgh) {
      auto u0a        = sycl::accessor{u0, cgh};
      auto u1a        = sycl::accessor{u1, cgh};
      auto u2a        = sycl::accessor{u2, cgh};
      auto inMask_acc = sycl::accessor{in_mask, cgh};
      auto airRange   = sycl::range<2>(Nx - 2, Ny - 2);

      cgh.parallel_for<struct AirUpdate>(airRange, [=](sycl::id<2> id) {
        auto const x   = id.get(0) + 1;
        auto const y   = id.get(1) + 1;
        auto const idx = x * Ny + y;

        if (inMask_acc[idx] == 0) {
          return;
        }

        auto const left   = u1a[x][y - 1];
        auto const right  = u1a[x][y + 1];
        auto const bottom = u1a[x - 1][y];
        auto const top    = u1a[x + 1][y];
        auto const last   = u2a[x][y];

        u0a[x][y] = 0.5 * (left + right + bottom + top) - last;
      });
    });

    queue.submit([&](sycl::handler& cgh) {
      auto u0a        = sycl::accessor{u0, cgh};
      auto u1a        = sycl::accessor{u1, cgh};
      auto u2a        = sycl::accessor{u2, cgh};
      auto bn_ixy_acc = sycl::accessor{bn_ixy, cgh};
      auto adj_bn_acc = sycl::accessor{adj_bn, cgh};
      auto rigidRange = sycl::range<1>(Nb);

      cgh.parallel_for<struct BoundaryRigid>(rigidRange, [=](sycl::id<1> id) {
        kernelBoundaryRigid(
            id,
            getPointer(u0a),
            getPointer(u1a),
            getPointer(u2a),
            getPointer(bn_ixy_acc),
            getPointer(adj_bn_acc),
            Ny
        );
      });
    });

    queue.submit([&](sycl::handler& cgh) {
      auto u0a        = sycl::accessor{u0, cgh};
      auto u2a        = sycl::accessor{u2, cgh};
      auto bn_ixy_acc = sycl::accessor{bn_ixy, cgh};
      auto adj_bn_acc = sycl::accessor{adj_bn, cgh};
      auto lossRange  = sycl::range<1>(Nb);

      cgh.parallel_for<struct BoundaryLoss>(lossRange, [=](sycl::id<1> id) {
        kernelBoundaryLoss(
            id,
            getPointer(u0a),
            getPointer(u2a),
            getPointer(bn_ixy_acc),
            getPointer(adj_bn_acc),
            loss_factor
        );
      });
    });

    queue.submit([&](sycl::handler& cgh) {
      auto u0a         = sycl::accessor{u0, cgh};
      auto src_sig_acc = sycl::accessor{src_sig, cgh};
      cgh.parallel_for<struct CopyInput>(sycl::range<1>(1), [=](sycl::id<1>) {
        u0a[inx][iny] += src_sig_acc[n];
      });
    });

    queue.submit([&](sycl::handler& cgh) {
      auto u0a         = sycl::accessor{u0, cgh};
      auto out_acc     = sycl::accessor{out, cgh};
      auto out_ixy_acc = sycl::accessor{out_ixy, cgh};
      auto range       = sycl::range<1>(Nr);

      cgh.parallel_for<struct CopyOutput>(range, [=](sycl::id<1> id) {
        auto r        = id[0];
        auto r_ixy    = out_ixy_acc[r];
        auto p0       = getPointer(u0a);
        out_acc[r][n] = p0[r_ixy];
      });
    });

    queue.submit([&](sycl::handler& cgh) {
      auto a0    = sycl::accessor{u0, cgh};
      auto a1    = sycl::accessor{u1, cgh};
      auto a2    = sycl::accessor{u2, cgh};
      auto range = sycl::range<1>(N);
      cgh.parallel_for<struct RotateBuffers>(range, [=](sycl::id<1> id) {
        auto p0 = getPointer(a0);
        auto p1 = getPointer(a1);
        auto p2 = getPointer(a2);
        p2[id]  = p1[id];
        p1[id]  = p0[id];
      });
    });

    if (shouldRenderVideo) {
      auto host = sycl::host_accessor{u0, sycl::read_only};
      for (auto i{0UL}; i < frame.size(); ++i) {
        frame[i] = std::abs(double(host.get_pointer()[i]));
      }
      push(*videoWriter, frame);
    }
  }

  if (shouldRenderVideo) {
    videoWriter->done.store(true);
    videoThread->join();
  }

  auto outputs = stdex::mdarray<double, stdex::dextents<size_t, 2>>(Nr, Nt);
  auto host    = sycl::host_accessor{out, sycl::read_only};
  for (auto it{0UL}; it < static_cast<size_t>(Nt); ++it) {
    for (auto ir{0UL}; ir < Nr; ++ir) {
      outputs(ir, it) = host[ir][it];
    }
  }

  fmt::println("");

  return outputs;
}
} // namespace pffdtd
