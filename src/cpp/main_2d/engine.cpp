#include "engine.hpp"

#include "pffdtd/sycl.hpp"
#include "pffdtd/video.hpp"

#include <fmt/format.h>
#include <fmt/os.h>

#include <atomic>
#include <queue>
#include <thread>

namespace pffdtd {

struct BackgroundWriter {
  cv::Size size;
  VideoWriter writer;
  std::queue<std::vector<double>> queue;
  std::mutex mutex;
  std::atomic<bool> done{false};
};

auto write(BackgroundWriter& writer) -> void {

  auto frame = std::vector<double>{};
  while (not writer.done or not writer.queue.empty()) {
    auto shouldSleep = false;
    {
      auto lock   = std::scoped_lock{writer.mutex};
      shouldSleep = writer.queue.empty();
    }

    if (shouldSleep) {
      std::this_thread::sleep_for(std::chrono::milliseconds{10});
      continue;
    }

    {
      auto lock = std::scoped_lock{writer.mutex};

      frame = writer.queue.front();
      writer.queue.pop();
      writer.writer.write(frame, writer.size.width, writer.size.height);
    }
  }
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

auto run(Simulation2D const& sim) -> std::vector<double> {

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

  auto videoFile = sim.file.parent_path() / "out.avi";
  auto video     = BackgroundWriter{
          .size   = cv::Size(Nx, Ny),
          .writer = VideoWriter{videoFile, sim.video_fps, 1000, 1000},
          .queue  = {},
          .mutex  = {},
          .done   = false,
  };
  auto videoThread = std::thread([&video] { write(video); });

  auto prop   = sycl::property_list{sycl::property::queue::in_order()};
  auto queue  = sycl::queue{prop};
  auto device = queue.get_device();
  pffdtd::summary(device);

  auto u0  = sycl::buffer<double, 2>(sycl::range<2>(sim.Nx, sim.Ny));
  auto u1  = sycl::buffer<double, 2>(sycl::range<2>(sim.Nx, sim.Ny));
  auto u2  = sycl::buffer<double, 2>(sycl::range<2>(sim.Nx, sim.Ny));
  auto out = sycl::buffer<double, 2>(sycl::range<2>(Nr, sim.Nt));

  auto in_mask = sycl::buffer<uint8_t, 1>{sim.in_mask};
  auto bn_ixy  = sycl::buffer<int64_t, 1>{sim.bn_ixy};
  auto adj_bn  = sycl::buffer<int64_t, 1>{sim.adj_bn};
  auto out_ixy = sycl::buffer<int64_t, 1>{sim.out_ixy};
  auto src_sig = sycl::buffer<double, 1>{sim.src_sig};

  auto frame = std::vector<double>(sim.Nx * sim.Ny);

  fmt::print(stdout, "111111111");
  for (auto i{0LL}; i < sim.Nt; ++i) {
    fmt::print(stdout, "\r\r\r\r\r\r\r\r\r");
    fmt::print(stdout, "{:04d}/{:04d}", i, sim.Nt);
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

    if (i == 0) {
      queue.submit([&](sycl::handler& cgh) {
        auto u0a = sycl::accessor{u0, cgh};
        cgh.parallel_for<struct Impulse>(sycl::range<1>(1), [=](sycl::id<1>) {
          auto const impulse = 1.0;
          u0a[inx][iny] += impulse;
        });
      });
    }

    queue.submit([&](sycl::handler& cgh) {
      auto u0a         = sycl::accessor{u0, cgh};
      auto out_acc     = sycl::accessor{out, cgh};
      auto out_ixy_acc = sycl::accessor{out_ixy, cgh};
      auto range       = sycl::range<1>(Nr);

      cgh.parallel_for<struct CopyOutput>(range, [=](sycl::id<1> id) {
        auto r        = id[0];
        auto r_ixy    = out_ixy_acc[r];
        auto p0       = getPointer(u0a);
        out_acc[r][i] = p0[r_ixy];
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

    auto host = sycl::host_accessor{u0, sycl::read_only};
    for (auto i{0UL}; i < frame.size(); ++i) {
      frame[i] = std::abs(double(host.get_pointer()[i]));
      // if (sim.in_mask[i] == 0) {
      //   frame[i] = 1.0;
      // }
    }

    {
      auto lock = std::scoped_lock{video.mutex};
      video.queue.push(frame);
    }
  }

  video.done.store(true);
  videoThread.join();

  auto save = std::vector<double>(Nt * Nr);
  auto host = sycl::host_accessor{out, sycl::read_only};
  auto max  = 0.0;
  auto min  = 0.0;
  for (auto i{0UL}; i < static_cast<size_t>(Nt); ++i) {
    for (auto r{0UL}; r < Nr; ++r) {
      auto const sample = host[r][i];
      save[i * Nr + r]  = sample;
      max               = std::max(max, sample);
      min               = std::min(min, sample);
    }
  }

  fmt::println("");
  fmt::println("MAX: {}", max);
  fmt::println("MIN: {}", min);

  return save;
}
} // namespace pffdtd
