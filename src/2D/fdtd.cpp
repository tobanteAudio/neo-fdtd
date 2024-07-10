#include "pffdtd/hdf.hpp"
#include "pffdtd/simulation_2d.hpp"
#include "pffdtd/sycl.hpp"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

int main(int, char **argv) {
  auto filePath = std::filesystem::path{argv[1]};
  if (not std::filesystem::exists(filePath)) {
    throw std::runtime_error{"invalid file: " + filePath.string()};
  }

  auto const sim = pffdtd::loadSimulation2D(filePath);
  auto const Nx = sim.Nx;
  auto const Ny = sim.Ny;
  auto const Nt = sim.Nt;
  auto const inx = sim.inx;
  auto const iny = sim.iny;
  auto const N = size_t(sim.Nx * sim.Ny);
  auto const Nr = sim.out_ixy.size();

  std::printf("Nt: %ld\n", static_cast<long>(sim.Nt));
  std::printf("Nx: %ld\n", static_cast<long>(sim.Nx));
  std::printf("Ny: %ld\n", static_cast<long>(sim.Ny));
  std::printf("N: %ld\n", static_cast<long>(N));
  std::printf("inx: %ld\n", static_cast<long>(sim.inx));
  std::printf("iny: %ld\n", static_cast<long>(sim.iny));
  std::printf("in_mask: %ld\n", static_cast<long>(sim.in_mask.size()));
  std::printf("bn_ixy: %ld\n", static_cast<long>(sim.bn_ixy.size()));
  std::printf("adj_bn: %ld\n", static_cast<long>(sim.adj_bn.size()));
  std::printf("out_ixy: %ld\n", static_cast<long>(Nr));
  std::printf("src_sig: %ld\n", static_cast<long>(sim.src_sig.size()));
  std::printf("loss_factor: %f\n", sim.loss_factor);

  for (auto dev : sycl::device::get_devices()) {
    pffdtd::printDeviceInfo(dev);
  }

  auto prop = sycl::property_list{sycl::property::queue::in_order()};
  auto queue = sycl::queue{prop};
  auto device = queue.get_device();
  pffdtd::printDeviceInfo(device);

  auto u0 = sycl::buffer<double, 2>(sycl::range<2>(sim.Nx, sim.Ny));
  auto u1 = sycl::buffer<double, 2>(sycl::range<2>(sim.Nx, sim.Ny));
  auto u2 = sycl::buffer<double, 2>(sycl::range<2>(sim.Nx, sim.Ny));
  auto out = sycl::buffer<double, 2>(sycl::range<2>(Nr, sim.Nt));

  auto in_mask_buf = sycl::buffer<uint8_t, 1>{sim.in_mask};
  auto bn_ixy_buf = sycl::buffer<int64_t, 1>{sim.bn_ixy};
  auto adj_bn_buf = sycl::buffer<int64_t, 1>{sim.adj_bn};
  auto out_ixy_buf = sycl::buffer<int64_t, 1>{sim.out_ixy};
  auto src_sig_buf = sycl::buffer<double, 1>{sim.src_sig};

  std::printf("111111111");
  for (auto i{0UL}; i < sim.Nt; ++i) {
    std::printf("\r\r\r\r\r\r\r\r\r");
    std::printf("%04d/%04d", int(i), int(sim.Nt));
    std::fflush(stdout);

    queue.submit([&](sycl::handler &cgh) {
      auto u0_acc = sycl::accessor{u0, cgh};
      auto u1_acc = sycl::accessor{u1, cgh};
      auto u2_acc = sycl::accessor{u2, cgh};
      auto inMask_acc = sycl::accessor{in_mask_buf, cgh};
      auto airRange = sycl::range<2>(Nx - 2, Ny - 2);

      cgh.parallel_for<struct AirUpdate>(airRange, [=](sycl::id<2> id) {
        auto const x = id.get(0) + 1;
        auto const y = id.get(1) + 1;
        auto const idx = x * Ny + y;

        if (inMask_acc[idx] == 0) {
          return;
        }

        auto const left = u1_acc[x][y - 1];
        auto const right = u1_acc[x][y + 1];
        auto const bottom = u1_acc[x - 1][y];
        auto const top = u1_acc[x + 1][y];
        auto const last = u2_acc[x][y];

        u0_acc[x][y] = 0.5 * (left + right + bottom + top) - last;
      });
    });

    if (i == 0) {
      queue.submit([&](sycl::handler &cgh) {
        auto u0_acc = sycl::accessor{u0, cgh};
        cgh.parallel_for<struct Impulse>(sycl::range<1>(1),
                                         [=](sycl::id<1> id) {
                                           auto const impulse = 1.0;
                                           u0_acc[inx][iny] += impulse;
                                         });
      });
    }

    queue.submit([&](sycl::handler &cgh) {
      auto u0_acc = sycl::accessor{u0, cgh};
      auto out_acc = sycl::accessor{out, cgh};
      auto out_ixy_acc = sycl::accessor{out_ixy_buf, cgh};

      cgh.parallel_for<struct CopyOutput>(
          sycl::range<1>(Nr), [=](sycl::id<1> id) {
            auto const r = id[0];
            auto const r_ixy = out_ixy_acc[r];
            auto flatPtr = u0_acc.get_multi_ptr<sycl::access::decorated::no>();
            out_acc[r][i] = flatPtr[r_ixy];
          });
    });

    queue.submit([&](sycl::handler &cgh) {
      auto u0_acc = sycl::accessor{u0, cgh};
      auto u1_acc = sycl::accessor{u1, cgh};
      auto u2_acc = sycl::accessor{u2, cgh};
      cgh.parallel_for<struct Rotate>(sycl::range<2>(Nx, Ny),
                                      [=](sycl::id<2> id) {
                                        u2_acc[id] = u1_acc[id];
                                        u1_acc[id] = u0_acc[id];
                                      });
    });

    queue.wait_and_throw();
  }

  auto save = std::vector<double>(Nt * Nr);
  auto host = sycl::host_accessor{out, sycl::read_only};
  auto max = 0.0;
  auto min = 0.0;
  for (auto i{0UL}; i < Nt; ++i) {
    for (auto r{0UL}; r < Nr; ++r) {
      auto const sample = host[r][i];
      save[i * Nr + r] = sample;
      max = std::max(max, sample);
      min = std::min(min, sample);
    }
  }

  std::puts("");
  std::printf("MAX: %f\n", max);
  std::printf("MIN: %f\n", min);

  auto dir = filePath.parent_path();
  auto outfile = dir / "out.h5";
  auto results = pffdtd::H5FWriter{outfile.string().c_str()};
  results.write("out", std::span{save}, Nr, Nt);

  return EXIT_SUCCESS;
}
