#include <sycl/sycl.hpp>

#include "hdf5.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T> inline constexpr auto isStdVector = false;

template <typename T> inline constexpr auto isStdVector<std::vector<T>> = true;

struct H5FFile {
  explicit H5FFile(char const *str)
      : _handle{H5Fopen(str, H5F_ACC_RDONLY, H5P_DEFAULT)} {}

  ~H5FFile() { H5Fclose(_handle); }

  template <typename T> [[nodiscard]] auto read(char const *dataset) -> T {
    if constexpr (isStdVector<T>) {
      return readBuffer<typename T::value_type>(dataset);
    }

    auto set = H5Dopen(_handle, dataset, H5P_DEFAULT);

    if constexpr (std::is_same_v<T, int64_t>) {
      auto val = T{};
      auto type = H5T_NATIVE_INT64;
      auto ptr = static_cast<void *>(&val);
      auto err = H5Dread(set, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptr);
      checkErrorAndCloseDataset(dataset, set, err);
      return val;
    }

    if constexpr (std::is_same_v<T, double>) {
      auto val = T{};
      auto type = H5T_NATIVE_DOUBLE;
      auto ptr = static_cast<void *>(&val);
      auto err = H5Dread(set, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptr);
      checkErrorAndCloseDataset(dataset, set, err);
      return val;
    }

    return {};
  }

  template <typename T>
  [[nodiscard]] auto readBuffer(char const *dataset) -> std::vector<T> {
    auto set = H5Dopen(_handle, dataset, H5P_DEFAULT);
    auto space = H5Dget_space(set);

    auto ndims = 1UL;
    auto dims = std::array<hsize_t, 3>{};
    assert(H5Sget_simple_extent_ndims(space) == ndims);
    H5Sget_simple_extent_dims(space, dims.data(), NULL);

    auto size = ndims == 1 ? dims[0] : dims[0] * dims[1];

    if constexpr (std::is_same_v<T, uint8_t>) {
      auto type = H5T_NATIVE_UINT8;
      auto buf = std::vector<T>(size);
      auto err = H5Dread(set, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
      checkErrorAndCloseDataset(dataset, set, err);
      return buf;
    }

    if constexpr (std::is_same_v<T, int64_t>) {
      auto type = H5T_NATIVE_INT64;
      auto buf = std::vector<T>(size);
      auto err = H5Dread(set, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
      checkErrorAndCloseDataset(dataset, set, err);
      return buf;
    }

    if constexpr (std::is_same_v<T, double>) {
      auto type = H5T_NATIVE_DOUBLE;
      auto buf = std::vector<T>(size);
      auto err = H5Dread(set, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
      checkErrorAndCloseDataset(dataset, set, err);
      return buf;
    }
  }

private:
  auto checkErrorAndCloseDataset(char const *name, hid_t set, herr_t err)
      -> void {
    if (err != 0) {
      throw std::runtime_error{"dataset read in: " + std::string{name}};
    }

    if (H5Dclose(set) != 0) {
      throw std::runtime_error{"dataset close in: " + std::string{name}};
    }
  }

  hid_t _handle;
};

int main(int, char **argv) {
  auto file = H5FFile{argv[1]};

  auto const Nx = file.read<int64_t>("Nx");
  auto const Ny = file.read<int64_t>("Ny");
  auto const Nt = file.read<int64_t>("Nt");
  auto const inx = file.read<int64_t>("inx");
  auto const iny = file.read<int64_t>("iny");
  auto const lf = file.read<double>("lf");
  auto const adj_bn = file.read<std::vector<int64_t>>("adj_bn");
  auto const bn_ixy = file.read<std::vector<int64_t>>("bn_ixy");
  auto const in_mask = file.read<std::vector<uint8_t>>("in_mask");
  auto const receiver_ixy = file.read<std::vector<uint8_t>>("receiver_ixy");
  auto const src_sig = file.read<std::vector<double>>("src_sig");

  auto const N = size_t(Nx * Ny);

  std::printf("Nt: %ld\n", static_cast<long>(Nt));
  std::printf("Nx: %ld\n", static_cast<long>(Nx));
  std::printf("Ny: %ld\n", static_cast<long>(Ny));
  std::printf("N: %ld\n", static_cast<long>(N));
  std::printf("inx: %ld\n", static_cast<long>(inx));
  std::printf("iny: %ld\n", static_cast<long>(iny));
  std::printf("in_mask: %ld\n", static_cast<long>(in_mask.size()));
  std::printf("bn_ixy: %ld\n", static_cast<long>(bn_ixy.size()));
  std::printf("adj_bn: %ld\n", static_cast<long>(adj_bn.size()));
  std::printf("receiver_ixy: %ld\n", static_cast<long>(receiver_ixy.size()));
  std::printf("src_sig: %ld\n", static_cast<long>(src_sig.size()));
  std::printf("lf: %f\n", lf);

  auto queueProp = sycl::property_list{sycl::property::queue::in_order()};
  auto queue = sycl::queue{queueProp};

  auto u0 = sycl::buffer<double, 2>(sycl::range<2>(Nx, Ny));
  auto u1 = sycl::buffer<double, 2>(sycl::range<2>(Nx, Ny));
  auto u2 = sycl::buffer<double, 2>(sycl::range<2>(Nx, Ny));

  auto in_mask_buf = sycl::buffer<uint8_t, 1>{in_mask};
  auto bn_ixy_buf = sycl::buffer<int64_t, 1>{bn_ixy};
  auto adj_bn_buf = sycl::buffer<int64_t, 1>{adj_bn};
  auto src_sig_buf = sycl::buffer<double, 1>{src_sig};

  for (auto i{0UL}; i < Nt; ++i) {
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
      auto u1_acc = sycl::accessor{u1, cgh};
      auto u2_acc = sycl::accessor{u2, cgh};
      cgh.parallel_for<struct Copy>(sycl::range<2>(Nx, Ny),
                                    [=](sycl::id<2> id) {
                                      auto const x = id[0];
                                      auto const y = id[1];
                                      u2_acc[x][y] = u1_acc[x][y];
                                      u1_acc[x][y] = u0_acc[x][y];
                                    });
    });
  }

  auto host = sycl::host_accessor{u0, sycl::read_only};
  std::printf("%f\n", double(host[inx][iny]));

  return EXIT_SUCCESS;
}
