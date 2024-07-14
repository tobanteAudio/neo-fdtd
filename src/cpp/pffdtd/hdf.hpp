#pragma once

#include "pffdtd/mdspan.hpp"

#include "hdf5.h"

#include <array>
#include <cassert>
#include <span>
#include <stdexcept>
#include <vector>

namespace pffdtd {

enum DataType : unsigned char {
  FLOAT64,
  FLOAT32,
  INT64,
  INT8,
  BOOL,
};

template<typename T>
inline constexpr auto isStdVector = false;

template<typename T>
inline constexpr auto isStdVector<std::vector<T>> = true;

struct H5FReader {
  explicit H5FReader(char const* str)
      : _handle{H5Fopen(str, H5F_ACC_RDONLY, H5P_DEFAULT)} {}

  ~H5FReader() { H5Fclose(_handle); }

  template<typename T>
  [[nodiscard]] auto read(char const* dataset) -> T {
    if constexpr (isStdVector<T>) {
      return readBuffer<typename T::value_type>(dataset);
    }

    auto set = H5Dopen(_handle, dataset, H5P_DEFAULT);

    if constexpr (std::is_same_v<T, int64_t>) {
      auto val  = T{};
      auto type = H5T_NATIVE_INT64;
      auto ptr  = static_cast<void*>(&val);
      auto err  = H5Dread(set, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptr);
      checkErrorAndCloseDataset(dataset, set, err);
      return val;
    }

    if constexpr (std::is_same_v<T, double>) {
      auto val  = T{};
      auto type = H5T_NATIVE_DOUBLE;
      auto ptr  = static_cast<void*>(&val);
      auto err  = H5Dread(set, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptr);
      checkErrorAndCloseDataset(dataset, set, err);
      return val;
    }

    return {};
  }

  template<typename T>
  [[nodiscard]] auto readBuffer(char const* dataset) -> std::vector<T> {
    auto set   = H5Dopen(_handle, dataset, H5P_DEFAULT);
    auto space = H5Dget_space(set);

    auto ndims = 1UL;
    auto dims  = std::array<hsize_t, 3>{};
    assert(H5Sget_simple_extent_ndims(space) == ndims);
    H5Sget_simple_extent_dims(space, dims.data(), NULL);

    auto size = ndims == 1 ? dims[0] : dims[0] * dims[1];

    if constexpr (std::is_same_v<T, uint8_t>) {
      auto type = H5T_NATIVE_UINT8;
      auto buf  = std::vector<T>(size);
      auto err  = H5Dread(set, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
      checkErrorAndCloseDataset(dataset, set, err);
      return buf;
    }

    if constexpr (std::is_same_v<T, int64_t>) {
      auto type = H5T_NATIVE_INT64;
      auto buf  = std::vector<T>(size);
      auto err  = H5Dread(set, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
      checkErrorAndCloseDataset(dataset, set, err);
      return buf;
    }

    if constexpr (std::is_same_v<T, double>) {
      auto type = H5T_NATIVE_DOUBLE;
      auto buf  = std::vector<T>(size);
      auto err  = H5Dread(set, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
      checkErrorAndCloseDataset(dataset, set, err);
      return buf;
    }
  }

  private:
  auto checkErrorAndCloseDataset(char const* name, hid_t set, herr_t err)
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

struct H5FWriter {
  explicit H5FWriter(char const* path)
      : _handle{H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)} {}

  ~H5FWriter() { H5Fclose(_handle); }

  auto write(
      char const* name,
      stdex::mdspan<double const, stdex::dextents<size_t, 2>> buf
  ) -> void {
    hsize_t dims[2]{
        static_cast<hsize_t>(buf.extent(0)),
        static_cast<hsize_t>(buf.extent(1)),
    };

    auto def  = H5P_DEFAULT;
    auto type = H5T_NATIVE_DOUBLE;

    auto space  = H5Screate_simple(2, dims, NULL);
    auto set    = H5Dcreate(_handle, name, type, space, def, def, def);
    auto status = H5Dwrite(set, type, H5S_ALL, H5S_ALL, def, buf.data_handle());

    if (status != 0) {
      throw std::runtime_error("error writing dataset\n");
    }

    status = H5Dclose(set);
    if (status != 0) {
      throw std::runtime_error("error closing dataset\n");
    }

    status = H5Sclose(space);
    if (status != 0) {
      throw std::runtime_error("error closing dataset space\n");
    }
  }

  private:
  hid_t _handle;
};

} // namespace pffdtd
