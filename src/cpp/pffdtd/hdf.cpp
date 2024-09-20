// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch
#include "hdf.hpp"

#include "pffdtd/utility.hpp"

#include <fmt/format.h>

#include <cstdio>
#include <cstdlib>

namespace pffdtd {

auto readDataset(hid_t file, char const* dset_str, int ndims, hsize_t* dims, void** out, DataType t) -> void {
  auto dset   = H5Dopen(file, dset_str, H5P_DEFAULT);
  auto dspace = H5Dget_space(dset);
  PFFDTD_ASSERT(H5Sget_simple_extent_ndims(dspace) == ndims);

  uint64_t N = 0;
  H5Sget_simple_extent_dims(dspace, dims, nullptr);
  if (ndims == 1) {
    N = dims[0];
  } else if (ndims == 2) {
    N = dims[0] * dims[1];
  } else {
    raisef<std::invalid_argument>("unexpected ndims = {} for dset = {}", ndims, dset_str);
  }

  switch (t) {
    case DataType::Float64: *out = allocate_zeros<double>(N); break;
    case DataType::Int64: *out = allocate_zeros<int64_t>(N); break;
    case DataType::Int8: *out = allocate_zeros<int8_t>(N); break;
    case DataType::Bool: *out = allocate_zeros<bool>(N); break;
  }
  if (*out == nullptr) {
    raise<std::bad_alloc>();
  }

  herr_t status = 0;
  switch (t) {
    case DataType::Float64: status = H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, *out); break;
    case DataType::Int64: status = H5Dread(dset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, *out); break;
    case DataType::Int8: // bool read in as DataType::Int8
    case DataType::Bool: status = H5Dread(dset, H5T_NATIVE_INT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, *out); break;
  }

  if (status != 0) {
    raisef<std::runtime_error>("error reading dataset: {}", dset_str);
  }
  if (H5Dclose(dset) != 0) {
    raisef<std::runtime_error>("error closing dataset: {}", dset_str);
  } else {
    fmt::println("read and closed dataset: {}", dset_str);
  }
}

} // namespace pffdtd
