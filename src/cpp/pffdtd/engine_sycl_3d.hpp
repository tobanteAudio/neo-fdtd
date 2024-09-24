// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch
#pragma once

#include "pffdtd/simulation_3d.hpp"

#if not defined(PFFDTD_HAS_SYCL)
  #error "SYCL must be enabled in CMake via PFFDTD_ENABLE_SYCL_ACPP or PFFDTD_ENABLE_SYCL_ONEAPI"
#endif

namespace pffdtd {

struct EngineSYCL3D {
  auto operator()(Simulation3D<float> const& sim) const -> void;
  auto operator()(Simulation3D<double> const& sim) const -> void;
};

} // namespace pffdtd
