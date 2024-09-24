// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton
// Description: GPU-based implementation of FDTD engine (using CUDA).

#pragma once

#include "pffdtd/simulation_3d.hpp"

#if not defined(PFFDTD_HAS_CUDA)
  #error "CUDA must be enabled in CMake via -D PFFDTD_ENABLE_CUDA"
#endif

namespace pffdtd {

struct EngineCUDA3D {
  auto operator()(Simulation3D<float>& sim) const -> void;
  auto operator()(Simulation3D<double>& sim) const -> void;
};

} // namespace pffdtd
