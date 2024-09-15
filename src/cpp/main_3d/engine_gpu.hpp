// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton
// Description: GPU-based implementation of FDTD engine (using CUDA).

#pragma once

#include "pffdtd/simulation_3d.hpp"

#if !PFFDTD_HAS_CUDA
  #error "CUDA must be enabled in the Makefile"
#endif

namespace pffdtd {
auto run(Simulation3D const& sd) -> double;
}
