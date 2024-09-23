// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton
// CPU-based implementation of FDTD engine, with OpenMP

#pragma once

#include "pffdtd/simulation_3d.hpp"

namespace pffdtd {

struct Engine3DCPU {
  auto operator()(Simulation3D<float>& sim) const -> void;
  auto operator()(Simulation3D<double>& sim) const -> void;
};

} // namespace pffdtd
