// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton
// CPU-based implementation of FDTD engine, with OpenMP

#pragma once

#include "pffdtd/config.hpp"
#include "pffdtd/simulation_3d.hpp"

#include <cstdint>

namespace pffdtd {

auto run(Simulation3D& sd) -> double;

} // namespace pffdtd
