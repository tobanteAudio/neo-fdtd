// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton
// CPU-based implementation of FDTD engine, with OpenMP

#pragma once

#include "pffdtd/simulation_3d.hpp"

namespace pffdtd {
auto run(Simulation3D& sd) -> double;
} // namespace pffdtd
