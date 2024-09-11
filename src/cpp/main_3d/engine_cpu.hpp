// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton
// CPU-based implementation of FDTD engine, with OpenMP

#pragma once

#include "pffdtd/config.hpp"
#include "pffdtd/simulation_3d.hpp"

#include <cstdint>

namespace pffdtd {

auto run(Simulation3D& sd) -> double;
double process_bnl_pts_fd(
    Real* u0b,
    Real const* u2b,
    Real const* ssaf_bnl,
    int8_t const* mat_bnl,
    int64_t Nbl,
    int8_t* Mb,
    Real lo2,
    Real* vh1,
    Real* gh1,
    MatQuad const* mat_quads,
    Real const* mat_beta
);

} // namespace pffdtd
