///////////////////////////////////////////////////////////////////////////////
// This file is a part of PFFDTD.
//
// PFFTD is released under the MIT License.
// For details see the LICENSE file.
//
// Copyright 2021 Brian Hamilton.
//
// File name: cpu_engine.h
//
// Description: CPU-based implementation of FDTD engine, with OpenMP
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "fdtd_common.hpp"
#include "fdtd_data.hpp"
#include "helper_funcs.hpp"

double runSim(Simulation3D* sd);
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
