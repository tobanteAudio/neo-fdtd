// vim: tabstop=3: ai
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

#ifndef _CPU_ENGINE_H
#define _CPU_ENGINE_H

#include "helper_funcs.h"
#include "fdtd_common.h"
#include "fdtd_data.h"

double run_sim(struct SimData *sd);
double process_bnl_pts_fd(Real *u0b, const Real *u2b, const Real *ssaf_bnl, const int8_t *mat_bnl, int64_t Nbl, int8_t *Mb, Real lo2, Real *vh1, Real *gh1, const struct MatQuad *mat_quads, const Real *mat_beta);

#endif