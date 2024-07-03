///////////////////////////////////////////////////////////////////////////////
// This file is a part of PFFDTD.
//
// PFFTD is released under the MIT License.
// For details see the LICENSE file.
//
// Copyright 2021 Brian Hamilton.
//
// File name: fdtd_common.h
//
// Description: Header-only misc function definitions related to FDTD simulation
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "helper_funcs.hpp"

#include <stdint.h> 
#include <string.h> 
#include <math.h> 
#include <float.h> 
#include <time.h> //date and time
#include <omp.h>
#include <sys/ioctl.h> //terminal width


//flag passed in at compilation (see Makefile)
#if PRECISION==2 //double
   typedef double Real;

   #define REAL_MAX_EXP DBL_MAX_EXP
   #define REAL_MIN_EXP DBL_MIN_EXP

   //using CUDA math intrinsics
   #define FMA_O __fma_rn 
   #define FMA_D __fma_rn
   #define ADD_O __dadd_rn
   #define ADD_D __dadd_rn
   #define EPS 0.0

#elif PRECISION==1 //float with safeguards
   typedef float Real;

   #define REAL_MAX_EXP FLT_MAX_EXP
   #define REAL_MIN_EXP FLT_MIN_EXP

   //using CUDA math intrinsics
   #define FMA_O __fmaf_rz //off-diag
   #define FMA_D __fmaf_rn //diag
   #define ADD_O __fadd_rz
   #define ADD_D __fadd_rn
   #define EPS 1.19209289e-07 //helps with stability in single
#else
   #error "PRECISION = 1 (single) or 2 (double)"
#endif

//declarations, defs below
void ind2sub3d(int64_t idx, int64_t Nx, int64_t Ny, int64_t Nz, int64_t *ix, int64_t *iy, int64_t *iz);
void check_inside_grid(int64_t *idx, int64_t N, int64_t Nx, int64_t Ny, int64_t Nz);
void print_progress(uint32_t n, uint32_t Nt, uint64_t Npts, uint64_t Nb, 
                     double time_elapsed, double time_elapsed_sample, 
                     double time_elapsed_air, double time_elapsed_sample_air, 
                     double time_elapsed_bn, double time_elapsed_sample_bn, int num_workers);


