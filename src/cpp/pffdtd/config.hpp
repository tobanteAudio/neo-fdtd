#pragma once

#include <omp.h>

#include <cfloat>

// flag passed in at compilation (see Makefile)
#if PRECISION == 2 // double
typedef double Real;

  #define REAL_MAX_EXP DBL_MAX_EXP
  #define REAL_MIN_EXP DBL_MIN_EXP

  // using CUDA math intrinsics
  #define FMA_O __fma_rn
  #define FMA_D __fma_rn
  #define ADD_O __dadd_rn
  #define ADD_D __dadd_rn
  #define EPS   0.0

#elif PRECISION == 1 // float with safeguards
typedef float Real;

  #define REAL_MAX_EXP FLT_MAX_EXP
  #define REAL_MIN_EXP FLT_MIN_EXP

  // using CUDA math intrinsics
  #define FMA_O        __fmaf_rz // off-diag
  #define FMA_D        __fmaf_rn // diag
  #define ADD_O        __fadd_rz
  #define ADD_D        __fadd_rn
  #define EPS          1.19209289e-07 // helps with stability in single
#else
  #error "PRECISION = 1 (single) or 2 (double)"
#endif
