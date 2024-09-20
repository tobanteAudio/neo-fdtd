// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#pragma once

#include <cfloat>

// flag passed in at compilation (see Makefile)
#if PFFDTD_PRECISION == 2 // double
namespace pffdtd {
using Real = double;
}

  #define EPS 0.0

#elif PFFDTD_PRECISION == 1 // float with safeguards
namespace pffdtd {
using Real = float;
}

  #define EPS 1.19209289e-07 // helps with stability in single
#else
  #error "PFFDTD_PRECISION = 1 (single) or 2 (double)"
#endif
