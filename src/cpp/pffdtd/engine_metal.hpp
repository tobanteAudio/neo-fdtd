// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#pragma once

#if not defined(__METAL__)
  #include <cstdint>
#endif

#include "mat_quad.hpp"

namespace pffdtd {

template<typename Real>
struct Constants2D {
  int64_t Ny;
  Real lossFactor;
};

template<typename Real>
struct Constants3D {
  int64_t n;
  int64_t Nx;
  int64_t Ny;
  int64_t Nz;
  int64_t NzNy;
  int64_t Nb;
  int64_t Nbl;
  int64_t Nba;
  int64_t Ns;
  int64_t Nr;
  int64_t Nt;
  Real l;
  Real lo2;
  Real sl2;
  Real a1;
  Real a2;
};

template<typename T>
auto get_bit(int64_t word, int64_t pos) -> T {
  return static_cast<T>((word >> pos) & 1);
}

} // namespace pffdtd
