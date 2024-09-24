// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton
// Function definitions for handling loading of simulation data from HDF5 files,
// preparing for simulation, and writing outputs
#pragma once

#include "pffdtd/assert.hpp"
#include "pffdtd/utility.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <vector>

namespace pffdtd {

inline constexpr auto MMb = 12; // maximum number of RLC branches in freq-dep (FD) boundaries
inline constexpr auto MNm = 64; // maximum number of materials allows

// see python code and 2016 ISMRA paper
template<typename Real>
struct MatQuad {
  Real b;   // b
  Real bd;  // b*d
  Real bDh; // b*D-hat
  Real bFh; // b*F-hat
};

// main sim data, on host
template<typename Real>
struct Simulation3D {
  std::vector<int64_t> bn_ixyz;         // boundary node indices
  std::vector<int64_t> bnl_ixyz;        // lossy boundary node indices
  std::vector<int64_t> bna_ixyz;        // absorbing boundary node indices
  int8_t* Q_bna;                        // integer for ABCs (wall 1,edge 2,corner 3)
  std::vector<int64_t> in_ixyz;         // input points
  std::vector<int64_t> out_ixyz;        // output points
  std::vector<int64_t> out_reorder;     // ordering for outputs point for final print/save
  std::vector<uint16_t> adj_bn;         // nearest-neighbour adjancencies for all boundary nodes
  std::vector<Real> ssaf_bnl;           // surface area corrections (with extra volume scaling)
  std::vector<uint8_t> bn_mask;         // bit mask for bounday nodes
  std::vector<int8_t> mat_bnl;          // material indices for lossy boundary nodes
  std::vector<int8_t> K_bn;             // number of adjacent neighbours, boundary nodesa
  std::vector<double> in_sigs;          // input signals
  std::unique_ptr<double[]> u_out;      // for output signals
  int64_t Ns;                           // number of input grid points
  int64_t Nr;                           // number of output grid points
  int64_t Nt;                           // number of samples simulation
  int64_t Npts;                         // number of Cartesian grid points
  int64_t Nx;                           // x-dim (non-continguous)
  int64_t Ny;                           // y-dim
  int64_t Nz;                           // z-dim (continguous)
  int64_t Nb;                           // number of boundary nodes
  int64_t Nbl;                          // number of lossy boundary nodes
  int64_t Nba;                          // number of ABC nodes
  double l;                             // Courant number (CFL)
  double l2;                            // CFL number squared
  int8_t fcc_flag;                      // boolean for FCC
  int8_t NN;                            // integer, neareast neighbours
  int8_t Nm;                            // number of materials used
  std::vector<int8_t> Mb;               // number of branches per material
  std::vector<MatQuad<Real>> mat_quads; // RLC coefficients (essentially)
  std::vector<Real> mat_beta;           // part of FD-boundaries one per material
  double infac;                         // rescaling of input (for numerical reason)
  Real sl2;                             // scaled l2 (for single precision)
  Real lo2;                             // 0.5*l
  Real a2;                              // update stencil coefficient
  Real a1;                              // update stencil coefficient
};

[[nodiscard]] auto loadSimulation3D_float(std::filesystem::path const& simDir) -> Simulation3D<float>;
[[nodiscard]] auto loadSimulation3D_double(std::filesystem::path const& simDir) -> Simulation3D<double>;

auto printLastSample(Simulation3D<float> const& sim) -> void;
auto printLastSample(Simulation3D<double> const& sim) -> void;

auto writeOutputs(Simulation3D<float> const& sim, std::filesystem::path const& simDir) -> void;
auto writeOutputs(Simulation3D<double> const& sim, std::filesystem::path const& simDir) -> void;

template<typename Real>
[[nodiscard]] auto loadSimulation3D(std::filesystem::path const& simDir) -> Simulation3D<Real> {
  if constexpr (std::is_same_v<Real, float>) {
    return loadSimulation3D_float(simDir);
  } else if constexpr (std::is_same_v<Real, double>) {
    return loadSimulation3D_double(simDir);
  } else {
    static_assert(always_false<Real>);
  }
}

// scale input to be in middle of floating-point range
template<typename Real>
void scaleInput(Simulation3D<Real>& sim) {
  auto* in_sigs    = sim.in_sigs.data();
  int64_t const Nt = sim.Nt;
  int64_t const Ns = sim.Ns;

  // normalise input signals (and save gain)
  double max_in = 0.0;
  for (int64_t n = 0; n < Nt; n++) {
    for (int64_t ns = 0; ns < Ns; ns++) {
      max_in = std::max(max_in, fabs(in_sigs[ns * Nt + n]));
    }
  }

  constexpr auto min_exp = static_cast<double>(std::numeric_limits<Real>::min_exponent);
  constexpr auto max_exp = static_cast<double>(std::numeric_limits<Real>::max_exponent);

  auto const aexp      = 0.5; // normalise to middle power of two
  auto const pow2      = static_cast<int32_t>(std::round(aexp * max_exp + (1 - aexp) * min_exp));
  auto const norm1     = std::pow(2.0, pow2);
  auto const inv_infac = norm1 / max_in;
  auto const infac     = 1.0 / inv_infac;

  std::printf(
      "max_in = %.16e, pow2 = %d, norm1 = %.16e, inv_infac = %.16e, infac = "
      "%.16e\n",
      max_in,
      pow2,
      norm1,
      inv_infac,
      infac
  );

  // normalise input data
  for (int64_t ns = 0; ns < Ns; ns++) {
    for (int64_t n = 0; n < Nt; n++) {
      in_sigs[ns * Nt + n] *= inv_infac;
    }
  }

  sim.infac = infac;
}

template<typename Real>
void rescaleOutput(Simulation3D<Real>& sim) {
  int64_t const Nt = sim.Nt;
  int64_t const Nr = sim.Nr;
  double infac     = sim.infac;
  double* u_out    = sim.u_out.get();

  std::transform(u_out, u_out + Nr * Nt, u_out, [infac](auto sample) { return sample * infac; });
}

template<typename Real>
void freeSimulation3D(Simulation3D<Real> const& sim) {
  delete[] sim.Q_bna;
}

} // namespace pffdtd
