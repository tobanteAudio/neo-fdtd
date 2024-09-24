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

namespace pffdtd {

// maximum number of RLC branches in freq-dep (FD) boundaries (needed at
// compile-time for CUDA kernels)
inline constexpr auto MMb = 12; // change as necssary
// maximum number of materials allows (needed at compile-time for CUDA)
inline constexpr auto MNm = 64; // change as necssary

// see python code and 2016 ISMRA paper
template<typename Float>
struct MatQuad {
  Float b;   // b
  Float bd;  // b*d
  Float bDh; // b*D-hat
  Float bFh; // b*F-hat
};

// main sim data, on host
template<typename Float>
struct Simulation3D {
  int64_t* bn_ixyz;          // boundary node indices
  int64_t* bnl_ixyz;         // lossy boundary node indices
  int64_t* bna_ixyz;         // absorbing boundary node indices
  int8_t* Q_bna;             // integer for ABCs (wall 1,edge 2,corner 3)
  int64_t* in_ixyz;          // input points
  int64_t* out_ixyz;         // output points
  int64_t* out_reorder;      // ordering for outputs point for final print/save
  uint16_t* adj_bn;          // nearest-neighbour adjancencies for all boundary nodes
  Float* ssaf_bnl;           // surface area corrections (with extra volume scaling)
  uint8_t* bn_mask;          // bit mask for bounday nodes
  int8_t* mat_bnl;           // material indices for lossy boundary nodes
  int8_t* K_bn;              // number of adjacent neighbours, boundary nodesa
  double* in_sigs;           // input signals
  double* u_out;             // for output signals
  int64_t Ns;                // number of input grid points
  int64_t Nr;                // number of output grid points
  int64_t Nt;                // number of samples simulation
  int64_t Npts;              // number of Cartesian grid points
  int64_t Nx;                // x-dim (non-continguous)
  int64_t Ny;                // y-dim
  int64_t Nz;                // z-dim (continguous)
  int64_t Nb;                // number of boundary nodes
  int64_t Nbl;               // number of lossy boundary nodes
  int64_t Nba;               // number of ABC nodes
  double l;                  // Courant number (CFL)
  double l2;                 // CFL number squared
  int8_t fcc_flag;           // boolean for FCC
  int8_t NN;                 // integer, neareast neighbours
  int8_t Nm;                 // number of materials used
  int8_t* Mb;                // number of branches per material
  MatQuad<Float>* mat_quads; // RLC coefficients (essentially)
  Float* mat_beta;           // part of FD-boundaries one per material
  double infac;              // rescaling of input (for numerical reason)
  Float sl2;                 // scaled l2 (for single precision)
  Float lo2;                 // 0.5*l
  Float a2;                  // update stencil coefficient
  Float a1;                  // update stencil coefficient
};

[[nodiscard]] auto loadSimulation3D_float(std::filesystem::path const& simDir) -> Simulation3D<float>;
[[nodiscard]] auto loadSimulation3D_double(std::filesystem::path const& simDir) -> Simulation3D<double>;

auto printLastSample(Simulation3D<float>& sim) -> void;
auto printLastSample(Simulation3D<double>& sim) -> void;

auto writeOutputs(Simulation3D<float>& sim, std::filesystem::path const& simDir) -> void;
auto writeOutputs(Simulation3D<double>& sim, std::filesystem::path const& simDir) -> void;

template<typename Float>
[[nodiscard]] auto loadSimulation3D(std::filesystem::path const& simDir) -> Simulation3D<Float> {
  if constexpr (std::is_same_v<Float, float>) {
    return loadSimulation3D_float(simDir);
  } else if constexpr (std::is_same_v<Float, double>) {
    return loadSimulation3D_double(simDir);
  } else {
    static_assert(always_false<Float>);
  }
}

// scale input to be in middle of floating-point range
template<typename Float>
void scaleInput(Simulation3D<Float>& sim) {
  double* in_sigs  = sim.in_sigs;
  int64_t const Nt = sim.Nt;
  int64_t const Ns = sim.Ns;

  // normalise input signals (and save gain)
  double max_in = 0.0;
  for (int64_t n = 0; n < Nt; n++) {
    for (int64_t ns = 0; ns < Ns; ns++) {
      max_in = std::max(max_in, fabs(in_sigs[ns * Nt + n]));
    }
  }

  constexpr auto min_exp = static_cast<double>(std::numeric_limits<Float>::min_exponent);
  constexpr auto max_exp = static_cast<double>(std::numeric_limits<Float>::max_exponent);

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

  sim.infac   = infac;
  sim.in_sigs = in_sigs;
}

template<typename Float>
void rescaleOutput(Simulation3D<Float>& sim) {
  int64_t const Nt = sim.Nt;
  int64_t const Nr = sim.Nr;
  double infac     = sim.infac;
  double* u_out    = sim.u_out;

  std::transform(u_out, u_out + Nr * Nt, u_out, [infac](auto sample) { return sample * infac; });
}

template<typename Float>
void freeSimulation3D(Simulation3D<Float>& sim) {
  std::free(sim.bn_ixyz);
  std::free(sim.bnl_ixyz);
  std::free(sim.bna_ixyz);
  std::free(sim.Q_bna);
  std::free(sim.adj_bn);
  std::free(sim.mat_bnl);
  std::free(sim.bn_mask);
  std::free(sim.ssaf_bnl);
  std::free(sim.K_bn);
  std::free(sim.in_ixyz);
  std::free(sim.out_ixyz);
  std::free(sim.out_reorder);
  std::free(sim.in_sigs);
  std::free(sim.u_out);
  std::free(sim.Mb);
  std::free(sim.mat_beta);
  std::free(sim.mat_quads);
}

} // namespace pffdtd
