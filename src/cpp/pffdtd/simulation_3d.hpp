// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton
// Function definitions for handling loading of simulation data from HDF5 files,
// preparing for simulation, and writing outputs
#pragma once

#include "pffdtd/config.hpp"
#include "pffdtd/hdf.hpp"
#include "pffdtd/utility.hpp"

#include <cstdint>
#include <filesystem>

// maximum number of RLC branches in freq-dep (FD) boundaries (needed at
// compile-time for CUDA kernels)
#define MMb 12 // change as necssary
// maximum number of materials allows (needed at compile-time for CUDA)
#define MNm 64 // change as necssary

namespace pffdtd {

// see python code and 2016 ISMRA paper
struct MatQuad {
  Real b;   // b
  Real bd;  // b*d
  Real bDh; // b*D-hat
  Real bFh; // b*F-hat
};

// main sim data, on host
struct Simulation3D {
  int64_t* bn_ixyz;     // boundary node indices
  int64_t* bnl_ixyz;    // lossy boundary node indices
  int64_t* bna_ixyz;    // absorbing boundary node indices
  int8_t* Q_bna;        // integer for ABCs (wall 1,edge 2,corner 3)
  int64_t* in_ixyz;     // input points
  int64_t* out_ixyz;    // output points
  int64_t* out_reorder; // ordering for outputs point for final print/save
  uint16_t* adj_bn;     // nearest-neighbour adjancencies for all boundary nodes
  Real* ssaf_bnl;       // surface area corrections (with extra volume scaling)
  uint8_t* bn_mask;     // bit mask for bounday nodes
  int8_t* mat_bnl;      // material indices for lossy boundary nodes
  int8_t* K_bn;         // number of adjacent neighbours, boundary nodesa
  double* in_sigs;      // input signals
  double* u_out;        // for output signals
  int64_t Ns;           // number of input grid points
  int64_t Nr;           // number of output grid points
  int64_t Nt;           // number of samples simulation
  int64_t Npts;         // number of Cartesian grid points
  int64_t Nx;           // x-dim (non-continguous)
  int64_t Ny;           // y-dim
  int64_t Nz;           // z-dim (continguous)
  int64_t Nb;           // number of boundary nodes
  int64_t Nbl;          // number of lossy boundary nodes
  int64_t Nba;          // number of ABC nodes
  double l;             // Courant number (CFL)
  double l2;            // CFL number squared
  int8_t fcc_flag;      // boolean for FCC
  int8_t NN;            // integer, neareast neighbours
  int8_t Nm;            // number of materials used
  int8_t* Mb;           // number of branches per material
  struct MatQuad* mat_quads; // RLC coefficients (essentially)
  Real* mat_beta;            // part of FD-boundaries one per material
  double infac;              // rescaling of input (for numerical reason)
  Real sl2;                  // scaled l2 (for single precision)
  Real lo2;                  // 0.5*l
  Real a2;                   // update stencil coefficient
  Real a1;                   // update stencil coefficient
};

[[nodiscard]] auto loadSimulation3D(std::filesystem::path const& simDir)
    -> Simulation3D;
void freeSimulation3D(Simulation3D& sim);
void printLastSample(Simulation3D& sim);
void scaleInput(Simulation3D& sim);
void rescaleOutput(Simulation3D& sim);
void writeOutputs(Simulation3D& sim, std::filesystem::path const& simDir);

void readH5Dataset(
    hid_t file,
    char* dset_str,
    int ndims,
    hsize_t* dims,
    void** data_array,
    DataType t
);
void readH5Constant(hid_t file, char* dset_str, void* out, DataType t);

} // namespace pffdtd
