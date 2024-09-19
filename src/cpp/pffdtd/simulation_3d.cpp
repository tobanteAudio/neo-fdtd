// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton

#include "simulation_3d.hpp"

#include "pffdtd/assert.hpp"
#include "pffdtd/hdf.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numbers>

namespace {

// linear indices to sub-indices in 3d, Nz continguous
void ind2sub3d(int64_t idx, int64_t Nx, int64_t Ny, int64_t Nz, int64_t* ix, int64_t* iy, int64_t* iz) {
  *iz = idx % Nz;
  *iy = (idx - (*iz)) / Nz % Ny;
  *ix = ((idx - (*iz)) / Nz - (*iy)) / Ny;
  PFFDTD_ASSERT(*ix > 0);
  PFFDTD_ASSERT(*iy > 0);
  PFFDTD_ASSERT(*iz > 0);
  PFFDTD_ASSERT(*ix < Nx - 1);
  PFFDTD_ASSERT(*iy < Ny - 1);
  PFFDTD_ASSERT(*iz < Nz - 1);
}

// double check some index inside grid
void check_inside_grid(int64_t* idx, int64_t N, int64_t Nx, int64_t Ny, int64_t Nz) {
  for (int64_t i = 0; i < N; i++) {
    int64_t iz = 0;
    int64_t iy = 0;
    int64_t ix = 0;
    ind2sub3d(idx[i], Nx, Ny, Nz, &ix, &iy, &iz);
  }
}
} // namespace

namespace pffdtd {

// load the sim data from Python-written HDF5 files
[[nodiscard]] auto loadSimulation3D(std::filesystem::path const& simDir) -> Simulation3D {
  // local values, to read in and attach to struct at end
  int64_t* bn_ixyz     = nullptr;
  int64_t* bnl_ixyz    = nullptr;
  int64_t* bna_ixyz    = nullptr;
  int8_t* Q_bna        = nullptr;
  int64_t* in_ixyz     = nullptr;
  int64_t* out_ixyz    = nullptr;
  int64_t* out_reorder = nullptr;
  bool* adj_bn_bool    = nullptr;
  int8_t* K_bn         = nullptr;
  uint16_t* adj_bn     = nullptr; // large enough for FCC
  uint8_t* bn_mask     = nullptr;
  int8_t* mat_bn       = nullptr;
  int8_t* mat_bnl      = nullptr;
  double* saf_bn       = nullptr;
  Real* ssaf_bn        = nullptr;
  Real* ssaf_bnl       = nullptr;
  double* in_sigs      = nullptr;
  double* u_out        = nullptr;
  int8_t* Mb           = nullptr;

  hsize_t dims[2]; // HDF5 type
  int expected_ndims = 0;

  ////////////////////////////////////////////////////////////////////////
  // Read constants HDF5 dataset
  ////////////////////////////////////////////////////////////////////////
  auto filename = simDir / "constants.h5";
  if (not std::filesystem::exists(filename)) {
    raisef<std::invalid_argument>("file '{}' does not exist", filename.string());
  }

  auto constants = H5FReader(filename);

  //////////////////
  // constants
  //////////////////
  auto const l        = constants.read<double>("l");
  auto const l2       = constants.read<double>("l2");
  auto const Ts       = constants.read<double>("Ts");
  auto const fcc_flag = constants.read<int8_t>("fcc_flag");
  fmt::println("l={:.16g}", l);
  fmt::println("l2={:.16g}", l2);
  fmt::println("Ts={:.16g}", Ts);
  fmt::println("fcc_flag={}", fcc_flag);
  PFFDTD_ASSERT((fcc_flag >= 0) && (fcc_flag <= 2));

  // FCC (1 is CPU-based, 2 is CPU or GPU)
  int8_t NN = 0;
  if (fcc_flag > 0) {
    PFFDTD_ASSERT(l2 <= 1.0);
    PFFDTD_ASSERT(l <= 1.0);
    NN = 12;
  } else {
    // simple Cartesian
    PFFDTD_ASSERT(l2 <= 1.0 / 3.0);
    PFFDTD_ASSERT(l <= sqrt(1.0 / 3.0));
    NN = 6;
  }

  // calculate some update coefficients
  double const lfac = (fcc_flag > 0) ? 0.25 : 1.0; // laplacian factor
  double const dsl2 = (1.0 + EPS) * lfac * l2;     // scale for stability (EPS in fdtd_common.hpp)
  double const da1  = (2.0 - dsl2 * NN);           // scaling for stability in single
  double const da2  = lfac * l2;

  Real const a1  = da1;
  Real const a2  = da2;
  Real const sl2 = dsl2;
  Real const lo2 = 0.5 * l;

  fmt::println("a2 (double): {:.16g}", da2);
  fmt::println("a2 (Real): {:.16g}", a2);
  fmt::println("a1 (double): {:.16g}", da1);
  fmt::println("a1 (Real): {:.16g}", a1);
  fmt::println("sl2 (double): {:.16g}", dsl2);
  fmt::println("sl2 (Real): {:.16g}", sl2);

  fmt::println("l2={:.16g}", l2);
  fmt::println("NN={}", NN);

  ////////////////////////////////////////////////////////////////////////
  // Read vox HDF5 dataset
  ////////////////////////////////////////////////////////////////////////
  filename = simDir / "vox_out.h5";
  if (not std::filesystem::exists(filename)) {
    raisef<std::invalid_argument>("file '{}' does not exist", filename.string());
  }

  auto vox_out = H5FReader(filename);

  //////////////////
  // integers
  //////////////////
  auto const Nx   = vox_out.read<int64_t>("Nx");
  auto const Ny   = vox_out.read<int64_t>("Ny");
  auto const Nz   = vox_out.read<int64_t>("Nz");
  auto const Npts = Nx * Ny * Nz;
  auto const Nb   = vox_out.read<int64_t>("Nb");
  fmt::println("Nx={}", Nx);
  fmt::println("Ny={}", Ny);
  fmt::println("Nz={}", Nz);
  fmt::println("Npts={}", Npts);
  fmt::println("Nb={}", Nb);

  //////////////////
  // bn_ixyz dataset
  //////////////////
  expected_ndims = 1;
  readDataset(vox_out.handle(), "bn_ixyz", expected_ndims, dims, (void**)&bn_ixyz, DataType::Int64);
  PFFDTD_ASSERT((int64_t)dims[0] == Nb);

  //////////////////
  // adj_bn dataset
  //////////////////
  expected_ndims = 2;
  readDataset(vox_out.handle(), "adj_bn", expected_ndims, dims, (void**)&adj_bn_bool, DataType::Bool);
  PFFDTD_ASSERT((int64_t)dims[0] == Nb);
  PFFDTD_ASSERT(dims[1] == (hsize_t)NN);

  //////////////////
  // mat_bn dataset
  //////////////////
  expected_ndims = 1;
  readDataset(vox_out.handle(), "mat_bn", expected_ndims, dims, (void**)&mat_bn, DataType::Int8);
  PFFDTD_ASSERT((int64_t)dims[0] == Nb);

  //////////////////
  // saf_bn dataset
  //////////////////
  expected_ndims = 1;
  readDataset(vox_out.handle(), "saf_bn", expected_ndims, dims, (void**)&saf_bn, DataType::Float64);
  PFFDTD_ASSERT((int64_t)dims[0] == Nb);

  allocate_zeros((void**)&ssaf_bn, Nb * sizeof(Real));
  for (int64_t i = 0; i < Nb; i++) {
    if (fcc_flag > 0) {
      ssaf_bn[i] = (Real)(0.5 / std::numbers::sqrt2) * saf_bn[i]; // rescale for S*h/V and cast
    } else {
      ssaf_bn[i] = (Real)saf_bn[i]; // just cast
    }
  }
  std::free(saf_bn);

  ////////////////////////////////////////////////////////////////////////
  // Read signals HDF5 dataset
  ////////////////////////////////////////////////////////////////////////
  filename = simDir / "signals.h5";
  if (not std::filesystem::exists(filename)) {
    raisef<std::invalid_argument>("file '{}' does not exist", filename.string());
  }

  auto signals = H5FReader(filename);

  //////////////////
  // integers
  //////////////////
  auto const Nt   = signals.read<int64_t>("Nt");
  auto const Ns   = signals.read<int64_t>("Ns");
  auto const Nr   = signals.read<int64_t>("Nr");
  auto const diff = signals.read<bool>("diff");
  fmt::println("Nt={}", Nt);
  fmt::println("Ns={}", Ns);
  fmt::println("Nr={}", Nr);
  fmt::println("diff={}", diff);

  //////////////////
  // in_ixyz dataset
  //////////////////
  expected_ndims = 1;
  readDataset(signals.handle(), "in_ixyz", expected_ndims, dims, (void**)&in_ixyz, DataType::Int64);
  PFFDTD_ASSERT((int64_t)dims[0] == Ns);

  //////////////////
  // out_ixyz dataset
  //////////////////
  expected_ndims = 1;
  readDataset(signals.handle(), "out_ixyz", expected_ndims, dims, (void**)&out_ixyz, DataType::Int64);
  PFFDTD_ASSERT((int64_t)dims[0] == Nr);

  expected_ndims = 1;
  readDataset(signals.handle(), "out_reorder", expected_ndims, dims, (void**)&out_reorder, DataType::Int64);
  PFFDTD_ASSERT((int64_t)dims[0] == Nr);

  //////////////////
  // in_sigs dataset
  //////////////////
  expected_ndims = 2;
  readDataset(signals.handle(), "in_sigs", expected_ndims, dims, (void**)&in_sigs, DataType::Float64);
  PFFDTD_ASSERT((int64_t)dims[0] == Ns);
  PFFDTD_ASSERT((int64_t)dims[1] == Nt);

  // not recommended to run single without differentiating input
  if (sizeof(Real) == 4) {
    PFFDTD_ASSERT(diff);
  }

  ////////////////////////////////////////////////////////////////////////
  // Read materials HDF5 dataset
  ////////////////////////////////////////////////////////////////////////
  filename = simDir / "materials.h5";
  if (not std::filesystem::exists(filename)) {
    raisef<std::invalid_argument>("file '{}' does not exist", filename.string());
  }

  auto materials = H5FReader(filename);

  //////////////////
  // integers
  //////////////////
  auto const Nm = materials.read<int8_t>("Nmat");
  fmt::println("Nm={}", Nm);
  PFFDTD_ASSERT(Nm <= MNm);

  expected_ndims = 1;
  readDataset(materials.handle(), "Mb", expected_ndims, dims, (void**)&Mb, DataType::Int8);

  for (int8_t i = 0; i < Nm; i++) {
    fmt::println("Mb[{}]={}", i, Mb[i]);
  }

  //////////////////
  // DEF (RLC) datasets
  //////////////////
  MatQuad<Real>* mat_quads = nullptr;
  Real* mat_beta           = nullptr; // one per material
  allocate_zeros((void**)&mat_quads, static_cast<unsigned long>(Nm * MMb) * sizeof(MatQuad<Real>));
  allocate_zeros((void**)&mat_beta, Nm * sizeof(Real));
  for (int8_t i = 0; i < Nm; i++) {
    double* DEF    = nullptr; // for one material
    auto id        = fmt::format("mat_{:02d}_DEF", i);
    expected_ndims = 2;
    readDataset(materials.handle(), id.c_str(), expected_ndims, dims, (void**)&DEF, DataType::Float64);
    PFFDTD_ASSERT((int8_t)dims[0] == Mb[i]);
    PFFDTD_ASSERT((int8_t)dims[1] == 3);
    PFFDTD_ASSERT(Mb[i] <= MMb);

    for (int8_t j = 0; j < Mb[i]; j++) {
      double const D = DEF[j * 3 + 0];
      double const E = DEF[j * 3 + 1];
      double const F = DEF[j * 3 + 2];
      fmt::println("DEF[{},{}]=[{:.16g}, {:.16g}, {:.16g}] ", i, j, D, E, F);

      // see 2016 ISMRA paper
      double const Dh = D / Ts;
      double const Eh = E;
      double const Fh = F * Ts;

      double const b   = 1.0 / (2.0 * Dh + Eh + 0.5 * Fh);
      double const bd  = b * (2.0 * Dh - Eh - 0.5 * Fh);
      double const bDh = b * Dh;
      double const bFh = b * Fh;
      PFFDTD_ASSERT(not std::isinf(b));
      PFFDTD_ASSERT(not std::isnan(b));
      PFFDTD_ASSERT(not std::isinf(bd));
      PFFDTD_ASSERT(not std::isnan(bd));

      int32_t const mij  = (int32_t)MMb * i + j;
      mat_quads[mij].b   = (Real)b;
      mat_quads[mij].bd  = (Real)bd;
      mat_quads[mij].bDh = (Real)bDh;
      mat_quads[mij].bFh = (Real)bFh;
      mat_beta[i] += (Real)b;
    }
    std::free(DEF);
  }

  ////////////////////////////////////////////////////////////////////////
  // Checks and repacking
  ////////////////////////////////////////////////////////////////////////

  //////////////////
  // check bn_ixyz
  //////////////////
  check_inside_grid(bn_ixyz, Nb, Nx, Ny, Nz);
  fmt::println("bn_ixyz checked");

  //////////////////
  // check adj_bn_bool and mat_bn
  //////////////////
  for (int64_t i = 0; i < Nb; i++) {
    bool at_least_one_not_adj = false;
    bool all_not_adj          = true;
    for (int8_t j = 0; j < NN; j++) {
      bool const adj = adj_bn_bool[i * NN + j];
      at_least_one_not_adj |= !adj;
      all_not_adj &= !adj;
    }
    PFFDTD_ASSERT(at_least_one_not_adj);
    if (all_not_adj) {
      PFFDTD_ASSERT(mat_bn[i] == -1);
    }
  }
  fmt::println("checked adj_bn against mat_bn.");

  //////////////////
  // bit-pack and check adj_bn
  //////////////////
  allocate_zeros((void**)&adj_bn, Nb * sizeof(uint16_t));

  for (int64_t i = 0; i < Nb; i++) {
    for (int8_t j = 0; j < NN; j++) {
      SET_BIT_VAL(adj_bn[i], j, adj_bn_bool[i * NN + j]);
    }
  }
  fmt::println("adj_bn filled");

  for (int64_t i = 0; i < Nb; i++) {
    for (int8_t j = 0; j < NN; j++) { // avoids race conditions
      PFFDTD_ASSERT(GET_BIT(adj_bn[i], j) == adj_bn_bool[i * NN + j]);
    }
  }
  fmt::println("adj_bn double checked");
  std::free(adj_bn_bool);

  //////////////////
  // calculate K_bn from adj_bn
  //////////////////
  allocate_zeros((void**)&K_bn, Nb * sizeof(int8_t));

  for (int64_t nb = 0; nb < Nb; nb++) {
    K_bn[nb] = 0;
    for (uint8_t nn = 0; nn < NN; nn++) {
      K_bn[nb] += GET_BIT(adj_bn[nb], nn);
    }
  }
  fmt::println("K_bn calculated");

  //////////////////
  // bit-pack and check bn_mask
  //////////////////
  // make compressed bit-mask
  int64_t const Nbm = (Npts - 1) / 8 + 1;
  allocate_zeros((void**)&bn_mask, Nbm); // one bit per
  for (int64_t i = 0; i < Nb; i++) {
    int64_t const ii = bn_ixyz[i];
    SET_BIT(bn_mask[ii >> 3], ii % 8);
  }

  // create bn_mask_raw to double check
  bool* bn_mask_raw = nullptr;
  allocate_zeros((void**)&bn_mask_raw, Npts * sizeof(bool));

  for (int64_t i = 0; i < Nb; i++) {
    int64_t const ii = bn_ixyz[i];
    PFFDTD_ASSERT(ii < Npts);
    bn_mask_raw[ii] = true;
  }
  fmt::println("bn_mask_raw filled");

  for (int64_t j = 0; j < Nbm; j++) {
    for (int64_t q = 0; q < 8; q++) { // avoid race conditions
      int64_t const i = j * 8 + q;
      if (i < Npts) {
        PFFDTD_ASSERT(GET_BIT(bn_mask[i >> 3], i % 8) == bn_mask_raw[i]);
      }
    }
  }
  fmt::println("bn_mask double checked");
  std::free(bn_mask_raw);

  // count Nbl
  int64_t Nbl = 0;
  for (int64_t i = 0; i < Nb; i++) {
    Nbl += static_cast<int64_t>(mat_bn[i] >= 0);
  }
  fmt::println("Nbl = {}", Nbl);
  allocate_zeros((void**)&mat_bnl, Nbl * sizeof(int8_t));
  allocate_zeros((void**)&bnl_ixyz, Nbl * sizeof(int64_t));
  allocate_zeros((void**)&ssaf_bnl, Nbl * sizeof(Real));
  {
    int64_t j = 0;
    for (int64_t i = 0; i < Nb; i++) {
      if (mat_bn[i] >= 0) {
        mat_bnl[j]  = mat_bn[i];
        ssaf_bnl[j] = ssaf_bn[i];
        bnl_ixyz[j] = bn_ixyz[i];
        j++;
      }
    }
    PFFDTD_ASSERT(j == Nbl);
  }
  std::free(mat_bn);
  std::free(ssaf_bn);

  fmt::println("separated non-rigid bn");

  // ABC ndoes
  int64_t Nyf = (fcc_flag == 2) ? 2 * (Ny - 1) : Ny; // full Ny dim, taking into account FCC fold
  int64_t Nba = 2 * (Nx * Nyf + Nx * Nz + Nyf * Nz) - 12 * (Nx + Nyf + Nz) + 56;
  if (fcc_flag > 0) {
    Nba /= 2;
  }

  allocate_zeros((void**)&bna_ixyz, Nba * sizeof(int64_t));
  allocate_zeros((void**)&Q_bna, Nba * sizeof(int8_t));
  {
    int64_t ii = 0;
    for (int64_t ix = 1; ix < Nx - 1; ix++) {
      for (int64_t iy = 1; iy < Nyf - 1; iy++) {
        for (int64_t iz = 1; iz < Nz - 1; iz++) {

          if ((fcc_flag > 0) && (ix + iy + iz) % 2 == 1) {
            continue;
          }

          int8_t Q = 0;
          Q += static_cast<int>((ix == 1) || (ix == Nx - 2));
          Q += static_cast<int>((iy == 1) || (iy == Nyf - 2));
          Q += static_cast<int>((iz == 1) || (iz == Nz - 2));
          if (Q > 0) {
            if ((fcc_flag == 2) && (iy >= Nyf / 2)) {
              bna_ixyz[ii] = ix * Nz * Ny + (Nyf - iy - 1) * Nz + iz; // index on folded grid
            } else {
              bna_ixyz[ii] = ix * Nz * Ny + iy * Nz + iz;
            }
            Q_bna[ii] = Q;
            ii += 1;
          }
        }
      }
    }
    PFFDTD_ASSERT(ii == Nba);
    fmt::println("ABC nodes");
    if (fcc_flag == 2) { // need to sort bna_ixyz
      int64_t* bna_sort_keys = nullptr;
      allocate_zeros((void**)&bna_sort_keys, Nba * sizeof(int64_t));
      sort_keys(bna_ixyz, bna_sort_keys, Nba);

      // now sort corresponding Q_bna array
      int8_t* Q_bna_sorted   = nullptr;
      int8_t* Q_bna_unsorted = nullptr;
      allocate_zeros((void**)&Q_bna_sorted, Nba * sizeof(int8_t));
      // swap pointers
      Q_bna_unsorted = Q_bna;
      Q_bna          = Q_bna_sorted;

      for (int64_t cc = 0; cc < Nba; cc++) {
        Q_bna_sorted[cc] = Q_bna_unsorted[bna_sort_keys[cc]];
      }
      std::free(bna_sort_keys);
      std::free(Q_bna_unsorted);
      fmt::println("sorted ABC nodes for FCC/GPU");
    }
  }

  // for outputs
  allocate_zeros((void**)&u_out, Nr * Nt * sizeof(double));

  /*------------------------
   * ATTACH
  ------------------------*/
  auto sim        = Simulation3D{};
  sim.Ns          = Ns;
  sim.Nr          = Nr;
  sim.Nt          = Nt;
  sim.Npts        = Npts;
  sim.Nx          = Nx;
  sim.Ny          = Ny;
  sim.Nz          = Nz;
  sim.Nb          = Nb;
  sim.Nbl         = Nbl;
  sim.Nba         = Nba;
  sim.l           = l;
  sim.l2          = l2;
  sim.fcc_flag    = fcc_flag;
  sim.Nm          = Nm;
  sim.NN          = NN;
  sim.a2          = a2;
  sim.a1          = a1;
  sim.sl2         = sl2;
  sim.lo2         = lo2;
  sim.Mb          = Mb;
  sim.bn_ixyz     = bn_ixyz;
  sim.bnl_ixyz    = bnl_ixyz;
  sim.bna_ixyz    = bna_ixyz;
  sim.Q_bna       = Q_bna;
  sim.adj_bn      = adj_bn;
  sim.ssaf_bnl    = ssaf_bnl;
  sim.bn_mask     = bn_mask;
  sim.mat_bnl     = mat_bnl;
  sim.K_bn        = K_bn;
  sim.out_ixyz    = out_ixyz;
  sim.out_reorder = out_reorder;
  sim.in_ixyz     = in_ixyz;
  sim.in_sigs     = in_sigs;
  sim.u_out       = u_out;
  sim.mat_beta    = mat_beta;
  sim.mat_quads   = mat_quads;
  return sim;
}

void freeSimulation3D(Simulation3D& sim) {
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
  fmt::println("sim data freed");
}

// print last samples of simulation (for correctness checking..)
void printLastSample(Simulation3D& sim) {
  int64_t const Nt     = sim.Nt;
  int64_t const Nr     = sim.Nr;
  double* u_out        = sim.u_out;
  int64_t* out_reorder = sim.out_reorder;
  // print last samples
  fmt::println("RAW OUTPUTS");
  for (int64_t nr = 0; nr < Nr; nr++) {
    fmt::println("receiver {}", nr);
    for (int64_t n = Nt - 5; n < Nt; n++) {
      printf("sample %ld: %.16e\n", long(n), u_out[out_reorder[nr] * Nt + n]);
    }
  }
}

// scale input to be in middle of floating-point range
void scaleInput(Simulation3D& sim) {
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
  double const aexp = 0.5; // normalise to middle power of two
  auto pow2         = (int32_t)round(aexp * REAL_MAX_EXP + (1 - aexp) * REAL_MIN_EXP);
  // int32_t pow2 = 0; //normalise to one
  double const norm1     = pow(2.0, pow2);
  double const inv_infac = norm1 / max_in;
  double const infac     = 1.0 / inv_infac;

  printf(
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

// undo that scaling
void rescaleOutput(Simulation3D& sim) {
  int64_t const Nt = sim.Nt;
  int64_t const Nr = sim.Nr;
  double infac     = sim.infac;
  double* u_out    = sim.u_out;

  std::transform(u_out, u_out + Nr * Nt, u_out, [infac](auto sample) { return sample * infac; });
}

void writeOutputs(Simulation3D& sim, std::filesystem::path const& simDir) {
  auto Nt           = static_cast<size_t>(sim.Nt);
  auto Nr           = static_cast<size_t>(sim.Nr);
  auto* out_reorder = sim.out_reorder;
  auto u_out        = stdex::mdarray<double, stdex::dextents<size_t, 2>>(Nr, Nt);

  // write outputs in correct order
  for (auto nr = size_t{0}; nr < Nr; ++nr) {
    for (auto n = size_t{0}; n < Nt; ++n) {
      u_out(nr, n) = sim.u_out[out_reorder[nr] * Nt + n];
    }
  }

  auto writer = H5FWriter{simDir / "sim_outs.h5"};
  writer.write("u_out", u_out);
  std::puts("wrote output dataset");
}

} // namespace pffdtd
