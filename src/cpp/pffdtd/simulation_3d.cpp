// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton

#include "simulation_3d.hpp"

#include "pffdtd/assert.hpp"

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
  int64_t Nx               = 0;
  int64_t Ny               = 0;
  int64_t Nz               = 0;
  int64_t Nb               = 0;
  int64_t Nbl              = 0;
  int64_t Nba              = 0;
  int64_t Npts             = 0;
  int64_t Ns               = 0;
  int64_t Nr               = 0;
  int64_t Nt               = 0;
  int64_t* bn_ixyz         = nullptr;
  int64_t* bnl_ixyz        = nullptr;
  int64_t* bna_ixyz        = nullptr;
  int8_t* Q_bna            = nullptr;
  int64_t* in_ixyz         = nullptr;
  int64_t* out_ixyz        = nullptr;
  int64_t* out_reorder     = nullptr;
  bool* adj_bn_bool        = nullptr;
  int8_t* K_bn             = nullptr;
  uint16_t* adj_bn         = nullptr; // large enough for FCC
  uint8_t* bn_mask         = nullptr;
  int8_t* mat_bn           = nullptr;
  int8_t* mat_bnl          = nullptr;
  double* saf_bn           = nullptr;
  Real* ssaf_bn            = nullptr;
  Real* ssaf_bnl           = nullptr;
  double* in_sigs          = nullptr;
  double* u_out            = nullptr;
  double l                 = NAN;
  double l2                = NAN;
  int8_t fcc_flag          = 0;
  int8_t NN                = 0;
  int8_t* Mb               = nullptr;
  int8_t Nm                = 0;
  MatQuad<Real>* mat_quads = nullptr;
  Real* mat_beta           = nullptr; // one per material

  double Ts = NAN;
  bool diff = false;

  hid_t file = 0;  // HDF5 type
  hsize_t dims[2]; // HDF5 type
  int expected_ndims = 0;
  char dset_str[80];
  // char filename[80];

  ////////////////////////////////////////////////////////////////////////
  //
  // Read constants HDF5 dataset
  //
  ////////////////////////////////////////////////////////////////////////
  auto filename = simDir / "constants.h5";
  if (not std::filesystem::exists(filename)) {
    PFFDTD_ASSERT(true == false);
  }

  file = H5Fopen(filename.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  //////////////////
  // constants
  //////////////////
  strcpy(dset_str, "l");
  readH5Constant(file, dset_str, (void*)&l, FLOAT64);
  printf("l=%.16g\n", l);

  strcpy(dset_str, "l2");
  readH5Constant(file, dset_str, (void*)&l2, FLOAT64);
  printf("l2=%.16g\n", l2);

  strcpy(dset_str, "Ts");
  readH5Constant(file, dset_str, (void*)&Ts, FLOAT64);
  printf("Ts=%.16g\n", Ts);

  strcpy(dset_str, "fcc_flag");
  readH5Constant(file, dset_str, (void*)&fcc_flag, INT8);
  printf("fcc_flag=%d\n", fcc_flag);
  PFFDTD_ASSERT((fcc_flag >= 0) && (fcc_flag <= 2));

  if (H5Fclose(file) != 0) {
    fmt::println("error closing file {}", filename.string());
    PFFDTD_ASSERT(true == false);
  } else {
    fmt::println("closed file {}", filename.string());
  }

  if (fcc_flag > 0) { // FCC (1 is CPU-based, 2 is CPU or GPU)
    PFFDTD_ASSERT(l2 <= 1.0);
    PFFDTD_ASSERT(l <= 1.0);
    NN = 12;
  } else { // simple Cartesian
    PFFDTD_ASSERT(l2 <= 1.0 / 3.0);
    PFFDTD_ASSERT(l <= sqrt(1.0 / 3.0));
    NN = 6;
  }

  // calculate some update coefficients
  double const lfac = (fcc_flag > 0) ? 0.25 : 1.0; // laplacian factor
  double const dsl2 = (1.0 + EPS) * lfac * l2;     // scale for stability (EPS in fdtd_common.hpp)
  double const da1  = (2.0 - dsl2 * NN);           // scaling for stability in single
  double const da2  = lfac * l2;
  // Real is defined in fdtd_common.hpp (float or double)
  Real const a1  = da1;
  Real const a2  = da2;
  Real const sl2 = dsl2;
  Real const lo2 = 0.5 * l;

  printf("a2 (double): %.16g\n", da2);
  printf("a2 (Real): %.16g\n", a2);
  printf("a1 (double): %.16g\n", da1);
  printf("a1 (Real): %.16g\n", a1);
  printf("sl2 (double): %.16g\n", dsl2);
  printf("sl2 (Real): %.16g\n", sl2);

  printf("l2=%.16g\n", l2);
  printf("NN=%d\n", NN);

  ////////////////////////////////////////////////////////////////////////
  //
  // Read vox HDF5 dataset
  //
  ////////////////////////////////////////////////////////////////////////
  filename = simDir / "vox_out.h5";
  if (not std::filesystem::exists(filename)) {
    PFFDTD_ASSERT(true == false);
  }

  file = H5Fopen(filename.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  //////////////////
  // integers
  //////////////////
  strcpy(dset_str, "Nx");
  readH5Constant(file, dset_str, (void*)&Nx, INT64);
  printf("Nx=%ld\n", static_cast<long>(Nx));

  strcpy(dset_str, "Ny");
  readH5Constant(file, dset_str, (void*)&Ny, INT64);
  printf("Ny=%ld\n", static_cast<long>(Ny));

  strcpy(dset_str, "Nz");
  readH5Constant(file, dset_str, (void*)&Nz, INT64);
  printf("Nz=%ld\n", static_cast<long>(Nz));

  Npts = Nx * Ny * Nz;
  printf("Npts=%ld\n", static_cast<long>(Npts));

  strcpy(dset_str, "Nb");
  readH5Constant(file, dset_str, (void*)&Nb, INT64);
  printf("Nb=%ld\n", static_cast<long>(Nb));

  //////////////////
  // bn_ixyz dataset
  //////////////////
  strcpy(dset_str, "bn_ixyz");
  expected_ndims = 1;
  readH5Dataset(file, dset_str, expected_ndims, dims, (void**)&bn_ixyz, INT64);
  PFFDTD_ASSERT((int64_t)dims[0] == Nb);

  //////////////////
  // adj_bn dataset
  //////////////////
  strcpy(dset_str, "adj_bn");
  expected_ndims = 2;
  readH5Dataset(file, dset_str, expected_ndims, dims, (void**)&adj_bn_bool, BOOL);
  PFFDTD_ASSERT((int64_t)dims[0] == Nb);
  PFFDTD_ASSERT(dims[1] == (hsize_t)NN);

  //////////////////
  // mat_bn dataset
  //////////////////
  strcpy(dset_str, "mat_bn");
  expected_ndims = 1;
  readH5Dataset(file, dset_str, expected_ndims, dims, (void**)&mat_bn, INT8);
  PFFDTD_ASSERT((int64_t)dims[0] == Nb);

  //////////////////
  // saf_bn dataset
  //////////////////
  strcpy(dset_str, "saf_bn");
  expected_ndims = 1;
  readH5Dataset(file, dset_str, expected_ndims, dims, (void**)&saf_bn, FLOAT64);
  PFFDTD_ASSERT((int64_t)dims[0] == Nb);

  allocate_zeros((void**)&ssaf_bn, Nb * sizeof(Real));
  for (int64_t i = 0; i < Nb; i++) {
    if (fcc_flag > 0) {
      ssaf_bn[i] = (Real)(0.5 / std::numbers::sqrt2) * saf_bn[i]; // rescale for S*h/V and cast
    } else {
      ssaf_bn[i] = (Real)saf_bn[i]; // just cast
    }
  }
  free(saf_bn);

  if (H5Fclose(file) != 0) {
    fmt::println("error closing file {}", filename.string());
    PFFDTD_ASSERT(true == false);
  } else {
    fmt::println("closed file {}", filename.string());
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // Read signals HDF5 dataset
  //
  ////////////////////////////////////////////////////////////////////////
  filename = simDir / "signals.h5";
  if (not std::filesystem::exists(filename)) {
    PFFDTD_ASSERT(true == false);
  }

  file = H5Fopen(filename.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  //////////////////
  // integers
  //////////////////
  strcpy(dset_str, "Nt");
  readH5Constant(file, dset_str, (void*)&Nt, INT64);
  printf("Nt=%ld\n", static_cast<long>(Nt));

  strcpy(dset_str, "Ns");
  readH5Constant(file, dset_str, (void*)&Ns, INT64);
  printf("Ns=%ld\n", static_cast<long>(Ns));

  strcpy(dset_str, "Nr");
  readH5Constant(file, dset_str, (void*)&Nr, INT64);
  printf("Nr=%ld\n", static_cast<long>(Nr));

  strcpy(dset_str, "Nr");
  readH5Constant(file, dset_str, (void*)&Nr, INT64);
  printf("Nr=%ld\n", static_cast<long>(Nr));

  strcpy(dset_str, "diff");
  readH5Constant(file, dset_str, (void*)&diff, BOOL);
  printf("diff=%d\n", static_cast<int>(diff));

  //////////////////
  // in_ixyz dataset
  //////////////////
  strcpy(dset_str, "in_ixyz");
  expected_ndims = 1;
  readH5Dataset(file, dset_str, expected_ndims, dims, (void**)&in_ixyz, INT64);
  PFFDTD_ASSERT((int64_t)dims[0] == Ns);

  //////////////////
  // out_ixyz dataset
  //////////////////
  strcpy(dset_str, "out_ixyz");
  expected_ndims = 1;
  readH5Dataset(file, dset_str, expected_ndims, dims, (void**)&out_ixyz, INT64);
  PFFDTD_ASSERT((int64_t)dims[0] == Nr);

  strcpy(dset_str, "out_reorder");
  expected_ndims = 1;
  readH5Dataset(file, dset_str, expected_ndims, dims, (void**)&out_reorder, INT64);
  PFFDTD_ASSERT((int64_t)dims[0] == Nr);

  //////////////////
  // in_sigs dataset
  //////////////////
  strcpy(dset_str, "in_sigs");
  expected_ndims = 2;
  readH5Dataset(file, dset_str, expected_ndims, dims, (void**)&in_sigs, FLOAT64);
  PFFDTD_ASSERT((int64_t)dims[0] == Ns);
  PFFDTD_ASSERT((int64_t)dims[1] == Nt);

  if (H5Fclose(file) != 0) {
    fmt::println("error closing file {}", filename.string());
    PFFDTD_ASSERT(true == false);
  } else {
    fmt::println("closed file {}", filename.string());
  }

  // not recommended to run single without differentiating input
  if (sizeof(Real) == 4) {
    PFFDTD_ASSERT(diff);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // Read materials HDF5 dataset
  //
  ////////////////////////////////////////////////////////////////////////
  filename = simDir / "materials.h5";
  if (not std::filesystem::exists(filename)) {
    PFFDTD_ASSERT(true == false);
  }

  file = H5Fopen(filename.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  //////////////////
  // integers
  //////////////////

  strcpy(dset_str, "Nmat");
  readH5Constant(file, dset_str, (void*)&Nm, INT8);
  printf("Nm=%d\n", Nm);

  PFFDTD_ASSERT(Nm <= MNm);

  strcpy(dset_str, "Mb");
  expected_ndims = 1;
  readH5Dataset(file, dset_str, expected_ndims, dims, (void**)&Mb, INT8);

  for (int8_t i = 0; i < Nm; i++) {
    printf("Mb[%d]=%d\n", i, Mb[i]);
  }

  //////////////////
  // DEF (RLC) datasets
  //////////////////
  allocate_zeros((void**)&mat_quads, static_cast<unsigned long>(Nm * MMb) * sizeof(MatQuad<Real>));
  allocate_zeros((void**)&mat_beta, Nm * sizeof(Real));
  for (int8_t i = 0; i < Nm; i++) {
    double* DEF = nullptr; // for one material
    // sprintf(dset_str, "mat_%02d_DEF", i);
    auto id        = fmt::format("mat_{:02d}_DEF", i);
    expected_ndims = 2;
    readH5Dataset(file, id.data(), expected_ndims, dims, (void**)&DEF, FLOAT64);
    PFFDTD_ASSERT((int8_t)dims[0] == Mb[i]);
    PFFDTD_ASSERT((int8_t)dims[1] == 3);
    PFFDTD_ASSERT(Mb[i] <= MMb);

    for (int8_t j = 0; j < Mb[i]; j++) {
      double const D = DEF[j * 3 + 0];
      double const E = DEF[j * 3 + 1];
      double const F = DEF[j * 3 + 2];
      printf("DEF[%d,%d]=[%.16g, %.16g, %.16g] \n", i, j, D, E, F);

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
    free(DEF);
  }

  if (H5Fclose(file) != 0) {
    fmt::println("error closing file {}", filename.string());
    PFFDTD_ASSERT(true == false);
  } else {
    fmt::println("closed file {}", filename.string());
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // Checks and repacking
  //
  ////////////////////////////////////////////////////////////////////////

  //////////////////
  // check bn_ixyz
  //////////////////
  check_inside_grid(bn_ixyz, Nb, Nx, Ny, Nz);
  printf("bn_ixyz checked\n");

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
  printf("checked adj_bn against mat_bn.\n");

  //////////////////
  // bit-pack and check adj_bn
  //////////////////
  allocate_zeros((void**)&adj_bn, Nb * sizeof(uint16_t));

  for (int64_t i = 0; i < Nb; i++) {
    for (int8_t j = 0; j < NN; j++) {
      SET_BIT_VAL(adj_bn[i], j, adj_bn_bool[i * NN + j]);
    }
  }
  printf("adj_bn filled\n");

  for (int64_t i = 0; i < Nb; i++) {
    for (int8_t j = 0; j < NN; j++) { // avoids race conditions
      PFFDTD_ASSERT(GET_BIT(adj_bn[i], j) == adj_bn_bool[i * NN + j]);
    }
  }
  printf("adj_bn double checked\n");
  free(adj_bn_bool);

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
  printf("K_bn calculated\n");

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
  printf("bn_mask_raw filled\n");

  for (int64_t j = 0; j < Nbm; j++) {
    for (int64_t q = 0; q < 8; q++) { // avoid race conditions
      int64_t const i = j * 8 + q;
      if (i < Npts) {
        PFFDTD_ASSERT(GET_BIT(bn_mask[i >> 3], i % 8) == bn_mask_raw[i]);
      }
    }
  }
  printf("bn_mask double checked\n");
  free(bn_mask_raw);

  // count Nbl
  Nbl = 0;
  for (int64_t i = 0; i < Nb; i++) {
    Nbl += static_cast<int64_t>(mat_bn[i] >= 0);
  }
  printf("Nbl = %ld\n", static_cast<long>(Nbl));
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
  free(mat_bn);
  free(ssaf_bn);

  printf("separated non-rigid bn\n");

  // ABC ndoes
  int64_t Nyf = 0;
  Nyf         = (fcc_flag == 2) ? 2 * (Ny - 1) : Ny; // full Ny dim, taking into account FCC fold
  Nba         = 2 * (Nx * Nyf + Nx * Nz + Nyf * Nz) - 12 * (Nx + Nyf + Nz) + 56;
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
    printf("ABC nodes\n");
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
      free(bna_sort_keys);
      free(Q_bna_unsorted);
      printf("sorted ABC nodes for FCC/GPU\n");
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

// free everything
void freeSimulation3D(Simulation3D& sim) {
  /*------------------------
   * FREE WILLY
  ------------------------*/
  free(sim.bn_ixyz);
  free(sim.bnl_ixyz);
  free(sim.bna_ixyz);
  free(sim.Q_bna);
  free(sim.adj_bn);
  free(sim.mat_bnl);
  free(sim.bn_mask);
  free(sim.ssaf_bnl);
  free(sim.K_bn);
  free(sim.in_ixyz);
  free(sim.out_ixyz);
  free(sim.out_reorder);
  free(sim.in_sigs);
  free(sim.u_out);
  free(sim.Mb);
  free(sim.mat_beta);
  free(sim.mat_quads);
  printf("sim data freed\n");
}

// read HDF5 files
void readH5Dataset(hid_t file, char* dset_str, int ndims, hsize_t* dims, void** out_array, DataType t) {
  hid_t dset   = 0;
  hid_t dspace = 0;
  uint64_t N   = 0;
  // herr_t status;

  dset   = H5Dopen(file, dset_str, H5P_DEFAULT);
  dspace = H5Dget_space(dset);
  PFFDTD_ASSERT(H5Sget_simple_extent_ndims(dspace) == ndims);
  H5Sget_simple_extent_dims(dspace, dims, nullptr);
  if (ndims == 1) {
    // printf("size dim 0 = %llu\n",dims[0]);
    N = dims[0];
  } else if (ndims == 2) {
    // printf("size dim 0 = %llu\n",dims[0]);
    // printf("size dim 1 = %llu\n",dims[1]);
    N = dims[0] * dims[1];
  } else {
    PFFDTD_ASSERT(true == false);
  }
  switch (t) {
    case FLOAT64: *out_array = (double*)malloc(N * sizeof(double)); break;
    case FLOAT32: *out_array = (double*)malloc(N * sizeof(float)); break;
    case INT64: *out_array = (int64_t*)malloc(N * sizeof(int64_t)); break;
    case INT8: *out_array = (int8_t*)malloc(N * sizeof(int8_t)); break;
    case BOOL: *out_array = (bool*)malloc(N * sizeof(bool)); break;
    default: PFFDTD_ASSERT(true == false);
  }
  if (*out_array == nullptr) {
    printf("Memory allocation failed");
    PFFDTD_ASSERT(true == false); // to break
  }
  herr_t status = 0;
  switch (t) {
    case FLOAT64: status = H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, *out_array); break;
    case FLOAT32: status = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, *out_array); break;
    case INT64: status = H5Dread(dset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, *out_array); break;
    case INT8: // bool read in as INT8
    case BOOL: status = H5Dread(dset, H5T_NATIVE_INT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, *out_array); break;
    default: PFFDTD_ASSERT(true == false);
  }

  if (status != 0) {
    printf("error reading dataset: %s\n", dset_str);
    PFFDTD_ASSERT(true == false);
  }
  if (H5Dclose(dset) != 0) {
    printf("error closing dataset: %s\n", dset_str);
    PFFDTD_ASSERT(true == false);
  } else {
    printf("read and closed dataset: %s\n", dset_str);
  }
}

// read scalars from HDF5 datasets
void readH5Constant(hid_t file, char* dset_str, void* out, DataType t) {
  hid_t dset   = 0;
  hid_t dspace = 0;

  dset   = H5Dopen(file, dset_str, H5P_DEFAULT);
  dspace = H5Dget_space(dset);
  PFFDTD_ASSERT(H5Sget_simple_extent_ndims(dspace) == 0);
  herr_t status = 0;
  switch (t) {
    case FLOAT64: status = H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out); break;
    case INT64: status = H5Dread(dset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, out); break;
    case INT8:
    case BOOL: status = H5Dread(dset, H5T_NATIVE_INT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, out); break;
    default: PFFDTD_ASSERT(true == false);
  }

  if (status != 0) {
    printf("error reading dataset: %s\n", dset_str);
    PFFDTD_ASSERT(true == false);
  }
  if (H5Dclose(dset) != 0) {
    printf("error closing dataset: %s\n", dset_str);
    PFFDTD_ASSERT(true == false);
  } else {
    printf("read constant: %s\n", dset_str);
  }
}

// print last samples of simulation (for correctness checking..)
void printLastSample(Simulation3D& sim) {
  int64_t const Nt     = sim.Nt;
  int64_t const Nr     = sim.Nr;
  double* u_out        = sim.u_out;
  int64_t* out_reorder = sim.out_reorder;
  // print last samples
  printf("RAW OUTPUTS\n");
  for (int64_t nr = 0; nr < Nr; nr++) {
    printf("receiver %ld\n", static_cast<long>(nr));
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
