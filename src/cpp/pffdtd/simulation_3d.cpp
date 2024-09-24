// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton

#include "simulation_3d.hpp"

#include "pffdtd/hdf.hpp"

#include <fmt/format.h>

#include <cfloat>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numbers>

namespace pffdtd {

namespace {

template<typename T>
constexpr auto EPS = 0.0;

template<>
constexpr auto EPS<float> = 1.19209289e-07;

// sort and return indices
void sort_keys(int64_t* val_arr, int64_t* key_arr, int64_t N) {
  // for sorting int64 arrays and returning keys
  struct Item {
    int64_t val;
    int64_t idx;
  };

  auto tmp = std::vector<Item>(static_cast<size_t>(N));
  for (int64_t i = 0; i < N; i++) {
    tmp[i].val = val_arr[i];
    tmp[i].idx = i;
  }

  // comparator with indice keys (for FCC ABC nodes)
  std::qsort(tmp.data(), N, sizeof(Item), [](void const* a, void const* b) -> int {
    auto const& as = *reinterpret_cast<Item const*>(a);
    auto const& bs = *reinterpret_cast<Item const*>(b);
    if (as.val < bs.val) {
      return -1;
    }
    if (as.val > bs.val) {
      return 1;
    }
    return 0;
  });

  for (int64_t i = 0; i < N; i++) {
    val_arr[i] = tmp[i].val;
    key_arr[i] = tmp[i].idx;
  }
}

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
void check_inside_grid(int64_t const* idx, int64_t N, int64_t Nx, int64_t Ny, int64_t Nz) {
  for (int64_t i = 0; i < N; i++) {
    int64_t iz = 0;
    int64_t iy = 0;
    int64_t ix = 0;
    ind2sub3d(idx[i], Nx, Ny, Nz, &ix, &iy, &iz);
  }
}

// load the sim data from Python-written HDF5 files
template<typename Real>
[[nodiscard]] auto loadSimulation3D_impl(std::filesystem::path const& simDir) -> Simulation3D<Real> {
  int expected_ndims = 0;
  hsize_t dims[2]    = {};

  ////////////////////////////////////////////////////////////////////////
  // Read constants HDF5 dataset
  ////////////////////////////////////////////////////////////////////////
  auto filename = simDir / "constants.h5";
  if (not std::filesystem::exists(filename)) {
    raisef<std::invalid_argument>("file '{}' does not exist", filename.string());
  }

  auto constants = H5FReader{filename};

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
  double const dsl2 = (1.0 + EPS<Real>)*lfac * l2; // scale for stability (EPS in fdtd_common.hpp)
  double const da1  = (2.0 - dsl2 * NN);           // scaling for stability in single
  double const da2  = lfac * l2;

  auto const a1  = static_cast<Real>(da1);
  auto const a2  = static_cast<Real>(da2);
  auto const sl2 = static_cast<Real>(dsl2);
  auto const lo2 = static_cast<Real>(0.5 * l);

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

  auto vox_out = H5FReader{filename};

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
  auto bn_ixyz = vox_out.read<std::vector<int64_t>>("bn_ixyz");
  PFFDTD_ASSERT(std::cmp_equal(bn_ixyz.size(), Nb));

  //////////////////
  // adj_bn dataset
  //////////////////
  expected_ndims    = 2;
  bool* adj_bn_bool = read<bool>(vox_out, "adj_bn", expected_ndims, dims);
  PFFDTD_ASSERT(static_cast<int64_t>(dims[0]) == Nb);
  PFFDTD_ASSERT(dims[1] == (hsize_t)NN);

  //////////////////
  // mat_bn dataset
  //////////////////
  auto mat_bn = vox_out.read<std::vector<int8_t>>("mat_bn");
  PFFDTD_ASSERT(std::cmp_equal(mat_bn.size(), Nb));

  //////////////////
  // saf_bn dataset
  //////////////////
  expected_ndims = 1;
  double* saf_bn = read<double>(vox_out, "saf_bn", expected_ndims, dims);
  PFFDTD_ASSERT(static_cast<int64_t>(dims[0]) == Nb);

  auto* ssaf_bn = allocate_zeros<Real>(Nb);
  for (int64_t i = 0; i < Nb; i++) {
    if (fcc_flag > 0) {
      ssaf_bn[i] = (Real)(0.5 / std::numbers::sqrt2) * saf_bn[i]; // rescale for S*h/V and cast
    } else {
      ssaf_bn[i] = (Real)saf_bn[i]; // just cast
    }
  }
  delete[] saf_bn;

  ////////////////////////////////////////////////////////////////////////
  // Read signals HDF5 dataset
  ////////////////////////////////////////////////////////////////////////
  filename = simDir / "signals.h5";
  if (not std::filesystem::exists(filename)) {
    raisef<std::invalid_argument>("file '{}' does not exist", filename.string());
  }

  auto signals = H5FReader{filename};

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
  auto in_ixyz = signals.read<std::vector<int64_t>>("in_ixyz");
  PFFDTD_ASSERT(std::cmp_equal(in_ixyz.size(), Ns));

  //////////////////
  // out_ixyz dataset
  //////////////////
  auto out_ixyz = signals.read<std::vector<int64_t>>("out_ixyz");
  PFFDTD_ASSERT(std::cmp_equal(out_ixyz.size(), Nr));

  auto out_reorder = signals.read<std::vector<int64_t>>("out_reorder");
  PFFDTD_ASSERT(std::cmp_equal(out_reorder.size(), Nr));

  //////////////////
  // in_sigs dataset
  //////////////////
  auto in_sigs = signals.read<std::vector<double>>("in_sigs");
  PFFDTD_ASSERT(std::cmp_equal(in_sigs.size(), Ns * Nt));

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

  auto materials = H5FReader{filename};

  //////////////////
  // integers
  //////////////////
  auto const Nm = materials.read<int8_t>("Nmat");
  fmt::println("Nm={}", Nm);
  PFFDTD_ASSERT(Nm <= MNm);

  auto Mb = materials.read<std::vector<int8_t>>("Mb");
  PFFDTD_ASSERT(std::cmp_equal(Mb.size(), Nm));

  for (int8_t i = 0; i < Nm; i++) {
    fmt::println("Mb[{}]={}", i, Mb[i]);
  }

  //////////////////
  // DEF (RLC) datasets
  //////////////////
  auto mat_beta  = std::vector<Real>(size_t(Nm));
  auto mat_quads = std::vector<MatQuad<Real>>(static_cast<unsigned long>(Nm) * size_t(MMb));
  for (int8_t i = 0; i < Nm; i++) {
    expected_ndims = 2;
    auto* DEF      = read<double>(materials, fmt::format("mat_{:02d}_DEF", i).c_str(), expected_ndims, dims);
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
    delete[] DEF;
  }

  ////////////////////////////////////////////////////////////////////////
  // Checks and repacking
  ////////////////////////////////////////////////////////////////////////

  //////////////////
  // check bn_ixyz
  //////////////////
  check_inside_grid(bn_ixyz.data(), Nb, Nx, Ny, Nz);
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
  auto adj_bn = std::vector<uint16_t>(size_t(Nb));

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
  delete[] adj_bn_bool;

  //////////////////
  // calculate K_bn from adj_bn
  //////////////////
  auto K_bn = std::vector<int8_t>(size_t(Nb));
  for (int64_t nb = 0; nb < Nb; nb++) {
    K_bn[nb] = 0;
    for (uint8_t nn = 0; nn < NN; nn++) {
      K_bn[nb] = static_cast<int8_t>(K_bn[nb] + get_bit_as<int>(adj_bn[nb], nn));
    }
  }
  fmt::println("K_bn calculated");

  //////////////////
  // bit-pack and check bn_mask
  //////////////////
  // make compressed bit-mask
  int64_t const Nbm = (Npts - 1) / 8 + 1;
  auto bn_mask      = std::vector<uint8_t>(size_t(Nbm)); // one bit per
  for (int64_t i = 0; i < Nb; i++) {
    int64_t const ii = bn_ixyz[i];
    SET_BIT(bn_mask[ii >> 3], ii % 8);
  }

  // create bn_mask_raw to double check
  auto bn_mask_raw = std::vector<bool>(size_t(Npts));

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

  // count Nbl
  int64_t Nbl = 0;
  for (int64_t i = 0; i < Nb; i++) {
    Nbl += static_cast<int64_t>(mat_bn[i] >= 0);
  }
  fmt::println("Nbl = {}", Nbl);
  auto mat_bnl  = std::vector<int8_t>(static_cast<size_t>(Nbl));
  auto bnl_ixyz = std::vector<int64_t>(static_cast<size_t>(Nbl));
  auto ssaf_bnl = std::vector<Real>(static_cast<size_t>(Nbl));
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
  delete[] ssaf_bn;

  fmt::println("separated non-rigid bn");

  // ABC ndoes
  int64_t const Nyf = (fcc_flag == 2) ? 2 * (Ny - 1) : Ny; // full Ny dim, taking into account FCC fold
  int64_t Nba       = 2 * (Nx * Nyf + Nx * Nz + Nyf * Nz) - 12 * (Nx + Nyf + Nz) + 56;
  if (fcc_flag > 0) {
    Nba /= 2;
  }

  auto bna_ixyz = std::vector<int64_t>(size_t(Nba));
  auto* Q_bna   = allocate_zeros<int8_t>(Nba);
  {
    int64_t ii = 0;
    for (int64_t ix = 1; ix < Nx - 1; ix++) {
      for (int64_t iy = 1; iy < Nyf - 1; iy++) {
        for (int64_t iz = 1; iz < Nz - 1; iz++) {

          if ((fcc_flag > 0) && (ix + iy + iz) % 2 == 1) {
            continue;
          }

          int8_t Q = 0;
          Q        = static_cast<int8_t>(Q + static_cast<int>((ix == 1) || (ix == Nx - 2)));
          Q        = static_cast<int8_t>(Q + static_cast<int>((iy == 1) || (iy == Nyf - 2)));
          Q        = static_cast<int8_t>(Q + static_cast<int>((iz == 1) || (iz == Nz - 2)));
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
      auto* bna_sort_keys = allocate_zeros<int64_t>(Nba);
      sort_keys(bna_ixyz.data(), bna_sort_keys, Nba);

      // now sort corresponding Q_bna array
      int8_t* Q_bna_unsorted = nullptr;
      auto* Q_bna_sorted     = allocate_zeros<int8_t>(Nba);
      // swap pointers
      Q_bna_unsorted = Q_bna;
      Q_bna          = Q_bna_sorted;

      for (int64_t cc = 0; cc < Nba; cc++) {
        Q_bna_sorted[cc] = Q_bna_unsorted[bna_sort_keys[cc]];
      }
      delete[] bna_sort_keys;
      delete[] Q_bna_unsorted;
      fmt::println("sorted ABC nodes for FCC/GPU");
    }
  }

  return Simulation3D<Real>{
      .bn_ixyz     = std::move(bn_ixyz),
      .bnl_ixyz    = std::move(bnl_ixyz),
      .bna_ixyz    = std::move(bna_ixyz),
      .Q_bna       = Q_bna,
      .in_ixyz     = std::move(in_ixyz),
      .out_ixyz    = std::move(out_ixyz),
      .out_reorder = std::move(out_reorder),
      .adj_bn      = std::move(adj_bn),
      .ssaf_bnl    = std::move(ssaf_bnl),
      .bn_mask     = std::move(bn_mask),
      .mat_bnl     = std::move(mat_bnl),
      .K_bn        = std::move(K_bn),
      .in_sigs     = std::move(in_sigs),
      .u_out       = std::unique_ptr<double[]>{allocate_zeros<double>(Nr * Nt)},
      .Ns          = Ns,
      .Nr          = Nr,
      .Nt          = Nt,
      .Npts        = Npts,
      .Nx          = Nx,
      .Ny          = Ny,
      .Nz          = Nz,
      .Nb          = Nb,
      .Nbl         = Nbl,
      .Nba         = Nba,
      .l           = l,
      .l2          = l2,
      .fcc_flag    = fcc_flag,
      .NN          = NN,
      .Nm          = Nm,
      .Mb          = std::move(Mb),
      .mat_quads   = std::move(mat_quads),
      .mat_beta    = std::move(mat_beta),
      .infac       = 1.0,
      .sl2         = sl2,
      .lo2         = lo2,
      .a2          = a2,
      .a1          = a1,
  };
}

template<typename Real>
void writeOutputs_impl(Simulation3D<Real> const& sim, std::filesystem::path const& simDir) {
  auto const* out_reorder = sim.out_reorder.data();
  auto Nt                 = static_cast<size_t>(sim.Nt);
  auto Nr                 = static_cast<size_t>(sim.Nr);
  auto u_out              = stdex::mdarray<double, stdex::dextents<size_t, 2>>(Nr, Nt);

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

// print last samples of simulation (for correctness checking..)
template<typename Real>
void printLastSample_impl(Simulation3D<Real> const& sim) {
  int64_t const Nt           = sim.Nt;
  int64_t const Nr           = sim.Nr;
  double* u_out              = sim.u_out.get();
  int64_t const* out_reorder = sim.out_reorder.data();
  // print last samples
  fmt::println("RAW OUTPUTS");
  for (int64_t nr = 0; nr < Nr; nr++) {
    fmt::println("receiver {}", nr);
    for (int64_t n = Nt - 5; n < Nt; n++) {
      printf("sample %ld: %.16e\n", long(n), u_out[out_reorder[nr] * Nt + n]);
    }
  }
}

} // namespace

[[nodiscard]] auto loadSimulation3D_float(std::filesystem::path const& simDir) -> Simulation3D<float> {
  return loadSimulation3D_impl<float>(simDir);
}

[[nodiscard]] auto loadSimulation3D_double(std::filesystem::path const& simDir) -> Simulation3D<double> {
  return loadSimulation3D_impl<double>(simDir);
}

void printLastSample(Simulation3D<float> const& sim) { printLastSample_impl(sim); }

void printLastSample(Simulation3D<double> const& sim) { printLastSample_impl(sim); }

void writeOutputs(Simulation3D<float> const& sim, std::filesystem::path const& simDir) {
  writeOutputs_impl(sim, simDir);
}

void writeOutputs(Simulation3D<double> const& sim, std::filesystem::path const& simDir) {
  writeOutputs_impl(sim, simDir);
}

} // namespace pffdtd
