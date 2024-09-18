// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton
// CPU-based implementation of FDTD engine, with OpenMP

#include "engine_cpu.hpp"

#include "pffdtd/progress.hpp"
#include "pffdtd/utility.hpp"

#include <fmt/format.h>

#include <omp.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>

namespace pffdtd {

namespace {

// function that does freq-dep RLC boundaries.  See 2016 ISMRA paper and
// accompanying webpage (slightly improved here)
template<typename Float>
auto process_bnl_pts_fd(
    Float* u0b,
    Float const* u2b,
    Float const* ssaf_bnl,
    int8_t const* mat_bnl,
    int64_t Nbl,
    int8_t const* Mb,
    Float lo2,
    Float* vh1,
    Float* gh1,
    MatQuad<Float> const* mat_quads,
    Float const* mat_beta
) -> double {
  auto const start = omp_get_wtime();
#pragma omp parallel for schedule(static)
  for (int64_t nb = 0; nb < Nbl; nb++) {
    Float _1        = 1.0;
    Float _2        = 2.0;
    int32_t const k = mat_bnl[nb];

    Float lo2Kbg = lo2 * ssaf_bnl[nb] * mat_beta[k];
    Float fac    = _2 * lo2 * ssaf_bnl[nb] / (_1 + lo2Kbg);

    Float u0bint = u0b[nb];
    Float u2bint = u2b[nb];

    u0bint = (u0bint + lo2Kbg * u2bint) / (_1 + lo2Kbg);

    Float vh1nb[MMb];
    for (int8_t m = 0; m < Mb[k]; m++) {
      int64_t const nbm        = nb * MMb + m;
      int32_t const mbk        = k * MMb + m;
      MatQuad<Float> const* tm = &(mat_quads[mbk]);
      vh1nb[m]                 = vh1[nbm];
      u0bint -= fac * (_2 * (tm->bDh) * vh1nb[m] - (tm->bFh) * gh1[nbm]);
    }

    Float du = u0bint - u2bint;

    for (int8_t m = 0; m < Mb[k]; m++) {
      int64_t const nbm        = nb * MMb + m;
      int32_t const mbk        = k * MMb + m;
      MatQuad<Float> const* tm = &(mat_quads[mbk]);
      Float vh0nbm             = (tm->b) * du + (tm->bd) * vh1nb[m] - _2 * (tm->bFh) * gh1[nbm];
      gh1[nbm] += (vh0nbm + vh1nb[m]) / _2;
      vh1[nbm] = vh0nbm;
    }

    u0b[nb] = u0bint;
  }
  return omp_get_wtime() - start;
}

} // namespace

auto run(Simulation3D& sd) -> double {
  // keep local ints, scalars
  int64_t const Ns   = sd.Ns;
  int64_t const Nr   = sd.Nr;
  int64_t const Nt   = sd.Nt;
  int64_t const Npts = sd.Npts;
  int64_t const Nx   = sd.Nx;
  int64_t const Ny   = sd.Ny;
  int64_t const Nz   = sd.Nz;
  int64_t const Nb   = sd.Nb;
  int64_t const Nbl  = sd.Nbl;
  int64_t const Nba  = sd.Nba;
  int8_t* Mb         = sd.Mb;

  // keep local copies of pointers (style choice)
  int64_t* bn_ixyz         = sd.bn_ixyz;
  int64_t* bnl_ixyz        = sd.bnl_ixyz;
  int64_t* bna_ixyz        = sd.bna_ixyz;
  int64_t* in_ixyz         = sd.in_ixyz;
  int64_t* out_ixyz        = sd.out_ixyz;
  uint16_t* adj_bn         = sd.adj_bn;
  uint8_t* bn_mask         = sd.bn_mask;
  int8_t* mat_bnl          = sd.mat_bnl;
  int8_t* Q_bna            = sd.Q_bna;
  double* in_sigs          = sd.in_sigs;
  double* u_out            = sd.u_out;
  int8_t const fcc_flag    = sd.fcc_flag;
  Real* ssaf_bnl           = sd.ssaf_bnl;
  Real* mat_beta           = sd.mat_beta;
  MatQuad<Real>* mat_quads = sd.mat_quads;

  // allocate memory
  auto u0_buf   = std::vector<Real>(static_cast<size_t>(Npts));
  auto u1_buf   = std::vector<Real>(static_cast<size_t>(Npts));
  auto u0b_buf  = std::vector<Real>(static_cast<size_t>(Nbl));
  auto u1b_buf  = std::vector<Real>(static_cast<size_t>(Nbl));
  auto u2b_buf  = std::vector<Real>(static_cast<size_t>(Nbl));
  auto u2ba_buf = std::vector<Real>(static_cast<size_t>(Nba));
  auto vh1_buf  = std::vector<Real>(static_cast<size_t>(Nbl * MMb));
  auto gh1_buf  = std::vector<Real>(static_cast<size_t>(Nbl * MMb));

  Real* u0   = u0_buf.data();
  Real* u1   = u1_buf.data();
  Real* u0b  = u0b_buf.data();
  Real* u1b  = u1b_buf.data();
  Real* u2b  = u2b_buf.data();
  Real* u2ba = u2ba_buf.data();
  Real* vh1  = vh1_buf.data();
  Real* gh1  = gh1_buf.data();

  // sim coefficients
  Real const lo2 = sd.lo2;
  Real const sl2 = sd.sl2;
  Real const l   = sd.l;
  Real const a1  = sd.a1;
  Real const a2  = sd.a2;

  // can control outside with OMP_NUM_THREADS env variable
  int const numWorkers = omp_get_max_threads();

  fmt::println("ENGINE: fcc_flag={}", fcc_flag);
  fmt::println("{}", (fcc_flag > 0) ? "fcc=true" : "fcc=false");

  // for timing
  double timeElapsed           = NAN;
  double timeElapsedAir        = 0.0;
  double timeElapsedBn         = 0.0;
  double timeElapsedSample     = NAN;
  double timeElapsedSample_air = 0.0;
  double timeElapsedSampleBn   = 0.0;
  double const startTime       = omp_get_wtime();

  int64_t const NzNy = Nz * Ny;
  for (int64_t n = 0; n < Nt; n++) {
    auto const sampleStartTime = omp_get_wtime();

// copy last state ABCs
#pragma omp parallel for
    for (int64_t nb = 0; nb < Nba; nb++) {
      u2ba[nb] = u0[bna_ixyz[nb]];
    }
    if (fcc_flag == 2) { // copy y-z face for FCC folded grid
#pragma omp parallel for
      for (int64_t ix = 0; ix < Nx; ix++) {
        for (int64_t iz = 0; iz < Nz; iz++) {
          u1[ix * NzNy + (Ny - 1) * Nz + iz] = u1[ix * NzNy + (Ny - 2) * Nz + iz];
        }
      }
    }

// halo flips for ABCs
#pragma omp parallel for
    for (int64_t ix = 0; ix < Nx; ix++) {
      for (int64_t iy = 0; iy < Ny; iy++) {
        u1[ix * NzNy + iy * Nz + 0]      = u1[ix * NzNy + iy * Nz + 2];
        u1[ix * NzNy + iy * Nz + Nz - 1] = u1[ix * NzNy + iy * Nz + Nz - 3];
      }
    }
#pragma omp parallel for
    for (int64_t ix = 0; ix < Nx; ix++) {
      for (int64_t iz = 0; iz < Nz; iz++) {
        u1[ix * NzNy + 0 * Nz + iz] = u1[ix * NzNy + 2 * Nz + iz];
      }
    }
    if (fcc_flag != 2) { // only this y-face if not FCC folded grid
#pragma omp parallel for
      for (int64_t ix = 0; ix < Nx; ix++) {
        for (int64_t iz = 0; iz < Nz; iz++) {
          u1[ix * NzNy + (Ny - 1) * Nz + iz] = u1[ix * NzNy + (Ny - 3) * Nz + iz];
        }
      }
    }
#pragma omp parallel for
    for (int64_t iy = 0; iy < Ny; iy++) {
      for (int64_t iz = 0; iz < Nz; iz++) {
        u1[0 * NzNy + iy * Nz + iz]        = u1[2 * NzNy + iy * Nz + iz];
        u1[(Nx - 1) * NzNy + iy * Nz + iz] = u1[(Nx - 3) * NzNy + iy * Nz + iz];
      }
    }

    // air update for schemes
    if (fcc_flag == 0) { // cartesian scheme
#pragma omp parallel for
      for (int64_t ix = 1; ix < Nx - 1; ix++) {
        for (int64_t iy = 1; iy < Ny - 1; iy++) {
          for (int64_t iz = 1; iz < Nz - 1; iz++) { // contiguous
            int64_t const ii = ix * NzNy + iy * Nz + iz;
            if ((GET_BIT(bn_mask[ii >> 3], ii % 8)) == 0) {
              Real partial = a1 * u1[ii] - u0[ii];
              partial += a2 * u1[ii + NzNy];
              partial += a2 * u1[ii - NzNy];
              partial += a2 * u1[ii + Nz];
              partial += a2 * u1[ii - Nz];
              partial += a2 * u1[ii + 1];
              partial += a2 * u1[ii - 1];
              u0[ii] = partial;
            }
          }
        }
      }
    } else if (fcc_flag > 0) {
#pragma omp parallel for
      for (int64_t ix = 1; ix < Nx - 1; ix++) {
        for (int64_t iy = 1; iy < Ny - 1; iy++) {
          // while loop iterates iterates over both types of FCC grids
          int64_t iz = (fcc_flag == 1) ? 2 - (ix + iy) % 2 : 1;
          while (iz < Nz - 1) {
            int64_t const ii = ix * NzNy + iy * Nz + iz;
            if ((GET_BIT(bn_mask[ii >> 3], ii % 8)) == 0) {
              Real partial = a1 * u1[ii] - u0[ii];
              partial += a2 * u1[ii + NzNy + Nz];
              partial += a2 * u1[ii - NzNy - Nz];
              partial += a2 * u1[ii + Nz + 1];
              partial += a2 * u1[ii - Nz - 1];
              partial += a2 * u1[ii + NzNy + 1];
              partial += a2 * u1[ii - NzNy - 1];
              partial += a2 * u1[ii + NzNy - Nz];
              partial += a2 * u1[ii - NzNy + Nz];
              partial += a2 * u1[ii + Nz - 1];
              partial += a2 * u1[ii - Nz + 1];
              partial += a2 * u1[ii + NzNy - 1];
              partial += a2 * u1[ii - NzNy + 1];
              u0[ii] = partial;
            }
            iz += ((fcc_flag == 1) ? 2 : 1);
          }
        }
      }
    }
    // ABC loss (2nd-order accurate first-order Engquist-Majda)
    for (int64_t nb = 0; nb < Nba; nb++) {
      Real const lQ    = l * Q_bna[nb];
      int64_t const ib = bna_ixyz[nb];
      u0[ib]           = (u0[ib] + lQ * u2ba[nb]) / (1.0 + lQ);
    }

    // rigid boundary nodes, using adj data
    timeElapsedSample_air = omp_get_wtime() - sampleStartTime;
    timeElapsedAir += timeElapsedSample_air;
    if (fcc_flag == 0) {
#pragma omp parallel for
      for (int64_t nb = 0; nb < Nb; nb++) {
        auto const ii   = bn_ixyz[nb];
        auto const adj  = adj_bn[nb];
        auto const Kint = std::popcount(adj);

        auto const _2 = static_cast<Real>(2.0);
        auto const K  = static_cast<Real>(Kint);
        auto const b2 = static_cast<Real>(a2);
        auto const b1 = (_2 - sl2 * K);

        auto partial = b1 * u1[ii] - u0[ii];
        partial += b2 * static_cast<Real>(GET_BIT(adj, 0)) * u1[ii + NzNy];
        partial += b2 * static_cast<Real>(GET_BIT(adj, 1)) * u1[ii - NzNy];
        partial += b2 * static_cast<Real>(GET_BIT(adj, 2)) * u1[ii + Nz];
        partial += b2 * static_cast<Real>(GET_BIT(adj, 3)) * u1[ii - Nz];
        partial += b2 * static_cast<Real>(GET_BIT(adj, 4)) * u1[ii + 1];
        partial += b2 * static_cast<Real>(GET_BIT(adj, 5)) * u1[ii - 1];
        u0[ii] = partial;
      }
    } else if (fcc_flag > 0) {
#pragma omp parallel for
      for (int64_t nb = 0; nb < Nb; nb++) {
        auto const ii   = bn_ixyz[nb];
        auto const adj  = adj_bn[nb];
        auto const Kint = std::popcount(adj);

        auto const _2 = static_cast<Real>(2.0);
        auto const K  = static_cast<Real>(Kint);
        auto const b2 = static_cast<Real>(a2);
        auto const b1 = (_2 - sl2 * K);

        auto partial = b1 * u1[ii] - u0[ii];
        partial += b2 * static_cast<Real>(GET_BIT(adj, 0)) * u1[ii + NzNy + Nz];
        partial += b2 * static_cast<Real>(GET_BIT(adj, 1)) * u1[ii - NzNy - Nz];
        partial += b2 * static_cast<Real>(GET_BIT(adj, 2)) * u1[ii + Nz + 1];
        partial += b2 * static_cast<Real>(GET_BIT(adj, 3)) * u1[ii - Nz - 1];
        partial += b2 * static_cast<Real>(GET_BIT(adj, 4)) * u1[ii + NzNy + 1];
        partial += b2 * static_cast<Real>(GET_BIT(adj, 5)) * u1[ii - NzNy - 1];
        partial += b2 * static_cast<Real>(GET_BIT(adj, 6)) * u1[ii + NzNy - Nz];
        partial += b2 * static_cast<Real>(GET_BIT(adj, 7)) * u1[ii - NzNy + Nz];
        partial += b2 * static_cast<Real>(GET_BIT(adj, 8)) * u1[ii + Nz - 1];
        partial += b2 * static_cast<Real>(GET_BIT(adj, 9)) * u1[ii - Nz + 1];
        partial += b2 * static_cast<Real>(GET_BIT(adj, 10)) * u1[ii + NzNy - 1];
        partial += b2 * static_cast<Real>(GET_BIT(adj, 11)) * u1[ii - NzNy + 1];
        u0[ii] = partial;
      }
    }

// read bn points (not strictly necessary, just mirrors CUDA implementation)
#pragma omp parallel for
    for (int64_t nb = 0; nb < Nbl; nb++) {
      u0b[nb] = u0[bnl_ixyz[nb]];
    }
    // process FD boundary nodes
    timeElapsedSampleBn = process_bnl_pts_fd(u0b, u2b, ssaf_bnl, mat_bnl, Nbl, Mb, lo2, vh1, gh1, mat_quads, mat_beta);
    timeElapsedBn += timeElapsedSampleBn;
// write back
#pragma omp parallel for
    for (int64_t nb = 0; nb < Nbl; nb++) {
      u0[bnl_ixyz[nb]] = u0b[nb];
    }

    // read output at current sample
    for (int64_t nr = 0; nr < Nr; nr++) {
      int64_t const ii   = out_ixyz[nr];
      u_out[nr * Nt + n] = static_cast<double>(u1[ii]);
    }

    // add current sample to next (as per update)
    for (int64_t ns = 0; ns < Ns; ns++) {
      int64_t const ii = in_ixyz[ns];
      u0[ii] += static_cast<Real>(in_sigs[ns * Nt + n]);
    }

    // swap pointers
    std::swap(u0, u1);

    // using extra state here for simplicity
    auto* tmp = u2b;
    u2b       = u1b;
    u1b       = u0b;
    u0b       = tmp;

    auto const now    = omp_get_wtime();
    timeElapsed       = now - startTime;
    timeElapsedSample = now - sampleStartTime;

    print_progress(
        n,
        Nt,
        Npts,
        Nb,
        timeElapsed,
        timeElapsedSample,
        timeElapsedAir,
        timeElapsedSample_air,
        timeElapsedBn,
        timeElapsedSampleBn,
        numWorkers
    );
  }
  fmt::println("");

  // timing
  auto const endTime = omp_get_wtime();
  timeElapsed        = endTime - startTime;

  /*------------------------
   * RETURN
  ------------------------*/
  fmt::println("Air update: {:.6}s, {:.2} Mvox/s", timeElapsedAir, Npts * Nt / 1e6 / timeElapsedAir);
  fmt::println("Boundary loop: {:.6}s, {:.2} Mvox/s", timeElapsedBn, Nb * Nt / 1e6 / timeElapsedBn);
  fmt::println("Combined (total): {:.6}s, {:.2} Mvox/s", timeElapsed, Npts * Nt / 1e6 / timeElapsed);

  return timeElapsed;
}

} // namespace pffdtd
