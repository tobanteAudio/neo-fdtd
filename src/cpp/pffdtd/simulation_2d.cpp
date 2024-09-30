// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#include "simulation_2d.hpp"

#include "pffdtd/hdf.hpp"
#include "pffdtd/print.hpp"

namespace pffdtd {

auto loadSimulation2D(std::filesystem::path const& dir) -> Simulation2D {
  auto sim = HDF5Reader{dir / "sim.h5"};

  return Simulation2D{
      .dir = dir,

      .Nx = sim.read<int64_t>("Nx"),
      .Ny = sim.read<int64_t>("Ny"),
      .Nt = sim.read<int64_t>("Nt"),

      .in_mask     = sim.read<std::vector<uint8_t>>("in_mask"),
      .adj_bn      = sim.read<std::vector<int64_t>>("adj_bn"),
      .bn_ixy      = sim.read<std::vector<int64_t>>("bn_ixy"),
      .loss_factor = sim.read<double>("loss_factor"),

      .in_sigs = sim.read<std::vector<double>>("in_sigs"),
      .in_ixy  = sim.read<std::vector<int64_t>>("in_ixy"),
      .out_ixy = sim.read<std::vector<int64_t>>("out_ixy"),
  };
}

auto summary(Simulation2D const& sim) -> void {
  println("Nt = {}", sim.Nt);
  println("Nx = {}", sim.Nx);
  println("Ny = {}", sim.Ny);
  println("N = {}", sim.Nx * sim.Ny);
  println("in_mask = {}", sim.in_mask.size());
  println("bn_ixy = {}", sim.bn_ixy.size());
  println("adj_bn = {}", sim.adj_bn.size());
  println("in_sigs = {}", sim.in_sigs.size());
  println("in_ixy = {}", sim.in_ixy.size());
  println("out_ixy = {}", sim.out_ixy.size());
  println("loss_factor = {}", sim.loss_factor);
}

} // namespace pffdtd
