// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#include "simulation_2d.hpp"

#include "pffdtd/hdf.hpp"

#include <fmt/format.h>

namespace pffdtd {

auto loadSimulation2D(std::filesystem::path const& dir) -> Simulation2D {
  auto sim = HDF5Reader{dir / "sim.h5"};

  auto const Nx  = sim.read<int64_t>("Nx");
  auto const Ny  = sim.read<int64_t>("Ny");
  auto const inx = sim.read<int64_t>("inx");
  auto const iny = sim.read<int64_t>("iny");

  auto in_ixy = std::vector<int64_t>{};
  in_ixy.push_back(inx * Ny + iny);

  return Simulation2D{
      .dir = dir,

      .Nx = Nx,
      .Ny = Ny,
      .Nt = sim.read<int64_t>("Nt"),

      .in_mask     = sim.read<std::vector<uint8_t>>("in_mask"),
      .adj_bn      = sim.read<std::vector<int64_t>>("adj_bn"),
      .bn_ixy      = sim.read<std::vector<int64_t>>("bn_ixy"),
      .loss_factor = sim.read<double>("loss_factor"),

      .in_sigs = sim.read<std::vector<double>>("in_sigs"),
      .in_ixy  = std::move(in_ixy),
      .out_ixy = sim.read<std::vector<int64_t>>("out_ixy"),
  };
}

auto summary(Simulation2D const& sim) -> void {
  fmt::println("Nt: {}", sim.Nt);
  fmt::println("Nx: {}", sim.Nx);
  fmt::println("Ny: {}", sim.Ny);
  fmt::println("N: {}", sim.Nx * sim.Ny);
  fmt::println("in_mask: {}", sim.in_mask.size());
  fmt::println("bn_ixy: {}", sim.bn_ixy.size());
  fmt::println("adj_bn: {}", sim.adj_bn.size());
  fmt::println("in_sigs: {}", sim.in_sigs.size());
  fmt::println("in_ixy: {}", sim.in_ixy.size());
  fmt::println("out_ixy: {}", sim.out_ixy.size());
  fmt::println("loss_factor: {}", sim.loss_factor);
}

} // namespace pffdtd
