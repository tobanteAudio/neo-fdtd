#include "simulation_2d.hpp"

#include "pffdtd/hdf.hpp"

#include <fmt/format.h>
#include <fmt/os.h>

namespace pffdtd {

auto loadSimulation2D(std::filesystem::path const& path) -> Simulation2D {
  auto file = H5FReader{path.string().c_str()};
  return Simulation2D{
      .Nx = file.read<int64_t>("Nx"),
      .Ny = file.read<int64_t>("Ny"),
      .Nt = file.read<int64_t>("Nt"),

      .in_mask     = file.read<std::vector<uint8_t>>("in_mask"),
      .adj_bn      = file.read<std::vector<int64_t>>("adj_bn"),
      .bn_ixy      = file.read<std::vector<int64_t>>("bn_ixy"),
      .loss_factor = file.read<double>("loss_factor"),

      .inx     = file.read<int64_t>("inx"),
      .iny     = file.read<int64_t>("iny"),
      .src_sig = file.read<std::vector<double>>("src_sig"),

      .out_ixy = file.read<std::vector<int64_t>>("out_ixy"),
  };
}

auto summary(Simulation2D const& sim) -> void {
  fmt::println("Nt: {}", sim.Nt);
  fmt::println("Nx: {}", sim.Nx);
  fmt::println("Ny: {}", sim.Ny);
  fmt::println("N: {}", sim.Nx * sim.Ny);
  fmt::println("inx: {}", sim.inx);
  fmt::println("iny: {}", sim.iny);
  fmt::println("in_mask: {}", sim.in_mask.size());
  fmt::println("bn_ixy: {}", sim.bn_ixy.size());
  fmt::println("adj_bn: {}", sim.adj_bn.size());
  fmt::println("out_ixy: {}", sim.out_ixy.size());
  fmt::println("src_sig: {}", sim.src_sig.size());
  fmt::println("loss_factor: {}", sim.loss_factor);
}

} // namespace pffdtd
