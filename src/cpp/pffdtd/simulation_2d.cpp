#include "simulation_2d.hpp"

#include "pffdtd/hdf.hpp"

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
  std::printf("Nt: %ld\n", static_cast<long>(sim.Nt));
  std::printf("Nx: %ld\n", static_cast<long>(sim.Nx));
  std::printf("Ny: %ld\n", static_cast<long>(sim.Ny));
  std::printf("N: %ld\n", static_cast<long>(sim.Nx * sim.Ny));
  std::printf("inx: %ld\n", static_cast<long>(sim.inx));
  std::printf("iny: %ld\n", static_cast<long>(sim.iny));
  std::printf("in_mask: %ld\n", static_cast<long>(sim.in_mask.size()));
  std::printf("bn_ixy: %ld\n", static_cast<long>(sim.bn_ixy.size()));
  std::printf("adj_bn: %ld\n", static_cast<long>(sim.adj_bn.size()));
  std::printf("out_ixy: %ld\n", static_cast<long>(sim.out_ixy.size()));
  std::printf("src_sig: %ld\n", static_cast<long>(sim.src_sig.size()));
  std::printf("loss_factor: %f\n", sim.loss_factor);
}

} // namespace pffdtd
