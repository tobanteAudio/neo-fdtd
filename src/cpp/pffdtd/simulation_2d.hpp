#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace pffdtd {

struct Simulation2D {
  int64_t Nx; // Number of x sample points
  int64_t Ny; // Number of y sample points
  int64_t Nt; // Number of time steps

  std::vector<uint8_t> in_mask; // Mask for interior points
  std::vector<int64_t> adj_bn;  // Adjacency points
  std::vector<int64_t> bn_ixy;  // Boundary points
  double loss_factor;           // Boundary loss

  int64_t inx;                 // Source position x
  int64_t iny;                 // Source position y
  std::vector<double> src_sig; // Source signal

  std::vector<int64_t> out_ixy; // Receiver points
};

[[nodiscard]] inline auto loadSimulation2D(std::filesystem::path const &path)
    -> Simulation2D {
  auto file = H5FReader{path.string().c_str()};
  return Simulation2D{
      .Nx = file.read<int64_t>("Nx"),
      .Ny = file.read<int64_t>("Ny"),
      .Nt = file.read<int64_t>("Nt"),

      .in_mask = file.read<std::vector<uint8_t>>("in_mask"),
      .adj_bn = file.read<std::vector<int64_t>>("adj_bn"),
      .bn_ixy = file.read<std::vector<int64_t>>("bn_ixy"),
      .loss_factor = file.read<double>("loss_factor"),

      .inx = file.read<int64_t>("inx"),
      .iny = file.read<int64_t>("iny"),
      .src_sig = file.read<std::vector<double>>("src_sig"),

      .out_ixy = file.read<std::vector<int64_t>>("out_ixy"),
  };
}

} // namespace pffdtd
