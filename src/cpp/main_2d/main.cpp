#include "engine.hpp"

#include "pffdtd/hdf.hpp"
#include "pffdtd/simulation_2d.hpp"

#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>

int main(int, char** argv) {
  auto filePath = std::filesystem::path{argv[1]};
  if (not std::filesystem::exists(filePath)) {
    throw std::runtime_error{"invalid file: " + filePath.string()};
  }

  auto const sim = pffdtd::loadSimulation2D(filePath);
  auto const out = pffdtd::run(sim);

  auto dir     = filePath.parent_path();
  auto outfile = dir / "out.h5";
  auto results = pffdtd::H5FWriter{outfile.string().c_str()};
  results.write("out", std::span{out}, sim.out_ixy.size(), sim.Nt);

  return EXIT_SUCCESS;
}
