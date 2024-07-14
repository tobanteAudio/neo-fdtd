#include "engine_native.hpp"
#include "engine_sycl.hpp"

#include "pffdtd/hdf.hpp"
#include "pffdtd/simulation_2d.hpp"

#include <CLI/CLI.hpp>

#include <fmt/format.h>

#include <chrono>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>

struct Arguments {
  std::string simDir{};
  std::string out{"out.h5"};
  bool video{false};
};

int main(int argc, char** argv) {
  auto app  = CLI::App{"pffdtd-2d"};
  auto args = Arguments{};
  app.add_option("-s,--sim_dir", args.simDir, "Folder path");
  app.add_option("-o,--out", args.out, "Filename");
  app.add_flag("-v,--video", args.video, "Export video");
  CLI11_PARSE(app, argc, argv);

  auto const start = std::chrono::steady_clock::now();

  auto filePath = std::filesystem::path{args.simDir};
  if (not std::filesystem::exists(filePath)) {
    throw std::runtime_error{"invalid file: " + filePath.string()};
  }

  auto const sim    = pffdtd::loadSimulation2D(filePath, args.video);
  auto const engine = pffdtd::EngineSYCL{};
  auto const out    = engine(sim);

  auto dir     = filePath.parent_path();
  auto outfile = dir / args.out;
  auto results = pffdtd::H5FWriter{outfile.string().c_str()};
  results.write("out", out);

  auto const stop = std::chrono::steady_clock::now();
  auto const sec  = std::chrono::duration<double>(stop - start);
  fmt::println("Simulation time: {} s", sec.count());

  return EXIT_SUCCESS;
}
