#include "engine_native.hpp"

#if defined(PFFDTD_HAS_SYCL)
  #include "engine_sycl.hpp"
#endif

#include "pffdtd/hdf.hpp"
#include "pffdtd/simulation_2d.hpp"

#include <CLI/CLI.hpp>

#include <fmt/format.h>

#include <oneapi/tbb/global_control.h>

#include <chrono>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>

struct Arguments {
  std::string simDir{};
  std::string out{"out.h5"};
  int jobs{-1};
  bool video{false};
};

int main(int argc, char** argv) {
  auto app  = CLI::App{"pffdtd-2d"};
  auto args = Arguments{};
  app.add_option("-s,--sim_dir", args.simDir, "Folder path");
  app.add_option("-o,--out", args.out, "Filename");
  app.add_option("-j,--jobs", args.jobs, "Num threads to use");
  app.add_flag("-v,--video", args.video, "Export video");
  CLI11_PARSE(app, argc, argv);

  if (args.jobs > 0) {
    oneapi::tbb::global_control global_control = oneapi::tbb::global_control(
        oneapi::tbb::global_control::max_allowed_parallelism,
        args.jobs
    );
  }

  auto const start = std::chrono::steady_clock::now();

  auto simDir = std::filesystem::path{args.simDir};
  if (not std::filesystem::exists(simDir)) {
    throw std::runtime_error{"invalid file: " + simDir.string()};
  }

  auto const sim    = pffdtd::loadSimulation2D(simDir, args.video);
  auto const engine = pffdtd::EngineNative{};
  auto const out    = engine(sim);

  auto outfile = simDir / args.out;
  auto results = pffdtd::H5FWriter{outfile.string().c_str()};
  results.write("out", out);

  auto const stop = std::chrono::steady_clock::now();
  auto const sec  = std::chrono::duration<double>(stop - start);
  fmt::println("Simulation time: {} s", sec.count());

  return EXIT_SUCCESS;
}
