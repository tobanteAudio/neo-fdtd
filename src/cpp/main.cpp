// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#include "pffdtd/engine_cpu_2d.hpp"
#include "pffdtd/engine_cpu_3d.hpp"
#include "pffdtd/exception.hpp"
#include "pffdtd/hdf.hpp"
#include "pffdtd/simulation_2d.hpp"
#include "pffdtd/simulation_3d.hpp"
#include "pffdtd/time.hpp"
#include "pffdtd/utility.hpp"

#if defined(PFFDTD_HAS_CUDA)
  #include "pffdtd/engine_cuda_3d.hpp"
#endif

#if defined(PFFDTD_HAS_SYCL)
  #include "pffdtd/engine_sycl_2d.hpp"
  #include "pffdtd/engine_sycl_3d.hpp"
#endif

#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include <filesystem>
#include <string>

namespace {

[[nodiscard]] auto getEngines() {
  using pffdtd::Simulation2D;
  using Callback    = std::function<stdex::mdarray<double, stdex::dextents<size_t, 2>>(Simulation2D const&)>;
  auto engines      = std::map<std::string, Callback>{};
  engines["native"] = pffdtd::EngineCPU2D{};
#if defined(PFFDTD_HAS_SYCL)
  engines["sycl"] = pffdtd::EngineSYCL2D{};
#endif
  return engines;
}

[[nodiscard]] auto toLower(std::string str) -> std::string {
  std::ranges::transform(str, str.begin(), [](auto ch) { return static_cast<char>(std::tolower(ch)); });
  return str;
}

struct Arguments {
  struct Sim2D {
    std::string engine{"native"};
    std::string simDir;
    std::string out{"out.h5"};
  };

  struct Sim3D {
    std::string simDir;
    std::string precision{"64"};
  };

  Sim2D sim2d;
  Sim3D sim3d;
};

template<typename Real>
auto run3D(Arguments::Sim3D const& args) {
  using namespace pffdtd;

  fmt::println("Running: {} with precision {}", args.simDir, args.precision);
  auto const simDir = std::filesystem::path{args.simDir};
  auto const start  = getTime();

  auto sim = loadSimulation3D<Real>(simDir);
  scaleInput(sim);

#if defined(PFFDTD_HAS_CUDA)
  EngineCUDA3D{}(sim);
#elif defined(PFFDTD_HAS_SYCL)
  EngineSYCL3D{}(sim);
#else
  EngineCPU3D{}(sim);
#endif

  rescaleOutput(sim);
  writeOutputs(sim, simDir);
  printLastSample(sim);

  auto const stop = getTime();
  auto const sec  = Seconds(stop - start);
  fmt::println("--Simulation time: {} s", sec.count());
}

} // namespace

auto main(int argc, char** argv) -> int {
  auto app  = CLI::App{"pffdtd-engine"};
  auto args = Arguments{};

  auto* sim2d = app.add_subcommand("sim2d", "Run 2D simulation");
  sim2d->add_option("-e,--engine", args.sim2d.engine)->transform(toLower);
  sim2d->add_option("-s,--sim_dir", args.sim2d.simDir)->check(CLI::ExistingDirectory)->required();
  sim2d->add_option("-o,--out", args.sim2d.out);

  auto* sim3d = app.add_subcommand("sim3d", "Run 3D simulation");
  sim3d->add_option("-s,--sim_dir", args.sim3d.simDir)->check(CLI::ExistingDirectory)->required();
  sim3d->add_option("-p,--precision", args.sim3d.precision)->transform(toLower);

  CLI11_PARSE(app, argc, argv);

  if (*sim2d) {
    fmt::println("Using engine: {}", args.sim2d.engine);
    auto const engines = getEngines();
    auto const& engine = engines.at(args.sim2d.engine);

    auto const start  = pffdtd::getTime();
    auto const simDir = std::filesystem::path{args.sim2d.simDir};
    auto const sim    = pffdtd::loadSimulation2D(simDir);
    auto const out    = engine(sim);

    auto results = pffdtd::H5FWriter{simDir / args.sim2d.out};
    results.write("out", out);

    auto const stop = pffdtd::getTime();
    auto const sec  = pffdtd::Seconds(stop - start);
    fmt::println("Simulation time: {} s", sec.count());
  }

  if (*sim3d) {
    if (args.sim3d.precision == "32") {
      run3D<float>(args.sim3d);
    } else if (args.sim3d.precision == "64") {
      run3D<double>(args.sim3d);
    } else {
      pffdtd::raisef<std::invalid_argument>("invalid precision '{}'", args.sim3d.precision);
    }
  }

  return EXIT_SUCCESS;
}
