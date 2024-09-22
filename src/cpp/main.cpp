// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#include "pffdtd/engine_2d_cpu.hpp"
#include "pffdtd/engine_3d_cpu.hpp"
#include "pffdtd/exception.hpp"
#include "pffdtd/hdf.hpp"
#include "pffdtd/simulation_2d.hpp"
#include "pffdtd/simulation_3d.hpp"
#include "pffdtd/time.hpp"
#include "pffdtd/utility.hpp"

#if defined(PFFDTD_HAS_CUDA)
  #include "pffdtd/engine_3d_cuda.hpp"
#endif

#if defined(PFFDTD_HAS_SYCL)
  #include "pffdtd/engine_2d_sycl.hpp"
  #include "pffdtd/engine_3d_sycl.hpp"
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
  engines["native"] = pffdtd::Engine2DCPU{};
#if defined(PFFDTD_HAS_SYCL)
  engines["sycl"] = pffdtd::Engine2DSYCL{};
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
  };

  Sim2D sim2d;
  Sim3D sim3d;
};
} // namespace

auto main(int argc, char** argv) -> int {
  auto app  = CLI::App{"pffdtd-2d"};
  auto args = Arguments{};

  auto* sim2d = app.add_subcommand("sim2d", "Run 2D simulation");
  sim2d->add_option("-e,--engine", args.sim2d.engine)->transform(toLower);
  sim2d->add_option("-s,--sim_dir", args.sim2d.simDir)->check(CLI::ExistingDirectory)->required();
  sim2d->add_option("-o,--out", args.sim2d.out);

  auto* sim3d = app.add_subcommand("sim3d", "Run 3D simulation");
  sim3d->add_option("-s,--sim_dir", args.sim3d.simDir)->check(CLI::ExistingDirectory)->required();

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
    fmt::println("Running: {} ...", args.sim3d.simDir);
    auto const simDir = std::filesystem::path{args.sim3d.simDir};
    auto const start  = pffdtd::getTime();

    auto sim = pffdtd::loadSimulation3D(simDir);
    pffdtd::scaleInput(sim);

#if defined(PFFDTD_HAS_CUDA)
    pffdtd::Engine3DCUDA{}(sim);
#elif defined(PFFDTD_HAS_SYCL)
    pffdtd::Engine3DSYCL{}(sim);
#else
    pffdtd::Engine3DCPU{}(sim);
#endif

    pffdtd::rescaleOutput(sim);
    pffdtd::writeOutputs(sim, simDir);
    pffdtd::printLastSample(sim);
    pffdtd::freeSimulation3D(sim);

    auto const stop = pffdtd::getTime();
    auto const sec  = pffdtd::Seconds(stop - start);
    fmt::println("--Simulation time: {} s", sec.count());
  }

  return EXIT_SUCCESS;
}
