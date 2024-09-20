// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#include "pffdtd/engine_2d_native.hpp"
#include "pffdtd/exception.hpp"
#include "pffdtd/hdf.hpp"
#include "pffdtd/simulation_2d.hpp"
#include "pffdtd/time.hpp"

#if defined(PFFDTD_HAS_SYCL)
  #include "pffdtd/engine_2d_sycl.hpp"
#endif

#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace {

[[nodiscard]] auto getEngines() {
  using pffdtd::Simulation2D;
  using Callback    = std::function<stdex::mdarray<double, stdex::dextents<size_t, 2>>(Simulation2D const&)>;
  auto engines      = std::map<std::string, Callback>{};
  engines["native"] = pffdtd::EngineNative{};
#if defined(PFFDTD_HAS_SYCL)
  engines["sycl"] = pffdtd::EngineSYCL{};
#endif
  return engines;
}

[[nodiscard]] auto toLower(std::string str) -> std::string {
  std::ranges::transform(str, str.begin(), [](auto ch) { return static_cast<char>(std::tolower(ch)); });
  return str;
}

struct Arguments {
  std::string engine{"native"};
  std::string simDir;
  std::string out{"out.h5"};
};
} // namespace

auto main(int argc, char** argv) -> int {
  auto app  = CLI::App{"pffdtd-2d"};
  auto args = Arguments{};
  app.add_option("-e,--engine", args.engine)->transform(toLower);
  app.add_option("-s,--sim_dir", args.simDir)->check(CLI::ExistingDirectory);
  app.add_option("-o,--out", args.out);
  CLI11_PARSE(app, argc, argv);

  fmt::println("Using engine: {}", args.engine);
  auto const engines = getEngines();
  auto const& engine = engines.at(args.engine);

  auto const start  = pffdtd::getTime();
  auto const simDir = std::filesystem::path{args.simDir};
  auto const sim    = pffdtd::loadSimulation2D(simDir);
  auto const out    = engine(sim);

  auto results = pffdtd::H5FWriter{simDir / args.out};
  results.write("out", out);

  auto const stop = pffdtd::getTime();
  auto const sec  = pffdtd::Seconds(stop - start);
  fmt::println("Simulation time: {} s", sec.count());

  return EXIT_SUCCESS;
}
