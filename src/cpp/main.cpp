// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#include "pffdtd/double.hpp"
#include "pffdtd/engine_cpu_2d.hpp"
#include "pffdtd/engine_cpu_3d.hpp"
#include "pffdtd/exception.hpp"
#include "pffdtd/hdf.hpp"
#include "pffdtd/precision.hpp"
#include "pffdtd/simulation_2d.hpp"
#include "pffdtd/simulation_3d.hpp"
#include "pffdtd/time.hpp"
#include "pffdtd/utility.hpp"

#if defined(PFFDTD_HAS_CUDA)
  #include "pffdtd/engine_cuda_3d.hpp"
#endif

#if defined(PFFDTD_HAS_METAL)
  #include "pffdtd/engine_metal_2d.hpp"
  #include "pffdtd/engine_metal_3d.hpp"
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

[[nodiscard]] auto precisionOptions() {
  return std::map<std::string, pffdtd::Precision>{
      {"32",  pffdtd::Precision::Float},
      {"64", pffdtd::Precision::Double},
  };
}

[[nodiscard]] auto toString(pffdtd::Precision precision) -> std::string {
  for (auto const& p : precisionOptions()) {
    if (p.second == precision) {
      return p.first;
    }
  }
  return "";
}

[[nodiscard]] auto getEngines2D() {
  using namespace pffdtd;
  using Callback = std::function<stdex::mdarray<double, stdex::dextents<size_t, 2>>(Simulation2D const&, Precision)>;
  auto engines   = std::map<std::string, Callback>{};
  engines["cpu"] = EngineCPU2D{};
#if defined(PFFDTD_HAS_METAL)
  engines["metal"] = EngineMETAL2D{};
#endif
#if defined(PFFDTD_HAS_SYCL)
  engines["sycl"] = EngineSYCL2D{};
#endif
  return engines;
}

template<typename Real>
[[nodiscard]] auto getEngines3D() {
  using namespace pffdtd;
  auto engines   = std::map<std::string, std::function<void(Simulation3D<Real> const&)>>{};
  engines["cpu"] = EngineCPU3D{};
#if defined(PFFDTD_HAS_CUDA)
  engines["cuda"] = EngineCUDA3D{};
#endif
#if defined(PFFDTD_HAS_METAL)
  if constexpr (std::same_as<Real, float>) {
    engines["metal"] = EngineMETAL3D{};
  }
#endif
#if defined(PFFDTD_HAS_SYCL)
  engines["sycl"] = EngineSYCL3D{};
#endif
  return engines;
}

[[nodiscard]] auto toLower(std::string str) -> std::string {
  std::ranges::transform(str, str.begin(), [](auto ch) { return static_cast<char>(std::tolower(ch)); });
  return str;
}

struct Arguments {
  struct Sim2D {
    std::string simDir;
    std::string engine{"cpu"};
    std::string out{"out.h5"};
    pffdtd::Precision precision{pffdtd::Precision::Double};
  };

  struct Sim3D {
    std::string simDir;
    std::string engine{"cpu"};
    pffdtd::Precision precision{pffdtd::Precision::Double};
  };

  Sim2D sim2d;
  Sim3D sim3d;
};

template<typename Real>
auto run3D(Arguments::Sim3D const& args) {
  using namespace pffdtd;
  fmt::println("Running: {} on {} with precision {}", args.simDir, args.engine, toString(args.precision));

  auto const engines = getEngines3D<Real>();
  auto const& engine = engines.at(args.engine);

  auto const simDir = std::filesystem::path{args.simDir};
  auto const start  = getTime();

  auto sim = loadSimulation3D<Real>(simDir);
  scaleInput(sim);
  engine(sim);
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
  sim2d->add_option("-s,--sim_dir", args.sim2d.simDir)->required()->check(CLI::ExistingDirectory);
  sim2d->add_option("-e,--engine", args.sim2d.engine)->transform(toLower);
  sim2d->add_option("-o,--out", args.sim2d.out);
  sim2d->add_option("-p,--precision", args.sim3d.precision)
      ->required()
      ->transform(CLI::CheckedTransformer(precisionOptions(), CLI::ignore_case));

  auto* sim3d = app.add_subcommand("sim3d", "Run 3D simulation");
  sim3d->add_option("-s,--sim_dir", args.sim3d.simDir)->required()->check(CLI::ExistingDirectory);
  sim3d->add_option("-e,--engine", args.sim3d.engine)->required()->transform(toLower);
  sim3d->add_option("-p,--precision", args.sim3d.precision)
      ->required()
      ->transform(CLI::CheckedTransformer(precisionOptions(), CLI::ignore_case));

  auto* test = app.add_subcommand("test", "Run unit tests");
  CLI11_PARSE(app, argc, argv);

  if (*sim2d) {
    fmt::println("Using engine: {} with precision {}", args.sim2d.engine, toString(args.sim2d.precision));
    auto const engines = getEngines2D();
    auto const& engine = engines.at(args.sim2d.engine);

    auto const start  = pffdtd::getTime();
    auto const simDir = std::filesystem::path{args.sim2d.simDir};
    auto const sim    = pffdtd::loadSimulation2D(simDir);
    auto const out    = engine(sim, args.sim2d.precision);

    auto results = pffdtd::HDF5Writer{simDir / args.sim2d.out};
    results.write("out", out);

    auto const stop = pffdtd::getTime();
    auto const sec  = pffdtd::Seconds(stop - start);
    fmt::println("Simulation time: {} s", sec.count());
  }

  if (*sim3d) {
    if (args.sim3d.precision == pffdtd::Precision::Float) {
      run3D<float>(args.sim3d);
    } else if (args.sim3d.precision == pffdtd::Precision::Double) {
      run3D<double>(args.sim3d);
    } else {
      pffdtd::raisef<std::invalid_argument>("invalid precision '{}'", toString(args.sim3d.precision));
    }
  }

  if (*test) {
    // NOLINTBEGIN
    using pffdtd::Double;
    PFFDTD_ASSERT(static_cast<float>(Double{42.0F} + Double{2.0F}) == 44.0F);
    PFFDTD_ASSERT(static_cast<float>(Double{42.0F} - Double{2.0F}) == 40.0F);
    PFFDTD_ASSERT(static_cast<float>(Double{42.0F} * Double{2.0F}) == 84.0F);
    PFFDTD_ASSERT(static_cast<float>(Double{42.0F} / Double{2.0F}) == 21.0F);

    PFFDTD_ASSERT(static_cast<double>(Double{42.0} + Double{2.0}) == 44.0);
    PFFDTD_ASSERT(static_cast<double>(Double{42.0} - Double{2.0}) == 40.0);
    PFFDTD_ASSERT(static_cast<double>(Double{42.0} * Double{2.0}) == 84.0);
    PFFDTD_ASSERT(static_cast<double>(Double{42.0} / Double{2.0}) == 21.0);
    // NOLINTEND
  }

  return EXIT_SUCCESS;
}
