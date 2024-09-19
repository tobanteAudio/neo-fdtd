// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton
// Main entry point of the CPU/GPU PFFDTD engines.

#include "pffdtd/simulation_3d.hpp"
#include "pffdtd/time.hpp"
#include "pffdtd/utility.hpp"

#ifndef PFFDTD_HAS_CUDA
  #error "forgot to define PFFDTD_HAS_CUDA"
#endif

#if PFFDTD_HAS_CUDA
  #include "pffdtd/engine_cuda.hpp"
#else
  #include "pffdtd/engine_openmp.hpp"
#endif

#include <fmt/format.h>

#include <chrono>

auto main(int argc, char** argv) -> int {
  if (argc != 2) {
    fmt::println(stderr, "Usage: {} path/to/sim", argv[0]);
    return EXIT_FAILURE;
  }

  auto const simDir = std::filesystem::path{argv[1]};
  auto const start  = pffdtd::getTime();

  auto sim = pffdtd::loadSimulation3D(simDir);
  pffdtd::scaleInput(sim);
  pffdtd::run(sim);
  pffdtd::rescaleOutput(sim);
  pffdtd::writeOutputs(sim, simDir);
  pffdtd::printLastSample(sim);
  pffdtd::freeSimulation3D(sim);

  auto const stop = pffdtd::getTime();
  auto const sec  = pffdtd::Seconds(stop - start);
  fmt::println("--Simulation time: {} s", sec.count());

  return EXIT_SUCCESS;
}
