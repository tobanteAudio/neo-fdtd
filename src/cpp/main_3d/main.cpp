///////////////////////////////////////////////////////////////////////////////
// This file is a part of PFFDTD.
//
// PFFTD is released under the MIT License.
// For details see the LICENSE file.
//
// Copyright 2021 Brian Hamilton.
//
// File name: fdtd_main.c
//
// Description: Main entry point of the CPU/GPU PFFDTD engines.
//
///////////////////////////////////////////////////////////////////////////////

#include "pffdtd/simulation_3d.hpp"
#include "pffdtd/utility.hpp"

#ifndef USING_CUDA
  #error "forgot to define USING_CUDA"
#endif

#if USING_CUDA
  #include "engine_cuda.hpp"
#else
  #include "engine_cpu.hpp"
#endif

#include <fmt/format.h>

#include <chrono>

auto main(int argc, char** argv) -> int {
  if (argc != 2) {
    fmt::println(stderr, "USAGE: pffdtd_3d path/to/sim");
    return EXIT_FAILURE;
  }

  auto const simDir = std::filesystem::path{argv[1]};
  auto const start  = std::chrono::steady_clock::now();

  auto sim = pffdtd::loadSimulation3D(simDir);
  pffdtd::scaleInput(sim);
  pffdtd::run(sim);
  pffdtd::rescaleOutput(sim);
  pffdtd::writeOutputs(sim, simDir);
  pffdtd::printLastSample(sim);
  pffdtd::freeSimulation3D(sim);

  auto const stop = std::chrono::steady_clock::now();
  auto const sec  = std::chrono::duration<double>(stop - start);
  fmt::println("--Simulation time: {} s", sec.count());

  return EXIT_SUCCESS;
}
