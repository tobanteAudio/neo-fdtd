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

#include "fdtd_common.hpp"
#include "fdtd_data.hpp"
#include "helper_funcs.hpp"

#ifndef USING_CUDA
  #error "forgot to define USING_CUDA"
#endif

#if USING_CUDA
  #include "gpu_engine.hpp"
#else
  #include "cpu_engine.hpp"
#endif

#include <fmt/format.h>
#include <fmt/os.h>

#include <chrono>

auto main(int /*argc*/, char** /*argv*/) -> int {
  auto const start = std::chrono::steady_clock::now();

  auto sim = Simulation3D{};
  loadSimulation3D(&sim);
  scaleInput(&sim);
  runSim(&sim);
  rescaleOutput(&sim);
  writeOutputs(&sim);
  printLastSample(&sim);
  freeSimulation3D(&sim);

  auto const stop = std::chrono::steady_clock::now();
  auto const sec  = std::chrono::duration<double>(stop - start);
  fmt::println("--Simulation time: {} s", sec.count());

  return 0;
}
