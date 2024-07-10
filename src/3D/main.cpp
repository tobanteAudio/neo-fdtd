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

#include <stdio.h>
#include <stdint.h> //uint types
#include <assert.h> //for assert
#include <time.h>   //date and time

#include "helper_funcs.hpp"
#include "fdtd_common.hpp"
#include "fdtd_data.hpp"

#ifndef USING_CUDA // this is defined in Makefile
#error "forgot to define USING_CUDA"
#endif

#if USING_CUDA
#include "gpu_engine.hpp"
#else
#include "cpu_engine.hpp"
#endif

int main(int argc, char **argv)
{
   (void)argc;
   (void)argv;

   // print date & time to start
   time_t t0;
   time(&t0);
   printf("--Date and time: %s", ctime(&t0)); // prints new line

   // double-check on 64-bit system
   assert(sizeof(size_t) == sizeof(int64_t));

    auto sim = Simulation3D{};

   loadSimulation3D(&sim);
   scaleInput(&sim);
   runSim(&sim);
   rescaleOutput(&sim);
   writeOutputs(&sim);
   printLastSample(&sim);
   freeSimulation3D(&sim);

   // print date & time to end
   time(&t0);
   printf("--Date and time: %s", ctime(&t0));
   return 0;
}
