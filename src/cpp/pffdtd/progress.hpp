// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton

#pragma once

#include <chrono>
#include <cstdint>

namespace pffdtd {

struct ProgressReport {
  int64_t n;
  int64_t Nt;
  int64_t Npts;
  int64_t Nb;
  std::chrono::nanoseconds elapsed;
  std::chrono::nanoseconds elapsedSample;
  std::chrono::nanoseconds elapsedAir;
  std::chrono::nanoseconds elapsedSampleAir;
  std::chrono::nanoseconds elapsedBoundary;
  std::chrono::nanoseconds elapsedSampleBoundary;
  int numWorkers;
};

auto print(ProgressReport const& progress) -> void;

} // namespace pffdtd
