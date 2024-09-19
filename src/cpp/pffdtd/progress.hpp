// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton

#pragma once

#include <cstdint>

namespace pffdtd {

struct ProgressReport {
  int64_t n;
  int64_t Nt;
  int64_t Npts;
  int64_t Nb;
  double elapsed;
  double elapsedSample;
  double elapsedAir;
  double elapsedSampleAir;
  double elapsedBoundary;
  double elapsedSampleBoundary;
  int numWorkers;
};

auto print(ProgressReport const progress) -> void;

} // namespace pffdtd
