// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#pragma once

#include <chrono>

namespace pffdtd {

using Seconds = std::chrono::duration<double>;

[[nodiscard]] inline auto getTime() -> std::chrono::steady_clock::time_point {
  return std::chrono::steady_clock::now();
}

[[nodiscard]] auto timeit(auto const& callback) -> std::chrono::nanoseconds {
  auto const start = getTime();
  callback();
  return getTime() - start;
}

} // namespace pffdtd
