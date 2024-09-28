// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#pragma once

#include "pffdtd/mdspan.hpp"
#include "pffdtd/precision.hpp"
#include "pffdtd/simulation_2d.hpp"

#include <cstddef>

namespace pffdtd {

struct EngineCPU2D {
  [[nodiscard]] auto operator()(Simulation2D const& sim, Precision precision) const
      -> stdex::mdarray<double, stdex::dextents<size_t, 2>>;
};

} // namespace pffdtd
