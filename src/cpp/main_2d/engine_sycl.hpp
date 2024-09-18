// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#pragma once

#include "pffdtd/mdspan.hpp"
#include "pffdtd/simulation_2d.hpp"

#include <cstddef>

namespace pffdtd {

struct EngineSYCL {
  [[nodiscard]] auto operator()(Simulation2D const& sim) const -> stdex::mdarray<double, stdex::dextents<size_t, 2>>;
};

} // namespace pffdtd
