// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#pragma once

#if not defined(PFFDTD_HAS_METAL)
  #error "METAL must be enabled in CMake via PFFDTD_ENABLE_METAL"
#endif

#include "pffdtd/mdspan.hpp"
#include "pffdtd/simulation_2d.hpp"

namespace pffdtd {

struct EngineMETAL2D {
  [[nodiscard]] auto operator()(Simulation2D const& sim) const -> stdex::mdarray<double, stdex::dextents<size_t, 2>>;
};

} // namespace pffdtd
