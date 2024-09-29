// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch
#pragma once

#include "pffdtd/simulation_3d.hpp"

#if not defined(PFFDTD_HAS_METAL)
  #error "METAL must be enabled in CMake via PFFDTD_ENABLE_METAL"
#endif

namespace pffdtd {

struct EngineMETAL3D {
  auto operator()(Simulation3D const& sim) const -> void;
};

} // namespace pffdtd
