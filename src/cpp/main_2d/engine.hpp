#pragma once

#include "pffdtd/simulation_2d.hpp"

#include <vector>

namespace pffdtd {

[[nodiscard]] auto run(Simulation2D const& sim) -> std::vector<double>;

} // namespace pffdtd
