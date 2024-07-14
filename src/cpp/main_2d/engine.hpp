#pragma once

#include "pffdtd/mdspan.hpp"
#include "pffdtd/simulation_2d.hpp"

#include <cstddef>

namespace pffdtd {

[[nodiscard]] auto run(Simulation2D const& sim)
    -> stdex::mdarray<double, stdex::dextents<size_t, 2>>;

} // namespace pffdtd
