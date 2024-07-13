#pragma once

#include <cstdint>

namespace pffdtd {

auto print_progress(
    uint32_t n,
    uint32_t Nt,
    uint64_t Npts,
    uint64_t Nb,
    double time_elapsed,
    double time_elapsed_sample,
    double time_elapsed_air,
    double time_elapsed_sample_air,
    double time_elapsed_bn,
    double time_elapsed_sample_bn,
    int num_workers
) -> void;

} // namespace pffdtd
