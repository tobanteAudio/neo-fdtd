// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#include "sycl.hpp"

#include <fmt/format.h>

namespace pffdtd {

auto elapsedTime(sycl::event const& startEvent, sycl::event const& endEvent) -> std::chrono::nanoseconds {
  auto const start = startEvent.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto const end   = endEvent.get_profiling_info<sycl::info::event_profiling::command_end>();
  return std::chrono::nanoseconds{end - start};
}

auto elapsedTime(sycl::event const& event) -> std::chrono::nanoseconds { return elapsedTime(event, event); }

auto toString(sycl::info::device_type type) -> std::string {
  switch (type) {
    case sycl::info::device_type::cpu: return "CPU";
    case sycl::info::device_type::gpu: return "GPU";
    case sycl::info::device_type::accelerator: return "Accelerator";
    case sycl::info::device_type::custom: return "Custom";
    case sycl::info::device_type::automatic: return "Automatic";
    case sycl::info::device_type::host: return "Host";
    case sycl::info::device_type::all: return "All";
  }

  return "Unkown";
}

auto summary(sycl::device const& dev) -> void {
  auto const vendor           = dev.get_info<sycl::info::device::vendor>();
  auto const name             = dev.get_info<sycl::info::device::name>();
  auto const type             = dev.get_info<sycl::info::device::device_type>();
  auto const driverVersion    = dev.get_info<sycl::info::device::driver_version>();
  auto const clockFrequency   = dev.get_info<sycl::info::device::max_clock_frequency>();
  auto const computeUnits     = dev.get_info<sycl::info::device::max_compute_units>();
  auto const workGroupSize    = dev.get_info<sycl::info::device::max_work_group_size>();
  auto const workItemSize     = dev.get_info<sycl::info::device::max_work_item_sizes<3>>();
  auto const allocSize        = dev.get_info<sycl::info::device::max_mem_alloc_size>();
  auto const globalMemorySize = dev.get_info<sycl::info::device::global_mem_size>();
  auto const localMemorySize  = dev.get_info<sycl::info::device::local_mem_size>();
  auto const hasProfiling     = dev.get_info<sycl::info::device::queue_profiling>();
  auto const extensions       = dev.get_info<sycl::info::device::extensions>();

  fmt::println("----------------------------------------");
  fmt::println("Vendor: {}", vendor);
  fmt::println("Name: {}", name);
  fmt::println("Type: {}", toString(type));
  fmt::println("Driver: {}", driverVersion);
  fmt::println("Has queue profiling: {}", hasProfiling ? "true" : "false");
  fmt::println("Global memory: {} GB", globalMemorySize / 1'000'000'000);
  fmt::println("Local memory: {} KB", localMemorySize / 1000);
  fmt::println("Max alloc size: {} GB", allocSize / 1'000'000'000);
  fmt::println("Max clock frequency: {} Hz", clockFrequency);
  fmt::println("Max compute units: {}", computeUnits);
  fmt::println("Max work group size: {}", workGroupSize);
  fmt::println("Max work item size: [{},{},{}]", workItemSize[0], workItemSize[1], workItemSize[2]);
  fmt::println("Extension:");
  for (auto const& ext : extensions) {
    fmt::println("  - '{}'", ext);
  }
  fmt::println("");
}

} // namespace pffdtd
