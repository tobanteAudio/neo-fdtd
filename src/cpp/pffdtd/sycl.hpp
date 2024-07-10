#pragma once

#include <sycl/sycl.hpp>

#include <cstdio>
#include <string>

namespace pffdtd {

inline auto toString(sycl::info::device_type type) -> std::string {
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

inline auto summary(sycl::device dev) -> void {
  auto name         = dev.get_info<sycl::info::device::name>();
  auto vendor       = dev.get_info<sycl::info::device::vendor>();
  auto type         = dev.get_info<sycl::info::device::device_type>();
  auto maxAllocSize = dev.get_info<sycl::info::device::max_mem_alloc_size>();

  std::printf("----------------------------------------\n");
  std::printf("Name: %s\n", name.c_str());
  std::printf("Vendor: %s\n", vendor.c_str());
  std::printf("Type: %s\n", toString(type).c_str());
  std::printf("Max alloc size: %zu MB\n", maxAllocSize / 1024 / 1024);
  for (auto groupSize : dev.get_info<sycl::info::device::sub_group_sizes>()) {
    std::printf("Subgroup size: %zu\n", groupSize);
  }
  std::printf("\n");
}

} // namespace pffdtd
