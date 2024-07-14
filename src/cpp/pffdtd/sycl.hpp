#pragma once

#include <sycl/sycl.hpp>

#include <fmt/format.h>
#include <fmt/os.h>

#include <string>
#include <utility>

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

  fmt::println("----------------------------------------");
  fmt::println("Name: {}", name.c_str());
  fmt::println("Vendor: {}", vendor.c_str());
  fmt::println("Type: {}", toString(type).c_str());
  fmt::println("Max alloc size: {} MB", maxAllocSize / 1024 / 1024);
  // for (auto groupSize : dev.get_info<sycl::info::device::sub_group_sizes>()) {
  //   fmt::println("Subgroup size: {}", groupSize);
  // }
  fmt::println("");
}

template<typename Accessor>
[[nodiscard]] auto getPtr(Accessor&& a) -> auto* {
  return std::forward<Accessor>(a)
      .template get_multi_ptr<sycl::access::decorated::no>()
      .get();
}

} // namespace pffdtd
