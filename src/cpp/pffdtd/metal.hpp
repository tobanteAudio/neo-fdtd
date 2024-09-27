// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#pragma once

#include "pffdtd/assert.hpp"
#include "pffdtd/time.hpp"

#if not defined(PFFDTD_HAS_METAL)
  #error "SYCL must be enabled in CMake via PFFDTD_ENABLE_METAL"
#endif

#include <fmt/format.h>

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

namespace pffdtd {

inline void summary(id<MTLDevice> device) {
  fmt::println("- Device {}", [device.name UTF8String]);
  fmt::println("  - Unified memory: {}", device.hasUnifiedMemory != 0 ? "true" : "false");
  fmt::println("  - Max buffer length {} MB", device.maxBufferLength / 1'000'000);
  fmt::println("  - Recommended max working set size: {} MB", device.recommendedMaxWorkingSetSize / 1'000'000);
  fmt::println("  - Max transfer rate: {}", device.maxTransferRate);
  fmt::println("  - Max threads per threadgroup: {}", device.maxThreadsPerThreadgroup.width);
  fmt::println("  - Max threadgroup memory size: {} bytes\n", device.maxThreadgroupMemoryLength);
}

inline id<MTLComputePipelineState> makeComputePipeline(id<MTLLibrary> library, char const* function) {
  id<MTLFunction> kernel = [library newFunctionWithName:[NSString stringWithUTF8String:function]];
  PFFDTD_ASSERT(kernel != nil);

  NSError* error                       = nullptr;
  id<MTLComputePipelineState> pipeline = [library.device newComputePipelineStateWithFunction:kernel error:&error];
  PFFDTD_ASSERT(error == nullptr);

  fmt::println("- {}:", function);
  fmt::println("  - threadExecutionWidth = {}", pipeline.threadExecutionWidth);
  fmt::println("  - maxTotalThreadsPerThreadgroup = {}", pipeline.maxTotalThreadsPerThreadgroup);
  fmt::println("  - staticThreadgroupMemoryLength = {}\n", pipeline.staticThreadgroupMemoryLength);

  return pipeline;
}

template<typename T, typename ModeT>
id<MTLBuffer> makeBuffer(id<MTLDevice> device, std::vector<T> const& data, ModeT mode) {
  auto const size   = data.size() * sizeof(T);
  id<MTLBuffer> buf = [device newBufferWithBytes:data.data() length:size options:mode];
  PFFDTD_ASSERT(buf != nil);
  return buf;
}

template<typename T, typename ModeT>
id<MTLBuffer> makeEmptyBuffer(id<MTLDevice> device, size_t count, ModeT mode) {
  auto const size   = count * sizeof(T);
  id<MTLBuffer> buf = [device newBufferWithLength:size options:mode];
  PFFDTD_ASSERT(buf != nil);
  return buf;
}

} // namespace pffdtd
