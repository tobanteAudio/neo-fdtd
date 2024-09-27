// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch
#include "engine_metal_3d.hpp"

#include "pffdtd/assert.hpp"
#include "pffdtd/engine_metal.hpp"
#include "pffdtd/progress.hpp"
#include "pffdtd/time.hpp"

#include <fmt/format.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <span>

namespace pffdtd {

namespace {

void summary(id<MTLDevice> device) {
  fmt::println("- Device {}", [device.name UTF8String]);
  fmt::println("  - Unified memory: {}", device.hasUnifiedMemory != 0 ? "true" : "false");
  fmt::println("  - Max buffer length {} MB", device.maxBufferLength / 1'000'000);
  fmt::println("  - Recommended max working set size: {} MB", device.recommendedMaxWorkingSetSize / 1'000'000);
  fmt::println("  - Max transfer rate: {}", device.maxTransferRate);
  fmt::println("  - Max threads per threadgroup: {}", device.maxThreadsPerThreadgroup.width);
  fmt::println("  - Max threadgroup memory size: {} bytes\n", device.maxThreadgroupMemoryLength);
}

id<MTLComputePipelineState> makeComputePipeline(id<MTLLibrary> library, char const* function) {
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

template<typename Real>
auto run(Simulation3D<Real> const& sim) {
  @autoreleasepool {
    // Device
    NSArray* devices     = MTLCopyAllDevices();
    id<MTLDevice> device = devices[0];
    PFFDTD_ASSERT(device != nil);
    summary(device);

    // Library
    id<MTLLibrary> library = [device newDefaultLibrary];
    PFFDTD_ASSERT(library != nil);

    id<MTLComputePipelineState> kernelAirCart         = makeComputePipeline(library, "pffdtd::sim3d::airUpdateCart");
    id<MTLComputePipelineState> kernelRigidCart       = makeComputePipeline(library, "pffdtd::sim3d::rigidUpdateCart");
    id<MTLComputePipelineState> kernelCopyFromGrid    = makeComputePipeline(library, "pffdtd::sim3d::copyFromGrid");
    id<MTLComputePipelineState> kernelCopyToGrid      = makeComputePipeline(library, "pffdtd::sim3d::copyToGrid");
    id<MTLComputePipelineState> kernelBoundaryLossFD  = makeComputePipeline(library, "pffdtd::sim3d::boundaryLossFD");
    id<MTLComputePipelineState> kernelBoundaryLossABC = makeComputePipeline(library, "pffdtd::sim3d::boundaryLossABC");
    id<MTLComputePipelineState> kernelFlipHaloXY      = makeComputePipeline(library, "pffdtd::sim3d::flipHaloXY");
    id<MTLComputePipelineState> kernelFlipHaloXZ      = makeComputePipeline(library, "pffdtd::sim3d::flipHaloXZ");
    id<MTLComputePipelineState> kernelFlipHaloYZ      = makeComputePipeline(library, "pffdtd::sim3d::flipHaloYZ");
    id<MTLComputePipelineState> kernelAddSource       = makeComputePipeline(library, "pffdtd::sim3d::addSource");
    id<MTLComputePipelineState> kernelReadOutput      = makeComputePipeline(library, "pffdtd::sim3d::readOutput");

    auto const Nx   = sim.Nx;
    auto const Ny   = sim.Ny;
    auto const Nz   = sim.Nz;
    auto const NzNy = Nz * Ny;
    auto const Npts = sim.Npts;
    auto const Nb   = sim.Nb;
    auto const Nbl  = sim.Nbl;
    auto const Nba  = sim.Nba;
    auto const Nr   = sim.Nr;
    auto const Ns   = sim.Ns;
    auto const Nt   = sim.Nt;

    auto const lo2 = static_cast<Real>(sim.lo2);
    auto const sl2 = static_cast<Real>(sim.sl2);
    auto const l   = static_cast<Real>(sim.l);
    auto const a1  = static_cast<Real>(sim.a1);
    auto const a2  = static_cast<Real>(sim.a2);

    fmt::println("HOST-BUFFERS");
    id<MTLBuffer> out_ixyz  = makeBuffer(device, sim.out_ixyz, MTLResourceStorageModeShared);
    id<MTLBuffer> in_ixyz   = makeBuffer(device, sim.in_ixyz, MTLResourceStorageModeShared);
    id<MTLBuffer> in_sigs   = makeBuffer(device, sim.in_sigs, MTLResourceStorageModeShared);
    id<MTLBuffer> adj_bn    = makeBuffer(device, sim.adj_bn, MTLResourceStorageModeShared);
    id<MTLBuffer> bn_mask   = makeBuffer(device, sim.bn_mask, MTLResourceStorageModeShared);
    id<MTLBuffer> bn_ixyz   = makeBuffer(device, sim.bn_ixyz, MTLResourceStorageModeShared);
    id<MTLBuffer> bnl_ixyz  = makeBuffer(device, sim.bnl_ixyz, MTLResourceStorageModeShared);
    id<MTLBuffer> bna_ixyz  = makeBuffer(device, sim.bna_ixyz, MTLResourceStorageModeShared);
    id<MTLBuffer> ssaf_bnl  = makeBuffer(device, sim.ssaf_bnl, MTLResourceStorageModeShared);
    id<MTLBuffer> mat_beta  = makeBuffer(device, sim.mat_beta, MTLResourceStorageModeShared);
    id<MTLBuffer> mat_bnl   = makeBuffer(device, sim.mat_bnl, MTLResourceStorageModeShared);
    id<MTLBuffer> Mb        = makeBuffer(device, sim.Mb, MTLResourceStorageModeShared);
    id<MTLBuffer> mat_quads = makeBuffer(device, sim.mat_quads, MTLResourceStorageModeShared);
    id<MTLBuffer> Q_bna     = makeBuffer(device, sim.Q_bna, MTLResourceStorageModeShared);

    fmt::println("DEVICE-BUFFERS");
    id<MTLBuffer> u0    = [device newBufferWithLength:sizeof(Real) * Npts options:MTLResourceStorageModeShared];
    id<MTLBuffer> u1    = [device newBufferWithLength:sizeof(Real) * Npts options:MTLResourceStorageModeShared];
    id<MTLBuffer> u0b   = [device newBufferWithLength:sizeof(Real) * Nbl options:MTLResourceStorageModeShared];
    id<MTLBuffer> u1b   = [device newBufferWithLength:sizeof(Real) * Nbl options:MTLResourceStorageModeShared];
    id<MTLBuffer> u2b   = [device newBufferWithLength:sizeof(Real) * Nbl options:MTLResourceStorageModeShared];
    id<MTLBuffer> u2ba  = [device newBufferWithLength:sizeof(Real) * Nba options:MTLResourceStorageModeShared];
    id<MTLBuffer> gh1   = [device newBufferWithLength:sizeof(Real) * Nbl * MMb options:MTLResourceStorageModeShared];
    id<MTLBuffer> vh1   = [device newBufferWithLength:sizeof(Real) * Nbl * MMb options:MTLResourceStorageModeShared];
    id<MTLBuffer> u_out = [device newBufferWithLength:sizeof(Real) * Nr * Nt options:MTLResourceStorageModeShared];

    // Queue
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    assert(commandQueue != nil);

    fmt::println("START");
    auto const start = getTime();
    for (int64_t n = 0; n < Nt; n++) {
      auto const sampleStart = getTime();

      auto const c = Constants3D<Real>{
          .n    = n,
          .Nx   = Nx,
          .Ny   = Ny,
          .Nz   = Nz,
          .NzNy = NzNy,
          .Nb   = Nb,
          .Nbl  = Nbl,
          .Nba  = Nba,
          .Ns   = Ns,
          .Nr   = Nr,
          .Nt   = Nt,
          .l    = l,
          .lo2  = lo2,
          .sl2  = sl2,
          .a1   = a1,
          .a2   = a2,
      };
      id<MTLBuffer> constants = [device newBufferWithBytes:&c length:sizeof(c) options:MTLResourceStorageModeShared];

      id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

      // Rigid
      id<MTLComputeCommandEncoder> rigidEncoder = [commandBuffer computeCommandEncoder];
      [rigidEncoder setComputePipelineState:kernelRigidCart];
      [rigidEncoder setBuffer:u0 offset:0 atIndex:0];
      [rigidEncoder setBuffer:u1 offset:0 atIndex:1];
      [rigidEncoder setBuffer:bn_ixyz offset:0 atIndex:2];
      [rigidEncoder setBuffer:adj_bn offset:0 atIndex:3];
      [rigidEncoder setBuffer:constants offset:0 atIndex:4];
      [rigidEncoder dispatchThreads:MTLSizeMake(Nb, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [rigidEncoder endEncoding];

      // Copy to boundary buffer
      id<MTLComputeCommandEncoder> copyToBoundaryEncoder = [commandBuffer computeCommandEncoder];
      [copyToBoundaryEncoder setComputePipelineState:kernelCopyFromGrid];
      [copyToBoundaryEncoder setBuffer:u0b offset:0 atIndex:0];
      [copyToBoundaryEncoder setBuffer:u0 offset:0 atIndex:1];
      [copyToBoundaryEncoder setBuffer:bnl_ixyz offset:0 atIndex:2];
      [copyToBoundaryEncoder dispatchThreads:MTLSizeMake(Nbl, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [copyToBoundaryEncoder endEncoding];

      // Apply rigid loss
      id<MTLComputeCommandEncoder> boundaryLossEncoder = [commandBuffer computeCommandEncoder];
      [boundaryLossEncoder setComputePipelineState:kernelBoundaryLossFD];
      [boundaryLossEncoder setBuffer:u0b offset:0 atIndex:0];
      [boundaryLossEncoder setBuffer:u2b offset:0 atIndex:1];
      [boundaryLossEncoder setBuffer:vh1 offset:0 atIndex:2];
      [boundaryLossEncoder setBuffer:gh1 offset:0 atIndex:3];
      [boundaryLossEncoder setBuffer:ssaf_bnl offset:0 atIndex:4];
      [boundaryLossEncoder setBuffer:mat_bnl offset:0 atIndex:5];
      [boundaryLossEncoder setBuffer:mat_beta offset:0 atIndex:6];
      [boundaryLossEncoder setBuffer:mat_quads offset:0 atIndex:7];
      [boundaryLossEncoder setBuffer:Mb offset:0 atIndex:8];
      [boundaryLossEncoder setBuffer:constants offset:0 atIndex:9];
      [boundaryLossEncoder dispatchThreads:MTLSizeMake(Nbl, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [boundaryLossEncoder endEncoding];

      // Copy from boundary buffer
      id<MTLComputeCommandEncoder> copyFromBoundaryEncoder = [commandBuffer computeCommandEncoder];
      [copyFromBoundaryEncoder setComputePipelineState:kernelCopyToGrid];
      [copyFromBoundaryEncoder setBuffer:u0 offset:0 atIndex:0];
      [copyFromBoundaryEncoder setBuffer:u0b offset:0 atIndex:1];
      [copyFromBoundaryEncoder setBuffer:bnl_ixyz offset:0 atIndex:2];
      [copyFromBoundaryEncoder dispatchThreads:MTLSizeMake(Nbl, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [copyFromBoundaryEncoder endEncoding];

      // Copy to ABC buffer
      id<MTLComputeCommandEncoder> copyToABCEncoder = [commandBuffer computeCommandEncoder];
      [copyToABCEncoder setComputePipelineState:kernelCopyFromGrid];
      [copyToABCEncoder setBuffer:u2ba offset:0 atIndex:0];
      [copyToABCEncoder setBuffer:u0 offset:0 atIndex:1];
      [copyToABCEncoder setBuffer:bna_ixyz offset:0 atIndex:2];
      [copyToABCEncoder dispatchThreads:MTLSizeMake(Nba, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [copyToABCEncoder endEncoding];

      // Flip halo XY
      id<MTLComputeCommandEncoder> flipHaloXYEncoder = [commandBuffer computeCommandEncoder];
      [flipHaloXYEncoder setComputePipelineState:kernelFlipHaloXY];
      [flipHaloXYEncoder setBuffer:u1 offset:0 atIndex:0];
      [flipHaloXYEncoder setBuffer:constants offset:0 atIndex:1];
      [flipHaloXYEncoder dispatchThreads:MTLSizeMake(Nx, Ny, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [flipHaloXYEncoder endEncoding];

      // Flip halo XZ
      id<MTLComputeCommandEncoder> flipHaloXZEncoder = [commandBuffer computeCommandEncoder];
      [flipHaloXZEncoder setComputePipelineState:kernelFlipHaloXZ];
      [flipHaloXZEncoder setBuffer:u1 offset:0 atIndex:0];
      [flipHaloXZEncoder setBuffer:constants offset:0 atIndex:1];
      [flipHaloXZEncoder dispatchThreads:MTLSizeMake(Nx, Nz, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [flipHaloXZEncoder endEncoding];

      // Flip halo YZ
      id<MTLComputeCommandEncoder> flipHaloYZEncoder = [commandBuffer computeCommandEncoder];
      [flipHaloYZEncoder setComputePipelineState:kernelFlipHaloYZ];
      [flipHaloYZEncoder setBuffer:u1 offset:0 atIndex:0];
      [flipHaloYZEncoder setBuffer:constants offset:0 atIndex:1];
      [flipHaloYZEncoder dispatchThreads:MTLSizeMake(Ny, Nz, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [flipHaloYZEncoder endEncoding];

      // Add source
      id<MTLComputeCommandEncoder> addSourceEncoder = [commandBuffer computeCommandEncoder];
      [addSourceEncoder setComputePipelineState:kernelAddSource];
      [addSourceEncoder setBuffer:u0 offset:0 atIndex:0];
      [addSourceEncoder setBuffer:in_sigs offset:0 atIndex:1];
      [addSourceEncoder setBuffer:in_ixyz offset:0 atIndex:2];
      [addSourceEncoder setBuffer:constants offset:0 atIndex:3];
      [addSourceEncoder dispatchThreads:MTLSizeMake(Ns, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [addSourceEncoder endEncoding];

      // Air update
      id<MTLComputeCommandEncoder> airUpdateEncoder = [commandBuffer computeCommandEncoder];
      [airUpdateEncoder setComputePipelineState:kernelAirCart];
      [airUpdateEncoder setBuffer:u0 offset:0 atIndex:0];
      [airUpdateEncoder setBuffer:u1 offset:0 atIndex:1];
      [airUpdateEncoder setBuffer:bn_mask offset:0 atIndex:2];
      [airUpdateEncoder setBuffer:constants offset:0 atIndex:3];
      [airUpdateEncoder dispatchThreads:MTLSizeMake(Nx - 2, Ny - 2, Nz - 2) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [airUpdateEncoder endEncoding];

      // ABC loss
      id<MTLComputeCommandEncoder> abcEncoder = [commandBuffer computeCommandEncoder];
      [abcEncoder setComputePipelineState:kernelBoundaryLossABC];
      [abcEncoder setBuffer:u0 offset:0 atIndex:0];
      [abcEncoder setBuffer:u2ba offset:0 atIndex:1];
      [abcEncoder setBuffer:Q_bna offset:0 atIndex:2];
      [abcEncoder setBuffer:bna_ixyz offset:0 atIndex:3];
      [abcEncoder setBuffer:constants offset:0 atIndex:4];
      [abcEncoder dispatchThreads:MTLSizeMake(Nba, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [abcEncoder endEncoding];

      // Read receiver
      id<MTLComputeCommandEncoder> readOutputEncoder = [commandBuffer computeCommandEncoder];
      [readOutputEncoder setComputePipelineState:kernelReadOutput];
      [readOutputEncoder setBuffer:u_out offset:0 atIndex:0];
      [readOutputEncoder setBuffer:u1 offset:0 atIndex:1];
      [readOutputEncoder setBuffer:out_ixyz offset:0 atIndex:2];
      [readOutputEncoder setBuffer:constants offset:0 atIndex:3];
      [readOutputEncoder dispatchThreads:MTLSizeMake(Nr, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [readOutputEncoder endEncoding];

      // Wait
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];

      // Swap buffers
      {
        auto tmp = u1;
        u1       = u0;
        u0       = tmp;
      }
      {
        auto tmp = u2b;
        u2b      = u1b;
        u1b      = u0b;
        u0b      = tmp;
      }

      auto const now           = getTime();
      auto const elapsed       = now - start;
      auto const elapsedSample = now - sampleStart;

      print(ProgressReport{
          .n                     = n,
          .Nt                    = Nt,
          .Npts                  = Npts,
          .Nb                    = Nb,
          .elapsed               = elapsed,
          .elapsedSample         = elapsedSample,
          .elapsedAir            = {},
          .elapsedSampleAir      = {},
          .elapsedBoundary       = {},
          .elapsedSampleBoundary = {},
          .numWorkers            = 1,
      });
    }

    float* output = (float*)[u_out contents];
    for (auto i{0UL}; i < static_cast<size_t>(Nr * Nt); ++i) {
      sim.u_out[i] = output[i];
    }
  }
}

} // namespace

auto EngineMETAL3D::operator()(Simulation3D<float> const& sim) const -> void { run(sim); }

} // namespace pffdtd
