#pragma once
#ifndef GCXX_RUNTIME_FLAGS_DEVICE_FLAGS_HPP_
#define GCXX_RUNTIME_FLAGS_DEVICE_FLAGS_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

GCXX_NAMESPACE_MAIN_FLAGS_BEGIN


enum class deviceAttribute : flag_t {
  MaxThreadsPerBlock        = GCXX_ATTRIBUTE_BACKEND(MaxThreadsPerBlock),
  MaxBlockDimX              = GCXX_ATTRIBUTE_BACKEND(MaxBlockDimX),
  MaxBlockDimY              = GCXX_ATTRIBUTE_BACKEND(MaxBlockDimY),
  MaxBlockDimZ              = GCXX_ATTRIBUTE_BACKEND(MaxBlockDimZ),
  MaxGridDimX               = GCXX_ATTRIBUTE_BACKEND(MaxGridDimX),
  MaxGridDimY               = GCXX_ATTRIBUTE_BACKEND(MaxGridDimY),
  MaxGridDimZ               = GCXX_ATTRIBUTE_BACKEND(MaxGridDimZ),
  MaxSharedMemoryPerBlock   = GCXX_ATTRIBUTE_BACKEND(MaxSharedMemoryPerBlock),
  TotalConstantMemory       = GCXX_ATTRIBUTE_BACKEND(TotalConstantMemory),
  WarpSize                  = GCXX_ATTRIBUTE_BACKEND(WarpSize),
  MaxPitch                  = GCXX_ATTRIBUTE_BACKEND(MaxPitch),
  MaxRegistersPerBlock      = GCXX_ATTRIBUTE_BACKEND(MaxRegistersPerBlock),
  ClockRate                 = GCXX_ATTRIBUTE_BACKEND(ClockRate),
  TextureAlignment          = GCXX_ATTRIBUTE_BACKEND(TextureAlignment),
  GpuOverlap                = GCXX_ATTRIBUTE_BACKEND(GpuOverlap),
  MultiProcessorCount       = GCXX_ATTRIBUTE_BACKEND(MultiProcessorCount),
  KernelExecTimeout         = GCXX_ATTRIBUTE_BACKEND(KernelExecTimeout),
  Integrated                = GCXX_ATTRIBUTE_BACKEND(Integrated),
  CanMapHostMemory          = GCXX_ATTRIBUTE_BACKEND(CanMapHostMemory),
  ComputeMode               = GCXX_ATTRIBUTE_BACKEND(ComputeMode),
  MaxTexture1DWidth         = GCXX_ATTRIBUTE_BACKEND(MaxTexture1DWidth),
  MaxTexture2DWidth         = GCXX_ATTRIBUTE_BACKEND(MaxTexture2DWidth),
  MaxTexture2DHeight        = GCXX_ATTRIBUTE_BACKEND(MaxTexture2DHeight),
  MaxTexture3DWidth         = GCXX_ATTRIBUTE_BACKEND(MaxTexture3DWidth),
  MaxTexture3DHeight        = GCXX_ATTRIBUTE_BACKEND(MaxTexture3DHeight),
  MaxTexture3DDepth         = GCXX_ATTRIBUTE_BACKEND(MaxTexture3DDepth),
  MaxTexture2DLayeredWidth  = GCXX_ATTRIBUTE_BACKEND(MaxTexture2DLayeredWidth),
  MaxTexture2DLayeredHeight = GCXX_ATTRIBUTE_BACKEND(MaxTexture2DLayeredHeight),
  MaxTexture2DLayeredLayers = GCXX_ATTRIBUTE_BACKEND(MaxTexture2DLayeredLayers),
  SurfaceAlignment          = GCXX_ATTRIBUTE_BACKEND(SurfaceAlignment),
  ConcurrentKernels         = GCXX_ATTRIBUTE_BACKEND(ConcurrentKernels),
  EccEnabled                = GCXX_ATTRIBUTE_BACKEND(EccEnabled),
  PciBusId                  = GCXX_ATTRIBUTE_BACKEND(PciBusId),
  PciDeviceId               = GCXX_ATTRIBUTE_BACKEND(PciDeviceId),
  TccDriver                 = GCXX_ATTRIBUTE_BACKEND(TccDriver),
  MemoryClockRate           = GCXX_ATTRIBUTE_BACKEND(MemoryClockRate),
  GlobalMemoryBusWidth      = GCXX_ATTRIBUTE_BACKEND(GlobalMemoryBusWidth),
  L2CacheSize               = GCXX_ATTRIBUTE_BACKEND(L2CacheSize),
  MaxThreadsPerMultiProcessor =
    GCXX_ATTRIBUTE_BACKEND(MaxThreadsPerMultiProcessor),
  AsyncEngineCount          = GCXX_ATTRIBUTE_BACKEND(AsyncEngineCount),
  UnifiedAddressing         = GCXX_ATTRIBUTE_BACKEND(UnifiedAddressing),
  MaxTexture1DLayeredWidth  = GCXX_ATTRIBUTE_BACKEND(MaxTexture1DLayeredWidth),
  MaxTexture1DLayeredLayers = GCXX_ATTRIBUTE_BACKEND(MaxTexture1DLayeredLayers),
  MaxTexture2DGatherWidth   = GCXX_ATTRIBUTE_BACKEND(MaxTexture2DGatherWidth),
  MaxTexture2DGatherHeight  = GCXX_ATTRIBUTE_BACKEND(MaxTexture2DGatherHeight),
  MaxTexture3DWidthAlt      = GCXX_ATTRIBUTE_BACKEND(MaxTexture3DWidthAlt),
  MaxTexture3DHeightAlt     = GCXX_ATTRIBUTE_BACKEND(MaxTexture3DHeightAlt),
  MaxTexture3DDepthAlt      = GCXX_ATTRIBUTE_BACKEND(MaxTexture3DDepthAlt),
  PciDomainId               = GCXX_ATTRIBUTE_BACKEND(PciDomainId),
  TexturePitchAlignment     = GCXX_ATTRIBUTE_BACKEND(TexturePitchAlignment),
  MaxTextureCubemapWidth    = GCXX_ATTRIBUTE_BACKEND(MaxTextureCubemapWidth),
  MaxTextureCubemapLayeredWidth =
    GCXX_ATTRIBUTE_BACKEND(MaxTextureCubemapLayeredWidth),
  MaxTextureCubemapLayeredLayers =
    GCXX_ATTRIBUTE_BACKEND(MaxTextureCubemapLayeredLayers),
  MaxSurface1DWidth         = GCXX_ATTRIBUTE_BACKEND(MaxSurface1DWidth),
  MaxSurface2DWidth         = GCXX_ATTRIBUTE_BACKEND(MaxSurface2DWidth),
  MaxSurface2DHeight        = GCXX_ATTRIBUTE_BACKEND(MaxSurface2DHeight),
  MaxSurface3DWidth         = GCXX_ATTRIBUTE_BACKEND(MaxSurface3DWidth),
  MaxSurface3DHeight        = GCXX_ATTRIBUTE_BACKEND(MaxSurface3DHeight),
  MaxSurface3DDepth         = GCXX_ATTRIBUTE_BACKEND(MaxSurface3DDepth),
  MaxSurface1DLayeredWidth  = GCXX_ATTRIBUTE_BACKEND(MaxSurface1DLayeredWidth),
  MaxSurface1DLayeredLayers = GCXX_ATTRIBUTE_BACKEND(MaxSurface1DLayeredLayers),
  MaxSurface2DLayeredWidth  = GCXX_ATTRIBUTE_BACKEND(MaxSurface2DLayeredWidth),
  MaxSurface2DLayeredHeight = GCXX_ATTRIBUTE_BACKEND(MaxSurface2DLayeredHeight),
  MaxSurface2DLayeredLayers = GCXX_ATTRIBUTE_BACKEND(MaxSurface2DLayeredLayers),
  MaxSurfaceCubemapWidth    = GCXX_ATTRIBUTE_BACKEND(MaxSurfaceCubemapWidth),
  MaxSurfaceCubemapLayeredWidth =
    GCXX_ATTRIBUTE_BACKEND(MaxSurfaceCubemapLayeredWidth),
  MaxSurfaceCubemapLayeredLayers =
    GCXX_ATTRIBUTE_BACKEND(MaxSurfaceCubemapLayeredLayers),
  MaxTexture1DLinearWidth  = GCXX_ATTRIBUTE_BACKEND(MaxTexture1DLinearWidth),
  MaxTexture2DLinearWidth  = GCXX_ATTRIBUTE_BACKEND(MaxTexture2DLinearWidth),
  MaxTexture2DLinearHeight = GCXX_ATTRIBUTE_BACKEND(MaxTexture2DLinearHeight),
  MaxTexture2DLinearPitch  = GCXX_ATTRIBUTE_BACKEND(MaxTexture2DLinearPitch),
  MaxTexture2DMipmappedWidth =
    GCXX_ATTRIBUTE_BACKEND(MaxTexture2DMipmappedWidth),
  MaxTexture2DMipmappedHeight =
    GCXX_ATTRIBUTE_BACKEND(MaxTexture2DMipmappedHeight),
  ComputeCapabilityMajor = GCXX_ATTRIBUTE_BACKEND(ComputeCapabilityMajor),
  ComputeCapabilityMinor = GCXX_ATTRIBUTE_BACKEND(ComputeCapabilityMinor),
  MaxTexture1DMipmappedWidth =
    GCXX_ATTRIBUTE_BACKEND(MaxTexture1DMipmappedWidth),
  StreamPrioritiesSupported = GCXX_ATTRIBUTE_BACKEND(StreamPrioritiesSupported),
  GlobalL1CacheSupported    = GCXX_ATTRIBUTE_BACKEND(GlobalL1CacheSupported),
  LocalL1CacheSupported     = GCXX_ATTRIBUTE_BACKEND(LocalL1CacheSupported),
  MaxSharedMemoryPerMultiprocessor =
    GCXX_ATTRIBUTE_BACKEND(MaxSharedMemoryPerMultiprocessor),
  MaxRegistersPerMultiprocessor =
    GCXX_ATTRIBUTE_BACKEND(MaxRegistersPerMultiprocessor),
  ManagedMemory             = GCXX_ATTRIBUTE_BACKEND(ManagedMemory),
  IsMultiGpuBoard           = GCXX_ATTRIBUTE_BACKEND(IsMultiGpuBoard),
  MultiGpuBoardGroupID      = GCXX_ATTRIBUTE_BACKEND(MultiGpuBoardGroupID),
  HostNativeAtomicSupported = GCXX_ATTRIBUTE_BACKEND(HostNativeAtomicSupported),
  SingleToDoublePrecisionPerfRatio =
    GCXX_ATTRIBUTE_BACKEND(SingleToDoublePrecisionPerfRatio),
  PageableMemoryAccess    = GCXX_ATTRIBUTE_BACKEND(PageableMemoryAccess),
  ConcurrentManagedAccess = GCXX_ATTRIBUTE_BACKEND(ConcurrentManagedAccess),
  ComputePreemptionSupported =
    GCXX_ATTRIBUTE_BACKEND(ComputePreemptionSupported),
  CanUseHostPointerForRegisteredMem =
    GCXX_ATTRIBUTE_BACKEND(CanUseHostPointerForRegisteredMem),
  Reserved92        = GCXX_ATTRIBUTE_BACKEND(Reserved92),
  Reserved93        = GCXX_ATTRIBUTE_BACKEND(Reserved93),
  Reserved94        = GCXX_ATTRIBUTE_BACKEND(Reserved94),
  CooperativeLaunch = GCXX_ATTRIBUTE_BACKEND(CooperativeLaunch),
  CooperativeMultiDeviceLaunch =
    GCXX_ATTRIBUTE_BACKEND(CooperativeMultiDeviceLaunch),
  MaxSharedMemoryPerBlockOptin =
    GCXX_ATTRIBUTE_BACKEND(MaxSharedMemoryPerBlockOptin),
  CanFlushRemoteWrites  = GCXX_ATTRIBUTE_BACKEND(CanFlushRemoteWrites),
  HostRegisterSupported = GCXX_ATTRIBUTE_BACKEND(HostRegisterSupported),
  PageableMemoryAccessUsesHostPageTables =
    GCXX_ATTRIBUTE_BACKEND(PageableMemoryAccessUsesHostPageTables),
  DirectManagedMemAccessFromHost =
    GCXX_ATTRIBUTE_BACKEND(DirectManagedMemAccessFromHost),
  MaxBlocksPerMultiprocessor =
    GCXX_ATTRIBUTE_BACKEND(MaxBlocksPerMultiprocessor),
  MaxPersistingL2CacheSize  = GCXX_ATTRIBUTE_BACKEND(MaxPersistingL2CacheSize),
  MaxAccessPolicyWindowSize = GCXX_ATTRIBUTE_BACKEND(MaxAccessPolicyWindowSize),
  ReservedSharedMemoryPerBlock =
    GCXX_ATTRIBUTE_BACKEND(ReservedSharedMemoryPerBlock),
  SparseCudaArraySupported = GCXX_ATTRIBUTE_BACKEND(SparseCudaArraySupported),
  HostRegisterReadOnlySupported =
    GCXX_ATTRIBUTE_BACKEND(HostRegisterReadOnlySupported),
  TimelineSemaphoreInteropSupported =
    GCXX_ATTRIBUTE_BACKEND(TimelineSemaphoreInteropSupported),
  MaxTimelineSemaphoreInteropSupported =
    GCXX_ATTRIBUTE_BACKEND(MaxTimelineSemaphoreInteropSupported),
  MemoryPoolsSupported   = GCXX_ATTRIBUTE_BACKEND(MemoryPoolsSupported),
  GPUDirectRDMASupported = GCXX_ATTRIBUTE_BACKEND(GPUDirectRDMASupported),
  GPUDirectRDMAFlushWritesOptions =
    GCXX_ATTRIBUTE_BACKEND(GPUDirectRDMAFlushWritesOptions),
  GPUDirectRDMAWritesOrdering =
    GCXX_ATTRIBUTE_BACKEND(GPUDirectRDMAWritesOrdering),
  MemoryPoolSupportedHandleTypes =
    GCXX_ATTRIBUTE_BACKEND(MemoryPoolSupportedHandleTypes),
  ClusterLaunch = GCXX_ATTRIBUTE_BACKEND(ClusterLaunch),
  DeferredMappingCudaArraySupported =
    GCXX_ATTRIBUTE_BACKEND(DeferredMappingCudaArraySupported),
  Reserved122        = GCXX_ATTRIBUTE_BACKEND(Reserved122),
  Reserved123        = GCXX_ATTRIBUTE_BACKEND(Reserved123),
  Reserved124        = GCXX_ATTRIBUTE_BACKEND(Reserved124),
  IpcEventSupport    = GCXX_ATTRIBUTE_BACKEND(IpcEventSupport),
  MemSyncDomainCount = GCXX_ATTRIBUTE_BACKEND(MemSyncDomainCount),
  Reserved127        = GCXX_ATTRIBUTE_BACKEND(Reserved127),
  Reserved128        = GCXX_ATTRIBUTE_BACKEND(Reserved128),
  Reserved129        = GCXX_ATTRIBUTE_BACKEND(Reserved129),
  NumaConfig         = GCXX_ATTRIBUTE_BACKEND(NumaConfig),
  NumaId             = GCXX_ATTRIBUTE_BACKEND(NumaId),
  Reserved132        = GCXX_ATTRIBUTE_BACKEND(Reserved132),
  MpsEnabled         = GCXX_ATTRIBUTE_BACKEND(MpsEnabled),
  HostNumaId         = GCXX_ATTRIBUTE_BACKEND(HostNumaId),
  D3D12CigSupported  = GCXX_ATTRIBUTE_BACKEND(D3D12CigSupported),
  VulkanCigSupported = GCXX_ATTRIBUTE_BACKEND(VulkanCigSupported),
  GpuPciDeviceId     = GCXX_ATTRIBUTE_BACKEND(GpuPciDeviceId),
  GpuPciSubsystemId  = GCXX_ATTRIBUTE_BACKEND(GpuPciSubsystemId),
  Reserved141        = GCXX_ATTRIBUTE_BACKEND(Reserved141),
  HostNumaMemoryPoolsSupported =
    GCXX_ATTRIBUTE_BACKEND(HostNumaMemoryPoolsSupported),
  HostNumaMultinodeIpcSupported =
    GCXX_ATTRIBUTE_BACKEND(HostNumaMultinodeIpcSupported),
  Max = GCXX_ATTRIBUTE_BACKEND(Max),
};


GCXX_NAMESPACE_MAIN_FLAGS_END

#endif