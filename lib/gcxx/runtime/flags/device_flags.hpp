#pragma once
#ifndef GCXX_RUNTIME_FLAGS_DEVICE_FLAGS_HPP_
#define GCXX_RUNTIME_FLAGS_DEVICE_FLAGS_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

GCXX_NAMESPACE_MAIN_FLAGS_BEGIN

enum class deviceLimit : flag_t {
  StackSize      = GCXX_RUNTIME_BACKEND(LimitStackSize),
  PrintfFifoSize = GCXX_RUNTIME_BACKEND(LimitPrintfFifoSize),
  MallocHeapSize = GCXX_RUNTIME_BACKEND(LimitMallocHeapSize),

#if GCXX_CUDA_MODE
  DevRuntimeSyncDepth = GCXX_RUNTIME_BACKEND(LimitDevRuntimeSyncDepth),
  DevRuntimePendingLaunchCount =
    GCXX_RUNTIME_BACKEND(LimitDevRuntimePendingLaunchCount),
  MaxL2FetchGranularity = GCXX_RUNTIME_BACKEND(LimitMaxL2FetchGranularity),
  PersistingL2CacheSize = GCXX_RUNTIME_BACKEND(LimitPersistingL2CacheSize),
#endif
};

enum class deviceAttribute : flag_t {
  AsyncEngineCount = GCXX_ATTRIBUTE_BACKEND(AsyncEngineCount),
  CanMapHostMemory = GCXX_ATTRIBUTE_BACKEND(CanMapHostMemory),
  CanUseHostPointerForRegisteredMem =
    GCXX_ATTRIBUTE_BACKEND(CanUseHostPointerForRegisteredMem),
  ClockRate              = GCXX_ATTRIBUTE_BACKEND(ClockRate),
  ComputeCapabilityMajor = GCXX_ATTRIBUTE_BACKEND(ComputeCapabilityMajor),
  ComputeCapabilityMinor = GCXX_ATTRIBUTE_BACKEND(ComputeCapabilityMinor),
  ComputeMode            = GCXX_ATTRIBUTE_BACKEND(ComputeMode),
  ComputePreemptionSupported =
    GCXX_ATTRIBUTE_BACKEND(ComputePreemptionSupported),
  ConcurrentKernels       = GCXX_ATTRIBUTE_BACKEND(ConcurrentKernels),
  ConcurrentManagedAccess = GCXX_ATTRIBUTE_BACKEND(ConcurrentManagedAccess),
  CooperativeLaunch       = GCXX_ATTRIBUTE_BACKEND(CooperativeLaunch),
  CooperativeMultiDeviceLaunch =
    GCXX_ATTRIBUTE_BACKEND(CooperativeMultiDeviceLaunch),
  DirectManagedMemAccessFromHost =
    GCXX_ATTRIBUTE_BACKEND(DirectManagedMemAccessFromHost),
  EccEnabled             = GCXX_ATTRIBUTE_BACKEND(EccEnabled),
  GlobalL1CacheSupported = GCXX_ATTRIBUTE_BACKEND(GlobalL1CacheSupported),
  GlobalMemoryBusWidth =
    GCXX_ATTRIBUTE_BACKEND_ALT(GlobalMemoryBusWidth, MemoryBusWidth),
  GpuOverlap = GCXX_ATTRIBUTE_BACKEND_ALT(GpuOverlap, AsyncEngineCount),
  HostNativeAtomicSupported = GCXX_ATTRIBUTE_BACKEND(HostNativeAtomicSupported),
  HostRegisterSupported     = GCXX_ATTRIBUTE_BACKEND(HostRegisterSupported),
  Integrated                = GCXX_ATTRIBUTE_BACKEND(Integrated),
  IsMultiGpuBoard           = GCXX_ATTRIBUTE_BACKEND(IsMultiGpuBoard),
  KernelExecTimeout         = GCXX_ATTRIBUTE_BACKEND(KernelExecTimeout),
  L2CacheSize               = GCXX_ATTRIBUTE_BACKEND(L2CacheSize),
  LocalL1CacheSupported     = GCXX_ATTRIBUTE_BACKEND(LocalL1CacheSupported),
  ManagedMemory             = GCXX_ATTRIBUTE_BACKEND(ManagedMemory),
  MaxBlockDimX              = GCXX_ATTRIBUTE_BACKEND(MaxBlockDimX),
  MaxBlockDimY              = GCXX_ATTRIBUTE_BACKEND(MaxBlockDimY),
  MaxBlockDimZ              = GCXX_ATTRIBUTE_BACKEND(MaxBlockDimZ),
  MaxBlocksPerMultiprocessor = GCXX_ATTRIBUTE_BACKEND_ALT(
    MaxBlocksPerMultiprocessor, MaxBlocksPerMultiProcessor),
  MaxGridDimX          = GCXX_ATTRIBUTE_BACKEND(MaxGridDimX),
  MaxGridDimY          = GCXX_ATTRIBUTE_BACKEND(MaxGridDimY),
  MaxGridDimZ          = GCXX_ATTRIBUTE_BACKEND(MaxGridDimZ),
  MaxPitch             = GCXX_ATTRIBUTE_BACKEND(MaxPitch),
  MaxRegistersPerBlock = GCXX_ATTRIBUTE_BACKEND(MaxRegistersPerBlock),
  MaxRegistersPerMultiprocessor =
    GCXX_ATTRIBUTE_BACKEND(MaxRegistersPerMultiprocessor),
  MaxSharedMemoryPerBlock = GCXX_ATTRIBUTE_BACKEND(MaxSharedMemoryPerBlock),
  MaxSharedMemoryPerBlockOptin = GCXX_ATTRIBUTE_BACKEND_ALT(
    MaxSharedMemoryPerBlockOptin, SharedMemPerBlockOptin),
  MaxSharedMemoryPerMultiprocessor =
    GCXX_ATTRIBUTE_BACKEND(MaxSharedMemoryPerMultiprocessor),
  MaxSurface1DLayeredWidth =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxSurface1DLayeredWidth, MaxSurface1DLayered),
  MaxSurface1DWidth =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxSurface1DWidth, MaxSurface1D),
  MaxSurface2DHeight =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxSurface2DHeight, MaxSurface2D),
  MaxSurface2DLayeredHeight =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxSurface2DLayeredHeight, MaxSurface2DLayered),
  MaxSurface2DLayeredWidth =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxSurface2DLayeredWidth, MaxSurface2DLayered),
  MaxSurface2DWidth =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxSurface2DWidth, MaxSurface2D),
  MaxSurface3DDepth =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxSurface3DDepth, MaxSurface3D),
  MaxSurface3DHeight =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxSurface3DHeight, MaxSurface3D),
  MaxSurface3DWidth =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxSurface3DWidth, MaxSurface3D),
  MaxSurfaceCubemapLayeredWidth = GCXX_ATTRIBUTE_BACKEND_ALT(
    MaxSurfaceCubemapLayeredWidth, MaxSurfaceCubemapLayered),
  MaxSurfaceCubemapWidth =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxSurfaceCubemapWidth, MaxSurfaceCubemap),
  MaxTexture1DLayeredWidth =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxTexture1DLayeredWidth, MaxTexture1DLayered),
  MaxTexture1DLinearWidth =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxTexture1DLinearWidth, MaxTexture1DLinear),
  MaxTexture1DMipmappedWidth =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxTexture1DMipmappedWidth, MaxTexture1DMipmap),
  MaxTexture1DWidth = GCXX_ATTRIBUTE_BACKEND(MaxTexture1DWidth),
  MaxTexture2DGatherHeight =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxTexture2DGatherHeight, MaxTexture2DGather),
  MaxTexture2DGatherWidth =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxTexture2DGatherWidth, MaxTexture2DGather),
  MaxTexture2DHeight = GCXX_ATTRIBUTE_BACKEND(MaxTexture2DHeight),
  MaxTexture2DLayeredHeight =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxTexture2DLayeredHeight, MaxTexture2DLayered),
  MaxTexture2DLayeredWidth =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxTexture2DLayeredWidth, MaxTexture2DLayered),
  MaxTexture2DLinearHeight =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxTexture2DLinearHeight, MaxTexture2DLinear),
  MaxTexture2DLinearPitch =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxTexture2DLinearPitch, MaxTexture2DLinear),
  MaxTexture2DLinearWidth =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxTexture2DLinearWidth, MaxTexture2DLinear),
  MaxTexture2DMipmappedHeight =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxTexture2DMipmappedHeight, MaxTexture2DMipmap),
  MaxTexture2DMipmappedWidth =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxTexture2DMipmappedWidth, MaxTexture2DMipmap),
  MaxTexture2DWidth = GCXX_ATTRIBUTE_BACKEND(MaxTexture2DWidth),
  MaxTexture3DDepth = GCXX_ATTRIBUTE_BACKEND(MaxTexture3DDepth),
  MaxTexture3DDepthAlt =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxTexture3DDepthAlt, MaxTexture3DAlt),
  MaxTexture3DHeight = GCXX_ATTRIBUTE_BACKEND(MaxTexture3DHeight),
  MaxTexture3DHeightAlt =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxTexture3DHeightAlt, MaxTexture3DAlt),
  MaxTexture3DWidth = GCXX_ATTRIBUTE_BACKEND(MaxTexture3DWidth),
  MaxTexture3DWidthAlt =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxTexture3DWidthAlt, MaxTexture3DAlt),
  MaxTextureCubemapLayeredWidth = GCXX_ATTRIBUTE_BACKEND_ALT(
    MaxTextureCubemapLayeredWidth, MaxTextureCubemapLayered),
  MaxTextureCubemapWidth =
    GCXX_ATTRIBUTE_BACKEND_ALT(MaxTextureCubemapWidth, MaxTextureCubemap),
  MaxThreadsPerBlock = GCXX_ATTRIBUTE_BACKEND(MaxThreadsPerBlock),
  MaxThreadsPerMultiProcessor =
    GCXX_ATTRIBUTE_BACKEND(MaxThreadsPerMultiProcessor),
  MemoryClockRate      = GCXX_ATTRIBUTE_BACKEND(MemoryClockRate),
  MemoryPoolsSupported = GCXX_ATTRIBUTE_BACKEND(MemoryPoolsSupported),
  MultiGpuBoardGroupID = GCXX_ATTRIBUTE_BACKEND(MultiGpuBoardGroupID),
  MultiProcessorCount =
    GCXX_ATTRIBUTE_BACKEND_ALT(MultiProcessorCount, MultiprocessorCount),
  PageableMemoryAccess = GCXX_ATTRIBUTE_BACKEND(PageableMemoryAccess),
  PageableMemoryAccessUsesHostPageTables =
    GCXX_ATTRIBUTE_BACKEND(PageableMemoryAccessUsesHostPageTables),
  PciBusId    = GCXX_ATTRIBUTE_BACKEND(PciBusId),
  PciDeviceId = GCXX_ATTRIBUTE_BACKEND(PciDeviceId),
  PciDomainId = GCXX_ATTRIBUTE_BACKEND(PciDomainId),
  Reserved94  = GCXX_ATTRIBUTE_BACKEND_ALT(Reserved94, CanUseStreamWaitValue),
  SingleToDoublePrecisionPerfRatio =
    GCXX_ATTRIBUTE_BACKEND(SingleToDoublePrecisionPerfRatio),
  StreamPrioritiesSupported = GCXX_ATTRIBUTE_BACKEND(StreamPrioritiesSupported),
  SurfaceAlignment          = GCXX_ATTRIBUTE_BACKEND(SurfaceAlignment),
  TccDriver                 = GCXX_ATTRIBUTE_BACKEND(TccDriver),
  TextureAlignment          = GCXX_ATTRIBUTE_BACKEND(TextureAlignment),
  TexturePitchAlignment     = GCXX_ATTRIBUTE_BACKEND(TexturePitchAlignment),
  TotalConstantMemory       = GCXX_ATTRIBUTE_BACKEND(TotalConstantMemory),
  UnifiedAddressing         = GCXX_ATTRIBUTE_BACKEND(UnifiedAddressing),
  WarpSize                  = GCXX_ATTRIBUTE_BACKEND(WarpSize),

#if GCXX_CUDA_MODE
  CanFlushRemoteWrites = GCXX_ATTRIBUTE_BACKEND(CanFlushRemoteWrites),
  ClusterLaunch        = GCXX_ATTRIBUTE_BACKEND(ClusterLaunch),
  D3D12CigSupported    = GCXX_ATTRIBUTE_BACKEND(D3D12CigSupported),
  DeferredMappingCudaArraySupported =
    GCXX_ATTRIBUTE_BACKEND(DeferredMappingCudaArraySupported),
  GPUDirectRDMAFlushWritesOptions =
    GCXX_ATTRIBUTE_BACKEND(GPUDirectRDMAFlushWritesOptions),
  GPUDirectRDMASupported = GCXX_ATTRIBUTE_BACKEND(GPUDirectRDMASupported),
  GPUDirectRDMAWritesOrdering =
    GCXX_ATTRIBUTE_BACKEND(GPUDirectRDMAWritesOrdering),
  GpuPciDeviceId    = GCXX_ATTRIBUTE_BACKEND(GpuPciDeviceId),
  GpuPciSubsystemId = GCXX_ATTRIBUTE_BACKEND(GpuPciSubsystemId),
  HostNumaId        = GCXX_ATTRIBUTE_BACKEND(HostNumaId),
  HostNumaMemoryPoolsSupported =
    GCXX_ATTRIBUTE_BACKEND(HostNumaMemoryPoolsSupported),
  HostNumaMultinodeIpcSupported =
    GCXX_ATTRIBUTE_BACKEND(HostNumaMultinodeIpcSupported),
  HostRegisterReadOnlySupported =
    GCXX_ATTRIBUTE_BACKEND(HostRegisterReadOnlySupported),
  IpcEventSupport           = GCXX_ATTRIBUTE_BACKEND(IpcEventSupport),
  Max                       = GCXX_ATTRIBUTE_BACKEND(Max),
  MaxAccessPolicyWindowSize = GCXX_ATTRIBUTE_BACKEND(MaxAccessPolicyWindowSize),
  MaxPersistingL2CacheSize  = GCXX_ATTRIBUTE_BACKEND(MaxPersistingL2CacheSize),
  MaxSurface1DLayeredLayers = GCXX_ATTRIBUTE_BACKEND(MaxSurface1DLayeredLayers),
  MaxSurface2DLayeredLayers = GCXX_ATTRIBUTE_BACKEND(MaxSurface2DLayeredLayers),
  MaxSurfaceCubemapLayeredLayers =
    GCXX_ATTRIBUTE_BACKEND(MaxSurfaceCubemapLayeredLayers),
  MaxTexture1DLayeredLayers = GCXX_ATTRIBUTE_BACKEND(MaxTexture1DLayeredLayers),
  MaxTexture2DLayeredLayers = GCXX_ATTRIBUTE_BACKEND(MaxTexture2DLayeredLayers),
  MaxTextureCubemapLayeredLayers =
    GCXX_ATTRIBUTE_BACKEND(MaxTextureCubemapLayeredLayers),
  MaxTimelineSemaphoreInteropSupported =
    GCXX_ATTRIBUTE_BACKEND(MaxTimelineSemaphoreInteropSupported),
  MemSyncDomainCount = GCXX_ATTRIBUTE_BACKEND(MemSyncDomainCount),
  MemoryPoolSupportedHandleTypes =
    GCXX_ATTRIBUTE_BACKEND(MemoryPoolSupportedHandleTypes),
  MpsEnabled  = GCXX_ATTRIBUTE_BACKEND(MpsEnabled),
  NumaConfig  = GCXX_ATTRIBUTE_BACKEND(NumaConfig),
  NumaId      = GCXX_ATTRIBUTE_BACKEND(NumaId),
  Reserved122 = GCXX_ATTRIBUTE_BACKEND(Reserved122),
  Reserved123 = GCXX_ATTRIBUTE_BACKEND(Reserved123),
  Reserved124 = GCXX_ATTRIBUTE_BACKEND(Reserved124),
  Reserved127 = GCXX_ATTRIBUTE_BACKEND(Reserved127),
  Reserved128 = GCXX_ATTRIBUTE_BACKEND(Reserved128),
  Reserved129 = GCXX_ATTRIBUTE_BACKEND(Reserved129),
  Reserved132 = GCXX_ATTRIBUTE_BACKEND(Reserved132),
  Reserved141 = GCXX_ATTRIBUTE_BACKEND(Reserved141),
  Reserved92  = GCXX_ATTRIBUTE_BACKEND(Reserved92),
  Reserved93  = GCXX_ATTRIBUTE_BACKEND(Reserved93),
  ReservedSharedMemoryPerBlock =
    GCXX_ATTRIBUTE_BACKEND(ReservedSharedMemoryPerBlock),
  SparseCudaArraySupported = GCXX_ATTRIBUTE_BACKEND(SparseCudaArraySupported),
  TimelineSemaphoreInteropSupported =
    GCXX_ATTRIBUTE_BACKEND(TimelineSemaphoreInteropSupported),
  VulkanCigSupported = GCXX_ATTRIBUTE_BACKEND(VulkanCigSupported),
#endif  // GCXX_CUDA_MODE

#if GCXX_HIP_MODE

#endif  // GCXX_HIP_MODE
};


GCXX_NAMESPACE_MAIN_FLAGS_END

#endif