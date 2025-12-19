#pragma once
#ifndef GCXX_RUNTIME_DEVICE_DEVICE_STRUCTS_HPP_
#define GCXX_RUNTIME_DEVICE_DEVICE_STRUCTS_HPP_

#include <string_view>

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

#include <gcxx/runtime/memory/span/span.hpp>


GCXX_NAMESPACE_MAIN_BEGIN


#if GCXX_CUDA_MODE
using DeviceProp = cudaDeviceProp;
#elif GCXX_HIP_MODE
using DeviceProp = hipDeviceProp_t;
#else
#error "Some horrible UB is happening now"
#endif

// struct uuid_st {
//   char bytes[16];
// };

// struct DeviceProp {
//   std::string_view name;
//   uuid_st uuid;
//   std::string_view luid;
//   unsigned int luidDeviceNodeMask;
//   std::size_t totalGlobalMem;
//   std::size_t sharedMemPerBlock;
//   int regsPerBlock;
//   int warpSize;
//   std::size_t memPitch;
//   int maxThreadsPerBlock;
//   gcxx::span<int, 3> maxThreadsDim;
//   gcxx::span<int, 3> maxGridSize;
//   int clockRate;
//   std::size_t totalConstMem;
//   int major;
//   int minor;
//   std::size_t textureAlignment;
//   std::size_t texturePitchAlignment;
//   int deviceOverlap;
//   int multiProcessorCount;
//   int kernelExecTimeoutEnabled;
//   int integrated;
//   int canMapHostMemory;
//   int computeMode;
//   int maxTexture1D;
//   int maxTexture1DMipmap;
//   int maxTexture1DLinear;
//   gcxx::span<int, 2> maxTexture2D;
//   gcxx::span<int, 2> maxTexture2DMipmap;
//   gcxx::span<int, 3> maxTexture2DLinear;
//   gcxx::span<int, 2> maxTexture2DGather;
//   gcxx::span<int, 3> maxTexture3D;
//   gcxx::span<int, 3> maxTexture3DAlt;
//   int maxTextureCubemap;
//   gcxx::span<int, 2> maxTexture1DLayered;
//   gcxx::span<int, 3> maxTexture2DLayered;
//   gcxx::span<int, 2> maxTextureCubemapLayered;
//   int maxSurface1D;
//   gcxx::span<int, 2> maxSurface2D;
//   gcxx::span<int, 3> maxSurface3D;
//   gcxx::span<int, 2> maxSurface1DLayered;
//   gcxx::span<int, 3> maxSurface2DLayered;
//   int maxSurfaceCubemap;
//   gcxx::span<int, 2> maxSurfaceCubemapLayered;
//   std::size_t surfaceAlignment;
//   int concurrentKernels;
//   int ECCEnabled;
//   int pciBusID;
//   int pciDeviceID;
//   int pciDomainID;
//   int tccDriver;
//   int asyncEngineCount;
//   int unifiedAddressing;
//   int memoryClockRate;
//   int memoryBusWidth;
//   int l2CacheSize;
//   int persistingL2CacheMaxSize;
//   int maxThreadsPerMultiProcessor;
//   int streamPrioritiesSupported;
//   int globalL1CacheSupported;
//   int localL1CacheSupported;
//   std::size_t sharedMemPerMultiprocessor;
//   int regsPerMultiprocessor;
//   int managedMemory;
//   int isMultiGpuBoard;
//   int multiGpuBoardGroupID;
//   int hostNativeAtomicSupported;
//   int singleToDoublePrecisionPerfRatio;
//   int pageableMemoryAccess;
//   int concurrentManagedAccess;
//   int computePreemptionSupported;
//   int canUseHostPointerForRegisteredMem;
//   int cooperativeLaunch;
//   int cooperativeMultiDeviceLaunch;
//   std::size_t sharedMemPerBlockOptin;
//   int pageableMemoryAccessUsesHostPageTables;
//   int directManagedMemAccessFromHost;
//   int maxBlocksPerMultiProcessor;
//   int accessPolicyMaxWindowSize;
//   std::size_t reservedSharedMemPerBlock;
//   int hostRegisterSupported;
//   int sparseCudaArraySupported;
//   int hostRegisterReadOnlySupported;
//   int timelineSemaphoreInteropSupported;
//   int memoryPoolsSupported;
//   int gpuDirectRDMASupported;
//   unsigned int gpuDirectRDMAFlushWritesOptions;
//   int gpuDirectRDMAWritesOrdering;
//   unsigned int memoryPoolSupportedHandleTypes;
//   int deferredMappingCudaArraySupported;
//   int ipcEventSupported;
//   int clusterLaunch;
//   int unifiedFunctionPointers;

//   GCXX_FH explicit DeviceProp(details_::deviceProp_t& raw_prop)
//       : name(raw_prop.name),
//         uuid(raw_prop.uuid),
//         luid(raw_prop.luid),
//         luidDeviceNodeMask(raw_prop.luidDeviceNodeMask),
//         totalGlobalMem(raw_prop.totalGlobalMem),
//         sharedMemPerBlock(raw_prop.sharedMemPerBlock),
//         regsPerBlock(raw_prop.regsPerBlock),
//         warpSize(raw_prop.warpSize),
//         memPitch(raw_prop.memPitch),
//         maxThreadsPerBlock(raw_prop.maxThreadsPerBlock),
//         maxThreadsDim(raw_prop.maxThreadsDim),
//         maxGridSize(raw_prop.maxGridSize),
//         clockRate(raw_prop.clockRate),
//         totalConstMem(raw_prop.totalConstMem),
//         major(raw_prop.major),
//         minor(raw_prop.minor),
//         textureAlignment(raw_prop.textureAlignment),
//         texturePitchAlignment(raw_prop.texturePitchAlignment),
//         deviceOverlap(raw_prop.deviceOverlap),
//         multiProcessorCount(raw_prop.multiProcessorCount),
//         kernelExecTimeoutEnabled(raw_prop.kernelExecTimeoutEnabled),
//         integrated(raw_prop.integrated),
//         canMapHostMemory(raw_prop.canMapHostMemory),
//         computeMode(raw_prop.computeMode),
//         maxTexture1D(raw_prop.maxTexture1D),
//         maxTexture1DMipmap(raw_prop.maxTexture1DMipmap),
//         maxTexture1DLinear(raw_prop.maxTexture1DLinear),
//         maxTexture2D(raw_prop.maxTexture2D),
//         maxTexture2DMipmap(raw_prop.maxTexture2DMipmap),
//         maxTexture2DLinear(raw_prop.maxTexture2DLinear),
//         maxTexture2DGather(raw_prop.maxTexture2DGather),
//         maxTexture3D(raw_prop.maxTexture3D),
//         maxTexture3DAlt(raw_prop.maxTexture3DAlt),
//         maxTextureCubemap(raw_prop.maxTextureCubemap),
//         maxTexture1DLayered(raw_prop.maxTexture1DLayered),
//         maxTexture2DLayered(raw_prop.maxTexture2DLayered),
//         maxTextureCubemapLayered(raw_prop.maxTextureCubemapLayered),
//         maxSurface1D(raw_prop.maxSurface1D),
//         maxSurface2D(raw_prop.maxSurface2D),
//         maxSurface3D(raw_prop.maxSurface3D),
//         maxSurface1DLayered(raw_prop.maxSurface1DLayered),
//         maxSurface2DLayered(raw_prop.maxSurface2DLayered),
//         maxSurfaceCubemap(raw_prop.maxSurfaceCubemap),
//         maxSurfaceCubemapLayered(raw_prop.maxSurfaceCubemapLayered),
//         surfaceAlignment(raw_prop.surfaceAlignment),
//         concurrentKernels(raw_prop.concurrentKernels),
//         ECCEnabled(raw_prop.ECCEnabled),
//         pciBusID(raw_prop.pciBusID),
//         pciDeviceID(raw_prop.pciDeviceID),
//         pciDomainID(raw_prop.pciDomainID),
//         tccDriver(raw_prop.tccDriver),
//         asyncEngineCount(raw_prop.asyncEngineCount),
//         unifiedAddressing(raw_prop.unifiedAddressing),
//         memoryClockRate(raw_prop.memoryClockRate),
//         memoryBusWidth(raw_prop.memoryBusWidth),
//         l2CacheSize(raw_prop.l2CacheSize),
//         persistingL2CacheMaxSize(raw_prop.persistingL2CacheMaxSize),
//         maxThreadsPerMultiProcessor(raw_prop.maxThreadsPerMultiProcessor),
//         streamPrioritiesSupported(raw_prop.streamPrioritiesSupported),
//         globalL1CacheSupported(raw_prop.globalL1CacheSupported),
//         localL1CacheSupported(raw_prop.localL1CacheSupported),
//         sharedMemPerMultiprocessor(raw_prop.sharedMemPerMultiprocessor),
//         regsPerMultiprocessor(raw_prop.regsPerMultiprocessor),
//         managedMemory(raw_prop.managedMemory),
//         isMultiGpuBoard(raw_prop.isMultiGpuBoard),
//         multiGpuBoardGroupID(raw_prop.multiGpuBoardGroupID),
//         hostNativeAtomicSupported(raw_prop.hostNativeAtomicSupported),
//         singleToDoublePrecisionPerfRatio(
//           raw_prop.singleToDoublePrecisionPerfRatio),
//         pageableMemoryAccess(raw_prop.pageableMemoryAccess),
//         concurrentManagedAccess(raw_prop.concurrentManagedAccess),
//         computePreemptionSupported(raw_prop.computePreemptionSupported),
//         canUseHostPointerForRegisteredMem(
//           raw_prop.canUseHostPointerForRegisteredMem),
//         cooperativeLaunch(raw_prop.cooperativeLaunch),
//         cooperativeMultiDeviceLaunch(raw_prop.cooperativeMultiDeviceLaunch),
//         sharedMemPerBlockOptin(raw_prop.sharedMemPerBlockOptin),
//         pageableMemoryAccessUsesHostPageTables(
//           raw_prop.pageableMemoryAccessUsesHostPageTables),
//         directManagedMemAccessFromHost(raw_prop.directManagedMemAccessFromHost),
//         maxBlocksPerMultiProcessor(raw_prop.maxBlocksPerMultiProcessor),
//         accessPolicyMaxWindowSize(raw_prop.accessPolicyMaxWindowSize),
//         reservedSharedMemPerBlock(raw_prop.reservedSharedMemPerBlock),
//         hostRegisterSupported(raw_prop.hostRegisterSupported),
//         sparseCudaArraySupported(raw_prop.sparseCudaArraySupported),
//         hostRegisterReadOnlySupported(raw_prop.hostRegisterReadOnlySupported),
//         timelineSemaphoreInteropSupported(
//           raw_prop.timelineSemaphoreInteropSupported),
//         memoryPoolsSupported(raw_prop.memoryPoolsSupported),
//         gpuDirectRDMASupported(raw_prop.gpuDirectRDMASupported),
//         gpuDirectRDMAFlushWritesOptions(
//           raw_prop.gpuDirectRDMAFlushWritesOptions),
//         gpuDirectRDMAWritesOrdering(raw_prop.gpuDirectRDMAWritesOrdering),
//         memoryPoolSupportedHandleTypes(raw_prop.memoryPoolSupportedHandleTypes),
//         deferredMappingCudaArraySupported(
//           raw_prop.deferredMappingCudaArraySupported),
//         ipcEventSupported(raw_prop.ipcEventSupported),
//         clusterLaunch(raw_prop.clusterLaunch),
//         unifiedFunctionPointers(raw_prop.unifiedFunctionPointers),
//   {}
// };

GCXX_NAMESPACE_MAIN_END


#endif