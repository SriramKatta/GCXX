// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Typed device-attribute / device-limit descriptors. This ports the pool_attr
// idea (see runtime/memory/mempool/memory_pool_attributes.hpp) to the
// device surface: instead of Device::getAttribute(enum) returning a raw int for
// everything, each attribute is a constexpr descriptor whose ::type is the
// right C++ type — bool for the boolean attributes, int for the numeric ones —
// so Device::attribute(desc) returns the typed value directly.
//
// Device attributes (cudaDeviceGetAttribute) are read-only and always stored as
// int by CUDA, so device_attr_impl fixes the storage type at int and only the
// exposed value type varies. Device limits (cudaDeviceGetLimit/SetLimit) are
// get/set and always size_t, and operate on the *current* device.

#pragma once
#ifndef GCXX_RUNTIME_DEVICE_DEVICE_ATTRIBUTES_HPP_
#define GCXX_RUNTIME_DEVICE_DEVICE_ATTRIBUTES_HPP_

#include <cstddef>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/flags/device_flags.hpp>
#include <gcxx/runtime_backend/backend_device.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

// ── device_attr: read-only device attribute descriptor ──────────────────────
// Attr        — the flags::deviceAttribute this describes.
// ValueType   — what the user sees (int by default; bool for the flags below).
// CUDA stores every attribute as int, so the storage type is fixed at int.
template <flags::deviceAttribute Attr, typename ValueType>
struct device_attr_impl {
  using type = ValueType;

  static inline constexpr flags::deviceAttribute attribute = Attr;

  // TODO : MAY NOT BE RIGHT BUT WORKS FOR NOW
  // Implicit conversion to the underlying enum value.
  GCXX_FH constexpr operator flags::deviceAttribute() const noexcept {
    // NOLINT(google-explicit-constructor)
    return Attr;
  }

  // Read the attribute for the given device (cudaDeviceGetAttribute takes the
  // device ordinal explicitly, so no current-device dependency).
  GCXX_FH auto operator()(int device) const -> type {
    return static_cast<type>(driver::deviceGetAttribute(
      static_cast<driver::deviceAttribute_t>(Attr), device));
  }
};

// Default: a numeric attribute exposed as int.
template <flags::deviceAttribute Attr>
struct device_attr : device_attr_impl<Attr, int> {};


#define GCXX_DEVICE_ATTR_SPECIALIZATION(DEV_ATTR_FLAG, TYPE) \
  template <>                                                \
  struct device_attr<flags::deviceAttribute::DEV_ATTR_FLAG>  \
      : device_attr_impl<flags::deviceAttribute::DEV_ATTR_FLAG, TYPE> {}


// Boolean attributes — CUDA stores 0/1, expose as bool. One specialization per
// documented boolean attribute; guards mirror flags::deviceAttribute exactly.
GCXX_DEVICE_ATTR_SPECIALIZATION(CanMapHostMemory, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(CanUseHostPointerForRegisteredMem, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(ComputePreemptionSupported, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(ConcurrentKernels, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(ConcurrentManagedAccess, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(CooperativeLaunch, bool);
#if GCXX_CUDA_VERSION_LESS_THAN(13, 0, 0)
GCXX_DEVICE_ATTR_SPECIALIZATION(CooperativeMultiDeviceLaunch, bool);
#endif
GCXX_DEVICE_ATTR_SPECIALIZATION(DirectManagedMemAccessFromHost, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(EccEnabled, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(GlobalL1CacheSupported, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(HostNativeAtomicSupported, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(HostRegisterSupported, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(Integrated, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(IsMultiGpuBoard, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(KernelExecTimeout, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(LocalL1CacheSupported, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(ManagedMemory, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(MemoryPoolsSupported, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(PageableMemoryAccess, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(PageableMemoryAccessUsesHostPageTables, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(StreamPrioritiesSupported, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(TccDriver, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(UnifiedAddressing, bool);

#if GCXX_CUDA_MODE()
// CUDA-mode-only boolean attributes.
GCXX_DEVICE_ATTR_SPECIALIZATION(CanFlushRemoteWrites, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(ClusterLaunch, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(D3D12CigSupported, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(DeferredMappingCudaArraySupported, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(GPUDirectRDMASupported, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(HostNumaMultinodeIpcSupported, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(HostRegisterReadOnlySupported, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(IpcEventSupport, bool);
#if GCXX_CUDA_VERSION_LESS_THAN(13, 0, 0)
GCXX_DEVICE_ATTR_SPECIALIZATION(MaxTimelineSemaphoreInteropSupported, bool);
#endif
GCXX_DEVICE_ATTR_SPECIALIZATION(MpsEnabled, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(SparseCudaArraySupported, bool);

#if GCXX_CUDA_VERSION_GREATER_EQUAL(12, 9, 0)
GCXX_DEVICE_ATTR_SPECIALIZATION(HostNumaMemoryPoolsSupported, bool);
GCXX_DEVICE_ATTR_SPECIALIZATION(VulkanCigSupported, bool);
#endif
#endif  // GCXX_CUDA_MODE()

#undef GCXX_DEVICE_ATTR_SPECIALIZATION


// dev_attr: named, typed, constexpr attribute objects passed to
// Device::attribute() / DeviceHandle::attribute(). Covers the usable device
// attributes; the MaxTexture*/MaxSurface* dimension families are omitted (they
// belong to the not-yet-wrapped texture/surface-object modules) and the
// Reserved* slots are skipped. Any attribute is still reachable directly as
// device_attr<flags::deviceAttribute::Name>{}.
namespace dev_attr {

#define GCXX_DEVICE_ATTR_DEFINE(ATTR_MEMBER, NAME)                   \
  using NAME##_t = device_attr<flags::deviceAttribute::ATTR_MEMBER>; \
  static inline constexpr NAME##_t NAME {}

  GCXX_DEVICE_ATTR_DEFINE(AsyncEngineCount, async_engine_count);

  GCXX_DEVICE_ATTR_DEFINE(CanMapHostMemory, can_map_host_memory);

  GCXX_DEVICE_ATTR_DEFINE(CanUseHostPointerForRegisteredMem,
                          can_use_host_pointer_for_registered_mem);

  GCXX_DEVICE_ATTR_DEFINE(ClockRate, clock_rate);

  GCXX_DEVICE_ATTR_DEFINE(ComputeCapabilityMajor, compute_capability_major);

  GCXX_DEVICE_ATTR_DEFINE(ComputeCapabilityMinor, compute_capability_minor);

  GCXX_DEVICE_ATTR_DEFINE(ComputeMode, compute_mode);

  GCXX_DEVICE_ATTR_DEFINE(ComputePreemptionSupported,
                          compute_preemption_supported);

  GCXX_DEVICE_ATTR_DEFINE(ConcurrentKernels, concurrent_kernels);

  GCXX_DEVICE_ATTR_DEFINE(ConcurrentManagedAccess, concurrent_managed_access);

  GCXX_DEVICE_ATTR_DEFINE(CooperativeLaunch, cooperative_launch);

#if GCXX_CUDA_VERSION_LESS_THAN(13, 0, 0)
  GCXX_DEVICE_ATTR_DEFINE(CooperativeMultiDeviceLaunch,
                          cooperative_multi_device_launch);
#endif

  GCXX_DEVICE_ATTR_DEFINE(DirectManagedMemAccessFromHost,
                          direct_managed_mem_access_from_host);

  GCXX_DEVICE_ATTR_DEFINE(EccEnabled, ecc_enabled);

  GCXX_DEVICE_ATTR_DEFINE(GlobalL1CacheSupported, global_l1_cache_supported);

  GCXX_DEVICE_ATTR_DEFINE(GlobalMemoryBusWidth, global_memory_bus_width);

  GCXX_DEVICE_ATTR_DEFINE(GpuOverlap, gpu_overlap);

  GCXX_DEVICE_ATTR_DEFINE(HostNativeAtomicSupported,
                          host_native_atomic_supported);

  GCXX_DEVICE_ATTR_DEFINE(HostRegisterSupported, host_register_supported);

  GCXX_DEVICE_ATTR_DEFINE(Integrated, integrated);

  GCXX_DEVICE_ATTR_DEFINE(IsMultiGpuBoard, is_multi_gpu_board);

  GCXX_DEVICE_ATTR_DEFINE(KernelExecTimeout, kernel_exec_timeout);

  GCXX_DEVICE_ATTR_DEFINE(L2CacheSize, l2_cache_size);

  GCXX_DEVICE_ATTR_DEFINE(LocalL1CacheSupported, local_l1_cache_supported);

  GCXX_DEVICE_ATTR_DEFINE(ManagedMemory, managed_memory);

  GCXX_DEVICE_ATTR_DEFINE(MaxBlockDimX, max_block_dim_x);

  GCXX_DEVICE_ATTR_DEFINE(MaxBlockDimY, max_block_dim_y);

  GCXX_DEVICE_ATTR_DEFINE(MaxBlockDimZ, max_block_dim_z);

  GCXX_DEVICE_ATTR_DEFINE(MaxBlocksPerMultiprocessor,
                          max_blocks_per_multiprocessor);

  GCXX_DEVICE_ATTR_DEFINE(MaxGridDimX, max_grid_dim_x);

  GCXX_DEVICE_ATTR_DEFINE(MaxGridDimY, max_grid_dim_y);

  GCXX_DEVICE_ATTR_DEFINE(MaxGridDimZ, max_grid_dim_z);

  GCXX_DEVICE_ATTR_DEFINE(MaxPitch, max_pitch);

  GCXX_DEVICE_ATTR_DEFINE(MaxRegistersPerBlock, max_registers_per_block);

  GCXX_DEVICE_ATTR_DEFINE(MaxRegistersPerMultiprocessor,
                          max_registers_per_multiprocessor);

  GCXX_DEVICE_ATTR_DEFINE(MaxSharedMemoryPerBlock, max_shared_memory_per_block);

  GCXX_DEVICE_ATTR_DEFINE(MaxSharedMemoryPerBlockOptin,
                          max_shared_memory_per_block_optin);

  GCXX_DEVICE_ATTR_DEFINE(MaxSharedMemoryPerMultiprocessor,
                          max_shared_memory_per_multiprocessor);

  GCXX_DEVICE_ATTR_DEFINE(MaxThreadsPerBlock, max_threads_per_block);

  GCXX_DEVICE_ATTR_DEFINE(MaxThreadsPerMultiProcessor,
                          max_threads_per_multi_processor);

  GCXX_DEVICE_ATTR_DEFINE(MemoryClockRate, memory_clock_rate);

  GCXX_DEVICE_ATTR_DEFINE(MemoryPoolsSupported, memory_pools_supported);

  GCXX_DEVICE_ATTR_DEFINE(MultiGpuBoardGroupID, multi_gpu_board_group_id);

  GCXX_DEVICE_ATTR_DEFINE(MultiProcessorCount, multiprocessor_count);

  GCXX_DEVICE_ATTR_DEFINE(PageableMemoryAccess, pageable_memory_access);

  GCXX_DEVICE_ATTR_DEFINE(PageableMemoryAccessUsesHostPageTables,
                          pageable_memory_access_uses_host_page_tables);

  GCXX_DEVICE_ATTR_DEFINE(PciBusId, pci_bus_id);

  GCXX_DEVICE_ATTR_DEFINE(PciDeviceId, pci_device_id);

  GCXX_DEVICE_ATTR_DEFINE(PciDomainId, pci_domain_id);

  GCXX_DEVICE_ATTR_DEFINE(SingleToDoublePrecisionPerfRatio,
                          single_to_double_precision_perf_ratio);

  GCXX_DEVICE_ATTR_DEFINE(StreamPrioritiesSupported,
                          stream_priorities_supported);

  GCXX_DEVICE_ATTR_DEFINE(TccDriver, tcc_driver);

  GCXX_DEVICE_ATTR_DEFINE(TotalConstantMemory, total_constant_memory);

  GCXX_DEVICE_ATTR_DEFINE(UnifiedAddressing, unified_addressing);

  GCXX_DEVICE_ATTR_DEFINE(WarpSize, warp_size);

#if GCXX_CUDA_MODE()
  // CUDA-mode-only attributes.
  GCXX_DEVICE_ATTR_DEFINE(CanFlushRemoteWrites, can_flush_remote_writes);

  GCXX_DEVICE_ATTR_DEFINE(ClusterLaunch, cluster_launch);

  GCXX_DEVICE_ATTR_DEFINE(D3D12CigSupported, d3d12_cig_supported);

  GCXX_DEVICE_ATTR_DEFINE(DeferredMappingCudaArraySupported,
                          deferred_mapping_cuda_array_supported);

  GCXX_DEVICE_ATTR_DEFINE(GPUDirectRDMAFlushWritesOptions,
                          gpu_direct_rdma_flush_writes_options);

  GCXX_DEVICE_ATTR_DEFINE(GPUDirectRDMASupported, gpu_direct_rdma_supported);

  GCXX_DEVICE_ATTR_DEFINE(GPUDirectRDMAWritesOrdering,
                          gpu_direct_rdma_writes_ordering);

  GCXX_DEVICE_ATTR_DEFINE(GpuPciDeviceId, gpu_pci_device_id);

  GCXX_DEVICE_ATTR_DEFINE(GpuPciSubsystemId, gpu_pci_subsystem_id);

  GCXX_DEVICE_ATTR_DEFINE(HostNumaId, host_numa_id);

  GCXX_DEVICE_ATTR_DEFINE(HostNumaMultinodeIpcSupported,
                          host_numa_multinode_ipc_supported);

  GCXX_DEVICE_ATTR_DEFINE(HostRegisterReadOnlySupported,
                          host_register_read_only_supported);

  GCXX_DEVICE_ATTR_DEFINE(IpcEventSupport, ipc_event_support);

  GCXX_DEVICE_ATTR_DEFINE(Max, max);

  GCXX_DEVICE_ATTR_DEFINE(MaxAccessPolicyWindowSize,
                          max_access_policy_window_size);

  GCXX_DEVICE_ATTR_DEFINE(MaxPersistingL2CacheSize,
                          max_persisting_l2_cache_size);

#if GCXX_CUDA_VERSION_LESS_THAN(13, 0, 0)
  GCXX_DEVICE_ATTR_DEFINE(MaxTimelineSemaphoreInteropSupported,
                          max_timeline_semaphore_interop_supported);
#endif

  GCXX_DEVICE_ATTR_DEFINE(MemSyncDomainCount, mem_sync_domain_count);

  GCXX_DEVICE_ATTR_DEFINE(MemoryPoolSupportedHandleTypes,
                          memory_pool_supported_handle_types);

  GCXX_DEVICE_ATTR_DEFINE(MpsEnabled, mps_enabled);

  GCXX_DEVICE_ATTR_DEFINE(NumaConfig, numa_config);

  GCXX_DEVICE_ATTR_DEFINE(NumaId, numa_id);

  GCXX_DEVICE_ATTR_DEFINE(SparseCudaArraySupported,
                          sparse_cuda_array_supported);

  GCXX_DEVICE_ATTR_DEFINE(TimelineSemaphoreInteropSupported,
                          timeline_semaphore_interop_supported);

#if GCXX_CUDA_VERSION_GREATER_EQUAL(12, 9, 0)
  GCXX_DEVICE_ATTR_DEFINE(HostNumaMemoryPoolsSupported,
                          host_numa_memory_pools_supported);

  GCXX_DEVICE_ATTR_DEFINE(VulkanCigSupported, vulkan_cig_supported);
#endif
#endif  // GCXX_CUDA_MODE()

#undef GCXX_DEVICE_ATTR_DEFINE

}  // namespace dev_attr

// ── device_limit: get/set device limit descriptor ───────────────────────────
// All device limits are size_t and operate on the *current* device
// (cudaDeviceGetLimit/SetLimit take no device ordinal), so the value/storage
// types are fixed at size_t and operator() takes no argument.
template <flags::deviceLimit Attr>
struct device_limit_impl {
  using type = std::size_t;

  static inline constexpr flags::deviceLimit attribute = Attr;

  GCXX_FH constexpr operator flags::deviceLimit()
    const noexcept {  // NOLINT(google-explicit-constructor)
    return Attr;
  }

  /// Read the limit for the current device.
  GCXX_FH auto operator()() const -> type {
    return driver::deviceGetLimit(static_cast<driver::deviceLimit_t>(Attr));
  }

  /// Write the limit for the current device.
  GCXX_FH static auto set(type value) -> void {
    driver::deviceSetLimit(static_cast<driver::deviceLimit_t>(Attr), value);
  }
};

template <flags::deviceLimit Attr>
struct device_limit : device_limit_impl<Attr> {};

// device_limits: named, typed, constexpr limit objects passed to
// Device::limit() / Device::set_limit() / DeviceHandle equivalents.
namespace device_limits {

#define GCXX_DEVICE_LIMIT_DEFINE(LIMIT_MEMBER, NAME)               \
  using NAME##_t = device_limit<flags::deviceLimit::LIMIT_MEMBER>; \
  static inline constexpr NAME##_t NAME {}

  GCXX_DEVICE_LIMIT_DEFINE(StackSize, stack_size);

  
  GCXX_DEVICE_LIMIT_DEFINE(MallocHeapSize, malloc_heap_size);
  
#if GCXX_CUDA_MODE()
  GCXX_DEVICE_LIMIT_DEFINE(PrintfFifoSize, printf_fifo_size);

  GCXX_DEVICE_LIMIT_DEFINE(DevRuntimeSyncDepth, dev_runtime_sync_depth);

  GCXX_DEVICE_LIMIT_DEFINE(DevRuntimePendingLaunchCount,
                           dev_runtime_pending_launch_count);

  GCXX_DEVICE_LIMIT_DEFINE(MaxL2FetchGranularity, max_l2_fetch_granularity);

  GCXX_DEVICE_LIMIT_DEFINE(PersistingL2CacheSize, persisting_l2_cache_size);
#endif  // GCXX_CUDA_MODE()

#undef GCXX_DEVICE_LIMIT_DEFINE

}  // namespace device_limits

GCXX_NAMESPACE_MAIN_END()

#endif
