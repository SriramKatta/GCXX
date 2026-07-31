// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Typed device-attribute / device-limit descriptors. This ports the pool_attr
// idea (see runtime/memory/memory_pool/memory_pool_attributes.hpp) to the
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

  static constexpr flags::deviceAttribute attribute = Attr;

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

// Boolean attributes — CUDA stores 0/1, expose as bool. One specialization per
// documented boolean attribute; guards mirror flags::deviceAttribute exactly.
template <>
struct device_attr<flags::deviceAttribute::CanMapHostMemory>
    : device_attr_impl<flags::deviceAttribute::CanMapHostMemory, bool> {};
template <>
struct device_attr<flags::deviceAttribute::CanUseHostPointerForRegisteredMem>
    : device_attr_impl<
        flags::deviceAttribute::CanUseHostPointerForRegisteredMem, bool> {};
template <>
struct device_attr<flags::deviceAttribute::ComputePreemptionSupported>
    : device_attr_impl<flags::deviceAttribute::ComputePreemptionSupported,
                       bool> {};
template <>
struct device_attr<flags::deviceAttribute::ConcurrentKernels>
    : device_attr_impl<flags::deviceAttribute::ConcurrentKernels, bool> {};
template <>
struct device_attr<flags::deviceAttribute::ConcurrentManagedAccess>
    : device_attr_impl<flags::deviceAttribute::ConcurrentManagedAccess, bool> {
};
template <>
struct device_attr<flags::deviceAttribute::CooperativeLaunch>
    : device_attr_impl<flags::deviceAttribute::CooperativeLaunch, bool> {};
#if GCXX_CUDA_VERSION_LESS_THAN(13, 0, 0)
template <>
struct device_attr<flags::deviceAttribute::CooperativeMultiDeviceLaunch>
    : device_attr_impl<flags::deviceAttribute::CooperativeMultiDeviceLaunch,
                       bool> {};
#endif
template <>
struct device_attr<flags::deviceAttribute::DirectManagedMemAccessFromHost>
    : device_attr_impl<flags::deviceAttribute::DirectManagedMemAccessFromHost,
                       bool> {};
template <>
struct device_attr<flags::deviceAttribute::EccEnabled>
    : device_attr_impl<flags::deviceAttribute::EccEnabled, bool> {};
template <>
struct device_attr<flags::deviceAttribute::GlobalL1CacheSupported>
    : device_attr_impl<flags::deviceAttribute::GlobalL1CacheSupported, bool> {};
template <>
struct device_attr<flags::deviceAttribute::HostNativeAtomicSupported>
    : device_attr_impl<flags::deviceAttribute::HostNativeAtomicSupported,
                       bool> {};
template <>
struct device_attr<flags::deviceAttribute::HostRegisterSupported>
    : device_attr_impl<flags::deviceAttribute::HostRegisterSupported, bool> {};
template <>
struct device_attr<flags::deviceAttribute::Integrated>
    : device_attr_impl<flags::deviceAttribute::Integrated, bool> {};
template <>
struct device_attr<flags::deviceAttribute::IsMultiGpuBoard>
    : device_attr_impl<flags::deviceAttribute::IsMultiGpuBoard, bool> {};
template <>
struct device_attr<flags::deviceAttribute::KernelExecTimeout>
    : device_attr_impl<flags::deviceAttribute::KernelExecTimeout, bool> {};
template <>
struct device_attr<flags::deviceAttribute::LocalL1CacheSupported>
    : device_attr_impl<flags::deviceAttribute::LocalL1CacheSupported, bool> {};
template <>
struct device_attr<flags::deviceAttribute::ManagedMemory>
    : device_attr_impl<flags::deviceAttribute::ManagedMemory, bool> {};
template <>
struct device_attr<flags::deviceAttribute::MemoryPoolsSupported>
    : device_attr_impl<flags::deviceAttribute::MemoryPoolsSupported, bool> {};
template <>
struct device_attr<flags::deviceAttribute::PageableMemoryAccess>
    : device_attr_impl<flags::deviceAttribute::PageableMemoryAccess, bool> {};
template <>
struct device_attr<
  flags::deviceAttribute::PageableMemoryAccessUsesHostPageTables>
    : device_attr_impl<
        flags::deviceAttribute::PageableMemoryAccessUsesHostPageTables, bool> {
};
template <>
struct device_attr<flags::deviceAttribute::StreamPrioritiesSupported>
    : device_attr_impl<flags::deviceAttribute::StreamPrioritiesSupported,
                       bool> {};
template <>
struct device_attr<flags::deviceAttribute::TccDriver>
    : device_attr_impl<flags::deviceAttribute::TccDriver, bool> {};
template <>
struct device_attr<flags::deviceAttribute::UnifiedAddressing>
    : device_attr_impl<flags::deviceAttribute::UnifiedAddressing, bool> {};

#if GCXX_CUDA_MODE()
// CUDA-mode-only boolean attributes.
template <>
struct device_attr<flags::deviceAttribute::CanFlushRemoteWrites>
    : device_attr_impl<flags::deviceAttribute::CanFlushRemoteWrites, bool> {};
template <>
struct device_attr<flags::deviceAttribute::ClusterLaunch>
    : device_attr_impl<flags::deviceAttribute::ClusterLaunch, bool> {};
template <>
struct device_attr<flags::deviceAttribute::D3D12CigSupported>
    : device_attr_impl<flags::deviceAttribute::D3D12CigSupported, bool> {};
template <>
struct device_attr<flags::deviceAttribute::DeferredMappingCudaArraySupported>
    : device_attr_impl<
        flags::deviceAttribute::DeferredMappingCudaArraySupported, bool> {};
template <>
struct device_attr<flags::deviceAttribute::GPUDirectRDMASupported>
    : device_attr_impl<flags::deviceAttribute::GPUDirectRDMASupported, bool> {};
template <>
struct device_attr<flags::deviceAttribute::HostNumaMultinodeIpcSupported>
    : device_attr_impl<flags::deviceAttribute::HostNumaMultinodeIpcSupported,
                       bool> {};
template <>
struct device_attr<flags::deviceAttribute::HostRegisterReadOnlySupported>
    : device_attr_impl<flags::deviceAttribute::HostRegisterReadOnlySupported,
                       bool> {};
template <>
struct device_attr<flags::deviceAttribute::IpcEventSupport>
    : device_attr_impl<flags::deviceAttribute::IpcEventSupport, bool> {};
#if GCXX_CUDA_VERSION_LESS_THAN(13, 0, 0)
template <>
struct device_attr<flags::deviceAttribute::MaxTimelineSemaphoreInteropSupported>
    : device_attr_impl<
        flags::deviceAttribute::MaxTimelineSemaphoreInteropSupported, bool> {};
#endif
template <>
struct device_attr<flags::deviceAttribute::MpsEnabled>
    : device_attr_impl<flags::deviceAttribute::MpsEnabled, bool> {};
template <>
struct device_attr<flags::deviceAttribute::SparseCudaArraySupported>
    : device_attr_impl<flags::deviceAttribute::SparseCudaArraySupported, bool> {
};

#if GCXX_CUDA_VERSION_GREATER_EQUAL(12, 9, 0)
template <>
struct device_attr<flags::deviceAttribute::HostNumaMemoryPoolsSupported>
    : device_attr_impl<flags::deviceAttribute::HostNumaMemoryPoolsSupported,
                       bool> {};
template <>
struct device_attr<flags::deviceAttribute::VulkanCigSupported>
    : device_attr_impl<flags::deviceAttribute::VulkanCigSupported, bool> {};
#endif
#endif  // GCXX_CUDA_MODE()

// dev_attr: named, typed, constexpr attribute objects passed to
// Device::attribute() / DeviceHandle::attribute(). Covers the usable device
// attributes; the MaxTexture*/MaxSurface* dimension families are omitted (they
// belong to the not-yet-wrapped texture/surface-object modules) and the
// Reserved* slots are skipped. Any attribute is still reachable directly as
// device_attr<flags::deviceAttribute::Name>{}.
namespace dev_attr {

  using async_engine_count_t =
    device_attr<flags::deviceAttribute::AsyncEngineCount>;
  static constexpr async_engine_count_t async_engine_count{};

  using can_map_host_memory_t =
    device_attr<flags::deviceAttribute::CanMapHostMemory>;
  static constexpr can_map_host_memory_t can_map_host_memory{};

  using can_use_host_pointer_for_registered_mem_t =
    device_attr<flags::deviceAttribute::CanUseHostPointerForRegisteredMem>;
  static constexpr can_use_host_pointer_for_registered_mem_t
    can_use_host_pointer_for_registered_mem{};

  using clock_rate_t = device_attr<flags::deviceAttribute::ClockRate>;
  static constexpr clock_rate_t clock_rate{};

  using compute_capability_major_t =
    device_attr<flags::deviceAttribute::ComputeCapabilityMajor>;
  static constexpr compute_capability_major_t compute_capability_major{};

  using compute_capability_minor_t =
    device_attr<flags::deviceAttribute::ComputeCapabilityMinor>;
  static constexpr compute_capability_minor_t compute_capability_minor{};

  using compute_mode_t = device_attr<flags::deviceAttribute::ComputeMode>;
  static constexpr compute_mode_t compute_mode{};

  using compute_preemption_supported_t =
    device_attr<flags::deviceAttribute::ComputePreemptionSupported>;
  static constexpr compute_preemption_supported_t
    compute_preemption_supported{};

  using concurrent_kernels_t =
    device_attr<flags::deviceAttribute::ConcurrentKernels>;
  static constexpr concurrent_kernels_t concurrent_kernels{};

  using concurrent_managed_access_t =
    device_attr<flags::deviceAttribute::ConcurrentManagedAccess>;
  static constexpr concurrent_managed_access_t concurrent_managed_access{};

  using cooperative_launch_t =
    device_attr<flags::deviceAttribute::CooperativeLaunch>;
  static constexpr cooperative_launch_t cooperative_launch{};

#if GCXX_CUDA_VERSION_LESS_THAN(13, 0, 0)
  using cooperative_multi_device_launch_t =
    device_attr<flags::deviceAttribute::CooperativeMultiDeviceLaunch>;
  static constexpr cooperative_multi_device_launch_t
    cooperative_multi_device_launch{};
#endif

  using direct_managed_mem_access_from_host_t =
    device_attr<flags::deviceAttribute::DirectManagedMemAccessFromHost>;
  static constexpr direct_managed_mem_access_from_host_t
    direct_managed_mem_access_from_host{};

  using ecc_enabled_t = device_attr<flags::deviceAttribute::EccEnabled>;
  static constexpr ecc_enabled_t ecc_enabled{};

  using global_l1_cache_supported_t =
    device_attr<flags::deviceAttribute::GlobalL1CacheSupported>;
  static constexpr global_l1_cache_supported_t global_l1_cache_supported{};

  using global_memory_bus_width_t =
    device_attr<flags::deviceAttribute::GlobalMemoryBusWidth>;
  static constexpr global_memory_bus_width_t global_memory_bus_width{};

  using gpu_overlap_t = device_attr<flags::deviceAttribute::GpuOverlap>;
  static constexpr gpu_overlap_t gpu_overlap{};

  using host_native_atomic_supported_t =
    device_attr<flags::deviceAttribute::HostNativeAtomicSupported>;
  static constexpr host_native_atomic_supported_t
    host_native_atomic_supported{};

  using host_register_supported_t =
    device_attr<flags::deviceAttribute::HostRegisterSupported>;
  static constexpr host_register_supported_t host_register_supported{};

  using integrated_t = device_attr<flags::deviceAttribute::Integrated>;
  static constexpr integrated_t integrated{};

  using is_multi_gpu_board_t =
    device_attr<flags::deviceAttribute::IsMultiGpuBoard>;
  static constexpr is_multi_gpu_board_t is_multi_gpu_board{};

  using kernel_exec_timeout_t =
    device_attr<flags::deviceAttribute::KernelExecTimeout>;
  static constexpr kernel_exec_timeout_t kernel_exec_timeout{};

  using l2_cache_size_t = device_attr<flags::deviceAttribute::L2CacheSize>;
  static constexpr l2_cache_size_t l2_cache_size{};

  using local_l1_cache_supported_t =
    device_attr<flags::deviceAttribute::LocalL1CacheSupported>;
  static constexpr local_l1_cache_supported_t local_l1_cache_supported{};

  using managed_memory_t = device_attr<flags::deviceAttribute::ManagedMemory>;
  static constexpr managed_memory_t managed_memory{};

  using max_block_dim_x_t = device_attr<flags::deviceAttribute::MaxBlockDimX>;
  static constexpr max_block_dim_x_t max_block_dim_x{};

  using max_block_dim_y_t = device_attr<flags::deviceAttribute::MaxBlockDimY>;
  static constexpr max_block_dim_y_t max_block_dim_y{};

  using max_block_dim_z_t = device_attr<flags::deviceAttribute::MaxBlockDimZ>;
  static constexpr max_block_dim_z_t max_block_dim_z{};

  using max_blocks_per_multiprocessor_t =
    device_attr<flags::deviceAttribute::MaxBlocksPerMultiprocessor>;
  static constexpr max_blocks_per_multiprocessor_t
    max_blocks_per_multiprocessor{};

  using max_grid_dim_x_t = device_attr<flags::deviceAttribute::MaxGridDimX>;
  static constexpr max_grid_dim_x_t max_grid_dim_x{};

  using max_grid_dim_y_t = device_attr<flags::deviceAttribute::MaxGridDimY>;
  static constexpr max_grid_dim_y_t max_grid_dim_y{};

  using max_grid_dim_z_t = device_attr<flags::deviceAttribute::MaxGridDimZ>;
  static constexpr max_grid_dim_z_t max_grid_dim_z{};

  using max_pitch_t = device_attr<flags::deviceAttribute::MaxPitch>;
  static constexpr max_pitch_t max_pitch{};

  using max_registers_per_block_t =
    device_attr<flags::deviceAttribute::MaxRegistersPerBlock>;
  static constexpr max_registers_per_block_t max_registers_per_block{};

  using max_registers_per_multiprocessor_t =
    device_attr<flags::deviceAttribute::MaxRegistersPerMultiprocessor>;
  static constexpr max_registers_per_multiprocessor_t
    max_registers_per_multiprocessor{};

  using max_shared_memory_per_block_t =
    device_attr<flags::deviceAttribute::MaxSharedMemoryPerBlock>;
  static constexpr max_shared_memory_per_block_t max_shared_memory_per_block{};

  using max_shared_memory_per_block_optin_t =
    device_attr<flags::deviceAttribute::MaxSharedMemoryPerBlockOptin>;
  static constexpr max_shared_memory_per_block_optin_t
    max_shared_memory_per_block_optin{};

  using max_shared_memory_per_multiprocessor_t =
    device_attr<flags::deviceAttribute::MaxSharedMemoryPerMultiprocessor>;
  static constexpr max_shared_memory_per_multiprocessor_t
    max_shared_memory_per_multiprocessor{};

  using max_threads_per_block_t =
    device_attr<flags::deviceAttribute::MaxThreadsPerBlock>;
  static constexpr max_threads_per_block_t max_threads_per_block{};

  using max_threads_per_multi_processor_t =
    device_attr<flags::deviceAttribute::MaxThreadsPerMultiProcessor>;
  static constexpr max_threads_per_multi_processor_t
    max_threads_per_multi_processor{};

  using memory_clock_rate_t =
    device_attr<flags::deviceAttribute::MemoryClockRate>;
  static constexpr memory_clock_rate_t memory_clock_rate{};

  using memory_pools_supported_t =
    device_attr<flags::deviceAttribute::MemoryPoolsSupported>;
  static constexpr memory_pools_supported_t memory_pools_supported{};

  using multi_gpu_board_group_id_t =
    device_attr<flags::deviceAttribute::MultiGpuBoardGroupID>;
  static constexpr multi_gpu_board_group_id_t multi_gpu_board_group_id{};

  using multiprocessor_count_t =
    device_attr<flags::deviceAttribute::MultiProcessorCount>;
  static constexpr multiprocessor_count_t multiprocessor_count{};

  using pageable_memory_access_t =
    device_attr<flags::deviceAttribute::PageableMemoryAccess>;
  static constexpr pageable_memory_access_t pageable_memory_access{};

  using pageable_memory_access_uses_host_page_tables_t =
    device_attr<flags::deviceAttribute::PageableMemoryAccessUsesHostPageTables>;
  static constexpr pageable_memory_access_uses_host_page_tables_t
    pageable_memory_access_uses_host_page_tables{};

  using pci_bus_id_t = device_attr<flags::deviceAttribute::PciBusId>;
  static constexpr pci_bus_id_t pci_bus_id{};

  using pci_device_id_t = device_attr<flags::deviceAttribute::PciDeviceId>;
  static constexpr pci_device_id_t pci_device_id{};

  using pci_domain_id_t = device_attr<flags::deviceAttribute::PciDomainId>;
  static constexpr pci_domain_id_t pci_domain_id{};

  using single_to_double_precision_perf_ratio_t =
    device_attr<flags::deviceAttribute::SingleToDoublePrecisionPerfRatio>;
  static constexpr single_to_double_precision_perf_ratio_t
    single_to_double_precision_perf_ratio{};

  using stream_priorities_supported_t =
    device_attr<flags::deviceAttribute::StreamPrioritiesSupported>;
  static constexpr stream_priorities_supported_t stream_priorities_supported{};

  using tcc_driver_t = device_attr<flags::deviceAttribute::TccDriver>;
  static constexpr tcc_driver_t tcc_driver{};

  using total_constant_memory_t =
    device_attr<flags::deviceAttribute::TotalConstantMemory>;
  static constexpr total_constant_memory_t total_constant_memory{};

  using unified_addressing_t =
    device_attr<flags::deviceAttribute::UnifiedAddressing>;
  static constexpr unified_addressing_t unified_addressing{};

  using warp_size_t = device_attr<flags::deviceAttribute::WarpSize>;
  static constexpr warp_size_t warp_size{};

#if GCXX_CUDA_MODE()
  // CUDA-mode-only attributes.
  using can_flush_remote_writes_t =
    device_attr<flags::deviceAttribute::CanFlushRemoteWrites>;
  static constexpr can_flush_remote_writes_t can_flush_remote_writes{};

  using cluster_launch_t = device_attr<flags::deviceAttribute::ClusterLaunch>;
  static constexpr cluster_launch_t cluster_launch{};

  using d3d12_cig_supported_t =
    device_attr<flags::deviceAttribute::D3D12CigSupported>;
  static constexpr d3d12_cig_supported_t d3d12_cig_supported{};

  using deferred_mapping_cuda_array_supported_t =
    device_attr<flags::deviceAttribute::DeferredMappingCudaArraySupported>;
  static constexpr deferred_mapping_cuda_array_supported_t
    deferred_mapping_cuda_array_supported{};

  using gpu_direct_rdma_flush_writes_options_t =
    device_attr<flags::deviceAttribute::GPUDirectRDMAFlushWritesOptions>;
  static constexpr gpu_direct_rdma_flush_writes_options_t
    gpu_direct_rdma_flush_writes_options{};

  using gpu_direct_rdma_supported_t =
    device_attr<flags::deviceAttribute::GPUDirectRDMASupported>;
  static constexpr gpu_direct_rdma_supported_t gpu_direct_rdma_supported{};

  using gpu_direct_rdma_writes_ordering_t =
    device_attr<flags::deviceAttribute::GPUDirectRDMAWritesOrdering>;
  static constexpr gpu_direct_rdma_writes_ordering_t
    gpu_direct_rdma_writes_ordering{};

  using gpu_pci_device_id_t =
    device_attr<flags::deviceAttribute::GpuPciDeviceId>;
  static constexpr gpu_pci_device_id_t gpu_pci_device_id{};

  using gpu_pci_subsystem_id_t =
    device_attr<flags::deviceAttribute::GpuPciSubsystemId>;
  static constexpr gpu_pci_subsystem_id_t gpu_pci_subsystem_id{};

  using host_numa_id_t = device_attr<flags::deviceAttribute::HostNumaId>;
  static constexpr host_numa_id_t host_numa_id{};

  using host_numa_multinode_ipc_supported_t =
    device_attr<flags::deviceAttribute::HostNumaMultinodeIpcSupported>;
  static constexpr host_numa_multinode_ipc_supported_t
    host_numa_multinode_ipc_supported{};

  using host_register_read_only_supported_t =
    device_attr<flags::deviceAttribute::HostRegisterReadOnlySupported>;
  static constexpr host_register_read_only_supported_t
    host_register_read_only_supported{};

  using ipc_event_support_t =
    device_attr<flags::deviceAttribute::IpcEventSupport>;
  static constexpr ipc_event_support_t ipc_event_support{};

  using max_t = device_attr<flags::deviceAttribute::Max>;
  static constexpr max_t max{};

  using max_access_policy_window_size_t =
    device_attr<flags::deviceAttribute::MaxAccessPolicyWindowSize>;
  static constexpr max_access_policy_window_size_t
    max_access_policy_window_size{};

  using max_persisting_l2_cache_size_t =
    device_attr<flags::deviceAttribute::MaxPersistingL2CacheSize>;
  static constexpr max_persisting_l2_cache_size_t
    max_persisting_l2_cache_size{};

#if GCXX_CUDA_VERSION_LESS_THAN(13, 0, 0)
  using max_timeline_semaphore_interop_supported_t =
    device_attr<flags::deviceAttribute::MaxTimelineSemaphoreInteropSupported>;
  static constexpr max_timeline_semaphore_interop_supported_t
    max_timeline_semaphore_interop_supported{};
#endif

  using mem_sync_domain_count_t =
    device_attr<flags::deviceAttribute::MemSyncDomainCount>;
  static constexpr mem_sync_domain_count_t mem_sync_domain_count{};

  using memory_pool_supported_handle_types_t =
    device_attr<flags::deviceAttribute::MemoryPoolSupportedHandleTypes>;
  static constexpr memory_pool_supported_handle_types_t
    memory_pool_supported_handle_types{};

  using mps_enabled_t = device_attr<flags::deviceAttribute::MpsEnabled>;
  static constexpr mps_enabled_t mps_enabled{};

  using numa_config_t = device_attr<flags::deviceAttribute::NumaConfig>;
  static constexpr numa_config_t numa_config{};

  using numa_id_t = device_attr<flags::deviceAttribute::NumaId>;
  static constexpr numa_id_t numa_id{};

  using sparse_cuda_array_supported_t =
    device_attr<flags::deviceAttribute::SparseCudaArraySupported>;
  static constexpr sparse_cuda_array_supported_t sparse_cuda_array_supported{};

  using timeline_semaphore_interop_supported_t =
    device_attr<flags::deviceAttribute::TimelineSemaphoreInteropSupported>;
  static constexpr timeline_semaphore_interop_supported_t
    timeline_semaphore_interop_supported{};

#if GCXX_CUDA_VERSION_GREATER_EQUAL(12, 9, 0)
  using host_numa_memory_pools_supported_t =
    device_attr<flags::deviceAttribute::HostNumaMemoryPoolsSupported>;
  static constexpr host_numa_memory_pools_supported_t
    host_numa_memory_pools_supported{};

  using vulkan_cig_supported_t =
    device_attr<flags::deviceAttribute::VulkanCigSupported>;
  static constexpr vulkan_cig_supported_t vulkan_cig_supported{};
#endif
#endif  // GCXX_CUDA_MODE()

}  // namespace dev_attr

// ── device_limit: get/set device limit descriptor ───────────────────────────
// All device limits are size_t and operate on the *current* device
// (cudaDeviceGetLimit/SetLimit take no device ordinal), so the value/storage
// types are fixed at size_t and operator() takes no argument.
template <flags::deviceLimit Attr>
struct device_limit_impl {
  using type = std::size_t;

  static constexpr flags::deviceLimit attribute = Attr;

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

  using stack_size_t = device_limit<flags::deviceLimit::StackSize>;
  static constexpr stack_size_t stack_size{};

  using printf_fifo_size_t = device_limit<flags::deviceLimit::PrintfFifoSize>;
  static constexpr printf_fifo_size_t printf_fifo_size{};

  using malloc_heap_size_t = device_limit<flags::deviceLimit::MallocHeapSize>;
  static constexpr malloc_heap_size_t malloc_heap_size{};

#if GCXX_CUDA_MODE()
  using dev_runtime_sync_depth_t =
    device_limit<flags::deviceLimit::DevRuntimeSyncDepth>;
  static constexpr dev_runtime_sync_depth_t dev_runtime_sync_depth{};

  using dev_runtime_pending_launch_count_t =
    device_limit<flags::deviceLimit::DevRuntimePendingLaunchCount>;
  static constexpr dev_runtime_pending_launch_count_t
    dev_runtime_pending_launch_count{};

  using max_l2_fetch_granularity_t =
    device_limit<flags::deviceLimit::MaxL2FetchGranularity>;
  static constexpr max_l2_fetch_granularity_t max_l2_fetch_granularity{};

  using persisting_l2_cache_size_t =
    device_limit<flags::deviceLimit::PersistingL2CacheSize>;
  static constexpr persisting_l2_cache_size_t persisting_l2_cache_size{};
#endif  // GCXX_CUDA_MODE()

}  // namespace device_limits

GCXX_NAMESPACE_MAIN_END()

#endif
