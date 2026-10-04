// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
// Audits the allocation-kind property tags: each default pool allocates one
// block and its runtime attribute classification (driver::queryMemoryKind)
// must match the *_allocated tag the pool advertises. This once-only
// verification is what lets consumers (e.g. the BLAS pointer-mode pairing)
// trust the tags entirely at compile time — no per-call pointer probing.

#include "tests_common.hpp"

#include <cstddef>

#include <gcxx/api.hpp>

TEST(PoolMemoryKind, AdvertisedTagsMatchKindVocabulary) {
  // Device pool: device memory, stream-ordered.
  static_assert(
    gcxx::has_property_v<gcxx::DeviceMemPoolView, gcxx::device_allocated>);
  static_assert(
    gcxx::has_property_v<gcxx::DeviceMemPoolView, gcxx::async_allocated>);
  static_assert(
    !gcxx::has_property_v<gcxx::DeviceMemPoolView, gcxx::host_allocated>);

  // Pinned pool: host storage that is also device-accessible. The lifecycle
  // tag is backend-conditional (stream-ordered pools on CUDA; the
  // hipMallocHost shim allocates synchronously on HIP).
  static_assert(
    gcxx::has_property_v<gcxx::PinnedMemPoolView, gcxx::pinned_allocated>);
  static_assert(
    gcxx::has_property_v<gcxx::PinnedMemPoolView, gcxx::device_accessible>);
#if GCXX_HIP_MODE()
  static_assert(
    gcxx::has_property_v<gcxx::PinnedMemPoolView, gcxx::sync_allocated>);
#else
  static_assert(
    gcxx::has_property_v<gcxx::PinnedMemPoolView, gcxx::async_allocated>);
#endif

#if GCXX_HAS_MANAGED_POOLS
  static_assert(
    gcxx::has_property_v<gcxx::ManagedMemPoolView, gcxx::unified_allocated>);
  static_assert(
    !gcxx::has_property_v<gcxx::ManagedMemPoolView, gcxx::host_allocated>);
#endif
}

TEST(PoolMemoryKind, PageableHostMemoryIsUnregisteredHost) {
  // The bijection's fourth leg: plain malloc/new/stack memory has no CUDA
  // allocation kind — the attribute query fails and reports host.
  double stack_value = 0.0;
  EXPECT_EQ(gcxx::driver::queryMemoryKind(&stack_value),
            gcxx::driver::memory_kind::host);
}

TEST(PoolMemoryKind, DevicePoolAllocatesDeviceMemory) {
  GCXX_SKIP_WITHOUT_DEVICE();

  gcxx::Stream str;
  auto pool = gcxx::device_default_memory_pool(gcxx::DeviceHandle{0});
  void* mem = pool.allocate(str, std::size_t{256});
  str.sync();
  ASSERT_NE(mem, nullptr);
  EXPECT_EQ(gcxx::driver::queryMemoryKind(mem),
            gcxx::driver::memory_kind::device);
  pool.deallocate(str, mem);
  str.sync();
}

TEST(PoolMemoryKind, PinnedPoolAllocatesMappedHostMemory) {
  GCXX_SKIP_WITHOUT_DEVICE();

  gcxx::Stream str;
  auto pool = gcxx::pinned_default_memory_pool();
  void* mem = pool.allocate(str, std::size_t{256});
  str.sync();
  ASSERT_NE(mem, nullptr);
  // The pinned pool claims device_accessible: its host-allocated memory
  // must carry a device (UVA/HMM) mapping. A `host` result here would mean
  // the resource's accessibility claim is wrong (HIP shim: depends on HMM).
  EXPECT_EQ(gcxx::driver::queryMemoryKind(mem),
            gcxx::driver::memory_kind::mapped_host);
  pool.deallocate(str, mem);
  str.sync();
}

// Managed pools exist from CUDA 13.0 (mempool/default_memory_pools.hpp).
#if GCXX_HAS_MANAGED_POOLS
TEST(PoolMemoryKind, ManagedPoolAllocatesUnifiedMemory) {
  GCXX_SKIP_WITHOUT_DEVICE();

  gcxx::Stream str;
  gcxx::ManagedMemPool pool{};
  void* mem = pool.as_ref().allocate(str, std::size_t{256});
  str.sync();
  ASSERT_NE(mem, nullptr);
  EXPECT_EQ(gcxx::driver::queryMemoryKind(mem),
            gcxx::driver::memory_kind::unified);
  pool.as_ref().deallocate(str, mem);
  str.sync();
}
#endif  // GCXX_HAS_MANAGED_POOLS
