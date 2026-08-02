// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Tests for PinnedMemPool. On CUDA the pool is a real cudaMemPool_t at a host
// (or host-NUMA) location; on HIP/ROCm there is no host memory pool, so the
// pinned pool is a hipMallocHost-backed shim (see PinnedMemPoolView). The shim
// intentionally carries NO pool handle — get() is null by design — and routes
// (de)allocation through hipMallocHost/hipFreeHost. Pool-management ops
// (trim_to / attribute) are unsupported on the shim (there is no pool to
// manage), so those cases are compiled in only for the CUDA backend.
//
// The construction ctor used is the NUMA-node one so the same ctor is exercised
// on every toolkit (the pinned-pool API is available from CUDA 12.8; on HIP the
// NUMA ctor ignores the node id and builds the shim).
#include "tests_common.hpp"

#include <cstddef>

namespace {
  struct not_a_resource {};  // for negative concept checks
}  // namespace

// ── Compile-time concept checks (run on every build) ─────────────────────────
static_assert(gcxx::resource_with<gcxx::PinnedMemPool, gcxx::device_accessible,
                                  gcxx::host_accessible>,
              "");
static_assert(
  gcxx::resource_with<gcxx::PinnedMemPoolView, gcxx::device_accessible,
                      gcxx::host_accessible>,
  "");
static_assert(!gcxx::resource_with<not_a_resource, gcxx::device_accessible,
                                   gcxx::host_accessible>,
              "a type without allocate/deallocate must not be a resource");

class PinnedMemoryPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available; skipping PinnedMemPool tests";
    }
  }
};

// On HIP the shim deliberately holds no handle; on CUDA the NUMA ctor creates a
// real pool. Either way the pool must be constructible and usable.
TEST_F(PinnedMemoryPoolTest, ConstructAndAllocate) {
  gcxx::PinnedMemPool pool{0};  // NUMA node 0 (ignored by the HIP shim)
#if GCXX_HIP_MODE()
  EXPECT_EQ(pool.get(), nullptr);  // shim: no underlying pool handle
#else
  EXPECT_NE(pool.get(), nullptr);
#endif
  void* ptr = pool.allocate(gcxx::StreamView::Null(), std::size_t{256});
  EXPECT_NE(ptr, nullptr);
  pool.deallocate(gcxx::StreamView::Null(), ptr);
}

TEST_F(PinnedMemoryPoolTest, NoInitIsEmpty) {
  gcxx::PinnedMemPool pool(gcxx::no_init);
  EXPECT_EQ(pool.get(), nullptr);
}

TEST_F(PinnedMemoryPoolTest, StreamOrderedAllocateDeallocate) {
  gcxx::PinnedMemPool pool{0};
  void* a = pool.allocate(gcxx::StreamView::Null(), 128);
  void* b = pool.allocate(gcxx::StreamView::Null(), 128);
  EXPECT_NE(a, nullptr);
  EXPECT_NE(b, nullptr);
  EXPECT_NE(a, b);
  pool.deallocate(gcxx::StreamView::Null(), a);
  pool.deallocate(gcxx::StreamView::Null(), b);
}

TEST_F(PinnedMemoryPoolTest, SyncAllocateDeallocate) {
  gcxx::PinnedMemPool pool{0};
  void* ptr = pool.allocate_sync(256);
  EXPECT_NE(ptr, nullptr);
  pool.deallocate_sync(ptr);
}

TEST_F(PinnedMemoryPoolTest, AsRefEquality) {
  gcxx::PinnedMemPool pool{0};
  gcxx::PinnedMemPoolView ref = pool.as_ref();
  EXPECT_TRUE(pool == ref);  // same underlying handle (null on the shim)
}

TEST_F(PinnedMemoryPoolTest, BacksABufferViaAsRef) {
  gcxx::PinnedMemPool pool{0};
  gcxx::PinnedMemPoolView ref = pool.as_ref();
  // Pinned memory is device-accessible, so it can back a device buffer.
  gcxx::buffer<int, gcxx::device_accessible> buf(gcxx::StreamView::Null(), ref,
                                                 16, gcxx::no_init);
  EXPECT_EQ(buf.size(), 16U);
}

TEST_F(PinnedMemoryPoolTest, ReleaseAndFromNativeHandle) {
  gcxx::PinnedMemPool pool{0};
  auto handle = pool.release();
  EXPECT_EQ(pool.get(), nullptr);  // ownership relinquished
#if GCXX_HIP_MODE()
  // The shim has no handle to release; there is nothing to re-adopt or destroy.
  EXPECT_EQ(handle, nullptr);
#else
  EXPECT_NE(handle, nullptr);
  gcxx::PinnedMemPool adopted = gcxx::PinnedMemPool::from_native_handle(handle);
  EXPECT_EQ(adopted.get(), handle);  // adopted owns it (dtor will destroy)
#endif
}

TEST_F(PinnedMemoryPoolTest, DefaultPinnedPoolRef) {
  auto ref = gcxx::pinned_default_memory_pool();
#if GCXX_HIP_MODE()
  EXPECT_EQ(ref.get(), nullptr);  // shim: no handle
#else
  EXPECT_NE(ref.get(), nullptr);
#endif
  // The default ref must be usable for allocation on every backend.
  void* ptr = ref.allocate(gcxx::StreamView::Null(), std::size_t{128});
  EXPECT_NE(ptr, nullptr);
  ref.deallocate(gcxx::StreamView::Null(), ptr);
}

// Pool-management ops only exist for a real pool (CUDA). On the HIP shim they
// would touch a null handle, so they are compiled out entirely.
#if GCXX_CUDA_MODE()
TEST_F(PinnedMemoryPoolTest, TypedAttributeRoundTrip) {
  using gcxx::memory_pool_attributes::release_threshold;
  gcxx::PinnedMemPool pool{0};
  pool.set_attribute(release_threshold, std::size_t{4096});
  EXPECT_EQ(pool.attribute(release_threshold), std::size_t{4096});
}

TEST_F(PinnedMemoryPoolTest, TrimTo) {
  gcxx::PinnedMemPool pool{0};
  EXPECT_NO_THROW(pool.trim_to(0));
}
#endif  // GCXX_CUDA_MODE()
