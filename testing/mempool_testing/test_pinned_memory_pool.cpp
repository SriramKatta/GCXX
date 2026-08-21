// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
// Tests for PinnedMemPool: a real cudaMemPool_t on CUDA, a hipMallocHost-
// backed shim on HIP (no handle); pool-management ops are CUDA-only.
#include "tests_common.hpp"

#include <cstddef>

GCXX_ASSERT_RAW_HANDLE(PinnedMemPoolView, gcxx::driver::deviceMemPool_t);
GCXX_ASSERT_RAW_HANDLE(PinnedMemPool, gcxx::driver::deviceMemPool_t);

namespace {
  struct not_a_resource {};  // for negative concept checks
}  // namespace

// Compile-time concept checks (run on every build).
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

// HIP shim deliberately holds no handle; CUDA default ctor is a real pool.
TEST_F(PinnedMemoryPoolTest, ConstructAndAllocate) {
  gcxx::PinnedMemPool pool{};  // default ctor on every backend
#if GCXX_HIP_MODE()
  EXPECT_EQ(pool.getRawHandle(), nullptr);  // shim: no underlying pool handle
#else
  EXPECT_NE(pool.getRawHandle(), nullptr);
#endif
  void* ptr = pool.allocate(gcxx::StreamView::Null(), std::size_t{256});
  EXPECT_NE(ptr, nullptr);
  pool.deallocate(gcxx::StreamView::Null(), ptr);
}

TEST_F(PinnedMemoryPoolTest, NoInitIsEmpty) {
  gcxx::PinnedMemPool pool(gcxx::no_init);
  EXPECT_EQ(pool.getRawHandle(), nullptr);
}

TEST_F(PinnedMemoryPoolTest, StreamOrderedAllocateDeallocate) {
  gcxx::PinnedMemPool pool{};
  void* a = pool.allocate(gcxx::StreamView::Null(), 128);
  void* b = pool.allocate(gcxx::StreamView::Null(), 128);
  EXPECT_NE(a, nullptr);
  EXPECT_NE(b, nullptr);
  EXPECT_NE(a, b);
  pool.deallocate(gcxx::StreamView::Null(), a);
  pool.deallocate(gcxx::StreamView::Null(), b);
}

TEST_F(PinnedMemoryPoolTest, SyncAllocateDeallocate) {
  gcxx::PinnedMemPool pool{};
  void* ptr = pool.allocate_sync(256);
  EXPECT_NE(ptr, nullptr);
  pool.deallocate_sync(ptr);
}

TEST_F(PinnedMemoryPoolTest, AsRefEquality) {
  gcxx::PinnedMemPool pool{};
  gcxx::PinnedMemPoolView ref = pool.as_ref();
  EXPECT_TRUE(pool == ref);  // same underlying handle (null on the shim)
}

TEST_F(PinnedMemoryPoolTest, BacksABufferViaAsRef) {
  gcxx::PinnedMemPool pool{};
  gcxx::PinnedMemPoolView ref = pool.as_ref();
  // Pinned memory is device-accessible, so it can back a device buffer.
  gcxx::buffer<int, gcxx::device_accessible> buf(gcxx::StreamView::Null(), ref,
                                                 16, gcxx::no_init);
  EXPECT_EQ(buf.size(), 16U);
}

TEST_F(PinnedMemoryPoolTest, ReleaseAndFromNativeHandle) {
  gcxx::PinnedMemPool pool{};
  auto handle = pool.release();
  EXPECT_EQ(pool.getRawHandle(), nullptr);  // ownership relinquished
#if GCXX_HIP_MODE()
  // The shim has no handle to release; there is nothing to re-adopt or destroy.
  EXPECT_EQ(handle, nullptr);
#else
  EXPECT_NE(handle, nullptr);
  gcxx::PinnedMemPool adopted = gcxx::PinnedMemPool::from_native_handle(handle);
  EXPECT_EQ(adopted.getRawHandle(),
            handle);  // adopted owns it (dtor will destroy)
#endif
}

TEST_F(PinnedMemoryPoolTest, DefaultPinnedPoolRef) {
  auto ref = gcxx::pinned_default_memory_pool();
#if GCXX_HIP_MODE()
  EXPECT_EQ(ref.getRawHandle(), nullptr);  // shim: no handle
#else
  EXPECT_NE(ref.getRawHandle(), nullptr);
#endif
  // The default ref must be usable for allocation on every backend.
  void* ptr = ref.allocate(gcxx::StreamView::Null(), std::size_t{128});
  EXPECT_NE(ptr, nullptr);
  ref.deallocate(gcxx::StreamView::Null(), ptr);
}

// Pool-management ops need a real pool; on the HIP shim they'd touch null.
#if GCXX_CUDA_MODE()
TEST_F(PinnedMemoryPoolTest, TypedAttributeRoundTrip) {
  using gcxx::memory_pool_attributes::release_threshold;
  gcxx::PinnedMemPool pool{};
  pool.set_attribute(release_threshold, std::size_t{4096});
  EXPECT_EQ(pool.attribute(release_threshold), std::size_t{4096});
}

TEST_F(PinnedMemoryPoolTest, TrimTo) {
  gcxx::PinnedMemPool pool{};
  EXPECT_NO_THROW(pool.trim_to(0));
}
#endif  // GCXX_CUDA_MODE()
