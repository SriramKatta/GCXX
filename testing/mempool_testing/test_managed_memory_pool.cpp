// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
// Tests for ManagedMemPool (CUDA 13.0+; empty TU otherwise). A real
// cudaMemPool_t with the full MemPoolView API; mirrors device-pool tests.
#include "tests_common.hpp"

#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)

#include <cstddef>

namespace {
  struct not_a_resource {};  // for negative concept checks
}  // namespace

// Compile-time concept checks (run on every CUDA-13+ build).
static_assert(gcxx::resource_with<gcxx::ManagedMemPool, gcxx::device_accessible,
                                  gcxx::host_accessible>,
              "");
static_assert(
  gcxx::resource_with<gcxx::ManagedMemPoolView, gcxx::device_accessible,
                      gcxx::host_accessible>,
  "");
static_assert(!gcxx::resource_with<not_a_resource, gcxx::device_accessible,
                                   gcxx::host_accessible>,
              "a type without allocate/deallocate must not be a resource");

// raw_handle_type contract (see tests_common.hpp).
GCXX_ASSERT_RAW_HANDLE(ManagedMemPoolView, gcxx::driver::deviceMemPool_t);
GCXX_ASSERT_RAW_HANDLE(ManagedMemPool, gcxx::driver::deviceMemPool_t);

class ManagedMemoryPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available; skipping ManagedMemPool tests";
    }
  }
};

TEST_F(ManagedMemoryPoolTest, ConstructAndAllocate) {
  gcxx::ManagedMemPool pool{};
  EXPECT_NE(pool.getRawHandle(), nullptr);
  void* ptr = pool.allocate(gcxx::StreamView::Null(), std::size_t{256});
  EXPECT_NE(ptr, nullptr);
  pool.deallocate(gcxx::StreamView::Null(), ptr);
}

TEST_F(ManagedMemoryPoolTest, NoInitIsEmpty) {
  gcxx::ManagedMemPool pool(gcxx::no_init);
  EXPECT_EQ(pool.getRawHandle(), nullptr);
}

TEST_F(ManagedMemoryPoolTest, StreamOrderedAllocateDeallocate) {
  gcxx::ManagedMemPool pool{};
  void* a = pool.allocate(gcxx::StreamView::Null(), 128);
  void* b = pool.allocate(gcxx::StreamView::Null(), 128);
  EXPECT_NE(a, nullptr);
  EXPECT_NE(b, nullptr);
  EXPECT_NE(a, b);
  pool.deallocate(gcxx::StreamView::Null(), a);
  pool.deallocate(gcxx::StreamView::Null(), b);
}

TEST_F(ManagedMemoryPoolTest, SyncAllocateDeallocate) {
  gcxx::ManagedMemPool pool{};
  void* ptr = pool.allocate_sync(256);
  EXPECT_NE(ptr, nullptr);
  pool.deallocate_sync(ptr);
}

TEST_F(ManagedMemoryPoolTest, TypedAttributeRoundTrip) {
  using gcxx::memory_pool_attributes::release_threshold;
  gcxx::ManagedMemPool pool{};
  pool.set_attribute(release_threshold, std::size_t{4096});
  EXPECT_EQ(pool.attribute(release_threshold), std::size_t{4096});
}

TEST_F(ManagedMemoryPoolTest, TrimTo) {
  gcxx::ManagedMemPool pool{};
  EXPECT_NO_THROW(pool.trim_to(0));
}

TEST_F(ManagedMemoryPoolTest, ReleaseAndFromNativeHandle) {
  gcxx::ManagedMemPool pool{};
  auto handle = pool.release();
  EXPECT_EQ(pool.getRawHandle(), nullptr);  // ownership relinquished
  EXPECT_NE(handle, nullptr);
  gcxx::ManagedMemPool adopted =
    gcxx::ManagedMemPool::from_native_handle(handle);
  EXPECT_EQ(adopted.getRawHandle(),
            handle);  // adopted owns it (dtor will destroy)
}

TEST_F(ManagedMemoryPoolTest, AsRefEquality) {
  gcxx::ManagedMemPool pool{};
  gcxx::ManagedMemPoolView ref = pool.as_ref();
  EXPECT_TRUE(pool == ref);  // same underlying handle
}

TEST_F(ManagedMemoryPoolTest, BacksABufferViaAsRef) {
  gcxx::ManagedMemPool pool{};
  gcxx::ManagedMemPoolView ref = pool.as_ref();
  // Managed memory is device-accessible, so it can back a device buffer.
  gcxx::buffer<int, gcxx::device_accessible> buf(gcxx::StreamView::Null(), ref,
                                                 16, gcxx::no_init);
  EXPECT_EQ(buf.size(), 16U);
}

TEST_F(ManagedMemoryPoolTest, DefaultManagedPoolRef) {
  auto ref = gcxx::managed_default_memory_pool();
  EXPECT_NE(ref.getRawHandle(), nullptr);
  void* ptr = ref.allocate(gcxx::StreamView::Null(), std::size_t{128});
  EXPECT_NE(ptr, nullptr);
  ref.deallocate(gcxx::StreamView::Null(), ptr);
}

#endif  // GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)
