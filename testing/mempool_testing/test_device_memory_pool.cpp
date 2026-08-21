// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
// Tests for the CCCL-parity DeviceMemPool (+ inherited MemPoolView API) and
// resource_with integration; GPU-dependent cases skip without a device.
#include "tests_common.hpp"

#include <gcxx/api.hpp>

namespace {
  struct not_a_resource {};  // for negative concept checks
}  // namespace

// Compile-time concept checks (run on every build).
static_assert(gcxx::resource_with<gcxx::DeviceMemPool, gcxx::device_accessible>,
              "");
static_assert(
  gcxx::resource_with<gcxx::DeviceMemPoolView, gcxx::device_accessible>, "");
static_assert(!gcxx::resource_with<not_a_resource, gcxx::device_accessible>,
              "a type without allocate/deallocate must not be a resource");

// raw_handle_type contract (see tests_common.hpp).
GCXX_ASSERT_RAW_HANDLE(DeviceMemPoolView, gcxx::driver::deviceMemPool_t);
GCXX_ASSERT_RAW_HANDLE(DeviceMemPool, gcxx::driver::deviceMemPool_t);

class DeviceMemoryPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available; skipping DeviceMemPool tests";
    }
  }
};

TEST_F(DeviceMemoryPoolTest, ConstructOwnsHandle) {
  gcxx::DeviceMemPool pool(gcxx::DeviceHandle{0});
  EXPECT_NE(pool.getRawHandle(), nullptr);
}

TEST_F(DeviceMemoryPoolTest, NoInitIsEmpty) {
  gcxx::DeviceMemPool pool(gcxx::no_init);
  EXPECT_EQ(pool.getRawHandle(), nullptr);
}

TEST_F(DeviceMemoryPoolTest, StreamOrderedAllocateDeallocate) {
  gcxx::DeviceMemPool pool(gcxx::DeviceHandle{0});
  void* ptr = pool.allocate(gcxx::StreamView::Null(), 256);
  EXPECT_NE(ptr, nullptr);
  pool.deallocate(gcxx::StreamView::Null(), ptr);
}

TEST_F(DeviceMemoryPoolTest, SyncAllocateDeallocate) {
  gcxx::DeviceMemPool pool(gcxx::DeviceHandle{0});
  void* ptr = pool.allocate_sync(256);
  EXPECT_NE(ptr, nullptr);
  pool.deallocate_sync(ptr, 256);
}

TEST_F(DeviceMemoryPoolTest, TypedAttributeRoundTrip) {
  using gcxx::memory_pool_attributes::release_threshold;
  gcxx::DeviceMemPool pool(gcxx::DeviceHandle{0});
  pool.set_attribute(release_threshold, std::size_t{4096});
  EXPECT_EQ(pool.attribute(release_threshold), std::size_t{4096});
}

// Peer/self-access checks (is_accessible_from) omitted: API is still a TODO.
TEST_F(DeviceMemoryPoolTest, TrimTo) {
  gcxx::DeviceMemPool pool(gcxx::DeviceHandle{0});
  EXPECT_NO_THROW(pool.trim_to(0));
}

TEST_F(DeviceMemoryPoolTest, ReleaseAndFromNativeHandle) {
  gcxx::DeviceMemPool pool(gcxx::DeviceHandle{0});
  auto handle = pool.release();
  EXPECT_EQ(pool.getRawHandle(), nullptr);  // ownership relinquished
  EXPECT_NE(handle, nullptr);
  gcxx::DeviceMemPool adopted = gcxx::DeviceMemPool::from_native_handle(handle);
  EXPECT_EQ(adopted.getRawHandle(),
            handle);  // adopted owns it now (dtor will destroy)
}

TEST_F(DeviceMemoryPoolTest, AsRefEquality) {
  gcxx::DeviceMemPool pool(gcxx::DeviceHandle{0});
  gcxx::DeviceMemPoolView ref = pool.as_ref();
  EXPECT_TRUE(pool == ref);  // same underlying handle
}

TEST_F(DeviceMemoryPoolTest, BacksABufferViaAsRef) {
  gcxx::DeviceMemPool pool(gcxx::DeviceHandle{0});
  gcxx::DeviceMemPoolView ref = pool.as_ref();
  gcxx::buffer<int, gcxx::device_accessible> buf(gcxx::StreamView::Null(), ref,
                                                 16, gcxx::no_init);
  EXPECT_EQ(buf.size(), 16U);
}

TEST_F(DeviceMemoryPoolTest, DefaultDevicePoolRef) {
  auto ref = gcxx::device_default_memory_pool(gcxx::DeviceHandle{0});
  EXPECT_NE(ref.getRawHandle(), nullptr);
}
