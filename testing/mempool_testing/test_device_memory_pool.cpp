// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Tests for the CCCL-parity device_memory_pool (+ the memory_pool_base API it
// inherits) and the resource_with concept integration. GPU-dependent cases are
// skipped on hosts without a usable device; the compile-time concept checks run
// unconditionally.
#include "tests_common.hpp"

#include <gcxx/api.hpp>

namespace {
  struct not_a_resource {};  // for negative concept checks
}  // namespace

// ── Compile-time concept checks (run on every build) ─────────────────────────
static_assert(gcxx::memory::resource_with<gcxx::memory::device_memory_pool,
                                          gcxx::memory::device_accessible>,
              "");
static_assert(gcxx::memory::resource_with<gcxx::memory::device_memory_pool_ref,
                                          gcxx::memory::device_accessible>,
              "");
static_assert(!gcxx::memory::resource_with<not_a_resource,
                                           gcxx::memory::device_accessible>,
              "a type without allocate/deallocate must not be a resource");

class DeviceMemoryPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available; skipping device_memory_pool tests";
    }
  }
};

TEST_F(DeviceMemoryPoolTest, ConstructOwnsHandle) {
  gcxx::memory::device_memory_pool pool(gcxx::DeviceHandle{0});
  EXPECT_NE(pool.get(), nullptr);
}

TEST_F(DeviceMemoryPoolTest, NoInitIsEmpty) {
  gcxx::memory::device_memory_pool pool(gcxx::memory::no_init);
  EXPECT_EQ(pool.get(), nullptr);
}

TEST_F(DeviceMemoryPoolTest, StreamOrderedAllocateDeallocate) {
  gcxx::memory::device_memory_pool pool(gcxx::DeviceHandle{0});
  void* ptr = pool.allocate(gcxx::StreamView::Null(), 256);
  EXPECT_NE(ptr, nullptr);
  pool.deallocate(gcxx::StreamView::Null(), ptr, 256);
}

TEST_F(DeviceMemoryPoolTest, SyncAllocateDeallocate) {
  gcxx::memory::device_memory_pool pool(gcxx::DeviceHandle{0});
  void* ptr = pool.allocate_sync(256);
  EXPECT_NE(ptr, nullptr);
  pool.deallocate_sync(ptr, 256);
}

TEST_F(DeviceMemoryPoolTest, TypedAttributeRoundTrip) {
  using gcxx::memory::memory_pool_attributes::release_threshold;
  gcxx::memory::device_memory_pool pool(gcxx::DeviceHandle{0});
  pool.set_attribute(release_threshold, std::size_t{4096});
  EXPECT_EQ(pool.attribute(release_threshold), std::size_t{4096});
}

TEST_F(DeviceMemoryPoolTest, TrimToAndSelfAccess) {
  gcxx::memory::device_memory_pool pool(gcxx::DeviceHandle{0});
  EXPECT_NO_THROW(pool.trim_to(0));
  // A device pool's allocations are reachable from its own device.
  EXPECT_TRUE(pool.is_accessible_from(0));
}

TEST_F(DeviceMemoryPoolTest, ReleaseAndFromNativeHandle) {
  gcxx::memory::device_memory_pool pool(gcxx::DeviceHandle{0});
  auto handle = pool.release();
  EXPECT_EQ(pool.get(), nullptr);  // ownership relinquished
  EXPECT_NE(handle, nullptr);
  gcxx::memory::device_memory_pool adopted =
    gcxx::memory::device_memory_pool::from_native_handle(handle);
  EXPECT_EQ(adopted.get(), handle);  // adopted owns it now (dtor will destroy)
}

TEST_F(DeviceMemoryPoolTest, AsRefEquality) {
  gcxx::memory::device_memory_pool pool(gcxx::DeviceHandle{0});
  gcxx::memory::device_memory_pool_ref ref = pool.as_ref();
  EXPECT_TRUE(pool == ref);  // same underlying handle
}

TEST_F(DeviceMemoryPoolTest, BacksABufferViaAsRef) {
  gcxx::memory::device_memory_pool pool(gcxx::DeviceHandle{0});
  gcxx::memory::device_memory_pool_ref ref = pool.as_ref();
  gcxx::memory::buffer<int, gcxx::memory::device_accessible> buf(
    gcxx::StreamView::Null(), ref, 16, gcxx::memory::no_init);
  EXPECT_EQ(buf.size(), 16U);
}

TEST_F(DeviceMemoryPoolTest, DefaultDevicePoolRef) {
  auto ref = gcxx::memory::device_default_memory_pool(gcxx::DeviceHandle{0});
  EXPECT_NE(ref.get(), nullptr);
}
