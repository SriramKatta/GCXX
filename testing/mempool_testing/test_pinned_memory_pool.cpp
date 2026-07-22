// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Tests for pinned_memory_pool. Pinned memory pools are CUDA 12.9+, so the
// entire body is gated; on older toolchains this compiles to an empty TU. Uses
// the NUMA-node ctor (available since 12.9) so it exercises the same path on
// 12.9 and 13.x.
#include "tests_common.hpp"

#if GCXX_CUDA_VERSION_GREATER_EQUAL(12, 9, 0)

#include <cstddef>

static_assert(gcxx::memory::resource_with<gcxx::memory::pinned_memory_pool,
                                          gcxx::memory::device_accessible,
                                          gcxx::memory::host_accessible>,
              "");
static_assert(gcxx::memory::resource_with<gcxx::memory::pinned_memory_pool_ref,
                                          gcxx::memory::device_accessible,
                                          gcxx::memory::host_accessible>,
              "");

class PinnedMemoryPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available; skipping pinned_memory_pool tests";
    }
  }
};

TEST_F(PinnedMemoryPoolTest, ConstructNumaAndAllocate) {
  gcxx::memory::pinned_memory_pool pool{0};  // NUMA node 0
  EXPECT_NE(pool.get(), nullptr);
  void* ptr = pool.allocate(gcxx::StreamView::Null(), std::size_t{256});
  EXPECT_NE(ptr, nullptr);
  pool.deallocate(gcxx::StreamView::Null(), ptr, std::size_t{256});
}

TEST_F(PinnedMemoryPoolTest, DefaultPinnedPoolRef) {
  auto ref = gcxx::memory::pinned_default_memory_pool();
  EXPECT_NE(ref.get(), nullptr);
}

#endif  // GCXX_CUDA_VERSION_GREATER_EQUAL(12, 9, 0)
