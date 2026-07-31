// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Tests for PinnedMemPool. Uses the NUMA-node ctor so it exercises the
// same path on every supported toolkit (the pinned-pool API is available from
// CUDA 12.8; creation succeeds on a capable driver).
#include "tests_common.hpp"

#include <cstddef>

static_assert(gcxx::resource_with<gcxx::PinnedMemPool, gcxx::device_accessible,
                                  gcxx::host_accessible>,
              "");
static_assert(
  gcxx::resource_with<gcxx::PinnedMemPoolView, gcxx::device_accessible,
                      gcxx::host_accessible>,
  "");

class PinnedMemoryPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available; skipping PinnedMemPool tests";
    }
  }
};

TEST_F(PinnedMemoryPoolTest, ConstructNumaAndAllocate) {
  gcxx::PinnedMemPool pool{0};  // NUMA node 0
  EXPECT_NE(pool.get(), nullptr);
  void* ptr = pool.allocate(gcxx::StreamView::Null(), std::size_t{256});
  EXPECT_NE(ptr, nullptr);
  pool.deallocate(gcxx::StreamView::Null(), ptr);
}

TEST_F(PinnedMemoryPoolTest, DefaultPinnedPoolRef) {
  auto ref = gcxx::pinned_default_memory_pool();
  EXPECT_NE(ref.get(), nullptr);
}
