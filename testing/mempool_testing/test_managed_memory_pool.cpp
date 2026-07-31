// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Tests for ManagedMemPool. Managed memory pools are CUDA 13.0+, so the
// entire body is gated; on older toolchains this compiles to an empty TU.
#include "tests_common.hpp"

#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)

#include <cstddef>

static_assert(gcxx::resource_with<gcxx::ManagedMemPool, gcxx::device_accessible,
                                  gcxx::host_accessible>,
              "");
static_assert(
  gcxx::resource_with<gcxx::ManagedMemPoolView, gcxx::device_accessible,
                      gcxx::host_accessible>,
  "");

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
  EXPECT_NE(pool.get(), nullptr);
  void* ptr = pool.allocate(gcxx::StreamView::Null(), std::size_t{256});
  EXPECT_NE(ptr, nullptr);
  pool.deallocate(gcxx::StreamView::Null(), ptr);
}

TEST_F(ManagedMemoryPoolTest, DefaultManagedPoolRef) {
  auto ref = gcxx::managed_default_memory_pool();
  EXPECT_NE(ref.get(), nullptr);
}

#endif  // GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)
