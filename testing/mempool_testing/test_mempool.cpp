// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
// Tests for gcxx::MemPool, the owning cudaMemPool_t handle (view/owner
// split like Stream); GPU-dependent cases skip without a device.
#include "tests_common.hpp"

#include <gcxx/api.hpp>

// raw_handle_type contract (see tests_common.hpp).
GCXX_ASSERT_RAW_HANDLE(MemPoolView, gcxx::driver::deviceMemPool_t);
GCXX_ASSERT_RAW_HANDLE(MemPool, gcxx::driver::deviceMemPool_t);

class MemPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP()
        << "No CUDA device available; skipping MemPool ownership tests";
    }
  }
};

TEST_F(MemPoolTest, ConstructAndDestroy) {
  {
    gcxx::MemPool pool;
    EXPECT_NE(pool.getRawHandle(), nullptr);
  }  // destroyed here; must not throw
}

TEST_F(MemPoolTest, MoveConstructorTransfersOwnership) {
  gcxx::MemPool pool1;
  const auto raw = pool1.getRawHandle();

  gcxx::MemPool pool2(std::move(pool1));

  EXPECT_EQ(pool1.getRawHandle(), nullptr);
  EXPECT_EQ(pool2.getRawHandle(), raw);
}

TEST_F(MemPoolTest, MoveAssignmentTransfersOwnership) {
  // Moved-from below; move-assign needs non-const (copy ctor is deleted).
  gcxx::MemPool pool1;  // NOLINT(misc-const-correctness)
  gcxx::MemPool pool2;
  const auto raw1 = pool1.getRawHandle();

  pool2 = std::move(pool1);

  EXPECT_EQ(pool1.getRawHandle(), nullptr);
  EXPECT_EQ(pool2.getRawHandle(),
            raw1);  // pool2's original handle was destroyed
}

TEST_F(MemPoolTest, ReleaseTransfersHandle) {
  gcxx::MemPool pool;
  const auto raw = pool.getRawHandle();

  const gcxx::MemPoolView view = pool.Release();

  EXPECT_EQ(pool.getRawHandle(), nullptr);
  EXPECT_EQ(view.getRawHandle(), raw);

  // Ownership left the pool (view is non-owning); destroy the handle manually.
  gcxx::driver::deviceMemPoolDestroy(raw);
}

TEST_F(MemPoolTest, FromNativeHandleAdoptsAndDestroys) {
  gcxx::MemPool pool;
  const auto raw = pool.getRawHandle();  // the handle, while pool still owns it
  pool.Release();  // relinquish ownership (view discarded)

  gcxx::MemPool adopted = gcxx::MemPool::from_native_handle(raw);
  EXPECT_EQ(adopted.getRawHandle(),
            raw);  // adopted owns raw; its dtor will destroy it
}
