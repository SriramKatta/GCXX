// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Tests for gcxx::MemPool — the owning cudaMemPool_t handle. Mirrors the
// Stream/StreamView relationship: MemPool : public MemPoolView, creates the
// pool in its ctor and destroys it in its dtor. GPU-dependent; skipped on hosts
// without a usable device.
#include "tests_common.hpp"

#include <gcxx/api.hpp>

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
    EXPECT_NE(pool.get(), nullptr);
  }  // destroyed here; must not throw
}

TEST_F(MemPoolTest, MoveConstructorTransfersOwnership) {
  gcxx::MemPool pool1;
  const auto raw = pool1.get();

  gcxx::MemPool pool2(std::move(pool1));

  EXPECT_EQ(pool1.get(), nullptr);
  EXPECT_EQ(pool2.get(), raw);
}

TEST_F(MemPoolTest, MoveAssignmentTransfersOwnership) {
  // pool1 is moved-from below; move-assignment needs a non-const source (copy
  // is deleted), so it stays non-const despite the const-correctness nudge.
  gcxx::MemPool pool1;  // NOLINT(misc-const-correctness)
  gcxx::MemPool pool2;
  const auto raw1 = pool1.get();

  pool2 = std::move(pool1);

  EXPECT_EQ(pool1.get(), nullptr);
  EXPECT_EQ(pool2.get(), raw1);  // pool2's original handle was destroyed
}

TEST_F(MemPoolTest, ReleaseTransfersHandle) {
  gcxx::MemPool pool;
  const auto raw = pool.get();

  const gcxx::MemPoolView view = pool.Release();

  EXPECT_EQ(pool.get(), nullptr);
  EXPECT_EQ(view.get(), raw);

  // Ownership left the pool (view is non-owning); destroy the handle manually.
  gcxx::driver::deviceMemPoolDestroy(raw);
}

TEST_F(MemPoolTest, FromNativeHandleAdoptsAndDestroys) {
  gcxx::MemPool pool;
  const auto raw = pool.get();  // the handle, while pool still owns it
  pool.Release();               // relinquish ownership (view discarded)

  gcxx::MemPool adopted = gcxx::MemPool::from_native_handle(raw);
  EXPECT_EQ(adopted.get(), raw);  // adopted owns raw; its dtor will destroy it
}
