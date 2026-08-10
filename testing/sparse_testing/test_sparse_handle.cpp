// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta

#include "tests_common.hpp"

#include <gcxx/sparse_api.hpp>

// Compile-time sanity: the owning handle is-a the non-owning view.
static_assert(std::is_base_of_v<gcxx::SparseHandleView, gcxx::SparseHandle>,
              "");

TEST(SparseHandle, CreateAndDestroy) {
  if (!gcxx::testing::haveCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  gcxx::SparseHandle handle;
  EXPECT_NE(handle.getHandle(), nullptr);
}

TEST(SparseHandle, VersionIsPositive) {
  if (!gcxx::testing::haveCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  gcxx::SparseHandle handle;
  EXPECT_GT(handle.getVersion(), 0);
}

TEST(SparseHandle, SetGetStream) {
  if (!gcxx::testing::haveCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  gcxx::SparseHandle handle;
  gcxx::Stream stream;
  handle.setStream(stream);
  EXPECT_EQ(handle.getStream().getRawStream(), stream.getRawStream());
}

TEST(SparseHandle, MoveTransfersOwnership) {
  if (!gcxx::testing::haveCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  gcxx::SparseHandle handle;
  auto raw = handle.getHandle();
  ASSERT_NE(raw, nullptr);
  gcxx::SparseHandle moved{std::move(handle)};
  EXPECT_EQ(moved.getHandle(), raw);
  EXPECT_EQ(handle.getHandle(), nullptr);  // NOLINT(bugprone-use-after-move)
}

TEST(SparseHandle, ReleaseYieldsView) {
  if (!gcxx::testing::haveCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  gcxx::SparseHandle handle;
  auto raw  = handle.getHandle();
  auto view = handle.release();
  EXPECT_EQ(view.getHandle(), raw);
  EXPECT_EQ(handle.getHandle(), nullptr);
  [[maybe_unused]] auto adopted = gcxx::SparseHandle::from_native_handle(raw);
}
