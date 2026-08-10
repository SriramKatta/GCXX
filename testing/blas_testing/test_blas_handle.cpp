// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta

#include "tests_common.hpp"

#include <gcxx/blas_api.hpp>

// Compile-time sanity: the owning handle is-a the non-owning view.
static_assert(std::is_base_of_v<gcxx::BlasHandleView, gcxx::BlasHandle>, "");

TEST(BlasHandle, CreateAndDestroy) {
  if (!gcxx::testing::haveCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  gcxx::BlasHandle handle;
  EXPECT_NE(handle.getHandle(), nullptr);
}

TEST(BlasHandle, VersionIsPositive) {
  if (!gcxx::testing::haveCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  gcxx::BlasHandle handle;
  EXPECT_GT(handle.getVersion(), 0);
}

TEST(BlasHandle, SetGetStream) {
  if (!gcxx::testing::haveCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  gcxx::BlasHandle handle;
  gcxx::Stream stream;
  handle.setStream(stream);
  EXPECT_EQ(handle.getStream().getRawStream(), stream.getRawStream());
}

TEST(BlasHandle, MoveTransfersOwnership) {
  if (!gcxx::testing::haveCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  gcxx::BlasHandle handle;
  auto raw = handle.getHandle();
  ASSERT_NE(raw, nullptr);
  gcxx::BlasHandle moved{std::move(handle)};
  EXPECT_EQ(moved.getHandle(), raw);
  EXPECT_EQ(handle.getHandle(), nullptr);  // NOLINT(bugprone-use-after-move)
}

TEST(BlasHandle, ReleaseYieldsView) {
  if (!gcxx::testing::haveCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  gcxx::BlasHandle handle;
  auto raw  = handle.getHandle();
  auto view = handle.release();
  EXPECT_EQ(view.getHandle(), raw);
  EXPECT_EQ(handle.getHandle(), nullptr);
  [[maybe_unused]] auto adopted = gcxx::BlasHandle::from_native_handle(raw);
}
