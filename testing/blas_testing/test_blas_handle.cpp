// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta

#include "tests_common.hpp"

#include <gcxx/blas_api.hpp>

// Compile-time sanity: the owning handle is-a the non-owning view.
static_assert(
  std::is_base_of_v<gcxx::blas::BlasHandleView, gcxx::blas::BlasHandle>, "");

// raw_handle_type contract (see tests_common.hpp).
GCXX_ASSERT_RAW_HANDLE(blas::BlasHandleView, gcxx::driver::deviceBlasHandle_t);
GCXX_ASSERT_RAW_HANDLE(blas::BlasHandle, gcxx::driver::deviceBlasHandle_t);

TEST(BlasHandle, CreateAndDestroy) {
  GCXX_SKIP_WITHOUT_DEVICE();
  gcxx::blas::BlasHandle handle;
  EXPECT_NE(handle.getRawHandle(), nullptr);
}

TEST(BlasHandle, VersionIsPositive) {
  GCXX_SKIP_WITHOUT_DEVICE();
  gcxx::blas::BlasHandle handle;
  EXPECT_GT(handle.getVersion(), 0);
}

TEST(BlasHandle, SetGetStream) {
  GCXX_SKIP_WITHOUT_DEVICE();
  gcxx::blas::BlasHandle handle;
  gcxx::Stream stream;
  handle.setStream(stream);
  EXPECT_EQ(handle.getStream().getRawHandle(), stream.getRawHandle());
}

TEST(BlasHandle, MoveTransfersOwnership) {
  GCXX_SKIP_WITHOUT_DEVICE();
  gcxx::blas::BlasHandle handle;
  auto raw = handle.getRawHandle();
  ASSERT_NE(raw, nullptr);
  gcxx::blas::BlasHandle moved{std::move(handle)};
  EXPECT_EQ(moved.getRawHandle(), raw);
  EXPECT_EQ(handle.getRawHandle(), nullptr);  // NOLINT(bugprone-use-after-move)
}

TEST(BlasHandle, ReleaseYieldsView) {
  GCXX_SKIP_WITHOUT_DEVICE();
  gcxx::blas::BlasHandle handle;
  auto raw  = handle.getRawHandle();
  auto view = handle.release();
  EXPECT_EQ(view.getRawHandle(), raw);
  EXPECT_EQ(handle.getRawHandle(), nullptr);
  [[maybe_unused]] auto adopted =
    gcxx::blas::BlasHandle::from_native_handle(raw);
}
