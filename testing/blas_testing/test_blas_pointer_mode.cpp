// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta

#include "tests_common.hpp"

#include <gcxx/blas/handle/blas_pointer_mode_guard.hpp>
#include <gcxx/blas_api.hpp>

TEST(BlasPointerMode, DefaultIsHost) {
  if (!gcxx::testing::haveCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  gcxx::blas::BlasHandle handle;
  EXPECT_EQ(handle.getPointerMode(), gcxx::blas::host_pointer_mode);
}

TEST(BlasPointerMode, SetGetRoundTrip) {
  if (!gcxx::testing::haveCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  gcxx::blas::BlasHandle handle;
  handle.setPointerMode(gcxx::blas::device_pointer_mode);
  EXPECT_EQ(handle.getPointerMode(), gcxx::blas::device_pointer_mode);
  handle.setPointerMode(gcxx::blas::host_pointer_mode);
  EXPECT_EQ(handle.getPointerMode(), gcxx::blas::host_pointer_mode);
}

TEST(BlasPointerMode, GuardRestoresOnDestruction) {
  if (!gcxx::testing::haveCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  gcxx::blas::BlasHandle handle;
  ASSERT_EQ(handle.getPointerMode(), gcxx::blas::host_pointer_mode);

  {
    gcxx::blas::details_::BlasPointerModeGuard guard{
      handle, gcxx::blas::device_pointer_mode};
    EXPECT_EQ(handle.getPointerMode(), gcxx::blas::device_pointer_mode);
  }
  EXPECT_EQ(handle.getPointerMode(), gcxx::blas::host_pointer_mode);
}

// Restores the *prior* mode, not a hard-coded host default.
TEST(BlasPointerMode, GuardRestoresPriorDeviceMode) {
  if (!gcxx::testing::haveCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  gcxx::blas::BlasHandle handle;
  handle.setPointerMode(gcxx::blas::device_pointer_mode);

  {
    gcxx::blas::details_::BlasPointerModeGuard guard{
      handle, gcxx::blas::host_pointer_mode};
    EXPECT_EQ(handle.getPointerMode(), gcxx::blas::host_pointer_mode);
  }
  EXPECT_EQ(handle.getPointerMode(), gcxx::blas::device_pointer_mode);
}
