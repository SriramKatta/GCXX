// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
// Owning gcxx scalars end-to-end through BLAS: direct alpha params and async
// reduction outputs; GPU-gated, must still compile everywhere.

#include "tests_common.hpp"

#include "blas_reference.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include <gcxx/blas_api.hpp>
#include <gcxx/runtime/memory/copy.hpp>
#include <gcxx/runtime/memory/scalars/scalar.hpp>
#include <gcxx/runtime/memory/spans/mdspan/mdspan.hpp>

TEST(BlasDeviceScalar, AxpyOwningScalarAlpha) {
  GCXX_SKIP_WITHOUT_DEVICE();

  constexpr std::size_t N = 16;
  const double alpha      = 2.5;

  std::vector<double> hx(N), hy(N), href(N);
  for (std::size_t i = 0; i < N; ++i) {
    hx[i]   = static_cast<double>(i);
    hy[i]   = 1.0;
    href[i] = alpha * hx[i] + hy[i];
  }

  gcxx::Stream str;
  auto pool = dev_pool();
  gcxx::uninit_device_buffer<double> dx(str, pool, N);
  gcxx::uninit_device_buffer<double> dy(str, pool, N);
  gcxx::Copy(str, dx.data(), hx.data(), N);
  gcxx::Copy(str, dy.data(), hy.data(), N);

  auto dAlpha =
    gcxx::make_device_scalar<double>(gcxx::DeviceHandle{0}, str, alpha);

  dvec_left<double, int> x(dx.data(), static_cast<int>(N));
  dvec_left<double, int> y(dy.data(), static_cast<int>(N));

  gcxx::blas::BlasHandle handle;
  handle.setStream(str);
  gcxx::blas::axpy(handle, dAlpha, x, y);
  str.sync();

  std::vector<double> hy_result(N);
  gcxx::Copy(str, hy_result.data(), dy.data(), N);
  str.sync();

  for (std::size_t i = 0; i < N; ++i) {
    EXPECT_NEAR(hy_result[i], href[i], 1e-9) << "mismatch at index " << i;
  }
}

TEST(BlasDeviceScalar, ReductionOutputsIntoOwningScalars) {
  GCXX_SKIP_WITHOUT_DEVICE();

  constexpr std::size_t N = 8;
  std::vector<double> hx(N);
  double expect_dot  = 0.0;
  double expect_nrm  = 0.0;
  double expect_asum = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    hx[i] = static_cast<double>(i) - 2.0;  // small signed values
    expect_dot += hx[i] * hx[i];
    expect_nrm += hx[i] * hx[i];
    expect_asum += (hx[i] < 0.0 ? -hx[i] : hx[i]);
  }
  expect_nrm = std::sqrt(expect_nrm);

  gcxx::Stream str;
  auto pool = dev_pool();
  gcxx::uninit_device_buffer<double> dx(str, pool, N);
  gcxx::Copy(str, dx.data(), hx.data(), N);

  auto dDot = gcxx::make_device_scalar<double>(gcxx::DeviceHandle{0}, str, 0.0);
  auto dNrm = gcxx::make_device_scalar<double>(gcxx::DeviceHandle{0}, str, 0.0);
  auto dAsum = gcxx::make_pinned_scalar<double>(str, 0.0);

  dvec_left<double, int> x(dx.data(), static_cast<int>(N));

  gcxx::blas::BlasHandle handle;
  handle.setStream(str);
  gcxx::blas::dot(handle, x, x, dDot);
  gcxx::blas::vector_two_norm(handle, x, dNrm);
  gcxx::blas::vector_abs_sum(handle, x, dAsum);
  str.sync();

  // value() syncs the scalar's stream (the same stream the writes rode on).
  EXPECT_NEAR(dDot.value(), expect_dot, 1e-9);
  EXPECT_NEAR(dNrm.value(), expect_nrm, 1e-9);
  EXPECT_NEAR(dAsum.value(), expect_asum, 1e-9);
}

TEST(BlasDeviceScalar, BatchedOpsRejectOwningDeviceScalars) {
  // Owning device/pinned scalars must classify as device-resident: host
  // pointer-mode-only entry points (e.g. the disabled batched gemm family)
  // and the scaled()-factor machinery both key off this detection.
  static_assert(
    gcxx::blas::details_::is_device_scalar_v<gcxx::device_scalar<double>>,
    "owning device scalars must select device pointer mode");
  static_assert(
    gcxx::blas::details_::is_device_scalar_v<gcxx::pinned_scalar<double>>,
    "pinned scalars are device-accessible and select device pointer mode");
  static_assert(!gcxx::blas::details_::is_device_scalar_v<double>,
                "plain host scalars keep host pointer mode");
  SUCCEED();
}
