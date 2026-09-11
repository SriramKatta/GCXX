// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
// Owning gcxx scalars end-to-end through BLAS: scaled() factors (device
// pointer mode, including pinned/managed storage), direct alpha params, and
// async reduction outputs; GPU-gated, must still compile everywhere.

#include "tests_common.hpp"

#include "blas_reference.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include <gcxx/blas_api.hpp>
#include <gcxx/runtime/memory/copy.hpp>
#include <gcxx/runtime/memory/scalars/default_scalar_pools.hpp>
#include <gcxx/runtime/memory/scalars/scalar.hpp>
#include <gcxx/runtime/memory/spans/mdspan/mdspan.hpp>

namespace {

  // Pinned and managed scalars are device-accessible: valid device
  // pointer-mode factors through the same scaled() bridge. The caller picks
  // the kind by passing the matching pool view (pinned vs managed).
  template <class ScalarT, class Resource>
  void run_gemm_scalar_factors(const Resource& resource) {
    GCXX_SKIP_WITHOUT_DEVICE();

    constexpr int M = 3;
    constexpr int K = 4;
    constexpr int N = 5;
    double alpha    = 2.0;
    double beta     = 1.0;

    std::vector<double> hA(M * K), hB(K * N), hC(M * N);
    for (int i = 0; i < M * K; ++i) {
      hA[i] = static_cast<double>(i + 1);
    }
    for (int i = 0; i < K * N; ++i) {
      hB[i] = static_cast<double>((i % 3) - 1);
    }
    for (int i = 0; i < M * N; ++i) {
      hC[i] = static_cast<double>(i % 5);
    }

    mat_left<double, int> hostA(hA.data(), M, K);
    mat_left<double, int> hostB(hB.data(), K, N);
    mat_left<double, int> hostCref(hC.data(), M, N);

    std::vector<double> href(M * N);
    mat_left<double, int> hostOut(href.data(), M, N);
    host_gemm(hostA, hostB, hostCref, hostOut, alpha, beta);

    gcxx::Stream str;
    const auto pool = dev_pool();
    gcxx::uninit_device_buffer<double> dA(str, pool,
                                          static_cast<std::size_t>(M * K));
    gcxx::uninit_device_buffer<double> dB(str, pool,
                                          static_cast<std::size_t>(K * N));
    gcxx::uninit_device_buffer<double> dC(str, pool,
                                          static_cast<std::size_t>(M * N));
    gcxx::Copy(str, dA.data(), hA.data(), static_cast<std::size_t>(M * K));
    gcxx::Copy(str, dB.data(), hB.data(), static_cast<std::size_t>(K * N));
    gcxx::Copy(str, dC.data(), hC.data(), static_cast<std::size_t>(M * N));

    ScalarT dAlpha(str, resource, 2.0);
    ScalarT dBeta(str, resource, 1.0);

    dmat_left<double, int> A(dA.data(), M, K);
    dmat_left<double, int> B(dB.data(), K, N);
    dmat_left<double, int> C(dC.data(), M, N);

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);
    gcxx::blas::matrix_product(handle, gcxx::scaled(dAlpha, A), B,
                               gcxx::scaled(dBeta, C), C);
    str.sync();

    std::vector<double> hC_result(M * N);
    gcxx::Copy(str, hC_result.data(), dC.data(),
               static_cast<std::size_t>(M * N));
    str.sync();

    for (int i = 0; i < M * N; ++i) {
      EXPECT_NEAR(hC_result[i], href[i], 1e-9)
        << "mismatch at linear index " << i;
    }
  }

}  // namespace

TEST(BlasDeviceScalar, GemmOwningScalarFactors) {
  GCXX_SKIP_WITHOUT_DEVICE();
  run_gemm_scalar_factors<gcxx::device_scalar<double>>(dev_pool());
}

TEST(BlasDeviceScalar, GemmPinnedScalarFactors) {
  // Skip before touching the pools: the default-pool accessors need a
  // driver even though the helper below also skips.
  GCXX_SKIP_WITHOUT_DEVICE();
  run_gemm_scalar_factors<gcxx::pinned_scalar<double>>(
    gcxx::pinned_default_memory_pool());
}

// Managed pools exist from CUDA 13.0 (see
// mempool/default_memory_pools.hpp); the managed scalar kind shares the
// pinned test's instantiation below 13.0.
#if GCXX_HAS_MANAGED_POOLS
TEST(BlasDeviceScalar, GemmManagedScalarFactors) {
  // Managed pool resource picks the managed allocation kind.
  GCXX_SKIP_WITHOUT_DEVICE();
  gcxx::ManagedMemPool mpool{};
  run_gemm_scalar_factors<gcxx::managed_scalar<double>>(mpool.as_ref());
}
#endif  // GCXX_HAS_MANAGED_POOLS

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
  // The batched pointer-array API requires host pointer mode; the
  // generalized detection must classify owning device scalars as
  // device-resident so its static_assert fires for them too.
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
