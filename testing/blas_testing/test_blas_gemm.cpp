// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
// End-to-end matrix_product (P1673 gemm) tests via cuBLAS with scaled()
// views and layout gates; GPU-gated, must still compile everywhere.

#include "tests_common.hpp"

#include "blas_reference.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gcxx/blas_api.hpp>
#include <gcxx/runtime/memory/copy.hpp>
#include <gcxx/runtime/memory/spans/mdspan/mdspan.hpp>

namespace {

  template <class T, class IndexT>
  using mat_right = gcxx::mdspan<T, dextents2d<IndexT>, gcxx::layout_right,
                                 gcxx::default_accessor<T>>;
  template <class T, class IndexT>
  using dmat_right =
    gcxx::device_mdspan<T, dextents2d<IndexT>, gcxx::layout_right>;

  // index_type picks the cu/hipblas entry: GemmEx vs GemmEx_64.
  template <class IndexT>
  void run_colmajor_double_ab() {
    GCXX_SKIP_WITHOUT_DEVICE();

    constexpr int M = 3;
    constexpr int K = 4;
    constexpr int N = 5;

    std::vector<double> hA(M * K), hB(K * N), hC(M * N, 0.0);
    for (int i = 0; i < M * K; ++i) {
      hA[i] = static_cast<double>(i + 1);
    }
    for (int i = 0; i < K * N; ++i) {
      hB[i] = static_cast<double>((i % 3) - 1);
    }

    mat_left<double, int> hostA(hA.data(), M, K);
    mat_left<double, int> hostB(hB.data(), K, N);
    mat_left<double, int> hostCref(hC.data(), M, N);

    std::vector<double> href(M * N);
    mat_left<double, int> hostOut(href.data(), M, N);
    host_gemm(hostA, hostB, hostCref, hostOut, 1.0, 0.0);

    gcxx::Stream str;
    gcxx::uninit_device_buffer<double> dA(str, dev_pool(),
                                          static_cast<std::size_t>(M * K));
    gcxx::uninit_device_buffer<double> dB(str, dev_pool(),
                                          static_cast<std::size_t>(K * N));
    gcxx::uninit_device_buffer<double> dC(str, dev_pool(),
                                          static_cast<std::size_t>(M * N));
    gcxx::Copy(str, dA.data(), hA.data(), static_cast<std::size_t>(M * K));
    gcxx::Copy(str, dB.data(), hB.data(), static_cast<std::size_t>(K * N));

    dmat_left<double, IndexT> A(dA.data(), M, K);
    dmat_left<double, IndexT> B(dB.data(), K, N);
    dmat_left<double, IndexT> C(dC.data(), M, N);

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);
    gcxx::blas::matrix_product(handle, A, B, C);
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

  // Regression gate: pre-fix this computed (A*B)^T for row-major C.
  template <class IndexT>
  void run_rowmajor_and_scaled() {
    GCXX_SKIP_WITHOUT_DEVICE();

    constexpr int M = 3;
    constexpr int K = 4;
    constexpr int N = 5;

    // filled in ROW-major order
    std::vector<double> hA(M * K), hB(K * N), hC(M * N);
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        hA[static_cast<std::size_t>(i * K + j)] =
          static_cast<double>(i + 1) - static_cast<double>(j);
      }
    }
    for (int i = 0; i < K; ++i) {
      for (int j = 0; j < N; ++j) {
        hB[static_cast<std::size_t>(i * N + j)] =
          static_cast<double>((i + j) % 3) - 1.0;
      }
    }
    for (int i = 0; i < M * N; ++i) {
      hC[i] = static_cast<double>(i % 5);
    }

    mat_right<double, int> hostA(hA.data(), M, K);
    mat_right<double, int> hostB(hB.data(), K, N);
    mat_right<double, int> hostCref(hC.data(), M, N);

    std::vector<double> href(M * N);
    mat_right<double, int> hostOut(href.data(), M, N);
    host_gemm(hostA, hostB, hostCref, hostOut, 1.0, 0.0);
    std::vector<double> href_acc(M * N);
    mat_right<double, int> hostOutAcc(href_acc.data(), M, N);
    host_gemm(hostA, hostB, hostCref, hostOutAcc, 2.0, 0.5);

    gcxx::Stream str;
    gcxx::uninit_device_buffer<double> dA(str, dev_pool(),
                                          static_cast<std::size_t>(M * K));
    gcxx::uninit_device_buffer<double> dB(str, dev_pool(),
                                          static_cast<std::size_t>(K * N));
    gcxx::uninit_device_buffer<double> dC(str, dev_pool(),
                                          static_cast<std::size_t>(M * N));
    gcxx::Copy(str, dA.data(), hA.data(), static_cast<std::size_t>(M * K));
    gcxx::Copy(str, dB.data(), hB.data(), static_cast<std::size_t>(K * N));
    gcxx::Copy(str, dC.data(), hC.data(), static_cast<std::size_t>(M * N));

    dmat_right<double, IndexT> A(dA.data(), M, K);
    dmat_right<double, IndexT> B(dB.data(), K, N);
    dmat_right<double, IndexT> C(dC.data(), M, N);

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);

    // Stage 1: write-only C = A*B (transposed-output dispatch, no masking).
    gcxx::blas::matrix_product(handle, A, B, C);
    str.sync();
    std::vector<double> hC_stage1(M * N);
    gcxx::Copy(str, hC_stage1.data(), dC.data(),
               static_cast<std::size_t>(M * N));
    str.sync();
    for (int i = 0; i < M * N; ++i) {
      EXPECT_NEAR(hC_stage1[i], href[i], 1e-9)
        << "row-major write-only mismatch at linear index " << i;
    }

    // Restore the original C so stage 2's beta reads the right addend.
    gcxx::Copy(str, dC.data(), hC.data(), static_cast<std::size_t>(M * N));

    // Stage 2: accumulate C = 2*A*B + 0.5*C via scaled() views.
    gcxx::blas::matrix_product(handle, gcxx::scaled(2.0, A), B,
                               gcxx::scaled(0.5, C), C);
    str.sync();

    std::vector<double> hC_result(M * N);
    gcxx::Copy(str, hC_result.data(), dC.data(),
               static_cast<std::size_t>(M * N));
    str.sync();

    for (int i = 0; i < M * N; ++i) {
      EXPECT_NEAR(hC_result[i], href_acc[i], 1e-9)
        << "row-major accumulate mismatch at linear index " << i;
    }
  }

  // Non-aliased addend E: split path via the in-place matrix_addition step.
  template <class IndexT>
  void run_colmajor_accumulate_unaliased() {
    GCXX_SKIP_WITHOUT_DEVICE();

    constexpr int M = 3;
    constexpr int K = 4;
    constexpr int N = 5;

    std::vector<double> hA(M * K), hB(K * N), hC(M * N, 0.0), hE(M * N);
    for (int i = 0; i < M * K; ++i) {
      hA[i] = static_cast<double>(i + 1);
    }
    for (int i = 0; i < K * N; ++i) {
      hB[i] = static_cast<double>((i % 3) - 1);
    }
    for (int i = 0; i < M * N; ++i) {
      hE[i] = static_cast<double>(2 * (i % 7) - 3);
    }

    mat_left<double, int> hostA(hA.data(), M, K);
    mat_left<double, int> hostB(hB.data(), K, N);
    mat_left<double, int> hostZero(hC.data(), M, N);
    mat_left<double, int> hostE(hE.data(), M, N);

    std::vector<double> href(M * N);
    mat_left<double, int> hostOut(href.data(), M, N);
    host_gemm(hostA, hostB, hostZero, hostOut, 1.0, 0.0);
    for (int i = 0; i < M * N; ++i) {
      href[i] += hE[i];
    }

    gcxx::Stream str;
    gcxx::uninit_device_buffer<double> dA(str, dev_pool(),
                                          static_cast<std::size_t>(M * K));
    gcxx::uninit_device_buffer<double> dB(str, dev_pool(),
                                          static_cast<std::size_t>(K * N));
    gcxx::uninit_device_buffer<double> dC(str, dev_pool(),
                                          static_cast<std::size_t>(M * N));
    gcxx::uninit_device_buffer<double> dE(str, dev_pool(),
                                          static_cast<std::size_t>(M * N));
    gcxx::Copy(str, dA.data(), hA.data(), static_cast<std::size_t>(M * K));
    gcxx::Copy(str, dB.data(), hB.data(), static_cast<std::size_t>(K * N));
    gcxx::Copy(str, dE.data(), hE.data(), static_cast<std::size_t>(M * N));

    dmat_left<double, IndexT> A(dA.data(), M, K);
    dmat_left<double, IndexT> B(dB.data(), K, N);
    dmat_left<double, IndexT> C(dC.data(), M, N);
    dmat_left<double, IndexT> E(dE.data(), M, N);

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);
    gcxx::blas::matrix_product(handle, A, B, E, C);
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

  // device_scalar selects device pointer mode; both factors device-resident.
  template <class IndexT>
  void run_colmajor_double_ab_device_scalar() {
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
    gcxx::uninit_device_buffer<double> dA(str, dev_pool(),
                                          static_cast<std::size_t>(M * K));
    gcxx::uninit_device_buffer<double> dB(str, dev_pool(),
                                          static_cast<std::size_t>(K * N));
    gcxx::uninit_device_buffer<double> dC(str, dev_pool(),
                                          static_cast<std::size_t>(M * N));
    auto dAlpha =
      gcxx::make_device_scalar<double>(gcxx::DeviceHandle{0}, str, alpha);
    auto dBeta =
      gcxx::make_device_scalar<double>(gcxx::DeviceHandle{0}, str, beta);
    gcxx::Copy(str, dA.data(), hA.data(), static_cast<std::size_t>(M * K));
    gcxx::Copy(str, dB.data(), hB.data(), static_cast<std::size_t>(K * N));
    gcxx::Copy(str, dC.data(), hC.data(), static_cast<std::size_t>(M * N));

    dmat_left<double, IndexT> A(dA.data(), M, K);
    dmat_left<double, IndexT> B(dB.data(), K, N);
    dmat_left<double, IndexT> C(dC.data(), M, N);

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

TEST(BlasGemm, ColMajorDouble_AB) {
  run_colmajor_double_ab<int>();
}

TEST(BlasGemm, ColMajorDouble_AB_64bitIndex) {
  run_colmajor_double_ab<std::int64_t>();
}

TEST(BlasGemm, RowMajorDouble_ScaledAccumulate) {
  run_rowmajor_and_scaled<int>();
}

TEST(BlasGemm, RowMajorDouble_ScaledAccumulate_64bitIndex) {
  run_rowmajor_and_scaled<std::int64_t>();
}

TEST(BlasGemm, ColMajorDouble_AB_AccumulateUnaliased) {
  run_colmajor_accumulate_unaliased<int>();
}

TEST(BlasGemm, ColMajorDouble_AB_DeviceScalar) {
  run_colmajor_double_ab_device_scalar<int>();
}
