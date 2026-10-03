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
    std::vector<double> href_scaled(M * N);
    mat_right<double, int> hostOutScaled(href_scaled.data(), M, N);
    host_gemm(hostA, hostB, hostCref, hostOutScaled, 2.0, 0.0);

    gcxx::Stream str;
    gcxx::uninit_device_buffer<double> dA(str, dev_pool(),
                                          static_cast<std::size_t>(M * K));
    gcxx::uninit_device_buffer<double> dB(str, dev_pool(),
                                          static_cast<std::size_t>(K * N));
    gcxx::uninit_device_buffer<double> dC(str, dev_pool(),
                                          static_cast<std::size_t>(M * N));
    gcxx::Copy(str, dA.data(), hA.data(), static_cast<std::size_t>(M * K));
    gcxx::Copy(str, dB.data(), hB.data(), static_cast<std::size_t>(K * N));

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

    // Stage 2: write-only C = 2*(A*B) via a scaled() input view.
    gcxx::blas::matrix_product(handle, gcxx::scaled(2.0, A), B, C);
    str.sync();

    std::vector<double> hC_result(M * N);
    gcxx::Copy(str, hC_result.data(), dC.data(),
               static_cast<std::size_t>(M * N));
    str.sync();

    for (int i = 0; i < M * N; ++i) {
      EXPECT_NEAR(hC_result[i], href_scaled[i], 1e-9)
        << "row-major scaled write-only mismatch at linear index " << i;
    }
  }

}  // namespace

TEST(BlasGemm, ColMajorDouble_AB) {
  run_colmajor_double_ab<int>();
}

TEST(BlasGemm, ColMajorDouble_AB_64bitIndex) {
  run_colmajor_double_ab<std::int64_t>();
}

TEST(BlasGemm, RowMajorDouble_ScaledWriteOnly) {
  run_rowmajor_and_scaled<int>();
}

TEST(BlasGemm, RowMajorDouble_ScaledWriteOnly_64bitIndex) {
  run_rowmajor_and_scaled<std::int64_t>();
}
