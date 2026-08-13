// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// End-to-end GEMM tests: C = alpha * op(A) * op(B) + beta * C via cuBLAS,
// compared against a host reference. GPU-gated — skipped when no device is
// present, but the template must still compile (it instantiates gemm and its
// cu/hipblasGemmEx dispatch, plus the GemmEx_64 64-bit-integer dispatch for the
// int64_t index_type variant).

#include "tests_common.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gcxx/blas_api.hpp>
#include <gcxx/runtime/memory/copy.hpp>
#include <gcxx/runtime/memory/smartpointers/pointers.hpp>
#include <gcxx/runtime/memory/spans/mdspan/mdspan.hpp>

namespace {

  template <class IndexT>
  using dextents2d = gcxx::dextents<IndexT, 2>;

  template <class T, class IndexT>
  using mat_left = gcxx::mdspan<T, dextents2d<IndexT>, gcxx::layout_left,
                                gcxx::default_accessor<T>>;

  // Column-major host reference: out = alpha * a * b + beta * cref.
  template <class T, class S>
  void host_gemm(const mat_left<T, int>& a, const mat_left<T, int>& b,
                 const mat_left<T, int>& cref, std::vector<T>& out, S alpha,
                 S beta) {
    const int m = a.extent(0);
    const int k = a.extent(1);
    const int n = b.extent(1);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        S acc{};
        for (int p = 0; p < k; ++p) {
          acc += static_cast<S>(a(i, p)) * static_cast<S>(b(p, j));
        }
        out[i + j * m] = alpha * acc + beta * static_cast<S>(cref(i, j));
      }
    }
  }

  // Runs C = A * B for column-major double operands whose device mdspan
  // index_type is IndexT — this is what selects the cu/hipblas integer
  // interface (GemmEx for int, GemmEx_64 for a 64-bit index_type).
  template <class IndexT>
  void run_colmajor_double_ab() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

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
    host_gemm<double, double>(hostA, hostB, hostCref, href, 1.0, 0.0);

    gcxx::Stream str;
    auto dA =
      gcxx::make_device_unique_ptr<double>(static_cast<std::size_t>(M * K));
    auto dB =
      gcxx::make_device_unique_ptr<double>(static_cast<std::size_t>(K * N));
    auto dC =
      gcxx::make_device_unique_ptr<double>(static_cast<std::size_t>(M * N));
    gcxx::Copy(str, dA.get(), hA.data(), static_cast<std::size_t>(M * K));
    gcxx::Copy(str, dB.get(), hB.data(), static_cast<std::size_t>(K * N));

    mat_left<double, IndexT> A(dA.get(), M, K);
    mat_left<double, IndexT> B(dB.get(), K, N);
    mat_left<double, IndexT> C(dC.get(), M, N);

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);
    gcxx::blas::gemm(handle, 1.0, A, B, 0.0, C);
    str.Synchronize();

    std::vector<double> hC_result(M * N);
    gcxx::Copy(str, hC_result.data(), dC.get(),
               static_cast<std::size_t>(M * N));
    str.Synchronize();

    for (int i = 0; i < M * N; ++i) {
      EXPECT_NEAR(hC_result[i], href[i], 1e-9)
        << "mismatch at linear index " << i;
    }
  }

  // Device-pointer-mode variant: alpha/beta live in device memory and are
  // passed via gcxx::blas::device_scalar, selecting device pointer mode. Uses
  // non-trivial alpha/beta and a non-zero C so both scalars are actually read.
  // (Also serves as the compile check for the device_scalar dispatch branch.)
  template <class IndexT>
  void run_colmajor_double_ab_device_scalar() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

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
    host_gemm<double, double>(hostA, hostB, hostCref, href, alpha, beta);

    gcxx::Stream str;
    auto dA =
      gcxx::make_device_unique_ptr<double>(static_cast<std::size_t>(M * K));
    auto dB =
      gcxx::make_device_unique_ptr<double>(static_cast<std::size_t>(K * N));
    auto dC =
      gcxx::make_device_unique_ptr<double>(static_cast<std::size_t>(M * N));
    auto dAlpha = gcxx::make_device_unique_ptr<double>(std::size_t{1});
    auto dBeta  = gcxx::make_device_unique_ptr<double>(std::size_t{1});
    gcxx::Copy(str, dA, hA.data(), static_cast<std::size_t>(M * K));
    gcxx::Copy(str, dB, hB.data(), static_cast<std::size_t>(K * N));
    gcxx::Copy(str, dC, hC.data(), static_cast<std::size_t>(M * N));
    gcxx::Copy(str, dAlpha, &alpha, std::size_t{1});
    gcxx::Copy(str, dBeta, &beta, std::size_t{1});

    mat_left<double, IndexT> A(dA.get(), M, K);
    mat_left<double, IndexT> B(dB.get(), K, N);
    mat_left<double, IndexT> C(dC.get(), M, N);

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);
    gcxx::blas::gemm(handle, gcxx::blas::device_scalar<double>{dAlpha.get()}, A,
                     B, gcxx::blas::device_scalar<double>{dBeta.get()}, C);
    str.Synchronize();

    std::vector<double> hC_result(M * N);
    gcxx::Copy(str, hC_result.data(), dC.get(),
               static_cast<std::size_t>(M * N));
    str.Synchronize();

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

TEST(BlasGemm, ColMajorDouble_AB_DeviceScalar) {
  run_colmajor_double_ab_device_scalar<int>();
}
