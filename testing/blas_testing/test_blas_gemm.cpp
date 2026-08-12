// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// End-to-end GEMM tests: C = alpha * op(A) * op(B) + beta * C via cuBLAS,
// compared against a host reference. GPU-gated — skipped when no device is
// present, but the template must still compile (it instantiates gemm and its
// typed cublas?gemm/hipblas?gemm dispatch).

#include "tests_common.hpp"

#include <cstddef>
#include <vector>

#include <gcxx/blas_api.hpp>
#include <gcxx/runtime/memory/copy.hpp>
#include <gcxx/runtime/memory/smartpointers/pointers.hpp>
#include <gcxx/runtime/memory/spans/mdspan/mdspan.hpp>

namespace {

  using dextents2d = gcxx::dextents<int, 2>;
  using def_acc_d  = gcxx::default_accessor<double>;

  template <class T>
  using mat_left =
    gcxx::mdspan<T, dextents2d, gcxx::layout_left, gcxx::default_accessor<T>>;

  // Column-major host reference: cref = alpha * a * b + beta * cref.
  template <class T, class S>
  void host_gemm(const mat_left<T>& a, const mat_left<T>& b,
                 const mat_left<T>& cref, std::vector<T>& out, S alpha,
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

}  // namespace

TEST(BlasGemm, ColMajorDouble_AB) {
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

  mat_left<double> hostA(hA.data(), M, K);
  mat_left<double> hostB(hB.data(), K, N);
  mat_left<double> hostCref(hC.data(), M, N);

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

  mat_left<double> A(dA.get(), M, K);
  mat_left<double> B(dB.get(), K, N);
  mat_left<double> C(dC.get(), M, N);

  gcxx::blas::BlasHandle handle;
  handle.setStream(str);
  gcxx::blas::gemm(handle, 1.0, A, B, 0.0, C);
  str.Synchronize();

  std::vector<double> hC_result(M * N);
  gcxx::Copy(str, hC_result.data(), dC.get(), static_cast<std::size_t>(M * N));
  str.Synchronize();

  for (int i = 0; i < M * N; ++i) {
    EXPECT_NEAR(hC_result[i], href[i], 1e-9)
      << "mismatch at linear index " << i;
  }
}
