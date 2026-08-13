// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// End-to-end GEMV tests: y = alpha * op(A) * x + beta * y via cuBLAS,
// compared against a host reference. GPU-gated — skipped when no device is
// present, but the template must still compile (it instantiates gemv and its
// cublasSgemv_v2 / Dgemv dispatch, plus the *_v2_64 64-bit-integer dispatch for
// the int64_t index_type variant).

#include "tests_common.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gcxx/blas_api.hpp>
#include <gcxx/runtime/memory/copy.hpp>
#include <gcxx/runtime/memory/smartpointers/pointers.hpp>
#include <gcxx/runtime/memory/spans/mdspan/make_mdspan.hpp>
#include <gcxx/runtime/memory/spans/mdspan/mdspan.hpp>

namespace {

  template <class IndexT>
  using dextents2d = gcxx::dextents<IndexT, 2>;
  template <class IndexT>
  using dextents1d = gcxx::dextents<IndexT, 1>;

  template <class T, class IndexT>
  using mat_left = gcxx::mdspan<T, dextents2d<IndexT>, gcxx::layout_left,
                                gcxx::default_accessor<T>>;
  template <class T, class IndexT>
  using vec = gcxx::mdspan<T, dextents1d<IndexT>, gcxx::layout_left,
                           gcxx::default_accessor<T>>;

  // Column-major host reference: out = alpha * a * x + beta * yref, where a is
  // (m x k), x is length k and out/yref are length m.
  template <class T, class S>
  void host_gemv(const mat_left<T, int>& a, const vec<T, int>& x,
                 const vec<T, int>& yref, std::vector<T>& out, S alpha, S beta) {
    const int m = a.extent(0);
    const int k = a.extent(1);
    for (int i = 0; i < m; ++i) {
      S acc{};
      for (int p = 0; p < k; ++p) {
        acc += static_cast<S>(a(i, p)) * static_cast<S>(x(p));
      }
      out[i] = alpha * acc + beta * static_cast<S>(yref(i));
    }
  }

  // Runs y = A * x for column-major double operands whose device mdspan
  // index_type is IndexT — this is what selects the cu/hipblas integer
  // interface (Sgemv_v2/Dgemv_v2 for int, the *_v2_64 entry for a 64-bit
  // index_type).
  template <class IndexT>
  void run_colmajor_double_ax() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    constexpr int M = 3;
    constexpr int K = 4;

    std::vector<double> hA(M * K), hX(K), hY(M, 0.0);
    for (int i = 0; i < M * K; ++i) {
      hA[i] = static_cast<double>(i + 1);
    }
    for (int i = 0; i < K; ++i) {
      hX[i] = static_cast<double>((i % 3) - 1);
    }

    mat_left<double, int> hostA(hA.data(), M, K);
    vec<double, int>      hostX(hX.data(), K);
    vec<double, int>      hostYref(hY.data(), M);

    std::vector<double> href(M);
    host_gemv<double, double>(hostA, hostX, hostYref, href, 1.0, 0.0);

    gcxx::Stream str;
    auto dA =
      gcxx::make_device_unique_ptr<double>(static_cast<std::size_t>(M * K));
    auto dX = gcxx::make_device_unique_ptr<double>(static_cast<std::size_t>(K));
    auto dY = gcxx::make_device_unique_ptr<double>(static_cast<std::size_t>(M));
    gcxx::Copy(str, dA.get(), hA.data(), static_cast<std::size_t>(M * K));
    gcxx::Copy(str, dX.get(), hX.data(), static_cast<std::size_t>(K));

    mat_left<double, IndexT> A(dA.get(), M, K);
    auto X = gcxx::make_vector<IndexT>(gcxx::span(dX.get(), K));
    auto Y = gcxx::make_vector<IndexT>(gcxx::span(dY.get(), M));

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);
    gcxx::blas::gemv(handle, 1.0, A, X, 0.0, Y);
    str.Synchronize();

    std::vector<double> hY_result(M);
    gcxx::Copy(str, hY_result.data(), dY.get(), static_cast<std::size_t>(M));
    str.Synchronize();

    for (int i = 0; i < M; ++i) {
      EXPECT_NEAR(hY_result[i], href[i], 1e-9)
        << "mismatch at index " << i;
    }
  }

}  // namespace

TEST(BlasGemv, ColMajorDouble_Ax) {
  run_colmajor_double_ax<int>();
}

TEST(BlasGemv, ColMajorDouble_Ax_64bitIndex) {
  run_colmajor_double_ax<std::int64_t>();
}
