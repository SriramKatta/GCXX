// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
<<<<<<< HEAD
// End-to-end matrix_vector_product (P1673 gemv) tests via cuBLAS with
// scaled() views and layout gates; GPU-gated, must still compile everywhere.
=======
//
// End-to-end GEMV tests: y = alpha * op(A) * x + beta * y via cuBLAS,
// compared against a host reference. GPU-gated — skipped when no device is
// present, but the template must still compile (it instantiates gemv and its
// cublasSgemv_v2 / Dgemv dispatch, plus the *_v2_64 64-bit-integer dispatch for
// the int64_t index_type variant).
>>>>>>> f6989c9 (Amending to new examples)

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
<<<<<<< HEAD
  using mat_right = gcxx::mdspan<T, dextents2d<IndexT>, gcxx::layout_right,
                                 gcxx::default_accessor<T>>;

  // Device-memory counterparts required by gcxx::blas::matrix_vector_product.
  template <class T, class IndexT>
  using dmat_left =
    gcxx::device_mdspan<T, dextents2d<IndexT>, gcxx::layout_left>;
  template <class T, class IndexT>
  using dmat_right =
    gcxx::device_mdspan<T, dextents2d<IndexT>, gcxx::layout_right>;
  template <class T, class IndexT>
  using vec = gcxx::mdspan<T, dextents1d<IndexT>, gcxx::layout_left,
                           gcxx::default_accessor<T>>;

  template <class MatT, class VecT, class T, class S>
  void host_gemv(const MatT& a, const VecT& x, const VecT& yref,
                 std::vector<T>& out, S alpha, S beta) {
    const int m = static_cast<int>(a.extent(0));
    const int k = static_cast<int>(a.extent(1));
=======
  using vec = gcxx::mdspan<T, dextents1d<IndexT>, gcxx::layout_left,
                           gcxx::default_accessor<T>>;

  // Column-major host reference: out = alpha * a * x + beta * yref, where a is
  // (m x k), x is length k and out/yref are length m.
  template <class T, class S>
  void host_gemv(const mat_left<T, int>& a, const vec<T, int>& x,
                 const vec<T, int>& yref, std::vector<T>& out, S alpha,
                 S beta) {
    const int m = a.extent(0);
    const int k = a.extent(1);
>>>>>>> f6989c9 (Amending to new examples)
    for (int i = 0; i < m; ++i) {
      S acc{};
      for (int p = 0; p < k; ++p) {
        acc += static_cast<S>(a(i, p)) * static_cast<S>(x(p));
      }
      out[i] = alpha * acc + beta * static_cast<S>(yref(i));
    }
  }

<<<<<<< HEAD
  // index_type picks the cu/hipblas entry: Sgemv_v2/Dgemv_v2 vs *_v2_64.
=======
  // Runs y = A * x for column-major double operands whose device mdspan
  // index_type is IndexT — this is what selects the cu/hipblas integer
  // interface (Sgemv_v2/Dgemv_v2 for int, the *_v2_64 entry for a 64-bit
  // index_type).
>>>>>>> f6989c9 (Amending to new examples)
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
    vec<double, int> hostX(hX.data(), K);
    vec<double, int> hostYref(hY.data(), M);

    std::vector<double> href(M);
<<<<<<< HEAD
    host_gemv(hostA, hostX, hostYref, href, 1.0, 0.0);
=======
    host_gemv<double, double>(hostA, hostX, hostYref, href, 1.0, 0.0);
>>>>>>> f6989c9 (Amending to new examples)

    gcxx::Stream str;
    auto dA =
      gcxx::make_device_unique_ptr<double>(static_cast<std::size_t>(M * K));
    auto dX = gcxx::make_device_unique_ptr<double>(static_cast<std::size_t>(K));
    auto dY = gcxx::make_device_unique_ptr<double>(static_cast<std::size_t>(M));
    gcxx::Copy(str, dA.get(), hA.data(), static_cast<std::size_t>(M * K));
    gcxx::Copy(str, dX.get(), hX.data(), static_cast<std::size_t>(K));

<<<<<<< HEAD
    dmat_left<double, IndexT> A(dA.get(), M, K);
    auto X = gcxx::make_device_vector<IndexT>(gcxx::span(dX.get(), K));
    auto Y = gcxx::make_device_vector<IndexT>(gcxx::span(dY.get(), M));

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);
    gcxx::blas::matrix_vector_product(handle, A, X, Y);
=======
    mat_left<double, IndexT> A(dA.get(), M, K);
    auto X = gcxx::make_vector<IndexT>(gcxx::span(dX.get(), K));
    auto Y = gcxx::make_vector<IndexT>(gcxx::span(dY.get(), M));

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);
    gcxx::blas::gemv(handle, 1.0, A, X, 0.0, Y);
>>>>>>> f6989c9 (Amending to new examples)
    str.Synchronize();

    std::vector<double> hY_result(M);
    gcxx::Copy(str, hY_result.data(), dY.get(), static_cast<std::size_t>(M));
    str.Synchronize();

    for (int i = 0; i < M; ++i) {
      EXPECT_NEAR(hY_result[i], href[i], 1e-9) << "mismatch at index " << i;
    }
  }

<<<<<<< HEAD
  // Layout gate: row-major buffer must give same A*x and scaled accumulate.
  template <class IndexT>
  void run_rowmajor_and_scaled() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    constexpr int M = 3;
    constexpr int K = 4;

    // hA is filled in ROW-major order
    std::vector<double> hA(M * K), hX(K), hY(M);
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        hA[static_cast<std::size_t>(i * K + j)] =
          static_cast<double>(i + 1) - static_cast<double>(j);
      }
    }
    for (int i = 0; i < K; ++i) {
      hX[i] = static_cast<double>((i % 3) - 1);
    }
    for (int i = 0; i < M; ++i) {
      hY[i] = static_cast<double>(i);
    }

    mat_right<double, int> hostA(hA.data(), M, K);
    vec<double, int> hostX(hX.data(), K);
    vec<double, int> hostY(hY.data(), M);

    std::vector<double> href(M);
    host_gemv(hostA, hostX, hostY, href, 1.0, 0.0);
    std::vector<double> href_acc(M);
    host_gemv(hostA, hostX, hostY, href_acc, 2.0, 0.5);

    gcxx::Stream str;
    auto dA =
      gcxx::make_device_unique_ptr<double>(static_cast<std::size_t>(M * K));
    auto dX = gcxx::make_device_unique_ptr<double>(static_cast<std::size_t>(K));
    auto dY = gcxx::make_device_unique_ptr<double>(static_cast<std::size_t>(M));
    gcxx::Copy(str, dA.get(), hA.data(), static_cast<std::size_t>(M * K));
    gcxx::Copy(str, dX.get(), hX.data(), static_cast<std::size_t>(K));
    gcxx::Copy(str, dY.get(), hY.data(), static_cast<std::size_t>(M));

    dmat_right<double, IndexT> A(dA.get(), M, K);
    auto X = gcxx::make_device_vector<IndexT>(gcxx::span(dX.get(), K));
    auto Y = gcxx::make_device_vector<IndexT>(gcxx::span(dY.get(), M));

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);

    // Stage 1: write-only y = A*x (row-major dispatch, no accumulate mask).
    gcxx::blas::matrix_vector_product(handle, A, X, Y);
    str.Synchronize();
    std::vector<double> hY_stage1(M);
    gcxx::Copy(str, hY_stage1.data(), dY.get(), static_cast<std::size_t>(M));
    str.Synchronize();
    for (int i = 0; i < M; ++i) {
      EXPECT_NEAR(hY_stage1[i], href[i], 1e-9)
        << "row-major write-only mismatch at index " << i;
    }

    // Restore the original y so stage 2's beta reads the right addend.
    gcxx::Copy(str, dY.get(), hY.data(), static_cast<std::size_t>(M));

    // Stage 2: accumulate y = 2*A*x + 0.5*y via scaled() views.
    gcxx::blas::matrix_vector_product(handle, gcxx::scaled(2.0, A), X,
                                      gcxx::scaled(0.5, Y), Y);
    str.Synchronize();

    std::vector<double> hY_result(M);
    gcxx::Copy(str, hY_result.data(), dY.get(), static_cast<std::size_t>(M));
    str.Synchronize();

    for (int i = 0; i < M; ++i) {
      EXPECT_NEAR(hY_result[i], href_acc[i], 1e-9)
        << "row-major accumulate mismatch at index " << i;
    }
  }

  // transposed(A): y = A^T*x with no data movement.
  template <class IndexT>
  void run_transposed_operand() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    constexpr int M = 4;  // A is (M x K) column-major
    constexpr int K = 3;

    std::vector<double> hA(M * K), hX(M), hY(K, 0.0);
    for (int i = 0; i < M * K; ++i) {
      hA[i] = static_cast<double>(2 * i - 5);
    }
    for (int i = 0; i < M; ++i) {
      hX[i] = static_cast<double>(i + 1);
    }

    mat_left<double, int> hostA(hA.data(), M, K);
    vec<double, int> hostX(hX.data(), M);
    vec<double, int> hostY(hY.data(), K);

    std::vector<double> href(K);
    host_gemv(gcxx::transposed(hostA), hostX, hostY, href, 1.0, 0.0);

    gcxx::Stream str;
    auto dA =
      gcxx::make_device_unique_ptr<double>(static_cast<std::size_t>(M * K));
    auto dX = gcxx::make_device_unique_ptr<double>(static_cast<std::size_t>(M));
    auto dY = gcxx::make_device_unique_ptr<double>(static_cast<std::size_t>(K));
    gcxx::Copy(str, dA.get(), hA.data(), static_cast<std::size_t>(M * K));
    gcxx::Copy(str, dX.get(), hX.data(), static_cast<std::size_t>(M));

    dmat_left<double, IndexT> A(dA.get(), M, K);
    auto X = gcxx::make_device_vector<IndexT>(gcxx::span(dX.get(), M));
    auto Y = gcxx::make_device_vector<IndexT>(gcxx::span(dY.get(), K));

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);
    gcxx::blas::matrix_vector_product(handle, gcxx::transposed(A), X, Y);
    str.Synchronize();

    std::vector<double> hY_result(K);
    gcxx::Copy(str, hY_result.data(), dY.get(), static_cast<std::size_t>(K));
    str.Synchronize();

    for (int i = 0; i < K; ++i) {
      EXPECT_NEAR(hY_result[i], href[i], 1e-9) << "mismatch at index " << i;
    }
  }

  // device_scalar selects device pointer mode; both factors device-resident.
=======
  // Device-pointer-mode variant: alpha/beta live in device memory and are
  // passed via gcxx::blas::device_scalar, selecting device pointer mode. Uses
  // non-trivial alpha/beta and a non-zero y so both scalars are actually read.
  // (Also serves as the compile check for the device_scalar dispatch branch.)
>>>>>>> f6989c9 (Amending to new examples)
  template <class IndexT>
  void run_colmajor_double_ax_device_scalar() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    constexpr int M = 3;
    constexpr int K = 4;
    double alpha    = 2.0;
    double beta     = 1.0;

    std::vector<double> hA(M * K), hX(K), hY(M);
    for (int i = 0; i < M * K; ++i) {
      hA[i] = static_cast<double>(i + 1);
    }
    for (int i = 0; i < K; ++i) {
      hX[i] = static_cast<double>((i % 3) - 1);
    }
    for (int i = 0; i < M; ++i) {
      hY[i] = static_cast<double>(i);
    }

    mat_left<double, int> hostA(hA.data(), M, K);
    vec<double, int> hostX(hX.data(), K);
    vec<double, int> hostYref(hY.data(), M);

    std::vector<double> href(M);
<<<<<<< HEAD
    host_gemv(hostA, hostX, hostYref, href, alpha, beta);
=======
    host_gemv<double, double>(hostA, hostX, hostYref, href, alpha, beta);
>>>>>>> f6989c9 (Amending to new examples)

    gcxx::Stream str;
    auto dA =
      gcxx::make_device_unique_ptr<double>(static_cast<std::size_t>(M * K));
    auto dX = gcxx::make_device_unique_ptr<double>(static_cast<std::size_t>(K));
    auto dY = gcxx::make_device_unique_ptr<double>(static_cast<std::size_t>(M));
    auto dAlpha = gcxx::make_device_unique_ptr<double>(std::size_t{1});
    auto dBeta  = gcxx::make_device_unique_ptr<double>(std::size_t{1});
    gcxx::Copy(str, dA, hA.data(), static_cast<std::size_t>(M * K));
    gcxx::Copy(str, dX, hX.data(), static_cast<std::size_t>(K));
    gcxx::Copy(str, dY, hY.data(), static_cast<std::size_t>(M));
    gcxx::Copy(str, dAlpha, &alpha, std::size_t{1});
    gcxx::Copy(str, dBeta, &beta, std::size_t{1});

<<<<<<< HEAD
    dmat_left<double, IndexT> A(dA.get(), M, K);
    auto X = gcxx::make_device_vector<IndexT>(gcxx::span(dX.get(), K));
    auto Y = gcxx::make_device_vector<IndexT>(gcxx::span(dY.get(), M));

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);
    gcxx::blas::matrix_vector_product(
      handle, gcxx::scaled(gcxx::blas::device_scalar<double>{dAlpha.get()}, A),
      X, gcxx::scaled(gcxx::blas::device_scalar<double>{dBeta.get()}, Y), Y);
=======
    mat_left<double, IndexT> A(dA.get(), M, K);
    auto X = gcxx::make_vector<IndexT>(gcxx::span(dX.get(), K));
    auto Y = gcxx::make_vector<IndexT>(gcxx::span(dY.get(), M));

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);
    gcxx::blas::gemv(handle, gcxx::blas::device_scalar<double>{dAlpha.get()}, A,
                     X, gcxx::blas::device_scalar<double>{dBeta.get()}, Y);
>>>>>>> f6989c9 (Amending to new examples)
    str.Synchronize();

    std::vector<double> hY_result(M);
    gcxx::Copy(str, hY_result.data(), dY.get(), static_cast<std::size_t>(M));
    str.Synchronize();

    for (int i = 0; i < M; ++i) {
      EXPECT_NEAR(hY_result[i], href[i], 1e-9) << "mismatch at index " << i;
    }
  }

}  // namespace

TEST(BlasGemv, ColMajorDouble_Ax) {
  run_colmajor_double_ax<int>();
}

TEST(BlasGemv, ColMajorDouble_Ax_64bitIndex) {
  run_colmajor_double_ax<std::int64_t>();
}

<<<<<<< HEAD
TEST(BlasGemv, RowMajorDouble_ScaledAccumulate) {
  run_rowmajor_and_scaled<int>();
}

TEST(BlasGemv, RowMajorDouble_ScaledAccumulate_64bitIndex) {
  run_rowmajor_and_scaled<std::int64_t>();
}

TEST(BlasGemv, TransposedOperand) {
  run_transposed_operand<int>();
}

=======
>>>>>>> f6989c9 (Amending to new examples)
TEST(BlasGemv, ColMajorDouble_Ax_DeviceScalar) {
  run_colmajor_double_ax_device_scalar<int>();
}
