// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// End-to-end GEAM / DGMM tests: C = alpha*op(A) + beta*op(B) and C =
// diag(x)*A / A*diag(x) via the typed cu/hipblasS/Dgeam and S/Ddgmm entry
// points, compared against a host reference. GPU-gated — skipped when no
// device is present, but the template must still compile (it instantiates
// both the int and the int64_t index_type dispatch, i.e. the *_64 entry
// points).

#include "tests_common.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <gcxx/blas_api.hpp>
#include <gcxx/runtime/memory/copy.hpp>
#include <gcxx/runtime/memory/smartpointers/pointers.hpp>
#include <gcxx/runtime/memory/spans/mdspan/make_mdspan.hpp>
#include <gcxx/runtime/memory/spans/mdspan/mdspan.hpp>

namespace {

  template <class IndexT>
  using mat_left =
    gcxx::mdspan<double, gcxx::dextents<IndexT, 2>, gcxx::layout_left,
                 gcxx::default_accessor<double>>;

  // Device-memory counterpart required by the gcxx::blas operations.
  template <class IndexT>
  using dmat_left =
    gcxx::device_mdspan<double, gcxx::dextents<IndexT, 2>, gcxx::layout_left>;

  template <class IndexT>
  void run_geam() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    constexpr int M    = 3;
    constexpr int N    = 4;
    const double alpha = 2.0, beta = -1.0;

    std::vector<double> hA(M * N), hB(M * N);
    for (int i = 0; i < M * N; ++i) {
      hA[i] = static_cast<double>(i + 1);
      hB[i] = static_cast<double>(2 * i);
    }

    gcxx::Stream str;
    auto dA = gcxx::make_device_unique_ptr<double>(std::size_t{M * N});
    auto dB = gcxx::make_device_unique_ptr<double>(std::size_t{M * N});
    auto dC = gcxx::make_device_unique_ptr<double>(std::size_t{M * N});
    gcxx::Copy(str, dA.get(), hA.data(), std::size_t{M * N});
    gcxx::Copy(str, dB.get(), hB.data(), std::size_t{M * N});

    dmat_left<IndexT> A(dA.get(), M, N);
    dmat_left<IndexT> B(dB.get(), M, N);
    dmat_left<IndexT> C(dC.get(), M, N);

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);
    gcxx::blas::geam(handle, alpha, A, beta, B, C);
    str.Synchronize();

    std::vector<double> hResult(M * N);
    gcxx::Copy(str, hResult.data(), dC.get(), std::size_t{M * N});
    str.Synchronize();

    for (int i = 0; i < M * N; ++i) {
      EXPECT_NEAR(hResult[i], alpha * hA[i] + beta * hB[i], 1e-9)
        << "geam mismatch at " << i;
    }
  }

  template <class IndexT, class Side>
  void run_dgmm() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    constexpr int M = 3;
    constexpr int N = 4;
    // the diagonal length follows from the side: rows (left) or cols (right)
    constexpr int xlen = std::is_same_v<Side, gcxx::blas::left_t> ? M : N;

    std::vector<double> hA(M * N), hX(xlen);
    for (int i = 0; i < M * N; ++i) {
      hA[i] = static_cast<double>(i + 1);
    }
    for (int i = 0; i < xlen; ++i) {
      hX[i] = static_cast<double>(i + 2);
    }

    gcxx::Stream str;
    auto dA = gcxx::make_device_unique_ptr<double>(std::size_t{M * N});
    auto dX =
      gcxx::make_device_unique_ptr<double>(static_cast<std::size_t>(xlen));
    auto dC = gcxx::make_device_unique_ptr<double>(std::size_t{M * N});
    gcxx::Copy(str, dA.get(), hA.data(), std::size_t{M * N});
    gcxx::Copy(str, dX.get(), hX.data(), static_cast<std::size_t>(xlen));

    dmat_left<IndexT> A(dA.get(), M, N);
    dmat_left<IndexT> C(dC.get(), M, N);
    auto X = gcxx::make_device_vector<IndexT>(
      gcxx::span(dX.get(), static_cast<std::size_t>(xlen)));

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);
    gcxx::blas::dgmm(handle, Side{}, A, X, C);
    str.Synchronize();

    std::vector<double> hResult(M * N);
    gcxx::Copy(str, hResult.data(), dC.get(), std::size_t{M * N});
    str.Synchronize();

    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        const double scale =
          std::is_same_v<Side, gcxx::blas::left_t> ? hX[i] : hX[j];
        EXPECT_NEAR(hResult[i + j * M], scale * hA[i + j * M], 1e-9)
          << "dgmm mismatch at (" << i << "," << j << ")";
      }
    }
  }

  // Layout-independence gate: a NON-SQUARE all-row-major geam must compute
  // C = alpha*A + beta*B mathematically. Before the output-orientation fix
  // this path flipped m/n against the operand ld's and read out of bounds.
  template <class IndexT>
  void run_geam_rowmajor() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    constexpr int M    = 3;
    constexpr int N    = 5;
    const double alpha = 2.0, beta = -1.0;

    // filled in ROW-major order
    std::vector<double> hA(M * N), hB(M * N);
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        hA[static_cast<std::size_t>(i * N + j)] =
          static_cast<double>(i + 1) - static_cast<double>(j);
        hB[static_cast<std::size_t>(i * N + j)] =
          static_cast<double>(2 * i) + 0.5 * static_cast<double>(j);
      }
    }

    using mat_right =
      gcxx::device_mdspan<double, gcxx::dextents<IndexT, 2>,
                          gcxx::layout_right>;

    gcxx::Stream str;
    auto dA = gcxx::make_device_unique_ptr<double>(std::size_t{M * N});
    auto dB = gcxx::make_device_unique_ptr<double>(std::size_t{M * N});
    auto dC = gcxx::make_device_unique_ptr<double>(std::size_t{M * N});
    gcxx::Copy(str, dA.get(), hA.data(), std::size_t{M * N});
    gcxx::Copy(str, dB.get(), hB.data(), std::size_t{M * N});

    mat_right A(dA.get(), M, N);
    mat_right B(dB.get(), M, N);
    mat_right C(dC.get(), M, N);

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);
    gcxx::blas::geam(handle, alpha, A, beta, B, C);
    str.Synchronize();

    std::vector<double> hResult(M * N);
    gcxx::Copy(str, hResult.data(), dC.get(), std::size_t{M * N});
    str.Synchronize();

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        const auto idx = static_cast<std::size_t>(i * N + j);
        EXPECT_NEAR(hResult[idx], alpha * hA[idx] + beta * hB[idx], 1e-9)
          << "row-major geam mismatch at (" << i << "," << j << ")";
      }
    }
  }

}  // namespace

TEST(BlasGeam, Double) {
  run_geam<int>();
}

TEST(BlasGeam, Double_64bitIndex) {
  run_geam<std::int64_t>();
}

TEST(BlasGeam, RowMajor_NonSquare_Double) {
  run_geam_rowmajor<int>();
}

TEST(BlasGeam, RowMajor_NonSquare_Double_64bitIndex) {
  run_geam_rowmajor<std::int64_t>();
}

TEST(BlasDgmm, Left_Double) {
  run_dgmm<int, gcxx::blas::left_t>();
}

TEST(BlasDgmm, Right_Double_64bitIndex) {
  run_dgmm<std::int64_t, gcxx::blas::right_t>();
}
