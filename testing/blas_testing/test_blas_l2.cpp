// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
// End-to-end Level-2 symv/trmv/trsv/ger/syr/syr2 tests vs host references;
// unread-triangle garbage pins tag semantics. GPU-gated, still compiles.

#include "tests_common.hpp"

#include <cmath>
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
  using dmat_left =
    gcxx::device_mdspan<T, dextents2d<IndexT>, gcxx::layout_left>;
  template <class T, class IndexT>
  using dmat_right =
    gcxx::device_mdspan<T, dextents2d<IndexT>, gcxx::layout_right>;

  // Garbage in the unread triangle pins the tag semantics.
  template <class IndexT, bool RowMajorA>
  void run_symv() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    constexpr int N = 4;
    std::vector<double> hFull(N * N);  // symmetric reference (column-major)
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < N; ++i) {
        hFull[i + j * N] =
          (i == j) ? 2.0 + i : 0.25 * static_cast<double>((i + 1) * (j + 1));
      }
    }
    // device buffer: upper triangle + diagonal from hFull, garbage below
    std::vector<double> hBuf(hFull);
    for (int j = 0; j < N; ++j) {
      for (int i = j + 1; i < N; ++i) {
        hBuf[i + j * N] = 1e9;  // unread lower triangle
      }
    }

    std::vector<double> hX{1.0, -2.0, 3.0, 0.5};
    std::vector<double> href(N);
    for (int i = 0; i < N; ++i) {
      double acc{};
      for (int j = 0; j < N; ++j) {
        acc += hFull[i + j * N] * hX[j];
      }
      href[i] = acc;
    }

    gcxx::Stream str;
    auto dA = gcxx::make_device_unique_ptr<double>(std::size_t{N * N});
    auto dX = gcxx::make_device_unique_ptr<double>(std::size_t{N});
    auto dY = gcxx::make_device_unique_ptr<double>(std::size_t{N});
    gcxx::Copy(str, dA.get(), hBuf.data(), std::size_t{N * N});
    gcxx::Copy(str, dX.get(), hX.data(), std::size_t{N});

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);

    if constexpr (RowMajorA) {
      dmat_right<double, IndexT> A(dA.get(), N, N);
      auto X = gcxx::make_device_vector<IndexT>(gcxx::span(dX.get(), N));
      auto Y = gcxx::make_device_vector<IndexT>(gcxx::span(dY.get(), N));
      // Row-major view: garbage sits where the tagged read never looks.
      std::vector<double> hRow(N * N);
      for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
          hRow[i * N + j] = hFull[i + j * N];
        }
      }
      for (int j = 0; j < N; ++j) {
        for (int i = j + 1; i < N; ++i) {
          hRow[i * N + j] = 1e9;  // unread lower triangle of the row-major view
        }
      }
      gcxx::Copy(str, dA.get(), hRow.data(), std::size_t{N * N});
      gcxx::blas::symmetric_matrix_vector_product(handle, A, gcxx::blas::upper,
                                                  X, Y);
    } else {
      dmat_left<double, IndexT> A(dA.get(), N, N);
      auto X = gcxx::make_device_vector<IndexT>(gcxx::span(dX.get(), N));
      auto Y = gcxx::make_device_vector<IndexT>(gcxx::span(dY.get(), N));
      gcxx::blas::symmetric_matrix_vector_product(handle, A, gcxx::blas::upper,
                                                  X, Y);
    }
    str.sync();

    std::vector<double> hResult(N);
    gcxx::Copy(str, hResult.data(), dY.get(), std::size_t{N});
    str.sync();
    for (int i = 0; i < N; ++i) {
      EXPECT_NEAR(hResult[i], href[i], 1e-9) << "symv mismatch at " << i;
    }
  }

  template <class IndexT>
  void run_trmv_trsv() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    constexpr int N = 4;
    std::vector<double> hTri(N * N, 0.0);  // upper triangle (column-major)
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i <= j; ++i) {
        hTri[i + j * N] = (i == j) ? 1.0 + j : 0.5 * static_cast<double>(i + j);
      }
    }
    std::vector<double> hBuf(hTri);
    for (int j = 0; j < N; ++j) {
      for (int i = j + 1; i < N; ++i) {
        hBuf[i + j * N] = 1e9;  // unread lower triangle
      }
    }

    std::vector<double> hX{1.0, -2.0, 3.0, 0.5};
    std::vector<double> hB(N), hYref(N);
    for (int i = 0; i < N; ++i) {
      double acc{};
      for (int j = 0; j < N; ++j) {
        acc += hTri[i + j * N] * hX[j];
      }
      hYref[i] = acc;
      hB[i]    = acc;  // solve with b = A*x must recover x
    }

    gcxx::Stream str;
    auto dA = gcxx::make_device_unique_ptr<double>(std::size_t{N * N});
    auto dX = gcxx::make_device_unique_ptr<double>(std::size_t{N});
    auto dY = gcxx::make_device_unique_ptr<double>(std::size_t{N});
    auto dB = gcxx::make_device_unique_ptr<double>(std::size_t{N});
    gcxx::Copy(str, dA.get(), hBuf.data(), std::size_t{N * N});
    gcxx::Copy(str, dX.get(), hX.data(), std::size_t{N});
    gcxx::Copy(str, dB.get(), hB.data(), std::size_t{N});

    dmat_left<double, IndexT> A(dA.get(), N, N);
    auto X = gcxx::make_device_vector<IndexT>(gcxx::span(dX.get(), N));
    auto Y = gcxx::make_device_vector<IndexT>(gcxx::span(dY.get(), N));
    auto B = gcxx::make_device_vector<IndexT>(gcxx::span(dB.get(), N));

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);

    gcxx::blas::triangular_matrix_vector_product(
      handle, A, gcxx::blas::upper, gcxx::blas::explicit_diagonal, X, Y);
    str.sync();

    std::vector<double> hResult(N);
    gcxx::Copy(str, hResult.data(), dY.get(), std::size_t{N});
    str.sync();
    for (int i = 0; i < N; ++i) {
      EXPECT_NEAR(hResult[i], hYref[i], 1e-9) << "trmv mismatch at " << i;
    }

    gcxx::blas::triangular_matrix_vector_solve(
      handle, A, gcxx::blas::upper, gcxx::blas::explicit_diagonal, B, Y);
    str.sync();

    gcxx::Copy(str, hResult.data(), dY.get(), std::size_t{N});
    str.sync();
    for (int i = 0; i < N; ++i) {
      EXPECT_NEAR(hResult[i], hX[i], 1e-9) << "trsv mismatch at " << i;
    }
  }

  template <class IndexT, bool RowMajorA>
  void run_ger() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    constexpr int M = 3, N = 4;
    std::vector<double> hA0(M * N);  // column-major base values
    for (int i = 0; i < M * N; ++i) {
      hA0[i] = 0.1 * static_cast<double>(i);
    }
    std::vector<double> hX{1.0, -2.0, 3.0};
    std::vector<double> hY{0.5, 1.5, -2.5, 4.0};

    std::vector<double> href(M * N);  // column-major result
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        href[i + j * M] = hA0[i + j * M] + hX[i] * hY[j];
      }
    }

    gcxx::Stream str;
    auto dA = gcxx::make_device_unique_ptr<double>(std::size_t{M * N});
    auto dX = gcxx::make_device_unique_ptr<double>(std::size_t{M});
    auto dY = gcxx::make_device_unique_ptr<double>(std::size_t{N});
    gcxx::Copy(str, dX.get(), hX.data(), std::size_t{M});
    gcxx::Copy(str, dY.get(), hY.data(), std::size_t{N});

    auto X = gcxx::make_device_vector<IndexT>(gcxx::span(dX.get(), M));
    auto Y = gcxx::make_device_vector<IndexT>(gcxx::span(dY.get(), N));

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);

    std::vector<double> hResult(M * N);
    if constexpr (RowMajorA) {
      std::vector<double> hRow(M * N);
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          hRow[i * N + j] = hA0[i + j * M];
        }
      }
      gcxx::Copy(str, dA.get(), hRow.data(), std::size_t{M * N});
      dmat_right<double, IndexT> A(dA.get(), M, N);
      gcxx::blas::matrix_rank_1_update(handle, X, Y, A);
      str.sync();
      gcxx::Copy(str, hResult.data(), dA.get(), std::size_t{M * N});
      str.sync();
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          EXPECT_NEAR(hResult[i * N + j], href[i + j * M], 1e-9)
            << "ger (row-major) mismatch at " << i << "," << j;
        }
      }
    } else {
      gcxx::Copy(str, dA.get(), hA0.data(), std::size_t{M * N});
      dmat_left<double, IndexT> A(dA.get(), M, N);
      gcxx::blas::matrix_rank_1_update(handle, X, Y, A);
      str.sync();
      gcxx::Copy(str, hResult.data(), dA.get(), std::size_t{M * N});
      str.sync();
      for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(hResult[i], href[i], 1e-9)
          << "ger mismatch at linear " << i;
      }

    }
  }

  // syr/syr2 write only the tagged triangle; check just that triangle.
  template <class IndexT>
  void run_syr_syr2() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    constexpr int N = 4;
    std::vector<double> hA0(N * N);  // symmetric base (column-major)
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < N; ++i) {
        hA0[i + j * N] = (i == j) ? static_cast<double>(i + 1)
                                  : 0.125 * static_cast<double>(i + 2 * j);
      }
    }
    std::vector<double> hX{1.0, -2.0, 3.0, 0.5};
    std::vector<double> hY{0.25, 0.5, -0.75, 1.25};

    gcxx::Stream str;
    auto dA = gcxx::make_device_unique_ptr<double>(std::size_t{N * N});
    auto dX = gcxx::make_device_unique_ptr<double>(std::size_t{N});
    auto dY = gcxx::make_device_unique_ptr<double>(std::size_t{N});
    gcxx::Copy(str, dX.get(), hX.data(), std::size_t{N});
    gcxx::Copy(str, dY.get(), hY.data(), std::size_t{N});

    dmat_left<double, IndexT> A(dA.get(), N, N);
    auto X = gcxx::make_device_vector<IndexT>(gcxx::span(dX.get(), N));
    auto Y = gcxx::make_device_vector<IndexT>(gcxx::span(dY.get(), N));

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);

    // syr into the lower triangle (garbage left in the upper one)
    std::vector<double> hLower(hA0);
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < j; ++i) {
        hLower[i + j * N] = 1e9;
      }
    }
    gcxx::Copy(str, dA.get(), hLower.data(), std::size_t{N * N});
    gcxx::blas::symmetric_matrix_rank_1_update(handle, X, A, gcxx::blas::lower);
    str.sync();

    std::vector<double> hResult(N * N);
    gcxx::Copy(str, hResult.data(), dA.get(), std::size_t{N * N});
    str.sync();
    for (int j = 0; j < N; ++j) {
      for (int i = j; i < N; ++i) {  // lower triangle + diagonal only
        const double want = hA0[i + j * N] + hX[i] * hX[j];
        EXPECT_NEAR(hResult[i + j * N], want, 1e-9)
          << "syr mismatch at " << i << "," << j;
      }
    }

    // syr2 into the upper triangle of the updated buffer; seed that triangle
    // with the symmetric update (syr only touched the lower one above)
    std::vector<double> hUpdated(N * N);
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i <= j; ++i) {  // upper triangle + diagonal
        hUpdated[i + j * N] = hA0[i + j * N] + hX[i] * hX[j];
      }
    }
    gcxx::Copy(str, dA.get(), hUpdated.data(), std::size_t{N * N});
    str.sync();
    gcxx::blas::symmetric_matrix_rank_2_update(handle, X, Y, A,
                                               gcxx::blas::upper);
    str.sync();
    gcxx::Copy(str, hResult.data(), dA.get(), std::size_t{N * N});
    str.sync();
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i <= j; ++i) {  // upper triangle + diagonal only
        const double want = hUpdated[i + j * N] + hX[i] * hY[j] + hY[i] * hX[j];
        EXPECT_NEAR(hResult[i + j * N], want, 1e-9)
          << "syr2 mismatch at " << i << "," << j;
      }
    }
  }

}  // namespace

TEST(BlasL2, Symv_ColMajor_Double) {
  run_symv<int, false>();
}
TEST(BlasL2, Symv_RowMajor_Double) {
  run_symv<int, true>();
}
TEST(BlasL2, Symv_RowMajor_Double_64bitIndex) {
  run_symv<std::int64_t, true>();
}
TEST(BlasL2, TrmvTrsv_Double) {
  run_trmv_trsv<int>();
}
TEST(BlasL2, TrmvTrsv_Double_64bitIndex) {
  run_trmv_trsv<std::int64_t>();
}
TEST(BlasL2, Ger_ColMajor_Double) {
  run_ger<int, false>();
}
TEST(BlasL2, Ger_RowMajor_Double) {
  run_ger<int, true>();
}
TEST(BlasL2, Ger_RowMajor_Double_64bitIndex) {
  run_ger<std::int64_t, true>();
}
TEST(BlasL2, SyrSyr2_Double) {
  run_syr_syr2<int>();
}
TEST(BlasL2, SyrSyr2_Double_64bitIndex) {
  run_syr_syr2<std::int64_t>();
}
