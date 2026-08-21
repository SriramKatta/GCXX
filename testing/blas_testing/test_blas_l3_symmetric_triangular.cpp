// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
// End-to-end Level-3 symm/syrk/syr2k/trmm/trsm tests vs host references;
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

  template <class IndexT, bool RowMajor>
  void run_symm() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    constexpr int K = 3, N = 4;
    std::vector<double> hSym(K * K);  // symmetric reference (column-major)
    for (int j = 0; j < K; ++j) {
      for (int i = 0; i < K; ++i) {
        hSym[i + j * K] =
          (i == j) ? 2.0 + i : 0.5 * static_cast<double>(i + 1) * (j + 1);
      }
    }
    std::vector<double> hSymBuf(hSym);
    for (int j = 0; j < K; ++j) {
      for (int i = j + 1; i < K; ++i) {
        hSymBuf[i + j * K] = 1e9;  // unread lower triangle
      }
    }
    std::vector<double> hB(K * N);
    for (int i = 0; i < K * N; ++i) {
      hB[i] = 0.25 * static_cast<double>(i) - 0.5;
    }

    std::vector<double> href(K * N);  // column-major A*B
    for (int i = 0; i < K; ++i) {
      for (int j = 0; j < N; ++j) {
        double acc{};
        for (int p = 0; p < K; ++p) {
          acc += hSym[i + p * K] * hB[p + j * K];
        }
        href[i + j * K] = acc;
      }
    }

    gcxx::Stream str;
    auto dA = gcxx::make_device_unique_ptr<double>(std::size_t{K * K});
    auto dB = gcxx::make_device_unique_ptr<double>(std::size_t{K * N});
    auto dC = gcxx::make_device_unique_ptr<double>(std::size_t{K * N});
    gcxx::Copy(str, dA.get(), hSymBuf.data(), std::size_t{K * K});

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);

    std::vector<double> hResult(K * N);
    if constexpr (RowMajor) {
      std::vector<double> hBRow(K * N), hCRow(K * N);
      for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
          hBRow[i * N + j] = hB[i + j * K];
        }
      }
      gcxx::Copy(str, dB.get(), hBRow.data(), std::size_t{K * N});
      dmat_left<double, IndexT> A(dA.get(), K, K);
      dmat_right<double, IndexT> B(dB.get(), K, N);
      dmat_right<double, IndexT> C(dC.get(), K, N);
      gcxx::blas::symmetric_matrix_product(handle, gcxx::blas::left, A,
                                           gcxx::blas::upper, B, C);
      str.Synchronize();
      gcxx::Copy(str, hCRow.data(), dC.get(), std::size_t{K * N});
      str.Synchronize();
      for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
          EXPECT_NEAR(hCRow[i * N + j], href[i + j * K], 1e-9)
            << "symm (row-major B/C) mismatch at " << i << "," << j;
        }
      }
    } else {
      gcxx::Copy(str, dB.get(), hB.data(), std::size_t{K * N});
      dmat_left<double, IndexT> A(dA.get(), K, K);
      dmat_left<double, IndexT> B(dB.get(), K, N);
      dmat_left<double, IndexT> C(dC.get(), K, N);
      gcxx::blas::symmetric_matrix_product(handle, gcxx::blas::left, A,
                                           gcxx::blas::upper, B, C);
      str.Synchronize();
      gcxx::Copy(str, hResult.data(), dC.get(), std::size_t{K * N});
      str.Synchronize();
      for (int i = 0; i < K * N; ++i) {
        EXPECT_NEAR(hResult[i], href[i], 1e-9)
          << "symm mismatch at linear " << i;
      }
    }
  }

  template <class IndexT>
  void run_syrk() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    constexpr int N = 3, K = 4;
    std::vector<double> hC0(N * N), hA(N * K);
    for (int i = 0; i < N * N; ++i) {
      hC0[i] = 0.5 * static_cast<double>(i) - 1.0;
    }
    for (int i = 0; i < N * K; ++i) {
      hA[i] = 0.125 * static_cast<double>(i + 1) - 0.75;
    }

    std::vector<double> href(N * N);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        double acc{};
        for (int p = 0; p < K; ++p) {
          acc += hA[i + p * N] * hA[j + p * N];
        }
        href[i + j * N] = hC0[i + j * N] + acc;
      }
    }

    gcxx::Stream str;
    auto dA = gcxx::make_device_unique_ptr<double>(std::size_t{N * K});
    auto dC = gcxx::make_device_unique_ptr<double>(std::size_t{N * N});
    gcxx::Copy(str, dA.get(), hA.data(), std::size_t{N * K});
    gcxx::Copy(str, dC.get(), hC0.data(), std::size_t{N * N});

    dmat_left<double, IndexT> A(dA.get(), N, K);
    dmat_left<double, IndexT> C(dC.get(), N, N);

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);
    gcxx::blas::symmetric_matrix_rank_k_update(handle, A, gcxx::blas::upper, C);
    str.Synchronize();

    std::vector<double> hResult(N * N);
    gcxx::Copy(str, hResult.data(), dC.get(), std::size_t{N * N});
    str.Synchronize();
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i <= j; ++i) {  // upper triangle + diagonal
        EXPECT_NEAR(hResult[i + j * N], href[i + j * N], 1e-9)
          << "syrk mismatch at " << i << "," << j;
      }
    }
  }

  // Row-major-like C exercises the transposed-problem dispatch.
  template <class IndexT, bool RowMajorC>
  void run_syr2k() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    constexpr int N = 3, K = 4;
    std::vector<double> hC0(N * N), hA(N * K), hB(N * K);
    for (int i = 0; i < N * N; ++i) {
      hC0[i] = 0.375 * static_cast<double>(i);
    }
    for (int i = 0; i < N * K; ++i) {
      hA[i] = 0.1 * static_cast<double>(i) - 0.4;
      hB[i] = 0.2 * static_cast<double>(N * K - i) - 1.1;
    }

    std::vector<double> href(N * N);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        double acc{};
        for (int p = 0; p < K; ++p) {
          acc += hA[i + p * N] * hB[j + p * N] + hB[i + p * N] * hA[j + p * N];
        }
        href[i + j * N] = hC0[i + j * N] + acc;
      }
    }

    gcxx::Stream str;
    auto dA = gcxx::make_device_unique_ptr<double>(std::size_t{N * K});
    auto dB = gcxx::make_device_unique_ptr<double>(std::size_t{N * K});
    auto dC = gcxx::make_device_unique_ptr<double>(std::size_t{N * N});
    gcxx::Copy(str, dA.get(), hA.data(), std::size_t{N * K});
    gcxx::Copy(str, dB.get(), hB.data(), std::size_t{N * K});

    dmat_left<double, IndexT> A(dA.get(), N, K);
    dmat_left<double, IndexT> B(dB.get(), N, K);

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);

    std::vector<double> hResult(N * N);
    if constexpr (RowMajorC) {
      std::vector<double> hC0Row(N * N);
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          hC0Row[i * N + j] = hC0[i + j * N];
        }
      }
      gcxx::Copy(str, dC.get(), hC0Row.data(), std::size_t{N * N});
      dmat_right<double, IndexT> C(dC.get(), N, N);
      gcxx::blas::symmetric_matrix_rank_2k_update(handle, A, B,
                                                  gcxx::blas::lower, C);
      str.Synchronize();
      gcxx::Copy(str, hResult.data(), dC.get(), std::size_t{N * N});
      str.Synchronize();
      for (int j = 0; j < N; ++j) {
        for (int i = j; i < N; ++i) {  // lower triangle + diagonal (row-major)
          EXPECT_NEAR(hResult[i * N + j], href[i + j * N], 1e-9)
            << "syr2k (row-major C) mismatch at " << i << "," << j;
        }
      }
    } else {
      gcxx::Copy(str, dC.get(), hC0.data(), std::size_t{N * N});
      dmat_left<double, IndexT> C(dC.get(), N, N);
      gcxx::blas::symmetric_matrix_rank_2k_update(handle, A, B,
                                                  gcxx::blas::lower, C);
      str.Synchronize();
      gcxx::Copy(str, hResult.data(), dC.get(), std::size_t{N * N});
      str.Synchronize();
      for (int j = 0; j < N; ++j) {
        for (int i = j; i < N; ++i) {
          EXPECT_NEAR(hResult[i + j * N], href[i + j * N], 1e-9)
            << "syr2k mismatch at " << i << "," << j;
        }
      }
    }
  }

  template <class IndexT, bool RowMajor>
  void run_trmm_trsm() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    constexpr int K = 3, N = 4;
    std::vector<double> hTri(K * K, 0.0);  // upper triangle
    for (int j = 0; j < K; ++j) {
      for (int i = 0; i <= j; ++i) {
        hTri[i + j * K] =
          (i == j) ? 1.5 + j : 0.25 * static_cast<double>(i + j + 1);
      }
    }
    std::vector<double> hTriBuf(hTri);
    for (int j = 0; j < K; ++j) {
      for (int i = j + 1; i < K; ++i) {
        hTriBuf[i + j * K] = 1e9;
      }
    }
    std::vector<double> hB(K * N);
    for (int i = 0; i < K * N; ++i) {
      hB[i] = 0.2 * static_cast<double>(i) - 0.3;
    }

    // references: C = A*B and X = A^-1*B (column-major)
    std::vector<double> hCref(K * N);
    for (int i = 0; i < K; ++i) {
      for (int j = 0; j < N; ++j) {
        double acc{};
        for (int p = 0; p < K; ++p) {
          acc += hTri[i + p * K] * hB[p + j * K];
        }
        hCref[i + j * K] = acc;
      }
    }
    std::vector<double> hXref(K * N);  // forward substitution per column
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < K; ++i) {
        double acc = hB[i + j * K];
        for (int p = 0; p < i; ++p) {
          acc -= hTri[i + p * K] * hXref[p + j * K];
        }
        hXref[i + j * K] = acc / hTri[i + i * K];
      }
    }

    gcxx::Stream str;
    auto dA = gcxx::make_device_unique_ptr<double>(std::size_t{K * K});
    auto dB = gcxx::make_device_unique_ptr<double>(std::size_t{K * N});
    auto dC = gcxx::make_device_unique_ptr<double>(std::size_t{K * N});
    auto dX = gcxx::make_device_unique_ptr<double>(std::size_t{K * N});
    gcxx::Copy(str, dA.get(), hTriBuf.data(), std::size_t{K * K});

    dmat_left<double, IndexT> A(dA.get(), K, K);

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);

    if constexpr (RowMajor) {
      std::vector<double> hBRow(K * N);
      for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
          hBRow[i * N + j] = hB[i + j * K];
        }
      }
      gcxx::Copy(str, dB.get(), hBRow.data(), std::size_t{K * N});
      dmat_right<double, IndexT> B(dB.get(), K, N);
      dmat_right<double, IndexT> C(dC.get(), K, N);
      dmat_right<double, IndexT> X(dX.get(), K, N);
      gcxx::blas::triangular_matrix_product(
        handle, gcxx::blas::left, A, gcxx::blas::upper,
        gcxx::blas::explicit_diagonal, B, C);
      gcxx::blas::triangular_matrix_matrix_solve(
        handle, gcxx::blas::left, A, gcxx::blas::upper,
        gcxx::blas::explicit_diagonal, B, X);
      str.Synchronize();

      std::vector<double> hRow(K * N);
      gcxx::Copy(str, hRow.data(), dC.get(), std::size_t{K * N});
      str.Synchronize();
      for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
          EXPECT_NEAR(hRow[i * N + j], hCref[i + j * K], 1e-9)
            << "trmm (row-major B/C) mismatch at " << i << "," << j;
        }
      }
      gcxx::Copy(str, hRow.data(), dX.get(), std::size_t{K * N});
      str.Synchronize();
      for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
          EXPECT_NEAR(hRow[i * N + j], hXref[i + j * K], 1e-9)
            << "trsm (row-major B/X) mismatch at " << i << "," << j;
        }
      }
    } else {
      gcxx::Copy(str, dB.get(), hB.data(), std::size_t{K * N});
      dmat_left<double, IndexT> B(dB.get(), K, N);
      dmat_left<double, IndexT> C(dC.get(), K, N);
      dmat_left<double, IndexT> X(dX.get(), K, N);
      gcxx::blas::triangular_matrix_product(
        handle, gcxx::blas::left, A, gcxx::blas::upper,
        gcxx::blas::explicit_diagonal, B, C);
      gcxx::blas::triangular_matrix_matrix_solve(
        handle, gcxx::blas::left, A, gcxx::blas::upper,
        gcxx::blas::explicit_diagonal, B, X);
      str.Synchronize();

      std::vector<double> hResult(K * N);
      gcxx::Copy(str, hResult.data(), dC.get(), std::size_t{K * N});
      str.Synchronize();
      for (int i = 0; i < K * N; ++i) {
        EXPECT_NEAR(hResult[i], hCref[i], 1e-9)
          << "trmm mismatch at linear " << i;
      }
      gcxx::Copy(str, hResult.data(), dX.get(), std::size_t{K * N});
      str.Synchronize();
      for (int i = 0; i < K * N; ++i) {
        EXPECT_NEAR(hResult[i], hXref[i], 1e-9)
          << "trsm mismatch at linear " << i;
      }
    }
  }

}  // namespace

TEST(BlasL3SymTri, Symm_ColMajor_Double) {
  run_symm<int, false>();
}
TEST(BlasL3SymTri, Symm_RowMajor_Double) {
  run_symm<int, true>();
}
TEST(BlasL3SymTri, Symm_RowMajor_Double_64bitIndex) {
  run_symm<std::int64_t, true>();
}
TEST(BlasL3SymTri, Syrk_Double) {
  run_syrk<int>();
}
TEST(BlasL3SymTri, Syrk_Double_64bitIndex) {
  run_syrk<std::int64_t>();
}
TEST(BlasL3SymTri, Syr2k_ColMajor_Double) {
  run_syr2k<int, false>();
}
TEST(BlasL3SymTri, Syr2k_RowMajor_Double) {
  run_syr2k<int, true>();
}
TEST(BlasL3SymTri, TrmmTrsm_ColMajor_Double) {
  run_trmm_trsm<int, false>();
}
TEST(BlasL3SymTri, TrmmTrsm_RowMajor_Double) {
  run_trmm_trsm<int, true>();
}
TEST(BlasL3SymTri, TrmmTrsm_RowMajor_Double_64bitIndex) {
  run_trmm_trsm<std::int64_t, true>();
}
