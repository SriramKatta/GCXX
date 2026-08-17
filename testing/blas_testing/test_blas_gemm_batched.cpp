// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// End-to-end batched GEMM tests, compared against a host reference:
//  - gemm_batched: C_i = A_i * B_i via cu/hipblasGemmBatchedEx, with each
//    batch matrix in its OWN device allocation (span-of-matrices operands,
//    i.e. non-uniform pointer placement — the capability that separates the
//    pointer-array entry point from the strided one).
//  - gemm_strided_batched: the same product via GemmStridedBatchedEx over a
//    rank-3 mdspan with extents (rows, cols, batch) and layout_left, i.e.
//    contiguous column-major matrices with a uniform batch stride.
// GPU-gated — skipped when no device is present, but the templates must
// still compile (they instantiate both the int and the int64_t index_type
// dispatch, i.e. the *_64 entry points).

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

  constexpr int M = 3;
  constexpr int K = 4;
  constexpr int N = 2;
  constexpr int B = 3;

  template <class IndexT>
  using mat2d = gcxx::mdspan<double, gcxx::dextents<IndexT, 2>,
                             gcxx::layout_left, gcxx::default_accessor<double>>;

  template <class IndexT>
  using mat3_left =
    gcxx::mdspan<double, gcxx::dextents<IndexT, 3>, gcxx::layout_left,
                 gcxx::default_accessor<double>>;

  // Device-memory counterparts required by the gcxx::blas operations.
  template <class IndexT>
  using dmat2d =
    gcxx::device_mdspan<double, gcxx::dextents<IndexT, 2>, gcxx::layout_left>;

  template <class IndexT>
  using dmat3_left =
    gcxx::device_mdspan<double, gcxx::dextents<IndexT, 3>, gcxx::layout_left>;

  // Column-major host reference: out_b = a_b * b_b, where a_b is (m x k), b_b
  // is (k x n) and out_b is (m x n), for each batch element b.
  void host_gemm3(const std::vector<double>& a, const std::vector<double>& b,
                  std::vector<double>& out, int m, int k, int n, int batch) {
    for (int bb = 0; bb < batch; ++bb) {
      const double* ab  = a.data() + bb * m * k;
      const double* bbp = b.data() + bb * k * n;
      double* ob        = out.data() + bb * m * n;
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
          double acc{};
          for (int p = 0; p < k; ++p) {
            acc += ab[i + p * m] * bbp[p + j * k];
          }
          ob[i + j * m] = acc;
        }
      }
    }
  }

  std::vector<double> make_a() {
    std::vector<double> hA(M * K * B);
    for (int i = 0; i < M * K * B; ++i) {
      hA[i] = static_cast<double>((i % 7) - 3);
    }
    return hA;
  }

  std::vector<double> make_b() {
    std::vector<double> hB(K * N * B);
    for (int i = 0; i < K * N * B; ++i) {
      hB[i] = static_cast<double>((i % 5) - 2);
    }
    return hB;
  }

  // gemm_batched over a host array of matrix views whose device storages are
  // all SEPARATE allocations, proving non-uniform pointer placement.
  template <class IndexT>
  void run_gemm_batched_separate_allocs() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    std::vector<double> hA = make_a();
    std::vector<double> hB = make_b();
    std::vector<double> href(M * N * B);
    host_gemm3(hA, hB, href, M, K, N, B);

    gcxx::Stream str;
    std::vector<gcxx::device_ptr<double>> dAs, dBs, dCs;
    std::vector<dmat2d<IndexT>> aViews, bViews, cViews;
    for (int i = 0; i < B; ++i) {
      dAs.push_back(gcxx::make_device_unique_ptr<double>(std::size_t{M * K}));
      dBs.push_back(gcxx::make_device_unique_ptr<double>(std::size_t{K * N}));
      dCs.push_back(gcxx::make_device_unique_ptr<double>(std::size_t{M * N}));
      gcxx::Copy(str, dAs.back().get(), hA.data() + i * M * K,
                 std::size_t{M * K});
      gcxx::Copy(str, dBs.back().get(), hB.data() + i * K * N,
                 std::size_t{K * N});
      aViews.emplace_back(dAs.back().get(), M, K);
      bViews.emplace_back(dBs.back().get(), K, N);
      cViews.emplace_back(dCs.back().get(), M, N);
    }

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);
    gcxx::blas::gemm_batched(handle, 1.0, aViews, bViews, 0.0, cViews);
    str.Synchronize();

    for (int i = 0; i < B; ++i) {
      std::vector<double> hResult(M * N);
      gcxx::Copy(str, hResult.data(), dCs[static_cast<std::size_t>(i)].get(),
                 std::size_t{M * N});
      str.Synchronize();
      for (int j = 0; j < M * N; ++j) {
        EXPECT_NEAR(hResult[static_cast<std::size_t>(j)],
                    href[static_cast<std::size_t>(i * M * N + j)], 1e-9)
          << "gemm_batched mismatch in batch " << i << " at " << j;
      }
    }
  }

  // gemm_strided_batched over one contiguous buffer per operand: rank-3
  // mdspan with extents (rows, cols, batch).
  template <class IndexT>
  void run_gemm_strided_batched() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    std::vector<double> hA = make_a();
    std::vector<double> hB = make_b();
    std::vector<double> href(M * N * B);
    host_gemm3(hA, hB, href, M, K, N, B);

    gcxx::Stream str;
    auto dA = gcxx::make_device_unique_ptr<double>(std::size_t{M * K * B});
    auto dB = gcxx::make_device_unique_ptr<double>(std::size_t{K * N * B});
    auto dC = gcxx::make_device_unique_ptr<double>(std::size_t{M * N * B});
    gcxx::Copy(str, dA.get(), hA.data(), std::size_t{M * K * B});
    gcxx::Copy(str, dB.get(), hB.data(), std::size_t{K * N * B});

    dmat3_left<IndexT> A(dA.get(), M, K, B);
    dmat3_left<IndexT> B3(dB.get(), K, N, B);
    dmat3_left<IndexT> C(dC.get(), M, N, B);

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);
    gcxx::blas::gemm_strided_batched(handle, 1.0, A, B3, 0.0, C);
    str.Synchronize();

    std::vector<double> hResult(M * N * B);
    gcxx::Copy(str, hResult.data(), dC.get(), std::size_t{M * N * B});
    str.Synchronize();

    for (int i = 0; i < M * N * B; ++i) {
      EXPECT_NEAR(hResult[static_cast<std::size_t>(i)],
                  href[static_cast<std::size_t>(i)], 1e-9)
        << "gemm_strided_batched mismatch at " << i;
    }
  }

}  // namespace

TEST(BlasGemmBatched, SeparateAllocs_Double) {
  run_gemm_batched_separate_allocs<int>();
}

TEST(BlasGemmBatched, SeparateAllocs_Double_64bitIndex) {
  run_gemm_batched_separate_allocs<std::int64_t>();
}

TEST(BlasGemmStridedBatched, Contiguous_Double) {
  run_gemm_strided_batched<int>();
}

TEST(BlasGemmStridedBatched, Contiguous_Double_64bitIndex) {
  run_gemm_strided_batched<std::int64_t>();
}
