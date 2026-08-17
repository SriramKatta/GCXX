// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// End-to-end batched GEMM tests, compared against a host reference:
//  - gemm_batched: C_i = A_i * B_i via cu/hipblasGemmBatchedEx, with each
//    batch matrix in its OWN device allocation (span-of-matrices operands,
//    i.e. non-uniform pointer placement — the capability that separates the
//    pointer-array entry point from the strided one).
//  - gemm_strided_batched: the same product via GemmStridedBatchedEx over
//    rank-3 mdspans with the P2901 leftmost-batch extents (batch, rows,
//    cols), in two storages: column-major inner matrices via layout_stride
//    (strides {m*k, 1, m} — no standard rank-3 layout packs
//    batch-outermost + column-major-inner), and row-major inner matrices via
//    layout_right, which also gates the transposed-output batched dispatch.
// GPU-gated — skipped when no device is present, but the templates must
// still compile (they instantiate both the int and the int64_t index_type
// dispatch, i.e. the *_64 entry points).

#include "tests_common.hpp"

#include <array>
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

  // Device-memory per-matrix views required by gcxx::blas::gemm_batched.
  template <class IndexT>
  using dmat2d =
    gcxx::device_mdspan<double, gcxx::dextents<IndexT, 2>, gcxx::layout_left>;

  // Host reference over leftmost-batch rank-3 views: C(b,i,j) = A(b,i,p) *
  // B(b,p,j); layout-agnostic (works for layout_stride and layout_right).
  template <class MDA, class MDB, class MDC>
  void host_gemm3_bfirst(const MDA& a, const MDB& b, MDC& c) {
    using idx_t = typename MDA::index_type;
    const idx_t batch = a.extent(0);
    const idx_t m     = a.extent(1);
    const idx_t k     = a.extent(2);
    const idx_t n     = b.extent(2);
    for (idx_t bb = 0; bb < batch; ++bb) {
      for (idx_t i = 0; i < m; ++i) {
        for (idx_t j = 0; j < n; ++j) {
          double acc{};
          for (idx_t p = 0; p < k; ++p) {
            acc += a(bb, i, p) * b(bb, p, j);
          }
          c(bb, i, j) = acc;
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

    // column-major per-batch reference (each batch element is (m x k)/(k x n)
    // column-major, matching the per-matrix views below)
    std::vector<double> href(M * N * B);
    {
      using ext3     = gcxx::dextents<int, 3>;
      using strides3 = std::array<int, 3>;
      using map_cm   = gcxx::layout_stride::mapping<ext3>;
      gcxx::mdspan<double, ext3, gcxx::layout_stride> A3(
        hA.data(), map_cm(ext3{B, M, K}, strides3{M * K, 1, M}));
      gcxx::mdspan<double, ext3, gcxx::layout_stride> B3(
        hB.data(), map_cm(ext3{B, K, N}, strides3{K * N, 1, K}));
      gcxx::mdspan<double, ext3, gcxx::layout_stride> C3(
        href.data(), map_cm(ext3{B, M, N}, strides3{M * N, 1, M}));
      host_gemm3_bfirst(A3, B3, C3);
    }

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

  // gemm_strided_batched over one contiguous buffer per operand, rank-3
  // extents (batch, rows, cols):
  //  - colmajor-inner: layout_stride with strides {m*k, 1, m} — contiguous
  //    column-major matrices with a uniform batch stride (the direct
  //    analogue of the pre-P2901 batch-last layout_left);
  //  - rowmajor-inner: layout_right — contiguous row-major matrices; for the
  //    OUTPUT this takes the transposed batched dispatch.
  template <class IndexT>
  void run_gemm_strided_batched(bool rowmajor_inner) {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    std::vector<double> hA = make_a();
    std::vector<double> hB = make_b();
    std::vector<double> href(M * N * B);

    using ext3     = gcxx::dextents<IndexT, 3>;
    using strides3 = std::array<IndexT, 3>;
    using map_cm   = gcxx::layout_stride::mapping<ext3>;

    if (rowmajor_inner) {
      gcxx::mdspan<double, ext3, gcxx::layout_right> A3(hA.data(), B, M, K);
      gcxx::mdspan<double, ext3, gcxx::layout_right> B3(hB.data(), B, K, N);
      gcxx::mdspan<double, ext3, gcxx::layout_right> C3(href.data(), B, M, N);
      host_gemm3_bfirst(A3, B3, C3);
    } else {
      gcxx::mdspan<double, ext3, gcxx::layout_stride> A3(
        hA.data(), map_cm(ext3{B, M, K}, strides3{M * K, 1, M}));
      gcxx::mdspan<double, ext3, gcxx::layout_stride> B3(
        hB.data(), map_cm(ext3{B, K, N}, strides3{K * N, 1, K}));
      gcxx::mdspan<double, ext3, gcxx::layout_stride> C3(
        href.data(), map_cm(ext3{B, M, N}, strides3{M * N, 1, M}));
      host_gemm3_bfirst(A3, B3, C3);
    }

    gcxx::Stream str;
    auto dA = gcxx::make_device_unique_ptr<double>(std::size_t{M * K * B});
    auto dB = gcxx::make_device_unique_ptr<double>(std::size_t{K * N * B});
    auto dC = gcxx::make_device_unique_ptr<double>(std::size_t{M * N * B});
    gcxx::Copy(str, dA.get(), hA.data(), std::size_t{M * K * B});
    gcxx::Copy(str, dB.get(), hB.data(), std::size_t{K * N * B});

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);

    if (rowmajor_inner) {
      gcxx::device_mdspan<double, ext3, gcxx::layout_right> A(dA.get(), B, M,
                                                              K);
      gcxx::device_mdspan<double, ext3, gcxx::layout_right> Bv(dB.get(), B, K,
                                                               N);
      gcxx::device_mdspan<double, ext3, gcxx::layout_right> C(dC.get(), B, M,
                                                              N);
      gcxx::blas::gemm_strided_batched(handle, 1.0, A, Bv, 0.0, C);
    } else {
      gcxx::device_mdspan<double, ext3, gcxx::layout_stride> A(
        dA.get(), map_cm(ext3{B, M, K}, strides3{M * K, 1, M}));
      gcxx::device_mdspan<double, ext3, gcxx::layout_stride> Bv(
        dB.get(), map_cm(ext3{B, K, N}, strides3{K * N, 1, K}));
      gcxx::device_mdspan<double, ext3, gcxx::layout_stride> C(
        dC.get(), map_cm(ext3{B, M, N}, strides3{M * N, 1, M}));
      gcxx::blas::gemm_strided_batched(handle, 1.0, A, Bv, 0.0, C);
    }
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

TEST(BlasGemmStridedBatched, ColMajorInner_Double) {
  run_gemm_strided_batched<int>(false);
}

TEST(BlasGemmStridedBatched, ColMajorInner_Double_64bitIndex) {
  run_gemm_strided_batched<std::int64_t>(false);
}

TEST(BlasGemmStridedBatched, RowMajorInner_Double) {
  run_gemm_strided_batched<int>(true);
}

TEST(BlasGemmStridedBatched, RowMajorInner_Double_64bitIndex) {
  run_gemm_strided_batched<std::int64_t>(true);
}
