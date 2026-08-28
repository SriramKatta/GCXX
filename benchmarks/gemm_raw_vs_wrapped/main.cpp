// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Compares a hand-written cuBLAS/hipBLAS GemmEx call (the exact entry point
// gcxx::blas::matrix_product dispatches to for these operands) against the
// wrapper itself:
//
//   • IssueOnly — no per-iteration sync: isolates the host-side cost of
//     issuing one gemm; the wrapper overhead (pointer-mode guard, scaled()
//     resolution, view inference, extent checks) lives here.
//   • WithSync  — stream sync every iteration: end-to-end time where GPU
//     work dominates; shows the wrapper delta drowning in kernel time.
//
// The IssueOnly problems stay small so the GPU drains faster than the host
// can issue; larger problems would back up the launch queue until issuing
// blocks, which equalizes both variants and hides the very overhead being
// measured.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include <gcxx/backend/backend_blas.hpp>
#include <gcxx/blas_api.hpp>
#include <gcxx/runtime/device.hpp>
#include <gcxx/runtime/memory/copy.hpp>
#include <gcxx/runtime/memory/smartpointers/pointers.hpp>
#include <gcxx/runtime/memory/spans/mdspan/mdspan.hpp>
#include <gcxx/runtime_backend/backend_blas_handles.hpp>
#include <gcxx/runtime_backend/backend_handles.hpp>

namespace {

  using gcxx::f32_t;

  // mdspan index_type: int32 selects the plain GemmEx entry point (the
  // wrapper picks GemmEx vs GemmEx_64 from this type).
  using index_t = std::int32_t;

  template <class IndexT>
  using dextents2d = gcxx::dextents<IndexT, 2>;

  // Column-major device views: layout_left strides line up with the BLAS
  // leading dimensions, so the wrapper takes its direct (non-transposed)
  // dispatch path — the same call the raw baseline makes.
  template <class T, class IndexT>
  using dmat_left =
    gcxx::device_mdspan<T, dextents2d<IndexT>, gcxx::layout_left>;

  // One shared stream + BLAS handle for every benchmark; creation is far
  // too heavy to sit inside a timed region, and sharing matches real usage.
  struct BlasEnv {
    gcxx::Stream stream{};
    gcxx::blas::BlasHandle handle{};

    BlasEnv() { handle.setStream(stream); }
  };

  auto blas_env() -> BlasEnv& {
    static BlasEnv env;
    return env;
  }

  // Deterministic small-magnitude fills, shared by the device upload and
  // the naive host reference in verify_equivalence().
  constexpr auto elem_a(std::size_t i) -> f32_t {
    return static_cast<f32_t>(static_cast<int>(i % 13) - 6) / 7.0F;
  }

  constexpr auto elem_b(std::size_t i) -> f32_t {
    return static_cast<f32_t>(static_cast<int>(i % 11) - 5) / 9.0F;
  }

  // Device buffers + column-major views for one C(m×n) = A(m×k) * B(k×n)
  // problem. A/B are uploaded; C is left uninitialized (beta = 0).
  struct GemmProblem {
    std::int32_t m;
    std::int32_t k;
    std::int32_t n;
    std::int32_t lda;
    std::int32_t ldb;
    std::int32_t ldc;
    gcxx::device_ptr<f32_t> a;
    gcxx::device_ptr<f32_t> b;
    gcxx::device_ptr<f32_t> c;
    dmat_left<f32_t, index_t> va;
    dmat_left<f32_t, index_t> vb;
    dmat_left<f32_t, index_t> vc;

    GemmProblem(std::int32_t m_, std::int32_t k_, std::int32_t n_,
                gcxx::device_ptr<f32_t> a_, gcxx::device_ptr<f32_t> b_,
                gcxx::device_ptr<f32_t> c_)
        : m(m_),
          k(k_),
          n(n_),
          lda(m_),  // column-major: ld == rows
          ldb(k_),
          ldc(m_),
          a(std::move(a_)),
          b(std::move(b_)),
          c(std::move(c_)),
          va(a.get(), m_, k_),
          vb(b.get(), k_, n_),
          vc(c.get(), m_, n_) {}
  };

  auto make_gemm(std::int32_t m, std::int32_t k, std::int32_t n)
    -> GemmProblem {
    auto& env = blas_env();

    const auto na = static_cast<std::size_t>(m) * static_cast<std::size_t>(k);
    const auto nb = static_cast<std::size_t>(k) * static_cast<std::size_t>(n);
    const auto nc = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);

    auto da = gcxx::make_device_unique_ptr<f32_t>(na);
    auto db = gcxx::make_device_unique_ptr<f32_t>(nb);
    auto dc = gcxx::make_device_unique_ptr<f32_t>(nc);

    std::vector<f32_t> ha(na), hb(nb);
    for (std::size_t i = 0; i < na; ++i) {
      ha[i] = elem_a(i);
    }
    for (std::size_t i = 0; i < nb; ++i) {
      hb[i] = elem_b(i);
    }
    gcxx::Copy(env.stream, da.get(), ha.data(), na);
    gcxx::Copy(env.stream, db.get(), hb.data(), nb);
    env.stream.sync();

    return GemmProblem(m, k, n, std::move(da), std::move(db), std::move(dc));
  }

  // The floor: one hand-written GemmEx with every argument spelled out —
  // no guards, no inference, no checks.
  auto raw_gemm_ex(const GemmProblem& p, const f32_t* alpha, const f32_t* beta)
    -> gcxx::driver::deviceBlasStatus_t {
    return ::GCXX_BLAS_BACKEND(GemmEx)(
      blas_env().handle.getRawHandle(), gcxx::driver::deviceBlasOpN,
      gcxx::driver::deviceBlasOpN, p.m, p.n, p.k, alpha, p.a.get(),
      gcxx::blas::cuda_datatype_v<f32_t>, p.lda, p.b.get(),
      gcxx::blas::cuda_datatype_v<f32_t>, p.ldb, beta, p.c.get(),
      gcxx::blas::cuda_datatype_v<f32_t>, p.ldc,
      gcxx::blas::blas_compute_type_v<f32_t>, GCXX_BLAS_GEMM(DEFAULT));
  }

  // items/s == FLOP/s for the WithSync variants.
  auto set_flops_counter(benchmark::State& state, const GemmProblem& p)
    -> void {
    const auto flops = 2.0 * static_cast<double>(p.m) *
                       static_cast<double>(p.k) * static_cast<double>(p.n) *
                       static_cast<double>(state.iterations());
    state.SetItemsProcessed(static_cast<std::int64_t>(flops));
  }

  // ── IssueOnly: host-side cost of issuing one gemm ─────────────────────────

  void BM_RawGemmEx_IssueOnly(benchmark::State& state) {
    auto& env    = blas_env();
    const auto n = static_cast<std::int32_t>(state.range(0));
    auto problem = make_gemm(n, n, n);
    const f32_t alpha{1.0F};
    const f32_t beta{0.0F};
    for (auto _ : state) {
      const auto status = raw_gemm_ex(problem, &alpha, &beta);
      benchmark::DoNotOptimize(status);
    }
    env.stream.sync();  // drain outside the timed region
  }

  void BM_WrappedMatrixProduct_IssueOnly(benchmark::State& state) {
    auto& env    = blas_env();
    const auto n = static_cast<std::int32_t>(state.range(0));
    auto problem = make_gemm(n, n, n);
    for (auto _ : state) {
      gcxx::blas::matrix_product(env.handle, problem.va, problem.vb,
                                 problem.vc);
    }
    env.stream.sync();  // drain outside the timed region
  }

  // ── WithSync: end-to-end, GPU-dominated ───────────────────────────────────

  void BM_RawGemmEx_WithSync(benchmark::State& state) {
    auto& env    = blas_env();
    const auto n = static_cast<std::int32_t>(state.range(0));
    auto problem = make_gemm(n, n, n);
    const f32_t alpha{1.0F};
    const f32_t beta{0.0F};
    for (auto _ : state) {
      const auto status = raw_gemm_ex(problem, &alpha, &beta);
      benchmark::DoNotOptimize(status);
      env.stream.sync();
    }
    set_flops_counter(state, problem);
  }

  void BM_WrappedMatrixProduct_WithSync(benchmark::State& state) {
    auto& env    = blas_env();
    const auto n = static_cast<std::int32_t>(state.range(0));
    auto problem = make_gemm(n, n, n);
    for (auto _ : state) {
      gcxx::blas::matrix_product(env.handle, problem.va, problem.vb,
                                 problem.vc);
      env.stream.sync();
    }
    set_flops_counter(state, problem);
  }

  // One-shot correctness gate on a non-square problem (catches mixed-up
  // dimensions/lds that square problems hide): both paths must agree with
  // each other and with a naive host reference.
  auto verify_equivalence() -> bool {
    auto& env                = blas_env();
    constexpr std::int32_t m = 13;
    constexpr std::int32_t k = 7;
    constexpr std::int32_t n = 11;
    const auto nc            = static_cast<std::size_t>(m) * n;

    auto problem = make_gemm(m, k, n);

    const f32_t alpha{1.0F};
    const f32_t beta{0.0F};

    if (raw_gemm_ex(problem, &alpha, &beta) !=
        gcxx::driver::deviceBlasStatusSuccess) {
      std::fputs("verify: raw GemmEx returned an error status\n", stderr);
      return false;
    }
    env.stream.sync();
    std::vector<f32_t> c_raw(nc);
    gcxx::Copy(env.stream, c_raw.data(), problem.c.get(), nc);
    env.stream.sync();

    gcxx::blas::matrix_product(env.handle, problem.va, problem.vb, problem.vc);
    env.stream.sync();
    std::vector<f32_t> c_wrapped(nc);
    gcxx::Copy(env.stream, c_wrapped.data(), problem.c.get(), nc);
    env.stream.sync();

    constexpr f32_t tol = 1.0e-4F;
    for (std::int32_t j = 0; j < n; ++j) {
      for (std::int32_t i = 0; i < m; ++i) {
        f32_t ref{};
        for (std::int32_t p = 0; p < k; ++p) {
          ref += elem_a(static_cast<std::size_t>(i + p * m)) *
                 elem_b(static_cast<std::size_t>(p + j * k));
        }
        const auto idx = static_cast<std::size_t>(i + j * m);
        if (std::fabs(c_raw[idx] - ref) > tol ||
            std::fabs(c_wrapped[idx] - ref) > tol) {
          std::fprintf(stderr,
                       "verify: mismatch at (%d,%d): raw %f wrapped %f "
                       "ref %f\n",
                       i, j, c_raw[idx], c_wrapped[idx], ref);
          return false;
        }
      }
    }
    return true;
  }

}  // namespace

// Geometric ranges (×2) rather than spelled-out Arg lists.
BENCHMARK(BM_RawGemmEx_IssueOnly)
  ->RangeMultiplier(4)
  ->Range(32, 2046)
  ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_WrappedMatrixProduct_IssueOnly)
  ->RangeMultiplier(4)
  ->Range(32, 2046)
  ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_RawGemmEx_WithSync)
  ->RangeMultiplier(2)
  ->Range(128, 2046)
  ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_WrappedMatrixProduct_WithSync)
  ->RangeMultiplier(2)
  ->Range(128, 2046)
  ->Unit(benchmark::kMicrosecond);

auto main(int argc, char** argv) -> int {
  if (!gcxx::Device::available()) {
    std::fputs(
      "SKIP: no CUDA/HIP device visible; this benchmark needs a "
      "GPU\n",
      stderr);
    return EXIT_SUCCESS;
  }
  if (!verify_equivalence()) {
    std::fputs("FATAL: matrix_product disagrees with raw GemmEx\n", stderr);
    return EXIT_FAILURE;
  }
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return EXIT_FAILURE;
  }
  benchmark::RunSpecifiedBenchmarks();
  return EXIT_SUCCESS;
}
