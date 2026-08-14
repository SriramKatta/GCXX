// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// End-to-end Level-1 tests: axpy / scal / dot / nrm2 / rot via the
// type-erased cu/hipblas*Ex entry points, compared against a host reference.
// GPU-gated — skipped when no device is present, but the template must still
// compile (it instantiates both the int and the int64_t index_type dispatch,
// i.e. the *_Ex and *_Ex_64 entry points).

#include "tests_common.hpp"

#include <cmath>
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
  void run_axpy_scal() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    constexpr int N = 5;
    const double alpha = 2.0, scale = 3.0;

    std::vector<double> hX{1.0, -2.0, 3.0, -4.0, 5.0};
    std::vector<double> hY{0.5, 0.5, 0.5, 0.5, 0.5};

    gcxx::Stream str;
    auto dX = gcxx::make_device_unique_ptr<double>(std::size_t{N});
    auto dY = gcxx::make_device_unique_ptr<double>(std::size_t{N});
    gcxx::Copy(str, dX.get(), hX.data(), std::size_t{N});
    gcxx::Copy(str, dY.get(), hY.data(), std::size_t{N});

    auto X = gcxx::make_vector<IndexT>(gcxx::span(dX.get(), N));
    auto Y = gcxx::make_vector<IndexT>(gcxx::span(dY.get(), N));

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);

    gcxx::blas::scal(handle, scale, X);
    gcxx::blas::axpy(handle, alpha, X, Y);
    str.Synchronize();

    std::vector<double> hResult(N);
    gcxx::Copy(str, hResult.data(), dY.get(), std::size_t{N});
    str.Synchronize();

    for (int i = 0; i < N; ++i) {
      const double want = alpha * scale * hX[i] + hY[i];
      EXPECT_NEAR(hResult[i], want, 1e-9) << "axpy/scal mismatch at " << i;
    }
  }

  template <class IndexT>
  void run_dot_nrm2_rot() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    constexpr int  N     = 5;
    constexpr double c = 0.8, s = 0.6;

    std::vector<double> hX{1.0, -2.0, 3.0, -4.0, 5.0};
    std::vector<double> hY{2.0, 1.0, 0.0, -1.0, -2.0};

    gcxx::Stream str;
    auto dX = gcxx::make_device_unique_ptr<double>(std::size_t{N});
    auto dY = gcxx::make_device_unique_ptr<double>(std::size_t{N});
    gcxx::Copy(str, dX.get(), hX.data(), std::size_t{N});
    gcxx::Copy(str, dY.get(), hY.data(), std::size_t{N});

    auto X = gcxx::make_vector<IndexT>(gcxx::span(dX.get(), N));
    auto Y = gcxx::make_vector<IndexT>(gcxx::span(dY.get(), N));

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);

    // dot + nrm2: host pointer mode writes the result synchronously
    double dot_r{}, nrm2_r{};
    gcxx::blas::dot(handle, X, Y, &dot_r);
    gcxx::blas::nrm2(handle, X, &nrm2_r);
    str.Synchronize();

    double dot_ref{}, nrm2_ref{};
    for (int i = 0; i < N; ++i) {
      dot_ref += hX[i] * hY[i];
      nrm2_ref += hX[i] * hX[i];
    }
    nrm2_ref = std::sqrt(nrm2_ref);
    EXPECT_NEAR(dot_r, dot_ref, 1e-9);
    EXPECT_NEAR(nrm2_r, nrm2_ref, 1e-9);

    gcxx::blas::rot(handle, c, s, X, Y);
    str.Synchronize();

    std::vector<double> hXr(N), hYr(N);
    gcxx::Copy(str, hXr.data(), dX.get(), std::size_t{N});
    gcxx::Copy(str, hYr.data(), dY.get(), std::size_t{N});
    str.Synchronize();

    for (int i = 0; i < N; ++i) {
      EXPECT_NEAR(hXr[i], c * hX[i] + s * hY[i], 1e-9) << "rot x mismatch at " << i;
      EXPECT_NEAR(hYr[i], c * hY[i] - s * hX[i], 1e-9) << "rot y mismatch at " << i;
    }
  }

}  // namespace

TEST(BlasL1, AxpyScal_Double) {
  run_axpy_scal<int>();
}

TEST(BlasL1, AxpyScal_Double_64bitIndex) {
  run_axpy_scal<std::int64_t>();
}

TEST(BlasL1, DotNrm2Rot_Double) {
  run_dot_nrm2_rot<int>();
}

TEST(BlasL1, DotNrm2Rot_Double_64bitIndex) {
  run_dot_nrm2_rot<std::int64_t>();
}
