// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// End-to-end Level-1 tests: axpy / scale / dot / vector_two_norm /
// apply_givens_rotation via the type-erased cu/hipblas*Ex entry points,
// compared against a host reference. Covers the P1673R13 shapes: the
// returning (synchronizing) reduction forms, their init-accumulating
// overloads, and the asynchronous device_scalar result forms.
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
  void run_axpy_scale() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    constexpr int N    = 5;
    const double alpha = 2.0, scale = 3.0;

    std::vector<double> hX{1.0, -2.0, 3.0, -4.0, 5.0};
    std::vector<double> hY{0.5, 0.5, 0.5, 0.5, 0.5};

    gcxx::Stream str;
    auto dX = gcxx::make_device_unique_ptr<double>(std::size_t{N});
    auto dY = gcxx::make_device_unique_ptr<double>(std::size_t{N});
    gcxx::Copy(str, dX.get(), hX.data(), std::size_t{N});
    gcxx::Copy(str, dY.get(), hY.data(), std::size_t{N});

    auto X = gcxx::make_device_vector<IndexT>(gcxx::span(dX.get(), N));
    auto Y = gcxx::make_device_vector<IndexT>(gcxx::span(dY.get(), N));

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);

    gcxx::blas::scale(handle, scale, X);
    gcxx::blas::axpy(handle, alpha, X, Y);
    str.Synchronize();

    std::vector<double> hResult(N);
    gcxx::Copy(str, hResult.data(), dY.get(), std::size_t{N});
    str.Synchronize();

    for (int i = 0; i < N; ++i) {
      const double want = alpha * scale * hX[i] + hY[i];
      EXPECT_NEAR(hResult[i], want, 1e-9) << "axpy/scale mismatch at " << i;
    }
  }

  template <class IndexT>
  void run_dot_two_norm_givens() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    constexpr int N    = 5;
    constexpr double c = 0.8, s = 0.6;

    std::vector<double> hX{1.0, -2.0, 3.0, -4.0, 5.0};
    std::vector<double> hY{2.0, 1.0, 0.0, -1.0, -2.0};

    gcxx::Stream str;
    auto dX = gcxx::make_device_unique_ptr<double>(std::size_t{N});
    auto dY = gcxx::make_device_unique_ptr<double>(std::size_t{N});
    gcxx::Copy(str, dX.get(), hX.data(), std::size_t{N});
    gcxx::Copy(str, dY.get(), hY.data(), std::size_t{N});

    auto X = gcxx::make_device_vector<IndexT>(gcxx::span(dX.get(), N));
    auto Y = gcxx::make_device_vector<IndexT>(gcxx::span(dY.get(), N));

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);

    // returning forms (synchronize before returning) + init accumulation
    const double dot_r     = gcxx::blas::dot(handle, X, Y);
    const double dot_init  = gcxx::blas::dot(handle, X, Y, 10.0);
    const double nrm2_r    = gcxx::blas::vector_two_norm(handle, X);
    const double nrm2_init = gcxx::blas::vector_two_norm(handle, X, 3.0);

    double dot_ref{}, nrm2_ref{};
    for (int i = 0; i < N; ++i) {
      dot_ref += hX[i] * hY[i];
      nrm2_ref += hX[i] * hX[i];
    }
    nrm2_ref = std::sqrt(nrm2_ref);
    EXPECT_NEAR(dot_r, dot_ref, 1e-9);
    EXPECT_NEAR(dot_init, 10.0 + dot_ref, 1e-9);
    EXPECT_NEAR(nrm2_r, nrm2_ref, 1e-9);
    EXPECT_NEAR(nrm2_init, std::sqrt(3.0 * 3.0 + nrm2_ref * nrm2_ref), 1e-9);

    // asynchronous device_scalar result forms
    auto dDot = gcxx::make_device_unique_ptr<double>(std::size_t{1});
    auto dNrm = gcxx::make_device_unique_ptr<double>(std::size_t{1});
    gcxx::blas::dot(handle, X, Y, gcxx::blas::device_scalar<double>{dDot.get()});
    gcxx::blas::vector_two_norm(
      handle, X, gcxx::blas::device_scalar<double>{dNrm.get()});
    str.Synchronize();

    double dot_d{}, nrm2_d{};
    gcxx::Copy(str, &dot_d, dDot.get(), std::size_t{1});
    gcxx::Copy(str, &nrm2_d, dNrm.get(), std::size_t{1});
    str.Synchronize();
    EXPECT_NEAR(dot_d, dot_ref, 1e-9);
    EXPECT_NEAR(nrm2_d, nrm2_ref, 1e-9);

    gcxx::blas::apply_givens_rotation(handle, X, Y, c, s);
    str.Synchronize();

    std::vector<double> hXr(N), hYr(N);
    gcxx::Copy(str, hXr.data(), dX.get(), std::size_t{N});
    gcxx::Copy(str, hYr.data(), dY.get(), std::size_t{N});
    str.Synchronize();

    for (int i = 0; i < N; ++i) {
      EXPECT_NEAR(hXr[i], c * hX[i] + s * hY[i], 1e-9)
        << "apply_givens_rotation x mismatch at " << i;
      EXPECT_NEAR(hYr[i], c * hY[i] - s * hX[i], 1e-9)
        << "apply_givens_rotation y mismatch at " << i;
    }
  }

}  // namespace

TEST(BlasL1, AxpyScale_Double) {
  run_axpy_scale<int>();
}

TEST(BlasL1, AxpyScale_Double_64bitIndex) {
  run_axpy_scale<std::int64_t>();
}

TEST(BlasL1, DotTwoNormGivens_Double) {
  run_dot_two_norm_givens<int>();
}

TEST(BlasL1, DotTwoNormGivens_Double_64bitIndex) {
  run_dot_two_norm_givens<std::int64_t>();
}
