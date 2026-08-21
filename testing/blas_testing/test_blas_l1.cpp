// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
// End-to-end Level-1 tests (axpy/scale/dot/norm/Givens/swap/idx) via the
// cu/hipblas*Ex entry points vs host references; GPU-gated, still compiles.

#include "tests_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
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

    // Asynchronous device_scalar result forms.
    auto dDot = gcxx::make_device_unique_ptr<double>(std::size_t{1});
    auto dNrm = gcxx::make_device_unique_ptr<double>(std::size_t{1});
    gcxx::blas::dot(handle, X, Y,
                    gcxx::blas::device_scalar<double>{dDot.get()});
    gcxx::blas::vector_two_norm(handle, X,
                                gcxx::blas::device_scalar<double>{dNrm.get()});
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

  template <class IndexT>
  void run_copy_swap_asum_idx() {
    if (!gcxx::testing::haveCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }

    constexpr int N = 6;

    std::vector<double> hX{0.5, -3.0, 2.0, -2.0, 3.0, -0.5};
    std::vector<double> hY{9.0, 9.0, 9.0, 9.0, 9.0, 9.0};

    gcxx::Stream str;
    auto dX = gcxx::make_device_unique_ptr<double>(std::size_t{N});
    auto dY = gcxx::make_device_unique_ptr<double>(std::size_t{N});
    gcxx::Copy(str, dX.get(), hX.data(), std::size_t{N});
    gcxx::Copy(str, dY.get(), hY.data(), std::size_t{N});

    auto X = gcxx::make_device_vector<IndexT>(gcxx::span(dX.get(), N));
    auto Y = gcxx::make_device_vector<IndexT>(gcxx::span(dY.get(), N));

    gcxx::blas::BlasHandle handle;
    handle.setStream(str);

    // copy: Y = X
    gcxx::blas::copy(handle, X, Y);
    str.Synchronize();

    std::vector<double> hResult(N);
    gcxx::Copy(str, hResult.data(), dY.get(), std::size_t{N});
    str.Synchronize();
    for (int i = 0; i < N; ++i) {
      EXPECT_DOUBLE_EQ(hResult[i], hX[i]) << "copy mismatch at " << i;
    }

    // returning + init + asynchronous device_scalar abs-sum forms
    double asum_ref = 0.0;
    for (double v : hX) {
      asum_ref += std::fabs(v);
    }
    const double asum_r    = gcxx::blas::vector_abs_sum(handle, X);
    const double asum_init = gcxx::blas::vector_abs_sum(handle, X, 1.5);
    EXPECT_NEAR(asum_r, asum_ref, 1e-9);
    EXPECT_NEAR(asum_init, 1.5 + asum_ref, 1e-9);

    auto dAsum = gcxx::make_device_unique_ptr<double>(std::size_t{1});
    gcxx::blas::vector_abs_sum(handle, X,
                               gcxx::blas::device_scalar<double>{dAsum.get()});
    str.Synchronize();
    double asum_d{};
    gcxx::Copy(str, &asum_d, dAsum.get(), std::size_t{1});
    str.Synchronize();
    EXPECT_NEAR(asum_d, asum_ref, 1e-9);

    // zero-based index-of-extreme forms (ties broken by FIRST occurrence)
    const IndexT imax = gcxx::blas::idx_abs_max(handle, X);
    const IndexT imin = gcxx::blas::idx_abs_min(handle, X);
    EXPECT_EQ(imax, IndexT{1});  // |−3.0| (index 1) ties |3.0| (index 4);
                                 // the backend reports the first
    EXPECT_EQ(imin, IndexT{0});  // |0.5| ties |−0.5|; the first wins

    // swap_elements: X <-> Y (Y holds X's values from the copy above)
    gcxx::blas::swap_elements(handle, X, Y);
    str.Synchronize();

    std::vector<double> hXs(N), hYs(N);
    gcxx::Copy(str, hXs.data(), dX.get(), std::size_t{N});
    gcxx::Copy(str, hYs.data(), dY.get(), std::size_t{N});
    str.Synchronize();
    for (int i = 0; i < N; ++i) {
      EXPECT_DOUBLE_EQ(hXs[i], hX[i]) << "swap x mismatch at " << i;
      EXPECT_DOUBLE_EQ(hYs[i], hY[i]) << "swap y mismatch at " << i;
    }
  }

  TEST(BlasL1, SetupGivensRotationHost) {
    const std::vector<std::pair<double, double>> cases{
      {3.0, 4.0},       {0.0, 0.0},      {0.0, 5.0},   {5.0, 0.0},
      {-3.0, 4.0},      {3.0, -4.0},     {-3.0, -4.0}, {1e150, 1e150},
      {1e-150, 1e-150}, {1e150, 1e-150}, {2.5, -0.75},
    };
    for (auto [a, b] : cases) {
      double c{}, s{}, r{};
      gcxx::blas::setup_givens_rotation(a, b, c, s, r);
      EXPECT_NEAR(c * a + s * b, r, 1e-9 * std::max(1.0, std::fabs(r)))
        << "c*a+s*b != r for a=" << a << " b=" << b;
      EXPECT_NEAR(c * b - s * a, 0.0, 1e-9 * std::max(1.0, std::fabs(r)))
        << "c*b-s*a != 0 for a=" << a << " b=" << b;
      if (a != 0.0 || b != 0.0) {
        EXPECT_NEAR(c * c + s * s, 1.0, 1e-12);
      } else {
        EXPECT_EQ(c, 1.0);
        EXPECT_EQ(s, 0.0);
        EXPECT_EQ(r, 0.0);
      }
    }

    // 4-argument form computes the same coefficients as the 5-argument one
    double c5{}, s5{}, r5{}, c4{}, s4{};
    gcxx::blas::setup_givens_rotation(3.0, 4.0, c5, s5, r5);
    gcxx::blas::setup_givens_rotation(3.0, 4.0, c4, s4);
    EXPECT_DOUBLE_EQ(c4, c5);
    EXPECT_DOUBLE_EQ(s4, s5);
  }

  // rotmg invariants: H*(x1,y1)=(x1',0), energy d1'*x1'^2, degenerate zeros.
  TEST(BlasL1, SetupModifiedGivensRotationHost) {
    std::vector<std::array<double, 4>> cases{
      {1.0, 1.0, 2.0, 3.0},  {2.0, 0.5, 1.0, -4.0}, {0.25, 4.0, 3.0, 1.0},
      {-1.0, 1.0, 2.0, 3.0}, {1.0, 0.0, 2.0, 3.0},  {1.0, 2.0, 0.0, 5.0},
      {1e-8, 1e8, 1.0, 1.0}, {1e8, 1e-8, 1.0, 1.0}, {1.0, 1.0, 1e-10, 1e10},
      {3.0, 3.0, 3.0, -3.0}, {5.0, 0.2, 0.5, 0.25}, {0.7, 0.7, 0.2, -0.9},
    };

    for (const auto& inp : cases) {
      double d1 = inp[0], d2 = inp[1], x1 = inp[2], y1 = inp[3];
      const double d1_0 = d1, d2_0 = d2, x1_0 = x1;

      std::array<double, 5> param{};
      gcxx::blas::setup_modified_givens_rotation(d1, d2, x1, y1, param);

      // degenerate branch: everything collapses to zero
      if (d1_0 < 0.0) {
        EXPECT_EQ(param[0], -1.0);
        EXPECT_EQ(d1, 0.0);
        EXPECT_EQ(d2, 0.0);
        EXPECT_EQ(x1, 0.0);
        continue;
      }

      // no rotation needed: H = I, d1/d2/x1 untouched
      if (d2_0 * y1 == 0.0) {
        EXPECT_EQ(param[0], -2.0);
        EXPECT_EQ(d1, d1_0);
        EXPECT_EQ(d2, d2_0);
        EXPECT_EQ(x1, x1_0);
        continue;
      }

      // reconstruct H from the flag convention
      double h11, h12, h21, h22;
      if (param[0] == -1.0) {
        h11 = param[1];
        h21 = param[2];
        h12 = param[3];
        h22 = param[4];
      } else if (param[0] == 0.0) {
        h11 = 1.0;
        h21 = param[2];
        h12 = param[3];
        h22 = 1.0;
      } else {
        h11 = param[1];
        h21 = -1.0;
        h12 = 1.0;
        h22 = param[4];
      }

      // energy preservation (scale-aware tolerance)
      const double energy = d1_0 * x1_0 * x1_0 + d2_0 * y1 * y1;
      const double eps    = 1e-12 * std::max(1.0, energy);
      EXPECT_GE(d1, 0.0) << "negative d1' for d1=" << d1_0 << " d2=" << d2_0;
      EXPECT_NEAR(d1 * x1 * x1, energy, eps)
        << "energy not preserved for d1=" << d1_0 << " d2=" << d2_0
        << " x1=" << x1_0 << " y1=" << y1;

      // Plain-pair annihilation for non-rescaled flag-0/1 outcomes.
      if (param[0] == 0.0 || param[0] == 1.0) {
        const double u1     = h11 * x1_0 + h12 * y1;
        const double u2     = h21 * x1_0 + h22 * y1;
        const double pscale = std::max({1.0, std::fabs(x1_0), std::fabs(y1)});
        EXPECT_NEAR(u2, 0.0, 1e-12 * pscale)
          << "H*(x1,y1) second component not zero for d1=" << d1_0;
        EXPECT_NEAR(u1, x1, 1e-12 * pscale)
          << "H*(x1,y1) first component != x1' for d1=" << d1_0;
      }
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

TEST(BlasL1, CopySwapAsumIdx_Double) {
  run_copy_swap_asum_idx<int>();
}

TEST(BlasL1, CopySwapAsumIdx_Double_64bitIndex) {
  run_copy_swap_asum_idx<std::int64_t>();
}
