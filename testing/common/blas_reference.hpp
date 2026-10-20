// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_TESTING_COMMON_BLAS_REFERENCE_HPP_
#define GCXX_TESTING_COMMON_BLAS_REFERENCE_HPP_

#include <gcxx/api.hpp>

// Shared column-major reference fixtures for the blas tests: host/device
// mdspan aliases and the naive host GEMM used as the golden model.

template <class IndexT>
using dextents2d = gcxx::dextents<IndexT, 2>;

template <class T, class IndexT>
using mat_left = gcxx::mdspan<T, dextents2d<IndexT>, gcxx::layout_left,
                              gcxx::default_accessor<T>>;

template <class T, class IndexT>
using vec_left = gcxx::mdspan<T, gcxx::dextents<IndexT, 1>, gcxx::layout_left,
                              gcxx::default_accessor<T>>;

// Device-memory counterparts required by gcxx::blas.
template <class T, class IndexT>
using dmat_left = gcxx::device_mdspan<T, dextents2d<IndexT>, gcxx::layout_left>;

template <class T, class IndexT>
using dvec_left =
  gcxx::device_mdspan<T, gcxx::dextents<IndexT, 1>, gcxx::layout_left>;

template <class MatA, class MatB, class MatC, class S>
auto host_gemm(const MatA& a, const MatB& b, const MatC& cref, MatC out,
               S alpha, S beta) -> void {
  const int m = static_cast<int>(a.extent(0));
  const int k = static_cast<int>(a.extent(1));
  const int n = static_cast<int>(b.extent(1));
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      S acc{};
      for (int p = 0; p < k; ++p) {
        acc += static_cast<S>(a(i, p)) * static_cast<S>(b(p, j));
      }
      out(i, j) = alpha * acc + beta * static_cast<S>(cref(i, j));
    }
  }
}

#endif
