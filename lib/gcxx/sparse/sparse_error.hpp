// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_SPARSE_SPARSE_ERROR_HPP_
#define GCXX_SPARSE_SPARSE_ERROR_HPP_


#include <gcxx/internal/prologue.hpp>
#include <gcxx/sparse/error/sparse_error.hpp>

// GCXX_SAFE_SPARSE_CALL — the SPARSE analogue of GCXX_SAFE_BLAS_CALL
#ifndef GCXX_DISABLE_RUNTIME_CHECKS
// =======================
// Checks ENABLED (default)
// =======================
#define GCXX_SAFE_SPARSE_CALL(BASEFUNCNAME, MSG, ...)         \
  do {                                                        \
    const auto sparse_status =                                \
      ::GCXX_SPARSE_BACKEND(BASEFUNCNAME)(__VA_ARGS__);       \
    if (sparse_status != driver::deviceSparseStatusSuccess) { \
      gcxx::details_::throwSparseError(sparse_status, MSG);   \
    }                                                         \
  } while (0)
#else
// =======================
// Checks DISABLED (opt in)
// =======================
#define GCXX_SAFE_SPARSE_CALL(BASEFUNCNAME, MSG, ...)       \
  do {                                                      \
    (void)::GCXX_SPARSE_BACKEND(BASEFUNCNAME)(__VA_ARGS__); \
  } while (0)
#endif

#endif
