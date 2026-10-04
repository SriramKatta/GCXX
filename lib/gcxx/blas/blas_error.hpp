// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_BLAS_ERROR_HPP_
#define GCXX_BLAS_BLAS_ERROR_HPP_


#include <gcxx/blas/error/blas_error.hpp>
#include <gcxx/internal/prologue.hpp>

// GCXX_SAFE_BLAS_CALL — the BLAS analogue of GCXX_SAFE_RUNTIME_CALL.
#ifdef GCXX_ENABLE_RUNTIME_CHECKS
// Checks ENABLED (opt in).
#define GCXX_SAFE_BLAS_CALL(BASEFUNCNAME, MSG, ...)                          \
  do {                                                                       \
    const auto blas_status = ::GCXX_BLAS_BACKEND(BASEFUNCNAME)(__VA_ARGS__); \
    if (blas_status != driver::deviceBlasStatusSuccess) {                    \
      gcxx::blas::details_::throwBlasError(blas_status, MSG);                \
    }                                                                        \
  } while (0)
#else
// Checks DISABLED (default).
#define GCXX_SAFE_BLAS_CALL(BASEFUNCNAME, MSG, ...)       \
  do {                                                    \
    (void)::GCXX_BLAS_BACKEND(BASEFUNCNAME)(__VA_ARGS__); \
  } while (0)
#endif

#endif
