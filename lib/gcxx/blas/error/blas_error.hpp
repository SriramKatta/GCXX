// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_ERROR_BLAS_ERROR_HPP_
#define GCXX_BLAS_ERROR_BLAS_ERROR_HPP_

#include <cstdlib>
#include <iostream>

#include <gcxx/blas/error/blas_exceptions.hpp>
#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_BEGIN()

[[noreturn]] inline auto throwBlasError(driver::deviceBlasStatus_t status,
                                        const char* msg) -> void {
#if defined(GCXX_WITH_EXCEPTIONS)
  throw gcxx::blas::BlasException(status, msg);
#else
  std::cerr << gcxx::blas::BlasException(status, msg).what() << "\n";
  std::abort();
#endif
}

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_END()

#endif
