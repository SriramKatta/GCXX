// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_SPARSE_ERROR_SPARSE_ERROR_HPP_
#define GCXX_SPARSE_ERROR_SPARSE_ERROR_HPP_

#include <cstdlib>
#include <iostream>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/sparse/error/sparse_exceptions.hpp>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN()

inline auto throwSparseError(driver::deviceSparseStatus_t status,
                             const char* msg) -> void {
#if defined(GCXX_WITH_EXCEPTIONS)
  throw gcxx::SparseException(status, msg);
#else
  std::cerr << gcxx::SparseException(status, msg).what() << "\n";
  std::abort();
#endif
}

GCXX_NAMESPACE_MAIN_DETAILS_END()

#endif
