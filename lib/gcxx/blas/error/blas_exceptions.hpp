// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_ERROR_BLAS_EXCEPTIONS_HPP_
#define GCXX_BLAS_ERROR_BLAS_EXCEPTIONS_HPP_

#include <stdexcept>
#include <string_view>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_blas_error.hpp>
#include <gcxx/runtime_backend/backend_blas_handles.hpp>

GCXX_NAMESPACE_MAIN_BLAS_BEGIN()

class BlasException : public std::runtime_error {
 public:
  BlasException(driver::deviceBlasStatus_t status, std::string_view context)
      : std::runtime_error(driver::make_message(status, context)),
        m_status(status) {}

  GCXX_FHC auto status() const noexcept -> driver::deviceBlasStatus_t {
    return m_status;
  }

 private:
  driver::deviceBlasStatus_t m_status;
};

GCXX_NAMESPACE_MAIN_BLAS_END()

#endif
