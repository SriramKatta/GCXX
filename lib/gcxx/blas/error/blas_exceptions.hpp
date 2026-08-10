// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_ERROR_BLAS_EXCEPTIONS_HPP_
#define GCXX_BLAS_ERROR_BLAS_EXCEPTIONS_HPP_

#include <stdexcept>
#include <string>
#include <string_view>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_blas_handles.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FH auto getBlasStatusName(driver::deviceBlasStatus_t status) -> const
  char* {
  switch (status) {
    case GCXX_BLAS_STATUS(SUCCESS):
      return "SUCCESS";
    case GCXX_BLAS_STATUS(NOT_INITIALIZED):
      return "NOT_INITIALIZED";
    case GCXX_BLAS_STATUS(ALLOC_FAILED):
      return "ALLOC_FAILED";
    case GCXX_BLAS_STATUS(INVALID_VALUE):
      return "INVALID_VALUE";
    case GCXX_BLAS_STATUS(ARCH_MISMATCH):
      return "ARCH_MISMATCH";
    case GCXX_BLAS_STATUS(MAPPING_ERROR):
      return "MAPPING_ERROR";
    case GCXX_BLAS_STATUS(EXECUTION_FAILED):
      return "EXECUTION_FAILED";
    case GCXX_BLAS_STATUS(INTERNAL_ERROR):
      return "INTERNAL_ERROR";
    case GCXX_BLAS_STATUS(NOT_SUPPORTED):
      return "NOT_SUPPORTED";
    default:
      return "UNKNOWN_BLAS_STATUS";
  }
}

class BlasException : public std::runtime_error {
 public:
  BlasException(driver::deviceBlasStatus_t status, std::string_view context)
      : std::runtime_error(make_message(status, context)), m_status(status) {}

  GCXX_FHC auto status() const noexcept -> driver::deviceBlasStatus_t {
    return m_status;
  }

 private:
  GCXX_FH static auto make_message(driver::deviceBlasStatus_t status,
                                   std::string_view context) -> std::string {
    std::string msg;
    msg.reserve(context.size() + 32);
    msg.append(context)
      .append(": ")
      .append(getBlasStatusName(status))
      .append(" (")
      .append(std::to_string(static_cast<int>(status)))
      .append(")");
    return msg;
  }

  driver::deviceBlasStatus_t m_status;
};

GCXX_NAMESPACE_MAIN_END()

#endif
