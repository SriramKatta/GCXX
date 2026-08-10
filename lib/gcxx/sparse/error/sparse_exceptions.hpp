// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_SPARSE_ERROR_SPARSE_EXCEPTIONS_HPP_
#define GCXX_SPARSE_ERROR_SPARSE_EXCEPTIONS_HPP_

#include <stdexcept>
#include <string>
#include <string_view>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_sparse_handles.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FH auto getSparseStatusName(driver::deviceSparseStatus_t status) -> const
  char* {
  switch (status) {
    case GCXX_SPARSE_STATUS(SUCCESS):
      return "SUCCESS";
    case GCXX_SPARSE_STATUS(NOT_INITIALIZED):
      return "NOT_INITIALIZED";
    case GCXX_SPARSE_STATUS(ALLOC_FAILED):
      return "ALLOC_FAILED";
    case GCXX_SPARSE_STATUS(INVALID_VALUE):
      return "INVALID_VALUE";
    case GCXX_SPARSE_STATUS(ARCH_MISMATCH):
      return "ARCH_MISMATCH";
    case GCXX_SPARSE_STATUS(MAPPING_ERROR):
      return "MAPPING_ERROR";
    case GCXX_SPARSE_STATUS(EXECUTION_FAILED):
      return "EXECUTION_FAILED";
    case GCXX_SPARSE_STATUS(INTERNAL_ERROR):
      return "INTERNAL_ERROR";
    case GCXX_SPARSE_STATUS(ZERO_PIVOT):
      return "ZERO_PIVOT";
    case GCXX_SPARSE_STATUS(NOT_SUPPORTED):
      return "NOT_SUPPORTED";
    case GCXX_SPARSE_STATUS(INSUFFICIENT_RESOURCES):
      return "INSUFFICIENT_RESOURCES";
    default:
      return "UNKNOWN_SPARSE_STATUS";
  }
}

class SparseException : public std::runtime_error {
 public:
  SparseException(driver::deviceSparseStatus_t status, std::string_view context)
      : std::runtime_error(make_message(status, context)), m_status(status) {}

  GCXX_FHC auto status() const noexcept -> driver::deviceSparseStatus_t {
    return m_status;
  }

 private:
  GCXX_FH static auto make_message(driver::deviceSparseStatus_t status,
                                   std::string_view context) -> std::string {
    std::string msg;
    msg.reserve(context.size() + 32);
    msg.append(context)
      .append(": ")
      .append(getSparseStatusName(status))
      .append(" (")
      .append(std::to_string(static_cast<int>(status)))
      .append(")");
    return msg;
  }

  driver::deviceSparseStatus_t m_status;
};

GCXX_NAMESPACE_MAIN_END()

#endif
