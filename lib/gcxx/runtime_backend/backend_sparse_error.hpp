// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_BACKEND_BACKEND_SPARSE_ERROR_HPP_
#define GCXX_RUNTIME_BACKEND_BACKEND_SPARSE_ERROR_HPP_

#include <string>
#include <string_view>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_error_common.hpp>
#include <gcxx/runtime_backend/backend_sparse_handles.hpp>

GCXX_NAMESPACE_MAIN_DRIVER_BEGIN()

GCXX_FH auto GetSparseStatusName(deviceSparseStatus_t status) -> const char* {
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

GCXX_FH auto make_message(deviceSparseStatus_t status,
                          std::string_view context) -> std::string {
  return make_status_message(context, GetSparseStatusName(status),
                             std::to_string(static_cast<int>(status)));
}

GCXX_NAMESPACE_MAIN_DRIVER_END()

#endif
