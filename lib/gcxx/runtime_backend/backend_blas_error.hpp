// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_BACKEND_BACKEND_BLAS_ERROR_HPP_
#define GCXX_RUNTIME_BACKEND_BACKEND_BLAS_ERROR_HPP_

#include <string>
#include <string_view>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_blas_handles.hpp>

GCXX_NAMESPACE_MAIN_DRIVER_BEGIN()

GCXX_FH auto GetBlasStatusName(deviceBlasStatus_t status) -> const char* {
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

GCXX_FH auto make_message(deviceBlasStatus_t status,
                          std::string_view context) -> std::string {
  std::string msg{};
  msg.reserve(context.size() + additional_message_size);

  msg.append(context)
    .append(": ")
    .append(GetBlasStatusName(status))
    .append(" (")
    .append(std::to_string(static_cast<int>(status)))
    .append(")");

  return msg;
}

GCXX_NAMESPACE_MAIN_DRIVER_END()

#endif
