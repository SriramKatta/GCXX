// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_BACKEND_BACKEND_ERROR_HPP_
#define GCXX_RUNTIME_BACKEND_BACKEND_ERROR_HPP_

#include <string>
#include <string_view>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_error_common.hpp>
#include <gcxx/runtime_backend/backend_handles.hpp>

GCXX_NAMESPACE_MAIN_DRIVER_BEGIN()

GCXX_FHD auto GetErrorName(deviceError_t err) -> const char* {
  return ::GCXX_RUNTIME_BACKEND(GetErrorName)(err);
}

GCXX_FHD auto GetErrorString(deviceError_t err) -> const char* {
  return ::GCXX_RUNTIME_BACKEND(GetErrorString)(err);
}

GCXX_FHD auto GetLastError() -> deviceError_t {
  return ::GCXX_RUNTIME_BACKEND(GetLastError)();
}

GCXX_FHD auto PeekAtLastError() -> deviceError_t {
  return ::GCXX_RUNTIME_BACKEND(PeekAtLastError)();
}

GCXX_FH auto make_message(deviceError_t err,
                          std::string_view context) -> std::string {
  return make_status_message(context, GetErrorName(err), GetErrorString(err));
}

GCXX_NAMESPACE_MAIN_DRIVER_END()


#endif
