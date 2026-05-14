// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_BACKEND_BACKEND_ERROR_HPP_
#define GCXX_RUNTIME_BACKEND_BACKEND_ERROR_HPP_

#include <string>
#include <string_view>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_handles.hpp>


GCXX_NAMESPACE_MAIN_DRIVER_BEGIN

GCXX_FHD const char* GetErrorName(deviceError_t err) {
  return ::GCXX_RUNTIME_BACKEND(GetErrorName)(err);
}

GCXX_FHD const char* GetErrorString(deviceError_t err) {
  return ::GCXX_RUNTIME_BACKEND(GetErrorString)(err);
}

GCXX_FH std::string make_message(deviceError_t err, std::string_view context) {
  std::string msg;
  constexpr size_t addtional_message_size = 128;
  msg.reserve(context.size() + addtional_message_size);

  msg.append(context)
    .append(": ")
    .append(GetErrorName(err))
    .append(" (")
    .append(GetErrorString(err))
    .append(")");

  return msg;
}

GCXX_NAMESPACE_MAIN_DRIVER_END


#endif
