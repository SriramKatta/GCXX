// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_BACKEND_BACKEND_ERROR_COMMON_HPP_
#define GCXX_RUNTIME_BACKEND_BACKEND_ERROR_COMMON_HPP_

#include <cstddef>
#include <string>
#include <string_view>

#include <gcxx/internal/prologue.hpp>

// Reserved headroom for the status text appended
constexpr std::size_t additional_message_size = 32;

GCXX_NAMESPACE_MAIN_DRIVER_BEGIN()

// Shared body of the per-backend make_message() overloads: renders
// "<context>: <name> (<detail>)" with a single allocation in the common case.
GCXX_FH auto make_status_message(std::string_view context, const char* name,
                                 std::string_view detail) -> std::string {
  std::string msg{};
  msg.reserve(context.size() + detail.size() + additional_message_size);

  msg.append(context)
    .append(": ")
    .append(name)
    .append(" (")
    .append(detail)
    .append(")");

  return msg;
}

GCXX_NAMESPACE_MAIN_DRIVER_END()

#endif
