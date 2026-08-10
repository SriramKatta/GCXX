// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_ERROR_RUNTIME_EXCEPTION_HPP_
#define GCXX_RUNTIME_ERROR_RUNTIME_EXCEPTION_HPP_

#include <stdexcept>
#include <string_view>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_error.hpp>
#include <gcxx/runtime_backend/backend_handles.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class Exception : public std::runtime_error {
 public:
  Exception(driver::deviceError_t err, std::string_view context)
      : std::runtime_error(driver::make_message(err, context)), m_error(err) {}

  driver::deviceError_t error() const noexcept { return m_error; }

 private:
  driver::deviceError_t m_error;
};

GCXX_NAMESPACE_MAIN_END()


#endif
