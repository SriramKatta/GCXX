// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_ERROR_GCXX_EXCEPTION_HPP_
#define GCXX_RUNTIME_ERROR_GCXX_EXCEPTION_HPP_

#include <exception>
#include <string_view>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/error/runtime_error_types.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class Exception : public std::runtime_error {
 public:
  Exception(details_::deviceError_t err, std::string_view context)
      : std::runtime_error(details_::make_message(err, context)),
        m_error(err) {}

  details_::deviceError_t error() const noexcept { return m_error; }

 private:
  details_::deviceError_t m_error;
};

GCXX_NAMESPACE_MAIN_END()


#endif
