// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_ERROR_RUNTIME_ERROR_HPP_
#define GCXX_RUNTIME_ERROR_RUNTIME_ERROR_HPP_

#include <iostream>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_handles.hpp>
#include <gcxx/runtime_backend/backend_error.hpp>

#if defined(GCXX_WITH_EXCEPTIONS)
#include <gcxx/runtime/error/runtime_exception.hpp>
#endif

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN()

inline auto throwGPUError(driver::deviceError_t err, const char* msg) -> void {
#if defined(GCXX_WITH_EXCEPTIONS)
  throw gcxx::Exception(err, msg);
#else
  std::cerr << driver::make_message(err, msg) << "\n";
  std::abort();
#endif
}

GCXX_FH auto nonFatalErrorQuery(driver::deviceError_t err) -> bool {
  switch (err) {
    case driver::deviceErrSuccess:
      return true;
    case driver::deviceErrNotReady:
      return false;
    default:
      details_::throwGPUError(err, /*msg=*/"Failed to query GPU Stream");
  }
  return false;
}

GCXX_NAMESPACE_MAIN_DETAILS_END()


#endif
