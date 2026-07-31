// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_RUNTIME_ERROR_HPP_
#define GCXX_RUNTIME_RUNTIME_ERROR_HPP_
#include <cassert>

#include <gcxx/runtime/error/runtime_error.hpp>

#ifndef GCXX_DISABLE_RUNTIME_CHECKS
// =======================
// Checks ENABLED (default)
// =======================
#define GCXX_SAFE_RUNTIME_CALL(BASEFUNCNAME, MSG, ...)                        \
  do {                                                                        \
    const auto err_state = ::GCXX_RUNTIME_BACKEND(BASEFUNCNAME)(__VA_ARGS__); \
    switch (err_state) {                                                      \
      case gcxx::driver::deviceErrSuccess:                                    \
        break;                                                                \
      default:                                                                \
        const auto last = driver::GetLastError();                             \
        gcxx::details_::throwGPUError(last, MSG);                             \
    }                                                                         \
  } while (0)
#else

// =======================
// Checks DISABLED (opt in)
// =======================

#define GCXX_DISCARD_RETURN_VALUES(expr) ((void)(expr))

#define GCXX_SAFE_RUNTIME_CALL(BASEFUNCNAME, MSG, ...)    \
  do {                                                    \
    GCXX_DISCARD_RETURN_VALUES(                           \
      ::GCXX_RUNTIME_BACKEND(BASEFUNCNAME)(__VA_ARGS__)); \
  } while (0)

#endif

#define GCXX_RUNTIME_EXPECT(COND, MSG) assert((COND) && (MSG))

#endif
