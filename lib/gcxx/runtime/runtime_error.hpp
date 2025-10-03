#pragma once
#ifndef GCXX_RUNTIME_RUNTIME_ERROR_HPP_
#define GCXX_RUNTIME_RUNTIME_ERROR_HPP_

#include <gcxx/runtime/error/runtime_error.hpp>

#define GCXX_SAFE_RUNTIME_CALL(FUNC, MSG, ...)                      \
  do {                                                              \
    const auto err_state = GCXX_RUNTIME_BACKEND(FUNC)(__VA_ARGS__); \
    switch (err_state) {                                            \
      case gcxx::details_::deviceSuccess:                           \
        break;                                                      \
      default:                                                      \
        gcxx::details_::deviceGetLastError();                       \
        gcxx::details_::throwGPUError(err_state, MSG);              \
    }                                                               \
  } while (0)


#endif
