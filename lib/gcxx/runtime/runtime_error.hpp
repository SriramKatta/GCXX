#pragma once
#ifndef GCXX_RUNTIME_RUNTIME_ERROR_HPP_
#define GCXX_RUNTIME_RUNTIME_ERROR_HPP_
#include <cassert>

#include <gcxx/runtime/error/runtime_error.hpp>

// TODO : Implement no ERR check style may increace the performance
#define GCXX_SAFE_RUNTIME_CALL(BASEFUNCNAME, MSG, ...)                      \
  do {                                                                      \
    const auto err_state = GCXX_RUNTIME_BACKEND(BASEFUNCNAME)(__VA_ARGS__); \
    switch (err_state) {                                                    \
      case gcxx::details_::deviceErrSuccess:                                \
        break;                                                              \
      default:                                                              \
        const auto err_state = GCXX_RUNTIME_BACKEND(GetLastError)();        \
        gcxx::details_::throwGPUError(err_state, MSG);                      \
    }                                                                       \
  } while (0)

#define GCXX_DYNAMIC_EXPECT(COND, MSG) assert((COND) && (MSG))

#define GCXX_STATIC_EXPECT(COND, MSG) static_assert(COND, MSG)


#endif
