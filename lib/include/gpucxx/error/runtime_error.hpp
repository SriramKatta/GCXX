#pragma once
#ifndef GPUCXX_API_RUNTIME_ERROR_HPP
#define GPUCXX_API_RUNTIME_ERROR_HPP

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/utils/define_specifiers.hpp>

#include <fmt/format.h>

GPUCXX_BEGIN_NAMESPACE

namespace details_ {
  using deviceError_t            = GPUCXX_RUNTIME_BACKEND(Error_t);
  GPUCXX_CA deviceSuccess        = GPUCXX_RUNTIME_BACKEND(Success);
  GPUCXX_CA deviceGetErrorstring = GPUCXX_RUNTIME_BACKEND(GetErrorString);

  GPUCXX_FH void checkDeviceError(const deviceError_t result,
                                  char const *const func,
                                  const char *const file, const int line) {
    if (result != deviceSuccess) {
      fmt::print(stderr,
                 "[{} Runtime Error]\n"
                 "  Location : {}:{}\n"
                 "  Function : {}\n"
                 "  Error    : code = {} ({})\n",
                 GPUCXX_RUNTIME_BACKEND_STR, file, line, func,
                 static_cast<unsigned int>(result),
                 deviceGetErrorstring(result));

      exit(EXIT_FAILURE);
    }
  }

  GPUCXX_FH void checkLastDeviceError(const char *errorMessage,
                                      const char *file, const int line) {
    auto err = GPUCXX_RUNTIME_BACKEND(GetLastError)();
    if (err != deviceSuccess) {
      fmt::print(stderr,
                 "[{} Last Error]\n"
                 "  Location : {}:{}\n"
                 "  Context  : {}\n"
                 "  Error    : code = {} ({})\n",
                 GPUCXX_RUNTIME_BACKEND_STR, file, line, errorMessage,
                 static_cast<int>(err), deviceGetErrorstring(err));
      exit(EXIT_FAILURE);
    }
  }

  GPUCXX_FH void peekLastDeviceError(const char *errorMessage, const char *file,
                                     const int line) {
    auto err = GPUCXX_RUNTIME_BACKEND(PeekAtLastError)();
    if (err != deviceSuccess) {
      fmt::print(stderr,
                 "[{} Peek Error]\n"
                 "  Location : {}:{}\n"
                 "  Context  : {}\n"
                 "  Error    : code = {} ({})\n",
                 GPUCXX_RUNTIME_BACKEND_STR, file, line, errorMessage,
                 static_cast<int>(err), deviceGetErrorstring(err));
    }
  }
}  // namespace details_

GPUCXX_END_NAMESPACE

#define GPUCXX_CHECK_DEVICE_ERR(val)                                     \
  do {                                                                   \
    gpuCXX::details_::checkDeviceError((val), #val, __FILE__, __LINE__); \
  } while (0)

#define GPUCXX_CHECK_DEVICE_LASTERR(...)                                      \
  do {                                                                        \
    gpuCXX::details_::checkLastDeviceError(#__VA_ARGS__, __FILE__, __LINE__); \
  } while (0)

#define GPUCXX_PEEK_DEVICE_LASTERR(...)                                      \
  do {                                                                       \
    gpuCXX::details_::peekLastDeviceError(#__VA_ARGS__, __FILE__, __LINE__); \
  } while (0)

#define GPUCXX_SAFE_RUNTIME_CALL(func, Args) \
  GPUCXX_CHECK_DEVICE_ERR(GPUCXX_RUNTIME_BACKEND(func) Args)

#endif