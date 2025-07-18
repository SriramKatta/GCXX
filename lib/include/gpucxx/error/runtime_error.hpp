#pragma once
#ifndef GPUCXX_API_RUNTIME_ERROR_HPP
#define GPUCXX_API_RUNTIME_ERROR_HPP

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/utils/define_specifiers.hpp>

#include <fmt/format.h>

namespace gpuCXX {
  namespace details_ {
    using deviceError_t          = GPUCXX_RUNTIME_BACKEND(Error_t);
    constexpr auto deviceSuccess = GPUCXX_RUNTIME_BACKEND(Success);

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
                   GPUCXX_RUNTIME_BACKEND(GetErrorString)(result));

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
                   static_cast<int>(err),
                   GPUCXX_RUNTIME_BACKEND(GetErrorString)(err));
        exit(EXIT_FAILURE);
      }
    }

    GPUCXX_FH void peekLastDeviceError(const char *errorMessage,
                                       const char *file, const int line) {
      auto err = GPUCXX_RUNTIME_BACKEND(PeekAtLastError)();
      if (err != deviceSuccess) {
        fmt::print(stderr,
                   "[{} Peek Error]\n"
                   "  Location : {}:{}\n"
                   "  Context  : {}\n"
                   "  Error    : code = {} ({})\n",
                   GPUCXX_RUNTIME_BACKEND_STR, file, line, errorMessage,
                   static_cast<int>(err),
                   GPUCXX_RUNTIME_BACKEND(GetErrorString)(err));
      }
    }
  }  // namespace details_
}  // namespace gpuCXX

#define CHECK_DEVICE_ERR(val)                                            \
  do {                                                                   \
    gpuCXX::details_::checkDeviceError((val), #val, __FILE__, __LINE__); \
  } while (0)

#define CHECK_DEVICE_LASTERR(...)                                             \
  do {                                                                        \
    gpuCXX::details_::checkLastDeviceError(#__VA_ARGS__, __FILE__, __LINE__); \
  } while (0)

#define PEEK_DEVICE_LASTERR(...)                                             \
  do {                                                                       \
    gpuCXX::details_::peekLastDeviceError(#__VA_ARGS__, __FILE__, __LINE__); \
  } while (0)


#endif