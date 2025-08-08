#pragma once
#ifndef GPUCXX_API_DETAILS_RUNTIME_ERROR_INL
#define GPUCXX_API_DETAILS_RUNTIME_ERROR_INL

#include <fmt/format.h>
#include <gpucxx/backend/backend.hpp>
#include <gpucxx/utils/define_specifiers.hpp>

GPUCXX_BEGIN_NAMESPACE

namespace details_ {

  auto checkDeviceError(const deviceError_t result, char const *const func,
                        const char *const file, const int line) -> void {
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

  auto checkLastDeviceError(const char *errorMessage, const char *file,
                            const int line) -> void {
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

  auto peekLastDeviceError(const char *errorMessage, const char *file,
                           const int line) -> void {
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

#endif