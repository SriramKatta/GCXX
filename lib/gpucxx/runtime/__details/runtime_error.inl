#pragma once
#ifndef GPUCXX_RUNTIME_DETAILS_RUNTIME_ERROR_INL_
#define GPUCXX_RUNTIME_DETAILS_RUNTIME_ERROR_INL_


#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>

#include <stdio.h>

GPUCXX_BEGIN_NAMESPACE

namespace details_ {

  auto checkDeviceError(const deviceError_t result, char const* const func,
                        const char* const file, const int line) -> void {
    if (result != deviceSuccess) {
      fprintf(stderr,
              "[%s Runtime Error]\n"
              "  Location : %s:%d\n"
              "  Function : %s\n"
              "  Error    : code = %d (%s)\n",
              GPUCXX_RUNTIME_BACKEND_STR, file, line, func,
              static_cast<unsigned int>(result), deviceGetErrorstring(result));

      exit(EXIT_FAILURE);
    }
  }

  auto checkLastDeviceError(const char* errorMessage, const char* file,
                            const int line) -> void {
    auto err = GPUCXX_RUNTIME_BACKEND(GetLastError)();
    if (err != deviceSuccess) {
      fprintf(stderr,
              "[%s Last Error]\n"
              "  Location : %s:%d\n"
              "  Function : %s\n"
              "  Error    : code = %d (%s)\n",
              GPUCXX_RUNTIME_BACKEND_STR, file, line, errorMessage,
              static_cast<int>(err), deviceGetErrorstring(err));
      exit(EXIT_FAILURE);
    }
  }

  auto peekLastDeviceError(const char* errorMessage, const char* file,
                           const int line) -> void {
    auto err = GPUCXX_RUNTIME_BACKEND(PeekAtLastError)();
    if (err != deviceSuccess) {
      fprintf(stderr,
              "[%s Peek Error]\n"
              "  Location : %s:%d\n"
              "  Function : %s\n"
              "  Error    : code = %d (%s)\n",
              GPUCXX_RUNTIME_BACKEND_STR, file, line, errorMessage,
              static_cast<int>(err), deviceGetErrorstring(err));
    }
  }
}  // namespace details_

GPUCXX_END_NAMESPACE

#endif