#pragma once
#ifndef GCXX_RUNTIME_DETAILS_RUNTIME_ERROR_INL_
#define GCXX_RUNTIME_DETAILS_RUNTIME_ERROR_INL_


#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

#include <stdio.h>
#include <cstdlib>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

auto checkDeviceError(const deviceError_t result, char const* const func,
                      const char* const file, const int line) -> void {
  if (result != deviceErrSuccess) {
    fprintf(stderr,
            "[%5s Runtime Error]\n"
            "  Location : %s:%4d\n"
            "  Function : %s\n"
            "  Error    : code = %4d (%s)\n",
            GCXX_RUNTIME_BACKEND_STR, file, line, func,
            static_cast<unsigned int>(result), deviceGetErrorstring(result));
    exit(EXIT_FAILURE);
  }
}

auto checkLastDeviceError(const char* errorMessage, const char* file,
                          const int line) -> void {
  auto err = GCXX_RUNTIME_BACKEND(GetLastError)();
  if (err != deviceErrSuccess) {
    fprintf(stderr,
            "[%5s Last Error]\n"
            "  Location : %s:%4d\n"
            "  Function : %s\n"
            "  Error    : code = %4d (%s)\n",
            GCXX_RUNTIME_BACKEND_STR, file, line, errorMessage,
            static_cast<int>(err), deviceGetErrorstring(err));
    exit(EXIT_FAILURE);
  }
}

auto peekLastDeviceError(const char* errorMessage, const char* file,
                         const int line) -> void {
  auto err = GCXX_RUNTIME_BACKEND(PeekAtLastError)();
  if (err != deviceErrSuccess) {
    fprintf(stderr,
            "[%5s Peek Last Error]\n"
            "  Location : %s:%4d\n"
            "  Function : %s\n"
            "  Error    : code = %4d (%s)\n",
            GCXX_RUNTIME_BACKEND_STR, file, line, errorMessage,
            static_cast<int>(err), deviceGetErrorstring(err));
  }
}

GCXX_NAMESPACE_MAIN_DETAILS_END

#endif