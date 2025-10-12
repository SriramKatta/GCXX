#pragma once
#ifndef GCXX_RUNTIME_ERROR_RUNTIME_ERROR_HPP_
#define GCXX_RUNTIME_ERROR_RUNTIME_ERROR_HPP_

#include <stdio.h>
#include <cstdlib>
#include <exception>

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

using deviceError_t                 = GCXX_RUNTIME_BACKEND(Error_t);
GCXX_CXPR auto deviceSuccess        = GCXX_RUNTIME_BACKEND(Success);
GCXX_CXPR auto deviceErrorNotReady  = GCXX_RUNTIME_BACKEND(ErrorNotReady);
GCXX_CXPR auto deviceGetErrorstring = GCXX_RUNTIME_BACKEND(GetErrorString);
GCXX_CXPR auto deviceGetLastError   = GCXX_RUNTIME_BACKEND(GetLastError);

// TODO : Implement an exception style throw
inline auto throwGPUError(deviceError_t err, const char* msg) -> void {
  fprintf(stderr, "code  %d : %s\n", err, msg);
  std::abort();
}

GCXX_NAMESPACE_MAIN_DETAILS_END

// #include <gcxx/runtime/details/runtime_error.inl>

#endif
