#pragma once
#ifndef GCXX_RUNTIME_ERROR_RUNTIME_ERROR_HPP_
#define GCXX_RUNTIME_ERROR_RUNTIME_ERROR_HPP_

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

using deviceError_t                 = GCXX_RUNTIME_BACKEND(Error_t);
GCXX_CXPR auto deviceSuccess        = GCXX_RUNTIME_BACKEND(Success);
GCXX_CXPR auto deviceGetErrorstring = GCXX_RUNTIME_BACKEND(GetErrorString);

GCXX_FH auto checkDeviceError(const deviceError_t result,
                              char const* const func, const char* const file,
                              const int line) -> void;

GCXX_FH auto checkLastDeviceError(const char* errorMessage, const char* file,
                                  const int line) -> void;

GCXX_FH auto peekLastDeviceError(const char* errorMessage, const char* file,
                                 const int line) -> void;

GCXX_NAMESPACE_MAIN_DETAILS_END

#include <gcxx/runtime/details/runtime_error.inl>

#endif
