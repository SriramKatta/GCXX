#pragma once
#ifndef GPUCXX_RUNTIME_ERROR_RUNTIME_ERROR_HPP_
#define GPUCXX_RUNTIME_ERROR_RUNTIME_ERROR_HPP_

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/macros/define_macros.hpp>

GPUCXX_DETAILS_BEGIN_NAMESPACE

using deviceError_t                   = GPUCXX_RUNTIME_BACKEND(Error_t);
GPUCXX_CXPR auto deviceSuccess        = GPUCXX_RUNTIME_BACKEND(Success);
GPUCXX_CXPR auto deviceGetErrorstring = GPUCXX_RUNTIME_BACKEND(GetErrorString);

GPUCXX_FH auto checkDeviceError(const deviceError_t result,
                                char const* const func, const char* const file,
                                const int line) -> void;

GPUCXX_FH auto checkLastDeviceError(const char* errorMessage, const char* file,
                                    const int line) -> void;

GPUCXX_FH auto peekLastDeviceError(const char* errorMessage, const char* file,
                                   const int line) -> void;

GPUCXX_DETAILS_END_NAMESPACE

#include <gpucxx/runtime/details/runtime_error.inl>

#endif
