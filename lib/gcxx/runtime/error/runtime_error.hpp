#pragma once
#ifndef GCXX_RUNTIME_ERROR_RUNTIME_ERROR_HPP_
#define GCXX_RUNTIME_ERROR_RUNTIME_ERROR_HPP_

#include <iostream>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/error/runtime_error_types.hpp>
#if defined(GCXX_WITH_EXCEPTIONS)
#include <gcxx/runtime/error/runtime_exception.hpp>
#endif

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

// TODO : Implement an exception style throw
inline auto throwGPUError(deviceError_t err, const char* msg) -> void {
#if defined(GCXX_WITH_EXCEPTIONS)
  throw gcxx::Exception(err, msg);
#else
  std::cerr << details_::make_message(err, msg) << "\n";
  std::abort();
#endif
}

GCXX_NAMESPACE_MAIN_DETAILS_END


#endif
