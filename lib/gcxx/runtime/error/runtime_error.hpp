#pragma once
#ifndef GCXX_RUNTIME_ERROR_RUNTIME_ERROR_HPP_
#define GCXX_RUNTIME_ERROR_RUNTIME_ERROR_HPP_

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <exception>

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

using deviceError_t                 = GCXX_RUNTIME_BACKEND(Error_t);
GCXX_CXPR auto deviceErrSuccess     = GCXX_RUNTIME_BACKEND(Success);
GCXX_CXPR auto deviceErrNotReady    = GCXX_RUNTIME_BACKEND(ErrorNotReady);

// TODO : Implement an exception style throw
inline auto throwGPUError(deviceError_t err, const char* msg) -> void {
  std::cerr << "code " << err << " : " << msg << "\n";
  std::abort();
}

GCXX_NAMESPACE_MAIN_DETAILS_END

#endif
