#pragma once
#ifndef GCXX_RUNTIME_ERROR_RUNTIME_ERROR_HPP_
#define GCXX_RUNTIME_ERROR_RUNTIME_ERROR_HPP_

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

using deviceError_t                 = GCXX_RUNTIME_BACKEND(Error_t);
GCXX_CXPR auto deviceErrSuccess     = GCXX_RUNTIME_BACKEND(Success);
GCXX_CXPR auto deviceErrNotReady    = GCXX_RUNTIME_BACKEND(ErrorNotReady);

// Custom exception class for GPU runtime errors
class gpu_runtime_error : public std::runtime_error {
 public:
  gpu_runtime_error(deviceError_t error_code, const char* user_msg)
      : std::runtime_error(build_error_message(error_code, user_msg)),
        error_code_(error_code) {}

  deviceError_t error_code() const noexcept { return error_code_; }

 private:
  deviceError_t error_code_;

  static std::string build_error_message(deviceError_t err, const char* msg) {
    const char* error_name = GCXX_RUNTIME_BACKEND(GetErrorName)(err);
    const char* error_string = GCXX_RUNTIME_BACKEND(GetErrorString)(err);
    
    std::string result = "GPU Runtime Error: ";
    result += msg;
    result += " [Error code: ";
    result += std::to_string(static_cast<int>(err));
    result += ", ";
    result += error_name ? error_name : "Unknown";
    result += ": ";
    result += error_string ? error_string : "No description available";
    result += "]";
    
    return result;
  }
};

// Exception-style throw for GPU errors
inline auto throwGPUError(deviceError_t err, const char* msg) -> void {
  throw gpu_runtime_error(err, msg);
}

GCXX_NAMESPACE_MAIN_DETAILS_END


#endif
