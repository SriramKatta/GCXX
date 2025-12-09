#pragma once
#ifndef GCXX_RUNTIME_ERROR_RUNTIME_ERROR_HPP_
#define GCXX_RUNTIME_ERROR_RUNTIME_ERROR_HPP_

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <exception>
#include <string>
#include <sstream>
#include <stdexcept>

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

using deviceError_t                 = GCXX_RUNTIME_BACKEND(Error_t);
GCXX_CXPR auto deviceErrSuccess     = GCXX_RUNTIME_BACKEND(Success);
GCXX_CXPR auto deviceErrNotReady    = GCXX_RUNTIME_BACKEND(ErrorNotReady);

// Custom GPU runtime exception class
class GPURuntimeError : public std::runtime_error {
private:
  deviceError_t error_code_;
  
public:
  GPURuntimeError(deviceError_t err, const char* msg) 
    : std::runtime_error(formatMessage(err, msg ? msg : "Unknown error")),
      error_code_(err) {
    // Note: msg is null-checked in the initializer list
  }
  
  deviceError_t getErrorCode() const noexcept {
    return error_code_;
  }
  
private:
  static std::string formatMessage(deviceError_t err, const char* msg) {
    std::ostringstream oss;
    const char* error_name = GCXX_RUNTIME_BACKEND(GetErrorName)(err);
    const char* error_string = GCXX_RUNTIME_BACKEND(GetErrorString)(err);
    
    oss << "GPU Runtime Error: " << msg << "\n"
        << "  Error Code: " << static_cast<int>(err) << "\n"
        << "  Error Name: " << (error_name ? error_name : "Unknown") << "\n"
        << "  Error Description: " << (error_string ? error_string : "Unknown");
    
    return oss.str();
  }
};

// Throw GPU runtime error as exception
inline auto throwGPUError(deviceError_t err, const char* msg) -> void {
  throw GPURuntimeError(err, msg);
}

GCXX_NAMESPACE_MAIN_DETAILS_END

#endif
