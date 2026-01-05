#pragma once
#ifndef GCXX_RUNTIME_ERROR_GCXX_EXCEPTION_HPP_
#define GCXX_RUNTIME_ERROR_GCXX_EXCEPTION_HPP_

#include <exception>
#include <string_view>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/error/runtime_error_types.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

class Exception : public std::runtime_error {
 public:
  Exception(details_::deviceError_t err, std::string_view context)
      : std::runtime_error(details_::make_message(err, context)), error_(err) {}

  details_::deviceError_t error() const noexcept { return error_; }
  private:
    details_::deviceError_t error_;
};

GCXX_NAMESPACE_MAIN_END


#endif
