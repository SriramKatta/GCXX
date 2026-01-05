#pragma once
#ifndef GCXX_RUNTIME_ERROR_RUNTIME_ERROR_TYPES_HPP_
#define GCXX_RUNTIME_ERROR_RUNTIME_ERROR_TYPES_HPP_

#include <string>
#include <string_view>

#include <gcxx/internal/prologue.hpp>


GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

using deviceError_t              = GCXX_RUNTIME_BACKEND(Error_t);
GCXX_CXPR auto deviceErrSuccess  = GCXX_RUNTIME_BACKEND(Success);
GCXX_CXPR auto deviceErrNotReady = GCXX_RUNTIME_BACKEND(ErrorNotReady);

GCXX_FH std::string_view GetErrorName(deviceError_t err) {
  return GCXX_RUNTIME_BACKEND(GetErrorName)(err);
}

GCXX_FH std::string_view GetErrorString(deviceError_t err) {
  return GCXX_RUNTIME_BACKEND(GetErrorString)(err);
}

GCXX_FH std::string make_message(deviceError_t err, std::string_view context) {
  std::string msg;
  msg.reserve(context.size() + 128);

  msg.append(context);
  msg.append(": ");
  msg.append(GetErrorName(err));
  msg.append(" (");
  msg.append(GetErrorString(err));
  msg.append(")");

  return msg;
}

GCXX_NAMESPACE_MAIN_DETAILS_END


#endif
