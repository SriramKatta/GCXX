#pragma once
#ifndef GCXX_RUNTIME_RUNTIME_ERROR_HPP_
#define GCXX_RUNTIME_RUNTIME_ERROR_HPP_

#include <gpucxx/runtime/error/runtime_error.hpp>

#define GCXX_CHECK_DEVICE_ERR(val)                                   \
  do {                                                                 \
    gcxx::details_::checkDeviceError((val), #val, __FILE__, __LINE__); \
  } while (0)

#define GCXX_CHECK_DEVICE_LASTERR(...)                                    \
  do {                                                                      \
    gcxx::details_::checkLastDeviceError(#__VA_ARGS__, __FILE__, __LINE__); \
  } while (0)

#define GCXX_PEEK_DEVICE_LASTERR(...)                                    \
  do {                                                                     \
    gcxx::details_::peekLastDeviceError(#__VA_ARGS__, __FILE__, __LINE__); \
  } while (0)

#define GCXX_SAFE_RUNTIME_CALL(func, Args) \
  GCXX_CHECK_DEVICE_ERR(GCXX_RUNTIME_BACKEND(func) Args)


#endif
