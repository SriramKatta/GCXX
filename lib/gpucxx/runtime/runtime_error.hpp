#pragma once
#ifndef GPUCXX_RUNTIME_RUNTIME_ERROR_HPP_
#define GPUCXX_RUNTIME_RUNTIME_ERROR_HPP_

#include <gpucxx/runtime/error/runtime_error.hpp>

#define GPUCXX_CHECK_DEVICE_ERR(val)                                   \
  do {                                                                 \
    gcxx::details_::checkDeviceError((val), #val, __FILE__, __LINE__); \
  } while (0)

#define GPUCXX_CHECK_DEVICE_LASTERR(...)                                    \
  do {                                                                      \
    gcxx::details_::checkLastDeviceError(#__VA_ARGS__, __FILE__, __LINE__); \
  } while (0)

#define GPUCXX_PEEK_DEVICE_LASTERR(...)                                    \
  do {                                                                     \
    gcxx::details_::peekLastDeviceError(#__VA_ARGS__, __FILE__, __LINE__); \
  } while (0)

#define GPUCXX_SAFE_RUNTIME_CALL(func, Args) \
  GPUCXX_CHECK_DEVICE_ERR(GPUCXX_RUNTIME_BACKEND(func) Args)


#endif
