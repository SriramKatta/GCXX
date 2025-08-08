#pragma once
#ifndef GPUCXX_API_RUNTIME_ERROR_HPP
#define GPUCXX_API_RUNTIME_ERROR_HPP

#include <gpucxx/backend/backend.hpp>
#include <gpucxx/utils/define_specifiers.hpp>

GPUCXX_BEGIN_NAMESPACE

namespace details_ {
  using deviceError_t            = GPUCXX_RUNTIME_BACKEND(Error_t);
  GPUCXX_CA deviceSuccess        = GPUCXX_RUNTIME_BACKEND(Success);
  GPUCXX_CA deviceGetErrorstring = GPUCXX_RUNTIME_BACKEND(GetErrorString);

  GPUCXX_FH auto checkDeviceError(const deviceError_t result,
                                  char const *const func,
                                  const char *const file, const int line)
    -> void;

  GPUCXX_FH auto checkLastDeviceError(const char *errorMessage,
                                      const char *file, const int line) -> void;

  GPUCXX_FH auto peekLastDeviceError(const char *errorMessage, const char *file,
                                     const int line) -> void;
}  // namespace details_

GPUCXX_END_NAMESPACE

#include <gpucxx/details/runtime/runtime_error.inl>


// macros for simpler library implementaion

#define GPUCXX_CHECK_DEVICE_ERR(val)                                     \
  do {                                                                   \
    gpuCXX::details_::checkDeviceError((val), #val, __FILE__, __LINE__); \
  } while (0)

#define GPUCXX_CHECK_DEVICE_LASTERR(...)                                      \
  do {                                                                        \
    gpuCXX::details_::checkLastDeviceError(#__VA_ARGS__, __FILE__, __LINE__); \
  } while (0)

#define GPUCXX_PEEK_DEVICE_LASTERR(...)                                      \
  do {                                                                       \
    gpuCXX::details_::peekLastDeviceError(#__VA_ARGS__, __FILE__, __LINE__); \
  } while (0)

#define GPUCXX_SAFE_RUNTIME_CALL(func, Args) \
  GPUCXX_CHECK_DEVICE_ERR(GPUCXX_RUNTIME_BACKEND(func) Args)


#endif
