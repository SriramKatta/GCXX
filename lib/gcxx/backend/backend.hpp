#pragma once
#ifndef GCXX_BACKEND_BACKEND_HPP_
#define GCXX_BACKEND_BACKEND_HPP_


#ifdef GCXX_CUDA_MODE
#include <gcxx/backend/cuda_backend.hpp>
#elif GCXX_HIP_MODE
#include <gcxx/backend/hip_backend.hpp>
#else
#error "One backend GCXX_CUDA_MODE or GCXX_HIP_MODE must be defined"
#endif


#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
constexpr auto GCXX_RUNTIME_BACKEND_STR = TOSTRING(RUNTIME_BACKEND);


#define STRINGIFY_AND_APPEND(a, b) a##b
#define APPEND_NAME(a, b) STRINGIFY_AND_APPEND(a, b)
#define GCXX_RUNTIME_BACKEND(name) APPEND_NAME(RUNTIME_BACKEND, name)
#define GCXX_ATTRIBUTE_BACKEND(name) APPEND_NAME(ATTRIBUTE_BACKEND, name)

// Macro to handle attributes with different names between CUDA and HIP
#if GCXX_CUDA_MODE
#define GCXX_ATTRIBUTE_BACKEND_ALT(cuda_name, hip_name) \
  GCXX_ATTRIBUTE_BACKEND(cuda_name)
#elif GCXX_HIP_MODE
#define GCXX_ATTRIBUTE_BACKEND_ALT(cuda_name, hip_name) \
  GCXX_ATTRIBUTE_BACKEND(hip_name)
#endif

#endif