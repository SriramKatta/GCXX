// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BACKEND_BACKEND_HPP_
#define GCXX_BACKEND_BACKEND_HPP_


#include <gcxx/macros/backend_mode_macros.hpp>


#if GCXX_CUDA_MODE()
#include <gcxx/backend/cuda_backend.hpp>
#elif GCXX_HIP_MODE()
#include <gcxx/backend/hip_backend.hpp>
#else
#error "One backend GCXX_CUDA_MODE or GCXX_HIP_MODE must be defined"
#endif

#if __cplusplus < 201703
#error "GCXX library needs atleast c++17 to compile"
#endif


#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
constexpr auto GCXX_RUNTIME_BACKEND_STR = TOSTRING(RUNTIME_BACKEND);


#define STRINGIFY_AND_APPEND(a, b) a##b
#define APPEND_NAME(a, b) STRINGIFY_AND_APPEND(a, b)
#define GCXX_RUNTIME_BACKEND(name) APPEND_NAME(RUNTIME_BACKEND, name)
#define GCXX_ATTRIBUTE_BACKEND(name) APPEND_NAME(ATTRIBUTE_BACKEND, name)

// Macro to handle Backend handles with different names between CUDA and HIP.
#if GCXX_CUDA_MODE()
#define GCXX_DIRECT_BACKEND_ALT(cuda_name, hip_name) cuda_name
#elif GCXX_HIP_MODE()
#define GCXX_DIRECT_BACKEND_ALT(cuda_name, hip_name) hip_name
#endif

// Macro to handle backend handles named differently, still cuda/hip-prefixed.
#if GCXX_CUDA_MODE()
#define GCXX_RUNTIME_BACKEND_ALT(cuda_name, hip_name) \
  GCXX_RUNTIME_BACKEND(cuda_name)
#elif GCXX_HIP_MODE()
#define GCXX_RUNTIME_BACKEND_ALT(cuda_name, hip_name) \
  GCXX_RUNTIME_BACKEND(hip_name)
#endif

// Macro to handle attributes named differently, still cuda/hip-prefixed.
#if GCXX_CUDA_MODE()
#define GCXX_ATTRIBUTE_BACKEND_ALT(cuda_name, hip_name) \
  GCXX_ATTRIBUTE_BACKEND(cuda_name)
#elif GCXX_HIP_MODE()
#define GCXX_ATTRIBUTE_BACKEND_ALT(cuda_name, hip_name) \
  GCXX_ATTRIBUTE_BACKEND(hip_name)
#endif

#endif