#pragma once
#ifndef GCXX_BACKEND_BACKEND_HPP_
#define GCXX_BACKEND_BACKEND_HPP_


#ifdef GCXX_CUDA_MODE
#include <gcxx/backend/cuda_backend.hpp>
#elif GCXX_HIP_MODE
#include <gcxx/backend/hip_backend.hpp>
#else
#error "Atleast one backend must be defined"
#endif


#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
constexpr auto GCXX_RUNTIME_BACKEND_STR = TOSTRING(RUNTIME_BACKEND);


#define STRINGIFY_AND_APPEND(a, b) a##b
#define APPEND_NAME(a, b) STRINGIFY_AND_APPEND(a, b)
#define GCXX_RUNTIME_BACKEND(name) APPEND_NAME(RUNTIME_BACKEND, name)

#endif