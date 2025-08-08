#pragma once
#ifndef GPUCXX_API_DETAILS_RUNTIME_BACKEND_HPP
#define GPUCXX_API_DETAILS_RUNTIME_BACKEND_HPP

// clang-format off
#ifdef GPUCXX_CUDA_MODE
    #include <gpucxx/backend/cuda_backend.hpp>
#elif GPUCXX_HIP_MODE
    #include <gpucxx/backend/hip_backend.hpp>
#endif

#define STRINGIFY(x)                #x
#define TOSTRING(x)                 STRINGIFY(x)
#define STRINGIFY_AND_APPEND(a, b)  a##b
#define APPEND_NAME(a, b)           STRINGIFY_AND_APPEND(a, b)

#define        GPUCXX_RUNTIME_BACKEND(name)    APPEND_NAME(RUNTIME_BACKEND, name)
constexpr auto GPUCXX_RUNTIME_BACKEND_STR    = TOSTRING(RUNTIME_BACKEND);
// clang-format on

#endif