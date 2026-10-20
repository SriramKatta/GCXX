// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_BACKEND_BACKEND_LAUNCH_HPP_
#define GCXX_RUNTIME_BACKEND_BACKEND_LAUNCH_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_handles.hpp>

GCXX_NAMESPACE_MAIN_DRIVER_BEGIN()

GCXX_FH auto launchHostFunc(deviceStream_t stream, deviceHostCallBackFn_t fn,
                            void* userData) -> void {
  GCXX_SAFE_RUNTIME_CALL(LaunchHostFunc, "Failed to launch hostfunc", stream,
                         fn, userData);
}

template <typename func_t>
GCXX_FH auto launchCooperativeKernel(func_t* func, dim3 gridDim, dim3 blockDim,
                                     void** args, size_t sharedMem = 0,
                                     deviceStream_t stream = nullptr) -> void {
  GCXX_SAFE_RUNTIME_CALL(LaunchCooperativeKernel,
                         "Failed to launch Cooprative GPU kernel", func,
                         gridDim, blockDim, args, sharedMem, stream);
}

template <typename func_t>
GCXX_FH auto launchKernel(func_t* func, dim3 gridDim, dim3 blockDim,
                          void** args, size_t sharedMem = 0,
                          deviceStream_t stream = nullptr) -> void {
  GCXX_SAFE_RUNTIME_CALL(LaunchKernel, "Failed to launch Cooprative GPU kernel",
                         func, gridDim, blockDim, args, sharedMem, stream);
}

template <typename... ExpTypes, typename... ActTypes>
GCXX_FH auto launchKernelEx(const deviceLaunchConfig_t* config,
                            void (*kernel)(ExpTypes...),
                            ActTypes&&... args) -> void {
  GCXX_SAFE_RUNTIME_CALL(LaunchKernelEx,
                         "Failed to launch Cooprative GPU kernel", config,
                         kernel, std::forward<ActTypes>(args)...);
}

GCXX_NAMESPACE_MAIN_DRIVER_END()


#endif
