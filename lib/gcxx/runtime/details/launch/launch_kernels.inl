// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_LAUNCH_LAUNCH_KERNELS_HPP
#define GCXX_RUNTIME_DETAILS_LAUNCH_LAUNCH_KERNELS_HPP

#include <utility>

#include <gcxx/internal/prologue.hpp>


#include <gcxx/runtime/launch/launch_config.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>
#include <gcxx/runtime_backend/backend_launch.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

namespace launch {
  template <typename... ExpTypes, typename... ActTypes>
  GCXX_FH void CooperativeKernel(LaunchConfig& config,
                                 void (*kernel)(ExpTypes...),
                                 ActTypes&&... args) {
    std::array<void*, sizeof...(args)> kernelArgs = {((void*)&args)...};
    driver::launchCooperativeKernel(kernel, config.config_.gridDim,
                                    config.config_.blockDim, kernelArgs.data(),
                                    config.config_.dynamicSmemBytes,
                                    config.config_.stream);
  }

  GCXX_FH void HostFunc(const StreamView sview, gcxxHostCallBackFn_t fn,
                        void* userData) {
    driver::launchHostFunc(sview.getRawStream(), fn, userData);
  }

  // TODO : add sfinae to check if the kernel is __global__
  template <typename... ExpTypes, typename... ActTypes>
  GCXX_FH void Kernel(StreamView sv, dim3 griddim, dim3 blockdim,
                      std::size_t smem_bytes, void (*kernel)(ExpTypes...),
                      ActTypes&&... args) {
    driver::deviceLaunchConfig_t config{};
    config.stream           = sv.getRawStream();
    config.blockDim         = blockdim;
    config.gridDim          = griddim;
    config.dynamicSmemBytes = smem_bytes;
    driver::launchKernelEx(&config, kernel, std::forward<ActTypes>(args)...);
  }
}  // namespace launch

GCXX_NAMESPACE_MAIN_END()

#endif