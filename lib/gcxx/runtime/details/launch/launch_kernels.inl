#pragma once
#ifndef GCXX_RUNTIME_DETAILS_LAUNCH_LAUNCH_KERNELS_HPP
#define GCXX_RUNTIME_DETAILS_LAUNCH_LAUNCH_KERNELS_HPP

#include <utility>

#include <gcxx/internal/prologue.hpp>


#include <gcxx/runtime/launch/launch_config.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

namespace launch {

  template <typename... ExpTypes, typename... ActTypes>
  GCXX_FH void CooperativeKernel(LaunchConfig& config,
                                 void (*kernel)(ExpTypes...),
                                 ActTypes&&... args) {
    std::array<void*, sizeof...(args)> kernelArgs = {((void*)&args)...};
    // GCXX_SAFE_RUNTIME_CALL(LaunchCooperativeKernel,
    //                        "failed to launch Cooprative kernel", kernel,
    //                        config.)
  }

  GCXX_FH void HostFunc(const StreamView sview, gcxxHostFn_t fn,
                        void* userData) {
    GCXX_SAFE_RUNTIME_CALL(LaunchHostFunc, "Failed to launch hostfunc",
                           sview.getRawStream(), fn, userData);
  }

  template <typename... ExpTypes, typename... ActTypes>
  GCXX_FH void Kernel(dim3 griddim, dim3 blockdim, void (*kernel)(ExpTypes...),
                      ActTypes&&... args) {
    Kernel(StreamView::Null(), griddim, blockdim, 0, kernel,
           std::forward<ActTypes>(args)...);
  }

  // TODO : add sfinae to check if the kernel is __global__
  template <typename... ExpTypes, typename... ActTypes>
  GCXX_FH void Kernel(StreamView sv, dim3 griddim, dim3 blockdim,
                      std::size_t smem_bytes, void (*kernel)(ExpTypes...),
                      ActTypes&&... args) {
    GCXX_RUNTIME_BACKEND(LaunchConfig_t) config{};
    config.stream           = sv.getRawStream();
    config.blockDim         = blockdim;
    config.gridDim          = griddim;
    config.dynamicSmemBytes = smem_bytes;
    GCXX_SAFE_RUNTIME_CALL(LaunchKernelEx, "Failed to launch GPU kernel",
                           &config, kernel, std::forward<ActTypes>(args)...);
  }
}  // namespace launch

GCXX_NAMESPACE_MAIN_END

#endif