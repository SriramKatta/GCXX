#pragma once
#ifndef GCXX_RUNTIME_DETAILS_LAUNCH_LAUNCH_KERNELS_HPP
#define GCXX_RUNTIME_DETAILS_LAUNCH_LAUNCH_KERNELS_HPP

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>


#include <gcxx/runtime/launch/launch_config.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

namespace launch {

  template <typename... ExpTypes, typename... ActTypes>
  GCXX_FH void CooperativeKernel(LaunchConfig& config,
                                 void (*kernel)(ExpTypes...),
                                 ActTypes&&... args) {
    void* kernelArgs[sizeof...(args) > 0 ? sizeof...(args) : 1] = {
      ((void*)&args)...};
    // GCXX_SAFE_RUNTIME_CALL(LaunchCooperativeKernel,
    //                        "failed to launch Cooprative kernel", kernel,
    //                        config.)
  }

  GCXX_FH void HostFunc(const StreamView sview, gcxxHostFn_t fn,
                        void* userData) {
    GCXX_SAFE_RUNTIME_CALL(LaunchHostFunc, "Failed to launch hostfunc",
                           sview.getRawStream(), fn, userData);
  }

  // TODO : add sfinae to check if the kernel is __global__
  template <typename... ExpTypes, typename... ActTypes>
  GCXX_FH void Kernel(StreamView sv, dim3 griddim, dim3 blockdim,
                      std::size_t smem_bytes, void (*kernel)(ExpTypes...),
                      ActTypes&&... args) {

    std::array<void*, sizeof...(ActTypes)> kernelArgs{
      static_cast<void*>(&args)...};
    GCXX_SAFE_RUNTIME_CALL(LaunchKernel, "Failed to launch GPU kernel",
#if GCXX_HIP_MODE
                           (void*)
#endif
                             kernel,
                           griddim, blockdim, kernelArgs.data(), smem_bytes,
                           sv.getRawStream());
  }
}  // namespace launch

GCXX_NAMESPACE_MAIN_END


#include <gcxx/runtime/details/launch_kernels.inl>

#endif