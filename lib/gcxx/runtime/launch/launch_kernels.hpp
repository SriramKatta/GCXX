#pragma once
#ifndef GCXX_RUNTIME_LAUNCH_LAUNCH_KERNELS_HPP
#define GCXX_RUNTIME_LAUNCH_LAUNCH_KERNELS_HPP

#include <gcxx/internal/prologue.hpp>


#include <gcxx/runtime/launch/launch_config.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

namespace launch {
  using gcxxHostFn_t = GCXX_RUNTIME_BACKEND(HostFn_t);

  template <typename... ExpTypes, typename... ActTypes>
  GCXX_FH void CooperativeKernel(LaunchConfig&, void (*)(ExpTypes...),
                                 ActTypes&&...);

  GCXX_FH void HostFunc(const StreamView, gcxxHostFn_t, void*);


  template <typename... ExpTypes, typename... ActTypes>
  GCXX_FH void Kernel(dim3 griddim, dim3 blockdim, void (*kernel)(ExpTypes...),
                      ActTypes&&... args);

  // TODO : add sfinae to check if the kernel is __global__
  template <typename... ExpTypes, typename... ActTypes>
  GCXX_FH void Kernel(StreamView stream, dim3 griddim, dim3 blockdim,
                      std::size_t smem_bytes, void (*kernel)(ExpTypes...),
                      ActTypes&&...);
}  // namespace launch

GCXX_NAMESPACE_MAIN_END

#include <gcxx/runtime/details/launch/launch_kernels.inl>

#endif