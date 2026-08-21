// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_LAUNCH_LAUNCH_KERNELS_HPP
#define GCXX_RUNTIME_LAUNCH_LAUNCH_KERNELS_HPP

#include <gcxx/internal/prologue.hpp>


#include <gcxx/runtime/launch/launch_config.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>
#include <gcxx/runtime_backend/backend_launch.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

namespace launch {
  using gcxxHostCallBackFn_t = driver::deviceHostCallBackFn_t;

  template <typename... ExpTypes, typename... ActTypes>
  GCXX_FH void CooperativeKernel(LaunchConfig&, void (*)(ExpTypes...),
                                 ActTypes&&...);

  GCXX_FH void HostFunc(const StreamView, gcxxHostCallBackFn_t, void*);

  // TODO: Add sfinae to check if the kernel is __global__.
  template <typename... ExpTypes, typename... ActTypes>
  GCXX_FH void Kernel(StreamView stream, dim3 griddim, dim3 blockdim,
                      std::size_t smem_bytes, void (*kernel)(ExpTypes...),
                      ActTypes&&...);
}  // namespace launch

GCXX_NAMESPACE_MAIN_END()

#include <gcxx/runtime/details/launch/launch_kernels.inl>

#endif