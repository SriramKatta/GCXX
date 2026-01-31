#pragma once
#ifndef GCXX_RUNTIME_LAUNCH_LAUNCH_CONFIG_HPP
#define GCXX_RUNTIME_LAUNCH_LAUNCH_CONFIG_HPP

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN
using deviceLaunchConfig_t = GCXX_RUNTIME_BACKEND(LaunchConfig_t);
GCXX_NAMESPACE_MAIN_DETAILS_END


GCXX_NAMESPACE_MAIN_BEGIN

class LaunchConfig {
 private:
  //  cudaLaunchConfig_t
  using deviceLaunchConfig_t   = details_::deviceLaunchConfig_t;
  deviceLaunchConfig_t config_ = {0};

 public:
  LaunchConfig(dim3 griddim = {1, 1, 1}, dim3 blockdim = {1, 1, 1},
               std::size_t smemBytes = 0,
               const StreamView& sv  = StreamView::Null())
      : config_({griddim, blockdim, smemBytes, sv.getRawStream(), nullptr, 0}) {
  }

  void print() {}
};

GCXX_NAMESPACE_MAIN_END


#endif