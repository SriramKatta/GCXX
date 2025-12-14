#pragma once
#ifndef GCXX_RUNTIME_LAUNCH_LAUNCH_CONFIG_HPP
#define GCXX_RUNTIME_LAUNCH_LAUNCH_CONFIG_HPP

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

class LaunchConfig {
 public:
 private:
  dim3 gridDim{1, 1, 1};
  dim3 blockDim{1, 1, 1};
  std::size_t smem_bytes{0};
  StreamView stream;
};

GCXX_NAMESPACE_MAIN_END

#endif