// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_LAUNCH_LAUNCH_CONFIG_HPP
#define GCXX_RUNTIME_LAUNCH_LAUNCH_CONFIG_HPP

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/stream/stream_view.hpp>
#include <gcxx/runtime_backend/backend_launch.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class LaunchConfig {
 public:
  using deviceLaunchConfig_t = driver::deviceLaunchConfig_t;
  LaunchConfig(dim3 griddim = {1, 1, 1}, dim3 blockdim = {1, 1, 1},
               std::size_t smemBytes = 0,
               const StreamView& sv  = StreamView::Null())
      : config_({griddim, blockdim, smemBytes, sv.getRawStream(), nullptr, 0}) {
  }

  void print() {}

 private:
  deviceLaunchConfig_t config_ = {0};
};

GCXX_NAMESPACE_MAIN_END()


#endif