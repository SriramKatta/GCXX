// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_OCCUPANCY_OCCUPANCY_HPP_
#define GCXX_RUNTIME_OCCUPANCY_OCCUPANCY_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/flags/occupancy_flags.hpp>
#include <gcxx/runtime_backend/backend_occupancy.hpp>

#include <cstddef>
#include <utility>

GCXX_NAMESPACE_MAIN_BEGIN

namespace Occupancy {
#if GCXX_CUDA_MODE
  template <typename func_t>
  GCXX_FH auto AvailableDynamicSMemPerBlock(func_t* func, int numBlocks,
                                            int blockSize) -> std::size_t {
    return driver::occupancyAvailableDynamicSMemPerBlock(func, numBlocks,
                                                         blockSize);
  }
#endif

  template <typename func_t>
  GCXX_FH auto MaxActiveBlocksPerMultiprocessor(
    func_t func, int blockSize, std::size_t dynamicSMemSizeBytes = 0) -> int {
    return MaxActiveBlocksPerMultiprocessorWithFlags(
      func, blockSize, dynamicSMemSizeBytes, flags::occupancyType::Default);
  }

  template <typename func_t>
  GCXX_FH auto MaxActiveBlocksPerMultiprocessorWithFlags(
    func_t func, int blockSize, std::size_t dynamicSMemSizeBytes = 0,
    flags::occupancyType flag = flags::occupancyType::Default) -> int {
    return driver::occupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      func, blockSize, dynamicSMemSizeBytes,
      static_cast<details_::flag_t>(flag));
  }

#if GCXX_CUDA_MODE
  template <typename func_t>
  GCXX_FH auto MaxActiveClusters(
    func_t* func, const driver::deviceLaunchConfig_t* config) -> int {
    return driver::occupancyMaxActiveClusters(func, config);
  }
#endif

  template <typename func_t>
  GCXX_FH auto MaxPotentialBlockSize(
    func_t func, std::size_t dynamicSMemSize = 0,
    int blockSizeLimit = 0) -> std::pair<int, int> {
    return driver::occupancyMaxPotentialBlockSize(func, dynamicSMemSize,
                                                  blockSizeLimit);
  }

  template <typename UnaryFunction, typename func_t>
  GCXX_FH auto MaxPotentialBlockSizeVariableSMem(
    func_t func, UnaryFunction blockSizeToDynamicSMemSize,
    int blockSizeLimit = 0) -> std::pair<int, int> {
    return driver::occupancyMaxPotentialBlockSizeVariableSMem(
      func, blockSizeToDynamicSMemSize, blockSizeLimit);
  }

  template <typename UnaryFunction, typename func_t>
  GCXX_FH auto MaxPotentialBlockSizeVariableSMemWithFlags(
    func_t func, UnaryFunction blockSizeToDynamicSMemSize,
    int blockSizeLimit        = 0,
    flags::occupancyType flag = flags::occupancyType::Default)
    -> std::pair<int, int> {
    return driver::occupancyMaxPotentialBlockSizeVariableSMemWithFlags(
      func, blockSizeToDynamicSMemSize, blockSizeLimit,
      static_cast<details_::flag_t>(flag));
  }

  template <typename func_t>
  GCXX_FH auto MaxPotentialBlockSizeWithFlags(
    func_t func, std::size_t dynamicSMemSize = 0, int blockSizeLimit = 0,
    flags::occupancyType flag = flags::occupancyType::Default)
    -> std::pair<int, int> {
    return driver::occupancyMaxPotentialBlockSizeWithFlags(
      func, dynamicSMemSize, blockSizeLimit,
      static_cast<details_::flag_t>(flag));
  }

#if GCXX_CUDA_MODE
  template <typename func_t>
  GCXX_FH auto MaxPotentialClusterSize(
    func_t* func, const driver::deviceLaunchConfig_t* config) -> int {
    return driver::occupancyMaxPotentialClusterSize(func, config);
  }
#endif

}  // namespace Occupancy

GCXX_NAMESPACE_MAIN_END


#endif
