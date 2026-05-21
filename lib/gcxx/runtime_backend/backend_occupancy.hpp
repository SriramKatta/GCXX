// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_BACKEND_BACKEND_OCCUPANCY_HPP_
#define GCXX_RUNTIME_BACKEND_BACKEND_OCCUPANCY_HPP_

#include <gcxx/internal/prologue.hpp>

#include <cstddef>
#include <utility>

#include <gcxx/runtime_backend/backend_handles.hpp>

GCXX_NAMESPACE_MAIN_DRIVER_BEGIN

#if GCXX_CUDA_MODE()
template <class T>
GCXX_FH auto occupancyAvailableDynamicSMemPerBlock(
  T* func, int numBlocks, int blockSize) -> std::size_t {
  std::size_t dynamicSmemSize{};
  GCXX_SAFE_RUNTIME_CALL(
    OccupancyAvailableDynamicSMemPerBlock,
    "Failed to query available dynamic shared memory per block",
    &dynamicSmemSize, func, numBlocks, blockSize);
  return dynamicSmemSize;
}
#endif
template <class T>
GCXX_FH auto occupancyMaxActiveBlocksPerMultiprocessorWithFlags(
  T func, int blockSize, std::size_t dynamicSMemSize,
  unsigned int flags) -> int {
  int numBlocks{};
  GCXX_SAFE_RUNTIME_CALL(OccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
                         "Failed to query max active blocks per multiprocessor",
                         &numBlocks, func, blockSize, dynamicSMemSize, flags);
  return numBlocks;
}
#if GCXX_CUDA_MODE()
template <class T>
GCXX_FH auto occupancyMaxActiveClusters(
  T* func, const deviceLaunchConfig_t* config) -> int {
  int numClusters{};
  GCXX_SAFE_RUNTIME_CALL(OccupancyMaxActiveClusters,
                         "Failed to query max active clusters", &numClusters,
                         func, config);
  return numClusters;
}
#endif
template <class T>
GCXX_FH auto occupancyMaxPotentialBlockSize(
  T func, std::size_t dynamicSMemSize = 0,
  int blockSizeLimit = 0) -> std::pair<int, int> {
  int minGridSize{};
  int blockSize{};
  GCXX_SAFE_RUNTIME_CALL(
    OccupancyMaxPotentialBlockSize, "Failed to query max potential block size",
    &minGridSize, &blockSize, func, dynamicSMemSize, blockSizeLimit);
  return {minGridSize, blockSize};
}

template <typename UnaryFunction, class T>
GCXX_FH auto occupancyMaxPotentialBlockSizeVariableSMem(
  T func, UnaryFunction blockSizeToDynamicSMemSize,
  int blockSizeLimit = 0) -> std::pair<int, int> {
  int minGridSize{};
  int blockSize{};
  GCXX_SAFE_RUNTIME_CALL(OccupancyMaxPotentialBlockSizeVariableSMem,
                         "Failed to query max potential block size",
                         &minGridSize, &blockSize, func,
                         blockSizeToDynamicSMemSize, blockSizeLimit);
  return {minGridSize, blockSize};
}

template <typename UnaryFunction, class T>
GCXX_FH auto occupancyMaxPotentialBlockSizeVariableSMemWithFlags(
  T func, UnaryFunction blockSizeToDynamicSMemSize, int blockSizeLimit = 0,
  unsigned int flags = 0) -> std::pair<int, int> {
  int minGridSize{};
  int blockSize{};
  GCXX_SAFE_RUNTIME_CALL(OccupancyMaxPotentialBlockSizeVariableSMemWithFlags,
                         "Failed to query max potential block size",
                         &minGridSize, &blockSize, func,
                         blockSizeToDynamicSMemSize, blockSizeLimit, flags);
  return {minGridSize, blockSize};
}

template <class T>
GCXX_FH auto occupancyMaxPotentialBlockSizeWithFlags(
  T func, std::size_t dynamicSMemSize = 0, int blockSizeLimit = 0,
  unsigned int flags = 0) -> std::pair<int, int> {
  int minGridSize{};
  int blockSize{};
  GCXX_SAFE_RUNTIME_CALL(OccupancyMaxPotentialBlockSizeWithFlags,
                         "Failed to query max potential block size",
                         &minGridSize, &blockSize, func, dynamicSMemSize,
                         blockSizeLimit, flags);
  return {minGridSize, blockSize};
}
#if GCXX_CUDA_MODE()
template <class T>
GCXX_FH auto occupancyMaxPotentialClusterSize(
  T* func, const deviceLaunchConfig_t* config) -> int {
  int clusterSize{};
  GCXX_SAFE_RUNTIME_CALL(OccupancyMaxPotentialClusterSize,
                         "Failed to query max potential cluster size",
                         &clusterSize, func, config);
  return clusterSize;
}
#endif

GCXX_NAMESPACE_MAIN_DRIVER_END

#endif
