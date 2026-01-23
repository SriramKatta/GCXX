#pragma once
#ifndef GCXX_RUNTIME_OCCUPANCY_OCCUPANCY_HPP_
#define GCXX_RUNTIME_OCCUPANCY_OCCUPANCY_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/flags/occupancy_flags.hpp>

GCXX_NAMESPACE_MAIN_BEGIN

namespace Occupancy {
  // TODO : needs to verify if func_t is actually an cuda kernel func type
  template <typename func_t>
  GCXX_FH auto AvailableDynamicSMemPerBlock(func_t func, int numBlocks,
                                            int blockSize) -> std::size_t {
    std::size_t smemsize{};
    cudaOccupancyAvailableDynamicSMemPerBlock GCXX_SAFE_RUNTIME_CALL(
      OccupancyAvailableDynamicSMemPerBlock,
      "Failed to query Avalible dynamic smem for given blocks and grid size",
      &smemsize, &func, numBlocks, blockSize);
    return smemsize;
  }

  // TODO : needs to verify if func_t is actually an cuda kernel func type
  template <typename func_t>
  GCXX_FH auto MaxActiveBlocksPerMultiprocessor(
    func_t* func, int blockSize, std::size_t dynamicSMemSizeBytes = 0,
    flags::occupancyType flag = flags::occupancyType::Default) -> int {
    int numBlocks{};
    GCXX_SAFE_RUNTIME_CALL(
      OccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
      "Failed to query Avalible numblocks for best occupancy", &numBlocks, func,
      blockSize, dynamicSMemSizeBytes, static_cast<details_::flag_t>(flag));
    return numBlocks;
  }

}  // namespace Occupancy

GCXX_NAMESPACE_MAIN_END


#endif