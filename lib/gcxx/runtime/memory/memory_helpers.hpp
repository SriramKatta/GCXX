#pragma once
#ifndef GCXX_API_RUNTIME_MEMORY_MEMORY_HELPERS_HPP_
#define GCXX_API_RUNTIME_MEMORY_MEMORY_HELPERS_HPP_

#include <gcxx/internal/prologue.hpp>


GCXX_NAMESPACE_MAIN_BEGIN

namespace memory {
  using devicePitchedPtr = GCXX_RUNTIME_BACKEND(PitchedPtr);
  using devicePos        = GCXX_RUNTIME_BACKEND(Pos);
  using deviceExtent     = GCXX_RUNTIME_BACKEND(Extent);

  template <typename VT>
  GCXX_FH auto makePitchedPtr(void* dPtr, size_t pitchelems = 1,
                              size_t xSize = 1, size_t ySize = 1)
    -> devicePitchedPtr {
    return
#if GCXX_CUDA_MODE
      make_cudaPitchedPtr
#else
      make_hipPitchedPtr
#endif
      (dPtr, pitchelems * sizeof(VT), xSize, ySize);
  }

  GCXX_FH auto makePos(size_t x, size_t y, size_t z) -> devicePos {
    return
#if GCXX_CUDA_MODE
      make_cudaPos
#else
      make_hipPos
#endif
      (x, y, z);
  }

  template <typename VT>
  GCXX_FH auto makeExtent(size_t xSize = 1, size_t ySize = 1, size_t zSize = 1)
    -> deviceExtent {
    return
#if GCXX_CUDA_MODE
      make_cudaExtent
#else
      make_hipExtent
#endif
      (xSize * sizeof(VT), ySize, zSize);
  }


}  // namespace memory

GCXX_NAMESPACE_MAIN_END


#endif