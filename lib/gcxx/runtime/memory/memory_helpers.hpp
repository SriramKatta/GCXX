#pragma once
#ifndef GCXX_API_RUNTIME_MEMORY_MEMORY_HELPERS_HPP_
#define GCXX_API_RUNTIME_MEMORY_MEMORY_HELPERS_HPP_

#include <gcxx/internal/prologue.hpp>


GCXX_NAMESPACE_MAIN_BEGIN

namespace memory {
  using devicePitchedPtr = GCXX_RUNTIME_BACKEND(PitchedPtr);
  using devicePos        = GCXX_RUNTIME_BACKEND(Pos);
  using deviceExtent     = GCXX_RUNTIME_BACKEND(Extent);

  GCXX_FH auto makePitchedPtr(void* d, size_t p, size_t xsz, size_t ysz)
    -> devicePitchedPtr {
#if GCXX_CUDA_MODE
    return make_cudaPitchedPtr(d, p, xsz, ysz);
#else
    return make_hipPitchedPtr(d, p, xsz, ysz);
#endif
  }

  GCXX_FH auto makePos(size_t x, size_t y, size_t z) -> devicePos {
#if GCXX_CUDA_MODE
    return make_cudaPos(x, y, z);
#else
    return make_hipPos(x, y, z);
#endif
  }

  GCXX_FH auto makeExtent(size_t w, size_t h, size_t d) -> deviceExtent {

#if GCXX_CUDA_MODE
    return make_cudaExtent(w, h, d);
#else
    return make_hipExtent(w, h, d);
#endif
  }


}  // namespace memory

GCXX_NAMESPACE_MAIN_END


#endif