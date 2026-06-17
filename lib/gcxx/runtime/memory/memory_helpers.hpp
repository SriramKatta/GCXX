// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_API_RUNTIME_MEMORY_MEMORY_HELPERS_HPP_
#define GCXX_API_RUNTIME_MEMORY_MEMORY_HELPERS_HPP_

#include <gcxx/internal/prologue.hpp>


GCXX_NAMESPACE_MAIN_BEGIN()

namespace memory {
  using devicePitchedPtr = driver::devicePitchedPtr;
  using devicePos        = driver::devicePos;
  using deviceExtent     = driver::deviceExtent;

  template <typename VT>
  GCXX_FH auto makePitchedPtr(void* dPtr, size_t pitchelems = 1,
                              size_t xSize = 1,
                              size_t ySize = 1) -> devicePitchedPtr {
    return GCXX_DIRECT_BACKEND_ALT(make_cudaPitchedPtr, make_hipPitchedPtr)(
      dPtr, pitchelems * sizeof(VT), xSize, ySize);
  }

  GCXX_FH auto makePos(size_t x, size_t y, size_t z) -> devicePos {
    return GCXX_DIRECT_BACKEND_ALT(make_cudaPos, make_hipPos)(x, y, z);
  }

  template <typename VT>
  GCXX_FH auto makeExtent(size_t xSize = 1, size_t ySize = 1,
                          size_t zSize = 1) -> deviceExtent {
    return GCXX_DIRECT_BACKEND_ALT(make_cudaExtent, make_hipExtent)(
      xSize * sizeof(VT), ySize, zSize);
  }
}  // namespace memory

GCXX_NAMESPACE_MAIN_END()


#endif