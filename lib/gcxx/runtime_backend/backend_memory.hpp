// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_BACKEND_BACKEND_MEMORY_HPP_
#define GCXX_RUNTIME_BACKEND_BACKEND_MEMORY_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_handles.hpp>

GCXX_NAMESPACE_MAIN_DRIVER_BEGIN()

GCXX_FH auto deviceMalloc(std::size_t numBytes) -> void* {
  void* ptr{nullptr};
  GCXX_SAFE_RUNTIME_CALL(Malloc, "Failed to allocate device memory", &ptr,
                         numBytes);
  return ptr;
}

GCXX_FH auto deviceFree(void* ptr) -> void {
  GCXX_SAFE_RUNTIME_CALL(Free, "Failed to free device memory", ptr);
}

GCXX_FH auto deviceMallocAsync(std::size_t numBytes,
                               deviceStream_t stream) -> void* {
  void* ptr = nullptr;
  GCXX_SAFE_RUNTIME_CALL(MallocAsync,
                         "Failed to allocate device memory asynchronously",
                         &ptr, numBytes, stream);
  return ptr;
}

GCXX_FH auto deviceFreeAsync(void* ptr, deviceStream_t stream) -> void {
  GCXX_SAFE_RUNTIME_CALL(
    FreeAsync, "Failed to free device memory asynchronously", ptr, stream);
}

GCXX_FH auto deviceMallocManaged(std::size_t numBytes) -> void* {
  void* ptr = nullptr;
  GCXX_SAFE_RUNTIME_CALL(MallocManaged, "Failed to allocate managed memory",
                         &ptr, numBytes);
  return ptr;
}

GCXX_FH auto deviceMallocHost(std::size_t numBytes) -> void* {
  void* ptr = nullptr;
  GCXX_SAFE_RUNTIME_CALL(GCXX_DIRECT_BACKEND_ALT(MallocHost, HostMalloc),
                         "Failed to allocate Pinned host memory", &ptr,
                         numBytes);
  return ptr;
}

GCXX_FH auto deviceFreeHost(void* ptr) -> void {
  GCXX_SAFE_RUNTIME_CALL(GCXX_DIRECT_BACKEND_ALT(FreeHost, HostFree),
                         "Failed to free pinned host memory", ptr);
}

GCXX_NAMESPACE_MAIN_DRIVER_END()

#endif
