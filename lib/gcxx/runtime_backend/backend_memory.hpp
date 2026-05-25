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

GCXX_FH auto deviceMemset(void* dev_ptr, const int value,
                          const std::size_t countinBytes) -> void {
  GCXX_SAFE_RUNTIME_CALL(Memset, "Failed to perform GPU memset", dev_ptr, value,
                         countinBytes);
}

GCXX_FH auto deviceMemsetAsync(void* dev_ptr, const int value,
                               const std::size_t countinBytes,
                               deviceStream_t stream) -> void {
  GCXX_SAFE_RUNTIME_CALL(MemsetAsync, "Failed to perform Async GPU memset",
                         dev_ptr, value, countinBytes, stream);
}

GCXX_FH auto deviceCopy(void* destination, const void* source,
                        const std::size_t countinBytes) -> void {
  GCXX_SAFE_RUNTIME_CALL(Memcpy, "Failed to perform GPU copy", destination,
                         source, countinBytes,
                         GCXX_RUNTIME_BACKEND(MemcpyDefault));
}

GCXX_FH auto deviceCopyAsync(void* destination, const void* source,
                             const std::size_t countinBytes,
                             deviceStream_t stream) -> void {
  GCXX_SAFE_RUNTIME_CALL(MemcpyAsync, "Failed to perform async GPU copy",
                         destination, source, countinBytes,
                         GCXX_RUNTIME_BACKEND(MemcpyDefault), stream);
}

GCXX_NAMESPACE_MAIN_DRIVER_END()

#endif
