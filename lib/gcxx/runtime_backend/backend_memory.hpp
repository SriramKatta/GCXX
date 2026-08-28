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

// Failed probes clear the sticky last-error and return false.
GCXX_FH auto isDeviceOrManagedMemory(const void* ptr) -> bool {
  if (ptr == nullptr) {
    return true;
  }
  devicePointerAttributes_t attrs{};
  const deviceError_t err =
    ::GCXX_RUNTIME_BACKEND(PointerGetAttributes)(&attrs, ptr);
  if (err != deviceErrSuccess) {
    (void)GetLastError();  // consume the recorded error; the probe is
                           // expected to fail for unregistered memory
    return false;
  }
  // Both backends spell the field `type`; managed memory maps to
  // cudaMemoryTypeManaged / hipMemoryTypeManaged (hipMemoryTypeUnified is an
  // AMD-specific unified-address-space concept, not managed memory).
  return attrs.type == GCXX_RUNTIME_BACKEND(MemoryTypeDevice) ||
         attrs.type == GCXX_RUNTIME_BACKEND(MemoryTypeManaged);
}

// Also accepts pinned host memory that has a device (UVA) mapping.
GCXX_FH auto isDeviceUsableMemory(const void* ptr) -> bool {
  if (ptr == nullptr) {
    return true;
  }
  devicePointerAttributes_t attrs{};
  const deviceError_t err =
    ::GCXX_RUNTIME_BACKEND(PointerGetAttributes)(&attrs, ptr);
  if (err != deviceErrSuccess) {
    (void)GetLastError();  // consume the recorded error; the probe is
                           // expected to fail for unregistered memory
    return false;
  }
  if (attrs.type == GCXX_RUNTIME_BACKEND(MemoryTypeDevice) ||
      attrs.type == GCXX_RUNTIME_BACKEND(MemoryTypeManaged)) {
    return true;
  }
  return attrs.type == GCXX_RUNTIME_BACKEND(MemoryTypeHost) &&
         attrs.devicePointer != nullptr;
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
