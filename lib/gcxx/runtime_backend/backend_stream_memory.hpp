// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_BACKEND_BACKEND_STREAM_MEMORY_HPP_
#define GCXX_RUNTIME_BACKEND_BACKEND_STREAM_MEMORY_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_handles.hpp>

GCXX_NAMESPACE_MAIN_DRIVER_BEGIN()

GCXX_FH auto deviceMallocAsync(std::size_t numBytes, deviceStream_t stream)
  -> void* {
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

GCXX_FH auto deviceMallocFromPoolAsync(std::size_t numBytes,
                                       deviceMemPool_t pool,
                                       deviceStream_t stream) -> void* {
  void* ptr = nullptr;
  GCXX_SAFE_RUNTIME_CALL(
    MallocFromPoolAsync,
    "Failed to allocate device memory from pool asynchronously", &ptr, numBytes,
    pool, stream);
  return ptr;
}

GCXX_FH auto deviceMemPoolCreate(const deviceMemPoolProps_t& poolProps)
  -> deviceMemPool_t {
  deviceMemPool_t pool{};
  GCXX_SAFE_RUNTIME_CALL(MemPoolCreate, "Failed to create memory pool", &pool,
                         &poolProps);
  return pool;
}

GCXX_FH auto deviceMemPoolDestroy(deviceMemPool_t pool) -> void {
  GCXX_SAFE_RUNTIME_CALL(MemPoolDestroy, "Failed to destroy memory pool", pool);
}

GCXX_FH auto deviceMemPoolExportPointer(void* ptr)
  -> deviceMemPoolPtrExportData_t {
  deviceMemPoolPtrExportData_t exportData{};
  GCXX_SAFE_RUNTIME_CALL(MemPoolExportPointer,
                         "Failed to export memory pool pointer", &exportData,
                         ptr);
  return exportData;
}

GCXX_FH auto deviceMemPoolExportToShareableHandle(
  void* shareableHandle, deviceMemPool_t pool,
  deviceMemAllocationHandleType_t handleType, unsigned int flags) -> void {
  GCXX_SAFE_RUNTIME_CALL(MemPoolExportToShareableHandle,
                         "Failed to export memory pool handle", shareableHandle,
                         pool, handleType, flags);
}

GCXX_FH auto deviceMemPoolGetAccess(deviceMemPool_t pool,
                                    deviceMemLocation_t* location)
  -> deviceMemAccessFlags_t {
  deviceMemAccessFlags_t flags{};
  GCXX_SAFE_RUNTIME_CALL(MemPoolGetAccess,
                         "Failed to get memory pool accessibility", &flags,
                         pool, location);
  return flags;
}

GCXX_FH auto deviceMemPoolGetAttribute(deviceMemPool_t pool,
                                       deviceMemPoolAttr_t attr, void* value)
  -> void {
  GCXX_SAFE_RUNTIME_CALL(MemPoolGetAttribute,
                         "Failed to get memory pool attribute", pool, attr,
                         value);
}

GCXX_FH auto deviceMemPoolImportFromShareableHandle(
  void* shareableHandle, deviceMemAllocationHandleType_t handleType,
  unsigned int flags) -> deviceMemPool_t {
  deviceMemPool_t pool{};
  GCXX_SAFE_RUNTIME_CALL(MemPoolImportFromShareableHandle,
                         "Failed to import memory pool handle", &pool,
                         shareableHandle, handleType, flags);
  return pool;
}

GCXX_FH auto deviceMemPoolImportPointer(
  deviceMemPool_t pool, deviceMemPoolPtrExportData_t* exportData) -> void* {
  void* ptr = nullptr;
  GCXX_SAFE_RUNTIME_CALL(MemPoolImportPointer,
                         "Failed to import memory pool pointer", &ptr, pool,
                         exportData);
  return ptr;
}

GCXX_FH auto deviceMemPoolSetAccess(deviceMemPool_t pool,
                                    const deviceMemAccessDesc_t* descList,
                                    std::size_t count) -> void {
  GCXX_SAFE_RUNTIME_CALL(MemPoolSetAccess,
                         "Failed to set memory pool accessibility", pool,
                         descList, count);
}

GCXX_FH auto deviceMemPoolSetAttribute(deviceMemPool_t pool,
                                       deviceMemPoolAttr_t attr, void* value)
  -> void {
  GCXX_SAFE_RUNTIME_CALL(MemPoolSetAttribute,
                         "Failed to set memory pool attribute", pool, attr,
                         value);
}

GCXX_FH auto deviceMemPoolTrimTo(deviceMemPool_t pool,
                                 std::size_t minBytesToKeep) -> void {
  GCXX_SAFE_RUNTIME_CALL(MemPoolTrimTo, "Failed to trim memory pool", pool,
                         minBytesToKeep);
}

#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)
GCXX_FH auto deviceMemPoolGetDefaultMemPool(deviceMemLocation_t* location,
                                            deviceMemAllocationType_t type)
  -> deviceMemPool_t {
  deviceMemPool_t pool{};
  GCXX_SAFE_RUNTIME_CALL(MemGetDefaultMemPool,
                         "Failed to get default memory pool", &pool, location,
                         type);
  return pool;
}

GCXX_FH auto deviceMemPoolGetMemPool(deviceMemLocation_t* location,
                                     deviceMemAllocationType_t type)
  -> deviceMemPool_t {
  deviceMemPool_t pool{};
  GCXX_SAFE_RUNTIME_CALL(MemGetMemPool, "Failed to get current memory pool",
                         &pool, location, type);
  return pool;
}

GCXX_FH auto deviceMemPoolSetMemPool(deviceMemLocation_t* location,
                                     deviceMemAllocationType_t type,
                                     deviceMemPool_t pool) -> void {
  GCXX_SAFE_RUNTIME_CALL(MemSetMemPool, "Failed to set current memory pool",
                         location, type, pool);
}
#endif

GCXX_NAMESPACE_MAIN_DRIVER_END()

#endif
