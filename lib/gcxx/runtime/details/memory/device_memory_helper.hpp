#pragma once
#ifndef GCXX_RUNTIME_DETAILS_MEMORY_DEVICE_MEMORY_HELPER_HPP
#define GCXX_RUNTIME_DETAILS_MEMORY_DEVICE_MEMORY_HELPER_HPP

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/error/runtime_error.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

GCXX_CXPR auto device_malloc = [](std::size_t numbytes) {
  void* ptr = nullptr;
  GCXX_SAFE_RUNTIME_CALL(Malloc, "Failed to allocate device memory", &ptr,
                         numbytes);
  return ptr;
};

GCXX_CXPR auto device_malloc_async = [](std::size_t numbytes,
                                        const StreamView& sv = NULL_STREAM) {
  void* ptr = nullptr;
  GCXX_SAFE_RUNTIME_CALL(MallocAsync,
                         "Failed to allocate device memory asynchronously",
                         &ptr, numbytes, sv.getRawStream());
  return ptr;
};


GCXX_CXPR auto device_managed_malloc = [](std::size_t numbytes) {
  void* ptr = nullptr;
  GCXX_SAFE_RUNTIME_CALL(
    MallocManaged, "Failed to allocate managed device memory", &ptr, numbytes);
  return ptr;
};

GCXX_CXPR auto host_malloc = [](std::size_t numbytes) {
  void* ptr = nullptr;
  GCXX_SAFE_RUNTIME_CALL(
#if GCXX_CUDA_MODE
    MallocHost
#elif GCXX_HIP_MODE
    HostMalloc
#endif
    ,
    "Failed to allocate Pinned host memory", &ptr, numbytes);
  return ptr;
};

GCXX_CXPR auto device_free = [](void* ptr) {
  GCXX_SAFE_RUNTIME_CALL(Free, "Failed to deallocate device memory", ptr);
};

GCXX_CXPR auto device_free_async = [](void* ptr,
                                      const StreamView& sv = NULL_STREAM) {
  GCXX_SAFE_RUNTIME_CALL(FreeAsync,
                         "Failed to deallocate device memory Asynchronysly",
                         ptr, sv.getRawStream());
};

GCXX_CXPR auto host_free = [](void* ptr) {
  GCXX_SAFE_RUNTIME_CALL(FreeHost, "Failed to deallocate Pinned host memory",
                         ptr);
};

GCXX_NAMESPACE_MAIN_DETAILS_END


#endif