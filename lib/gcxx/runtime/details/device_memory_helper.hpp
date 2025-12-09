#pragma once
#ifndef GCXX_RUNTIME_DETAILS_DEVICE_MEMORY_HELPER_HPP
#define GCXX_RUNTIME_DETAILS_DEVICE_MEMORY_HELPER_HPP

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>
#include <gcxx/runtime/error/runtime_error.hpp>

GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

auto device_malloc = [](std::size_t numbytes) {
  void* ptr = nullptr;
  GCXX_SAFE_RUNTIME_CALL(Malloc, "Failed to allocate device memory", &ptr,
                         numbytes);
  return ptr;
};


auto device_managed_malloc = [](std::size_t numbytes) {
  void* ptr = nullptr;
  GCXX_SAFE_RUNTIME_CALL(
    MallocManaged, "Failed to allocate manged device memory", &ptr, numbytes);
  return ptr;
};

auto host_malloc = [](std::size_t numbytes) {
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

auto device_free = [](void* ptr) {
  GCXX_SAFE_RUNTIME_CALL(Free, "Failed to deallocate device memory", ptr);
};

auto host_free = [](void* ptr) {
  GCXX_SAFE_RUNTIME_CALL(FreeHost, "Failed to deallocate Pinned host memory",
                         ptr);
};

GCXX_NAMESPACE_MAIN_DETAILS_END


#endif