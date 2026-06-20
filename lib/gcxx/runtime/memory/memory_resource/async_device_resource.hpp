#pragma once
#ifndef GCXX_RUNTIME_MEMEORY_MEMORY_RESOURCE_ASYNC_DEVICE_RESOURCE_HPP_
#define GCXX_RUNTIME_MEMEORY_MEMORY_RESOURCE_ASYNC_DEVICE_RESOURCE_HPP_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime_backend/backend_memory.hpp>

#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()


GCXX_NAMESPACE_MEMORY_BEGIN()

class async_device_resource {

 public:
  GCXX_FH auto allocate(std::size_t num_bytes, gcxx::StreamView sv) -> void* {
    return driver::deviceMallocAsync(num_bytes, sv.getRawStream());
  }

  GCXX_FH auto deallocate(void* ptr, gcxx::StreamView sv) -> void {
    driver::deviceFreeAsync(ptr, sv.getRawStream());
  }

  GCXX_FHDC static auto is_device() -> bool { return true; }
};

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif