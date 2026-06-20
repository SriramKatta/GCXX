// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMORY_RESOURCE_SYNC_HOST_RESOURCE_HPP_
#define GCXX_RUNTIME_MEMORY_MEMORY_RESOURCE_SYNC_HOST_RESOURCE_HPP_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime_backend/backend_memory.hpp>

#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_MEMORY_BEGIN()

class sync_host_resource {
 public:
  GCXX_FH auto allocate(std::size_t num_bytes, gcxx::StreamView) -> void* {
    return driver::deviceMallocHost(num_bytes);
  }

  GCXX_FH auto deallocate(void* ptr, gcxx::StreamView) -> void {
    driver::deviceFreeHost(ptr);
  }

  GCXX_FHDC static auto is_device() -> bool { return false; }
};

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif
