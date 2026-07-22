// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Derived from NVIDIA CCCL libcudacxx:
//   https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__memory_resource/__memory_pool/pinned_memory_pool.h
// Copyright (c) NVIDIA Corporation and affiliates.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// pinned_memory_pool(_ref): a pinned (page-locked) host memory pool. Pinned
// memory enables faster host<->device transfers and is reachable from all
// devices. CUDA 12.9+ only. The owning pool creates a cudaMemPool_t at a host
// (or host-NUMA) location and, like CCCL, immediately grants access from every
// device.
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMORY_POOL_PINNED_MEMORY_POOL_HPP_
#define GCXX_RUNTIME_MEMORY_MEMORY_POOL_PINNED_MEMORY_POOL_HPP_

#include <cstddef>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/memory/buffers/no_init.hpp>
#include <gcxx/runtime/memory/buffers/properties.hpp>
#include <gcxx/runtime/memory/memory_pool/memory_pool_base.hpp>
#include <gcxx/runtime/memory/memory_pool/memory_pool_properties.hpp>
#include <gcxx/runtime/memory/memory_resource/resource_concepts.hpp>
#include <gcxx/runtime_backend/backend_stream_memory.hpp>

#if GCXX_CUDA_VERSION_GREATER_EQUAL(12, 9, 0)

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_MEMORY_BEGIN()

// pinned_memory_pool_ref: non-owning pinned host memory pool. Pinned memory is
// reachable from both host and device.
class pinned_memory_pool_ref : public memory_pool_base {
 public:
  using properties = TypeSet<device_accessible, host_accessible>;

  GCXX_FH explicit pinned_memory_pool_ref(driver::deviceMemPool_t pool) noexcept
      : memory_pool_base(pool) {}

  using memory_pool_base::allocate;
  using memory_pool_base::deallocate;

  // gcxx resource-concept adapters (bytes-first arg order).
  GCXX_FH auto allocate(std::size_t num_bytes, gcxx::StreamView stream) -> void* {
    return memory_pool_base::allocate(stream, num_bytes);
  }
  GCXX_FH auto deallocate(void* ptr, gcxx::StreamView stream) -> void {
    memory_pool_base::deallocate(stream, ptr, 0);
  }
};

// pinned_memory_pool: owning pinned host memory pool. After creation it grants
// read/write access from every device (matching CCCL), so its allocations can be
// used in peer transfers without further setup.
struct pinned_memory_pool : pinned_memory_pool_ref {
  using reference_type = pinned_memory_pool_ref;

  GCXX_FH explicit pinned_memory_pool(no_init_t) noexcept
      : pinned_memory_pool_ref(driver::deviceMemPool_t{}) {}

#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)
  /// Create a pinned pool on the default host location (CUDA 13.0+).
  GCXX_FH pinned_memory_pool(memory_pool_properties props = {})
      : pinned_memory_pool_ref(create_memory_pool(flags::MemLocation::Host, 0,
                                                  flags::MemAllocation::Pinned,
                                                  props)) {
    enable_access_from_all();
  }
#endif

  /// Create a pinned pool bound to a specific host NUMA node (CUDA 12.9+).
  GCXX_FH pinned_memory_pool(int numa_id, memory_pool_properties props = {})
      : pinned_memory_pool_ref(create_memory_pool(flags::MemLocation::HostNuma,
                                                  numa_id,
                                                  flags::MemAllocation::Pinned,
                                                  props)) {
    enable_access_from_all();
  }

  GCXX_FH ~pinned_memory_pool() noexcept {
    if (m_pool_ != nullptr) {
      driver::deviceMemPoolDestroy(m_pool_);
    }
  }

  GCXX_FH static auto from_native_handle(driver::deviceMemPool_t pool) noexcept
    -> pinned_memory_pool {
    return pinned_memory_pool(pool);
  }

  GCXX_FH constexpr auto release() noexcept -> driver::deviceMemPool_t {
    driver::deviceMemPool_t pool = m_pool_;
    m_pool_                      = nullptr;
    return pool;
  }

  GCXX_FH auto as_ref() noexcept -> pinned_memory_pool_ref& {
    return static_cast<pinned_memory_pool_ref&>(*this);
  }

  pinned_memory_pool(const pinned_memory_pool&)            = delete;
  pinned_memory_pool& operator=(const pinned_memory_pool&) = delete;

 private:
  GCXX_FH explicit pinned_memory_pool(driver::deviceMemPool_t pool) noexcept
      : pinned_memory_pool_ref(pool) {}
};

static_assert(resource_with<pinned_memory_pool_ref, device_accessible, host_accessible>,
              "pinned_memory_pool_ref must model the gcxx resource concept");
static_assert(resource_with<pinned_memory_pool, device_accessible, host_accessible>,
              "pinned_memory_pool must model the gcxx resource concept");

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif  // GCXX_CUDA_VERSION_GREATER_EQUAL(12, 9, 0)

#endif
