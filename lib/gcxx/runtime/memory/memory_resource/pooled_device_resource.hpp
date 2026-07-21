// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMORY_RESOURCE_POOLED_DEVICE_RESOURCE_HPP_
#define GCXX_RUNTIME_MEMORY_MEMORY_RESOURCE_POOLED_DEVICE_RESOURCE_HPP_

#include <cstddef>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/memory/buffers/properties.hpp>
#include <gcxx/runtime/memory/mempool/mempool_view.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>
#include <gcxx/runtime_backend/backend_stream_memory.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()


GCXX_NAMESPACE_MEMORY_BEGIN()

// Memory resource that allocates from a device MemPoolView via the async
// pool allocator and returns memory to the pool via the async free path.
// Satisfies the buffer<VT, Resource> resource concept:
//   void* allocate(std::size_t num_bytes, gcxx::StreamView)
//   void  deallocate(void* ptr, gcxx::StreamView)
//
// Advertises device_accessible via `using properties` so the buffer ctor's
// static_assert on execution-space properties accepts it. Pool memory lives in
// device memory; not host-accessible. (Stream-ordered alloc/free — no extra
// sync-before-free needed; the pool/driver handle ordering.)
class pooled_device_resource {
 public:
  /// Pool memory is always device-visible.
  using properties = TypeSet<device_accessible>;

  pooled_device_resource() = default;

  GCXX_FH explicit pooled_device_resource(MemPoolView pool) : m_pool(pool) {}

  GCXX_FH auto allocate(std::size_t num_bytes, gcxx::StreamView sv) -> void* {
    return driver::deviceMallocFromPoolAsync(num_bytes, m_pool.getRawMemPool(),
                                             sv.getRawStream());
  }

  GCXX_FH auto deallocate(void* ptr, gcxx::StreamView sv) -> void {
    driver::deviceFreeAsync(ptr, sv.getRawStream());
  }

  GCXX_FH auto pool() const -> MemPoolView { return m_pool; }

 private:
  MemPoolView m_pool{};
};

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif
