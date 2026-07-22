// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Derived from NVIDIA CCCL libcudacxx:
//   https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__memory_resource/__memory_pool/device_memory_pool.h
// Copyright (c) NVIDIA Corporation and affiliates.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// device_memory_pool(_ref): a device memory pool. Allocations come from device
// memory via cudaMallocFromPoolAsync/cudaFreeAsync. The owning
// device_memory_pool creates and owns a cudaMemPool_t located on a device; the
// non-owning device_memory_pool_ref wraps an existing handle. Both expose the
// full CCCL pool API (allocate/allocate_sync/trim_to/attributes/peer access)
// AND the gcxx resource concept (allocate(size_t, StreamView) /
// deallocate(void*, StreamView)) so the ref can back a buffer.
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMORY_POOL_DEVICE_MEMORY_POOL_HPP_
#define GCXX_RUNTIME_MEMORY_MEMORY_POOL_DEVICE_MEMORY_POOL_HPP_

#include <cstddef>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/device/device_handle.hpp>
#include <gcxx/runtime/memory/buffers/no_init.hpp>
#include <gcxx/runtime/memory/buffers/properties.hpp>
#include <gcxx/runtime/memory/memory_pool/memory_pool_base.hpp>
#include <gcxx/runtime/memory/memory_pool/memory_pool_properties.hpp>
#include <gcxx/runtime/memory/memory_resource/resource_concepts.hpp>
#include <gcxx/runtime_backend/backend_stream_memory.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_MEMORY_BEGIN()

// ─────────────────────────────────────────────────────────────────────────────
// device_memory_pool_ref: a non-owning device memory pool. Does NOT own the
// underlying cudaMemPool_t; the caller must keep the pool alive longer than the
// ref. Copyable (shallow handle copy), so it can back a buffer.
// ─────────────────────────────────────────────────────────────────────────────
class device_memory_pool_ref : public memory_pool_base {
 public:
  /// Pool memory is always device-visible.
  using properties = TypeSet<device_accessible>;

  GCXX_FH explicit device_memory_pool_ref(driver::deviceMemPool_t pool) noexcept
      : memory_pool_base(pool) {}

  device_memory_pool_ref(int)            = delete;
  device_memory_pool_ref(std::nullptr_t) = delete;

  // Re-expose the base's CCCL-style allocate/deallocate (stream-first), which
  // would otherwise be hidden by the resource-concept overloads below.
  using memory_pool_base::allocate;
  using memory_pool_base::deallocate;

  // gcxx resource-concept adapters (bytes-first arg order). Overload resolution
  // disambiguates from the stream-first overloads because StreamView is not
  // convertible to std::size_t (and vice versa).
  GCXX_FH auto allocate(std::size_t num_bytes, gcxx::StreamView stream) -> void* {
    return memory_pool_base::allocate(stream, num_bytes);
  }
  GCXX_FH auto deallocate(void* ptr, gcxx::StreamView stream) -> void {
    memory_pool_base::deallocate(stream, ptr, 0);
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// device_memory_pool: an owning device memory pool. Creates a cudaMemPool_t
// located on the given device (cudaMemAllocationTypePinned) and destroys it on
// destruction. Non-copyable, non-movable; transfer the handle via release() /
// from_native_handle(). Satisfies resource_with but cannot back a buffer
// directly (not copyable) — use as_ref() for that.
// ─────────────────────────────────────────────────────────────────────────────
struct device_memory_pool : device_memory_pool_ref {
  using reference_type = device_memory_pool_ref;

  /// Construct an empty pool with no underlying handle.
  GCXX_FH explicit device_memory_pool(no_init_t) noexcept
      : device_memory_pool_ref(driver::deviceMemPool_t{}) {}

  /// Construct and own a device memory pool on `device`.
  GCXX_FH device_memory_pool(gcxx::DeviceHandle device,
                             memory_pool_properties props = {})
      : device_memory_pool_ref(create_memory_pool(
          flags::MemLocation::Device, device.id(), flags::MemAllocation::Pinned,
          props)) {}

  GCXX_FH ~device_memory_pool() noexcept {
    if (m_pool_ != nullptr) {
      driver::deviceMemPoolDestroy(m_pool_);
    }
  }

  /// Adopt an existing cudaMemPool_t without creating a new one.
  GCXX_FH static auto from_native_handle(driver::deviceMemPool_t pool) noexcept
    -> device_memory_pool {
    return device_memory_pool(pool);
  }

  /// Relinquish ownership of the handle and return it (pool left empty).
  GCXX_FH constexpr auto release() noexcept -> driver::deviceMemPool_t {
    driver::deviceMemPool_t pool = m_pool_;
    m_pool_                      = nullptr;
    return pool;
  }

  /// A non-owning ref to this pool (the buffer-compatible view).
  GCXX_FH auto as_ref() noexcept -> device_memory_pool_ref& {
    return static_cast<device_memory_pool_ref&>(*this);
  }

  device_memory_pool(const device_memory_pool&)            = delete;
  device_memory_pool& operator=(const device_memory_pool&) = delete;

 private:
  GCXX_FH explicit device_memory_pool(driver::deviceMemPool_t pool) noexcept
      : device_memory_pool_ref(pool) {}
};

static_assert(resource_with<device_memory_pool_ref, device_accessible>,
              "device_memory_pool_ref must model the gcxx resource concept");
static_assert(resource_with<device_memory_pool, device_accessible>,
              "device_memory_pool must model the gcxx resource concept");

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif
