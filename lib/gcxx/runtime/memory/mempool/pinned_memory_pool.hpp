// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Derived from NVIDIA CCCL libcudacxx:
//   https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__memory_resource/__memory_pool/pinned_memory_pool.h
// Copyright (c) NVIDIA Corporation and affiliates.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// PinnedMemPool / PinnedMemPoolView: a pinned (page-locked) host memory pool.
// Pinned memory enables faster host<->device transfers and is reachable from
// all devices. The owning pool creates a cudaMemPool_t at a host (or host-NUMA)
// location and, like CCCL, immediately grants access from every device. Both
// the view and the owning pool expose the full CCCL pool API
// (allocate/allocate_sync/trim_to/attributes) inherited from MemPoolView AND
// the gcxx resource concept (allocate(StreamView, size_t) /
// deallocate(StreamView, void*)) so the view can back a buffer.
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMORY_POOL_PINNED_MEMORY_POOL_HPP_
#define GCXX_RUNTIME_MEMORY_MEMORY_POOL_PINNED_MEMORY_POOL_HPP_

#include <cstddef>
#include <utility>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/memory/buffers/no_init.hpp>
#include <gcxx/runtime/memory/buffers/properties.hpp>
#include <gcxx/runtime/memory/memory_resource/resource_concepts.hpp>
#include <gcxx/runtime/memory/mempool/memory_pool_properties.hpp>
#include <gcxx/runtime/memory/mempool/mempool_view.hpp>
#include <gcxx/runtime_backend/backend_stream_memory.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()


// ─────────────────────────────────────────────────────────────────────────────
// PinnedMemPoolView: a non-owning pinned host memory pool. Does NOT own the
// underlying cudaMemPool_t; the caller must keep the pool alive longer than the
// view. Copyable (shallow handle copy), so it can back a buffer. Pinned memory
// is reachable from both host and device, so it advertises both.
// ─────────────────────────────────────────────────────────────────────────────
class PinnedMemPoolView : public MemPoolView {
 public:
  /// Pinned memory is always host- and device-visible.
  using properties = TypeSet<device_accessible, host_accessible>;

  GCXX_FH explicit PinnedMemPoolView(driver::deviceMemPool_t pool) noexcept
      : MemPoolView(pool) {}

  // Block `PinnedMemPoolView{0}` / `{nullptr}`, which would otherwise
  // null-pointer-convert into the explicit handle ctor above and silently build
  // a null-handle view (the base's deleted nullptr_t ctor does not catch int
  // 0).
  PinnedMemPoolView(int)            = delete;
  PinnedMemPoolView(std::nullptr_t) = delete;
};

// ─────────────────────────────────────────────────────────────────────────────
// PinnedMemPool: an owning pinned host memory pool. Creates a cudaMemPool_t at
// a host (or host-NUMA) location and destroys it on destruction, then grants
// read/write access from every device (matching CCCL) so its allocations can be
// used in peer transfers without further setup. Non-copyable, non-movable;
// transfer the handle via release() / from_native_handle(). Satisfies
// resource_with but cannot back a buffer directly (not copyable) — use
// as_ref().
// ─────────────────────────────────────────────────────────────────────────────
struct PinnedMemPool : PinnedMemPoolView {
  using reference_type = PinnedMemPoolView;

  /// Construct an empty pool with no underlying handle.
  GCXX_FH explicit PinnedMemPool(no_init_t) noexcept
      : PinnedMemPoolView(driver::deviceMemPool_t{}) {}

#if GCXX_CUDA_VERSION_GREATER_EQUAL(13, 0, 0)
  /// Create a pinned pool on the default host location (CUDA 13.0+).
  GCXX_FH PinnedMemPool(memory_pool_properties props = {})
      : PinnedMemPoolView(create_memory_pool(
          flags::MemLocation::Host, 0, flags::MemAllocation::Pinned, props)) {
  }
#endif

  /// Create a pinned pool bound to a specific host NUMA node.
  GCXX_FH PinnedMemPool(int numa_id, memory_pool_properties props = {})
      : PinnedMemPoolView(
          create_memory_pool(flags::MemLocation::HostNuma, numa_id,
                             flags::MemAllocation::Pinned, props)) {
  }

  GCXX_FH ~PinnedMemPool() noexcept {
    if (m_pool_ != nullptr) {
      driver::deviceMemPoolDestroy(m_pool_);
    }
  }

  /// Adopt an existing cudaMemPool_t without creating a new one.
  GCXX_FH static auto from_native_handle(driver::deviceMemPool_t pool) noexcept
    -> PinnedMemPool {
    return PinnedMemPool(pool);
  }

  /// Relinquish ownership of the handle and return it (pool left empty).
  // std::exchange is not constexpr until C++20
  GCXX_FH constexpr auto release() noexcept -> driver::deviceMemPool_t {
    auto old = m_pool_;
    m_pool_  = nullptr;
    return old;
  }

  /// A non-owning ref to this pool (the buffer-compatible view).
  GCXX_FH auto as_ref() noexcept -> PinnedMemPoolView& {
    return static_cast<PinnedMemPoolView&>(*this);
  }

  PinnedMemPool(const PinnedMemPool&)                = delete;
  PinnedMemPool& operator=(const PinnedMemPool&)     = delete;
  PinnedMemPool(PinnedMemPool&&) noexcept            = delete;
  PinnedMemPool& operator=(PinnedMemPool&&) noexcept = delete;

 private:
  GCXX_FH explicit PinnedMemPool(driver::deviceMemPool_t pool) noexcept
      : PinnedMemPoolView(pool) {}
};

static_assert(
  resource_with<PinnedMemPoolView, device_accessible, host_accessible>,
  "PinnedMemPoolView must model the gcxx resource concept");
static_assert(resource_with<PinnedMemPool, device_accessible, host_accessible>,
              "PinnedMemPool must model the gcxx resource concept");


GCXX_NAMESPACE_MAIN_END()

#endif
