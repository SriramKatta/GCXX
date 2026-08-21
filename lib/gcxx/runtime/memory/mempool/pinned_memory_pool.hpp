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
// all devices. The owning pool creates a cudaMemPool_t at a host location and,
// like CCCL, immediately grants access from every device. Both
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
#include <gcxx/runtime_backend/backend_memory.hpp>
#include <gcxx/runtime_backend/backend_stream_memory.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()


// Non-owning view; caller must keep the pool alive longer than the view.
class PinnedMemPoolView : public MemPoolView {
 public:
  // Pinned memory is always host- and device-visible.
  using properties = TypeSet<device_accessible, host_accessible>;

  GCXX_FH explicit PinnedMemPoolView(driver::deviceMemPool_t pool) noexcept
      : MemPoolView(pool) {}

  // Blocks {0}/{nullptr} silently building a null-handle view.
  PinnedMemPoolView(int)            = delete;
  PinnedMemPoolView(std::nullptr_t) = delete;

#if GCXX_HIP_MODE()
  // HIP shim: ROCm has no host hipMemPool_t; routes through hipMallocHost.
  struct hip_shim_handle_t {};

  GCXX_FH explicit PinnedMemPoolView(hip_shim_handle_t) noexcept
      : MemPoolView(driver::deviceMemPool_t{}) {}

  GCXX_FH auto allocate(gcxx::StreamView /*stream*/, std::size_t bytes,
                        std::size_t alignment = default_cuda_malloc_alignment)
    -> void* {
    assert(is_valid_alignment(alignment) &&
           "Invalid alignment passed to PinnedMemPoolView::allocate.");
    (void)alignment;
    return driver::deviceMallocHost(bytes);
  }

  GCXX_FH void deallocate(
    gcxx::StreamView /*stream*/, void* ptr, std::size_t /*bytes*/ = 0,
    std::size_t alignment = default_cuda_malloc_alignment) noexcept {
    assert(is_valid_alignment(alignment) &&
           "Invalid alignment passed to PinnedMemPoolView::deallocate.");
    driver::deviceFreeHost(ptr);
  }

  GCXX_FH auto allocate_sync(
    std::size_t bytes,
    std::size_t alignment = default_cuda_malloc_alignment) -> void* {
    assert(is_valid_alignment(alignment) &&
           "Invalid alignment passed to PinnedMemPoolView::allocate_sync.");
    (void)alignment;
    return driver::deviceMallocHost(bytes);
  }

  GCXX_FH void deallocate_sync(
    void* ptr, std::size_t /*bytes*/ = 0,
    std::size_t alignment = default_cuda_malloc_alignment) noexcept {
    assert(is_valid_alignment(alignment) &&
           "Invalid alignment passed to PinnedMemPoolView::deallocate_sync.");
    driver::deviceFreeHost(ptr);
  }
#endif  // GCXX_HIP_MODE()
};

// Owning pinned pool; non-copyable, so back buffers via as_ref().
struct PinnedMemPool : PinnedMemPoolView {
  using reference_type = PinnedMemPoolView;

  GCXX_FH explicit PinnedMemPool(no_init_t) noexcept
      : PinnedMemPoolView(driver::deviceMemPool_t{}) {}

#if GCXX_HIP_MODE()
  // Props are ignored: HIP has no host pool to configure.
  GCXX_FH PinnedMemPool(memory_pool_properties /*props*/ = {})
      : PinnedMemPoolView(PinnedMemPoolView::hip_shim_handle_t{}) {}
#else
  // Generic Host location on purpose; HostNuma ids are a portability trap.
  GCXX_FH PinnedMemPool(memory_pool_properties props = {})
      : PinnedMemPoolView(
          create_memory_pool(flags::MemLocation::Host, /*location_id=*/0,
                             flags::MemAllocation::Pinned, props)) {
    // Host-location pool allocations are only GPU-mapped once
    // cudaMemPoolSetAccess grants access (unlike cudaMallocHost), so grant
    // read/write access from every device up front, matching CCCL.
    enable_access_from_all();
  }
#endif

  GCXX_FH ~PinnedMemPool() noexcept {
    if (m_pool_ != nullptr) {
      driver::deviceMemPoolDestroy(m_pool_);
    }
  }

  GCXX_FH static auto from_native_handle(driver::deviceMemPool_t pool) noexcept
    -> PinnedMemPool {
    return PinnedMemPool(pool);
  }

  // Hand-rolled instead of std::exchange: not constexpr until C++20.
  GCXX_FH constexpr auto release() noexcept -> driver::deviceMemPool_t {
    auto old = m_pool_;
    m_pool_  = nullptr;
    return old;
  }

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
