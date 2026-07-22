// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Derived from NVIDIA CCCL libcudacxx:
//   https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__memory_resource/__memory_pool/memory_pool_base.h
// Copyright (c) NVIDIA Corporation and affiliates.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// memory_pool_base: the non-templated, non-owning core of the gcxx memory-pool
// hierarchy. It holds a raw cudaMemPool_t and exposes the CCCL pool API
// (allocate / allocate_sync / trim_to / typed attributes / peer access / get)
// by calling the gcxx driver wrappers directly — it does NOT depend on the
// older gcxx::MemPool / MemPoolView handle classes. The owning device/managed/
// pinned pools and their *_ref views derive from this.
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMORY_POOL_MEMORY_POOL_BASE_HPP_
#define GCXX_RUNTIME_MEMORY_MEMORY_POOL_MEMORY_POOL_BASE_HPP_

#include <cstddef>
#include <vector>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/flags/memory_flags.hpp>
#include <gcxx/runtime/memory/mempool/mempool_props.hpp>
#include <gcxx/runtime/memory/memory_pool/memory_pool_attributes.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>
#include <gcxx/runtime_backend/backend_device.hpp>
#include <gcxx/runtime_backend/backend_stream.hpp>
#include <gcxx/runtime_backend/backend_stream_memory.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_MEMORY_BEGIN()

// The alignment guarantee CUDA makes for pool allocations (cudaMallocFromPoolAsync
// returns at least this-aligned memory). Valid requested alignments are the
// powers of two up to this value. Mirrors CCCL's cuda::mr::default_cuda_malloc_alignment.
inline constexpr std::size_t default_cuda_malloc_alignment = 256;

class memory_pool_base {
 protected:
  driver::deviceMemPool_t m_pool_{nullptr};

  /// true iff alignment is a power of two <= default_cuda_malloc_alignment.
  GCXX_FH static constexpr auto is_valid_alignment(std::size_t alignment) noexcept
    -> bool {
    return alignment <= default_cuda_malloc_alignment
        && (default_cuda_malloc_alignment % alignment == 0);
  }

 public:
  memory_pool_base(std::nullptr_t) = delete;

  GCXX_FH explicit memory_pool_base(driver::deviceMemPool_t pool) noexcept
      : m_pool_(pool) {}

  // ── Stream-ordered allocation ──────────────────────────────────────────────

  /// Allocate at least `bytes` from the pool, ordered on `stream`, with the
  /// requested alignment (validated; must be a power of two <= the default).
  GCXX_FH auto allocate(gcxx::StreamView stream, std::size_t bytes,
                        std::size_t alignment) -> void* {
    assert(is_valid_alignment(alignment) &&
           "Invalid alignment passed to memory_pool_base::allocate.");
    return allocate(stream, bytes);
  }

  /// Allocate at least `bytes` from the pool, ordered on `stream`.
  GCXX_FH auto allocate(gcxx::StreamView stream, std::size_t bytes) -> void* {
    return driver::deviceMallocFromPoolAsync(bytes, m_pool_, stream.getRawStream());
  }

  /// Return `ptr` to the pool, ordered on `stream`.
  GCXX_FH void deallocate(gcxx::StreamView stream, void* ptr, std::size_t bytes,
                          std::size_t alignment) noexcept {
    assert(is_valid_alignment(alignment) &&
           "Invalid alignment passed to memory_pool_base::deallocate.");
    deallocate(stream, ptr, bytes);
  }

  GCXX_FH void deallocate(gcxx::StreamView stream, void* ptr,
                          std::size_t /*bytes*/) noexcept {
    driver::deviceFreeAsync(ptr, stream.getRawStream());
  }

  // ── Synchronous allocation (default stream) ────────────────────────────────
  // Uses the default (legacy) stream + synchronize, so the returned pointer is
  // immediately usable. Mirrors CCCL's allocate_sync/deallocate_sync semantics.

  GCXX_FH auto allocate_sync(std::size_t bytes,
                             std::size_t alignment = default_cuda_malloc_alignment)
    -> void* {
    assert(is_valid_alignment(alignment) &&
           "Invalid alignment passed to memory_pool_base::allocate_sync.");
    void* ptr = driver::deviceMallocFromPoolAsync(bytes, m_pool_, nullptr);
    driver::streamSynchronize(nullptr);
    return ptr;
  }

  GCXX_FH void deallocate_sync(void* ptr, std::size_t /*bytes*/,
                               std::size_t alignment = default_cuda_malloc_alignment)
    noexcept {
    assert(is_valid_alignment(alignment) &&
           "Invalid alignment passed to memory_pool_base::deallocate_sync.");
    driver::deviceFreeAsync(ptr, nullptr);
    driver::streamSynchronize(nullptr);
  }

  // ── Pool management ────────────────────────────────────────────────────────

  /// Release pool memory down to at least `min_bytes_to_keep` reserved bytes.
  GCXX_FH auto trim_to(std::size_t min_bytes_to_keep) -> void {
    driver::deviceMemPoolTrimTo(m_pool_, min_bytes_to_keep);
  }

  /// Read a typed attribute (see memory_pool_attributes).
  template <typename Attr>
  GCXX_FH auto attribute(Attr attr) const -> typename Attr::type {
    return attr(m_pool_);
  }

  /// Write a typed attribute (see memory_pool_attributes). Read-only attributes
  /// are a no-op.
  template <typename Attr>
  GCXX_FH auto set_attribute(Attr attr, typename Attr::type value) -> void {
    Attr::set(m_pool_, value);
  }

  /// The underlying cudaMemPool_t handle.
  GCXX_FH constexpr auto get() const noexcept -> driver::deviceMemPool_t {
    return m_pool_;
  }

  // ── Peer / cross-device access ─────────────────────────────────────────────

  /// Enable read/write access to this pool's allocations from `device_id`.
  GCXX_FH auto enable_access_from(int device_id) -> void {
    MemAccessDesc desc{flags::MemLocation::Device, device_id,
                       flags::MemAccessFlags::ReadWrite};
    auto raw = desc.getRawMemAccessDesc();
    driver::deviceMemPoolSetAccess(m_pool_, &raw, 1);
  }

  /// Disable access to this pool's allocations from `device_id`.
  GCXX_FH auto disable_access_from(int device_id) -> void {
    MemAccessDesc desc{flags::MemLocation::Device, device_id,
                       flags::MemAccessFlags::None};
    auto raw = desc.getRawMemAccessDesc();
    driver::deviceMemPoolSetAccess(m_pool_, &raw, 1);
  }

  /// Enable read/write access from every device in the system.
  GCXX_FH auto enable_access_from_all() -> void {
    const int count = driver::deviceGetCount();
    std::vector<driver::deviceMemAccessDesc_t> descs;
    descs.reserve(static_cast<std::size_t>(count));
    for (int dev = 0; dev < count; ++dev) {
      MemAccessDesc desc{flags::MemLocation::Device, dev,
                         flags::MemAccessFlags::ReadWrite};
      descs.push_back(desc.getRawMemAccessDesc());
    }
    if (!descs.empty()) {
      driver::deviceMemPoolSetAccess(m_pool_, descs.data(), descs.size());
    }
  }

  /// true iff `device_id` has read/write access to this pool's allocations.
  GCXX_FH auto is_accessible_from(int device_id) -> bool {
    MemAccessDesc location{flags::MemLocation::Device, device_id,
                           flags::MemAccessFlags::None};
    auto rawLoc = location.getRawMemLocation();
    auto flags = driver::deviceMemPoolGetAccess(m_pool_, &rawLoc);
    return flags == static_cast<driver::deviceMemAccessFlags_t>(
                      flags::MemAccessFlags::ReadWrite);
  }

  // ── Comparison ─────────────────────────────────────────────────────────────

  GCXX_FH auto operator==(const memory_pool_base& rhs) const noexcept -> bool {
    return m_pool_ == rhs.m_pool_;
  }
  GCXX_FH auto operator!=(const memory_pool_base& rhs) const noexcept -> bool {
    return m_pool_ != rhs.m_pool_;
  }
};

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif
