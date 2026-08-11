// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Derived from NVIDIA CCCL libcudacxx:
//   https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__memory_resource/__memory_pool/memory_pool_base.h
// Copyright (c) NVIDIA Corporation and affiliates.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMPOOL_MEMPOOL_VIEW_HPP_
#define GCXX_RUNTIME_MEMORY_MEMPOOL_MEMPOOL_VIEW_HPP_

#include <cstddef>
#include <vector>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/flags/memory_flags.hpp>
#include <gcxx/runtime/memory/mempool/memory_pool_attributes.hpp>
#include <gcxx/runtime/memory/mempool/mempool_props.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>
#include <gcxx/runtime_backend/backend_device.hpp>
#include <gcxx/runtime_backend/backend_stream.hpp>
#include <gcxx/runtime_backend/backend_stream_memory.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

// The alignment guarantee CUDA makes for pool allocations
inline constexpr std::size_t default_cuda_malloc_alignment = 256;

class MemPoolView {
 protected:
  using deviceMemPool_t = driver::deviceMemPool_t;

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  deviceMemPool_t m_pool_{nullptr};

  GCXX_FH static constexpr auto is_valid_alignment(
    std::size_t alignment) noexcept -> bool {
    return alignment != 0 && alignment <= default_cuda_malloc_alignment &&
           (default_cuda_malloc_alignment % alignment == 0);
  }

 public:
  using raw_handle_type = driver::deviceMemPool_t;

  MemPoolView(std::nullptr_t) = delete;
  MemPoolView(int)            = delete;

  GCXX_FH explicit MemPoolView(deviceMemPool_t pool) noexcept : m_pool_(pool) {}

  // ╔════════════════════════════════════════════════════════╗
  // ║               Stream-ordered (de)allocation            ║
  // ╚════════════════════════════════════════════════════════╝

  // Allocate at least `bytes` from the pool, ordered on `stream`, with the
  // requested alignment (validated; must be a power of two <= the default).
  GCXX_FH auto allocate(gcxx::StreamView stream, std::size_t bytes,
                        [[maybe_unused]] std::size_t alignment =
                          default_cuda_malloc_alignment) -> void* {
    assert(is_valid_alignment(alignment) &&
           "Invalid alignment passed to MemPoolView::allocate.");
    return driver::deviceMallocFromPoolAsync(bytes, m_pool_,
                                             stream.getRawHandle());
  }

  // Return `ptr` to the pool, ordered on `stream`.
  GCXX_FH void deallocate(gcxx::StreamView stream, void* ptr,
                          [[maybe_unused]] std::size_t bytes = 0,
                          [[maybe_unused]] std::size_t alignment =
                            default_cuda_malloc_alignment) noexcept {
    assert(is_valid_alignment(alignment) &&
           "Invalid alignment passed to MemPoolView::deallocate.");
    driver::deviceFreeAsync(ptr, stream.getRawHandle());
  }


  // ╔════════════════════════════════════════════════════════╗
  // ║        Synchronous (de)allocation (default stream)     ║
  // ╚════════════════════════════════════════════════════════╝
  GCXX_FH auto allocate_sync(
    std::size_t bytes,
    std::size_t alignment = default_cuda_malloc_alignment) -> void* {
    assert(is_valid_alignment(alignment) &&
           "Invalid alignment passed to MemPoolView::allocate_sync.");
    void* ptr = allocate(StreamView::Null(), bytes, alignment);
    StreamView::Null().Synchronize();
    return ptr;
  }

  GCXX_FH void deallocate_sync(
    void* ptr, std::size_t bytes = 0,
    std::size_t alignment = default_cuda_malloc_alignment) noexcept {
    assert(is_valid_alignment(alignment) &&
           "Invalid alignment passed to MemPoolView::deallocate_sync.");
    deallocate(StreamView::Null(), ptr, bytes, alignment);
    StreamView::Null().Synchronize();
  }

  // ╔════════════════════════════════════════════════════════╗
  // ║                    Pool management                     ║
  // ╚════════════════════════════════════════════════════════╝

  // Release pool memory down to at least `min_bytes_to_keep` reserved bytes.
  GCXX_FH auto trim_to(std::size_t min_bytes_to_keep) -> void {
    driver::deviceMemPoolTrimTo(m_pool_, min_bytes_to_keep);
  }

  // Read a typed attribute (see memory_pool_attributes).
  template <typename Attr>
  GCXX_FH auto attribute(Attr attr) const -> typename Attr::type {
    return attr(m_pool_);
  }

  template <typename Attr>
  GCXX_FH auto set_attribute(Attr /*attr*/, typename Attr::type value) -> void {
    // Read-only attributes are a no-op.
    Attr::set(m_pool_, value);
  }
  GCXX_FH constexpr auto getRawHandle() const noexcept -> deviceMemPool_t {
    return m_pool_;
  }

  // ╔════════════════════════════════════════════════════════╗
  // ║               TODO : Peer / cross-device access        ║
  // ╚════════════════════════════════════════════════════════╝
  // GCXX_FH auto enable_access_from(int device_id) -> void {
  //   const MemAccessDesc desc{flags::MemLocation::Device, device_id,
  //                            flags::MemAccessFlags::ReadWrite};
  //   auto raw = desc.getRawMemAccessDesc();
  //   driver::deviceMemPoolSetAccess(m_pool_, &raw, /*count=*/1);
  // }

  // // Disable access to this pool's allocations from `device_id`.
  // GCXX_FH auto disable_access_from(int device_id) -> void {
  //   const MemAccessDesc desc{flags::MemLocation::Device, device_id,
  //                            flags::MemAccessFlags::None};
  //   auto raw = desc.getRawMemAccessDesc();
  //   driver::deviceMemPoolSetAccess(m_pool_, &raw, /*count=*/1);
  // }

  // // Enable read/write access from every device in the system.
  // GCXX_FH auto enable_access_from_all() -> void {
  //   const int count = driver::deviceGetCount();
  //   std::vector<driver::deviceMemAccessDesc_t> descs;
  //   descs.reserve(static_cast<std::size_t>(count));
  //   for (int dev = 0; dev < count; ++dev) {
  //     const MemAccessDesc desc{flags::MemLocation::Device, dev,
  //                              flags::MemAccessFlags::ReadWrite};
  //     descs.push_back(desc.getRawMemAccessDesc());
  //   }
  //   if (!descs.empty()) {
  //     driver::deviceMemPoolSetAccess(m_pool_, descs.data(), descs.size());
  //   }
  // }

  // // true iff `device_id` has read/write access to this pool's allocations.
  // GCXX_FH auto is_accessible_from(int device_id) -> bool {
  //   const MemAccessDesc location{flags::MemLocation::Device, device_id,
  //                                flags::MemAccessFlags::None};
  //   auto rawLoc = location.getRawMemLocation();
  //   auto flags  = driver::deviceMemPoolGetAccess(m_pool_, &rawLoc);
  //   return flags == static_cast<driver::deviceMemAccessFlags_t>(
  //                     flags::MemAccessFlags::ReadWrite);
  // }

  // ╔════════════════════════════════════════════════════════╗
  // ║                       Comparison                       ║
  // ╚════════════════════════════════════════════════════════╝
  GCXX_FH auto operator==(const MemPoolView& rhs) const noexcept -> bool {
    return m_pool_ == rhs.m_pool_;
  }
  GCXX_FH auto operator!=(const MemPoolView& rhs) const noexcept -> bool {
    return m_pool_ != rhs.m_pool_;
  }
};

GCXX_NAMESPACE_MAIN_END()

#endif
