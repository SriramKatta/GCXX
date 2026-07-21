// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMORY_RESOURCE_SYNCHRONOUS_RESOURCE_HPP_
#define GCXX_RUNTIME_MEMORY_MEMORY_RESOURCE_SYNCHRONOUS_RESOURCE_HPP_

#include <cstddef>
#include <type_traits>
#include <utility>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/memory/device_memory_helper.hpp>
#include <gcxx/runtime/memory/buffers/properties.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_MEMORY_BEGIN()

// ─────────────────────────────────────────────────────────────────────────────
// gcxx::memory::synchronous_resource<AllocFn, FreeFn, Properties...>
//
// Explicit wrapper for *synchronous* allocators (cudaMalloc/cudaFree,
// cudaMallocHost/cudaFreeHost, cudaMallocManaged/cudaFree): composes two 1-arg
// function objects (from device_memory_helper.hpp) into the buffer resource
// concept shape:
//   void* allocate(std::size_t num_bytes, gcxx::StreamView)
//   void  deallocate(void* ptr, gcxx::StreamView)
//
// Synchronous resources are stream-agnostic: on allocate the stream is ignored,
// and on deallocate the stream is Synchronize()'d BEFORE the sync free. That
// sync-before-free is required for correctness when the buffer was used on a
// stream but the underlying allocator is stream-agnostic (cudaMalloc,
// ncclMemAlloc, nvshmem_malloc). Skipping it = use-after-free.
//
// Properties... are the resource's advertised accessibility, exposed via
// `using properties = TypeSet<Properties...>` so the buffer ctor can validate
// (static_assert) that the resource's properties ⊇ the buffer's. No ADL
// get_property.
//
// Async/stream-ordered allocations go through pooled_device_resource, not here.
// ─────────────────────────────────────────────────────────────────────────────
template <typename AllocFn, typename FreeFn, typename... Properties>
class synchronous_resource {
  static_assert(
    std::is_invocable_v<AllocFn, std::size_t>,
    "synchronous_resource requires a 1-arg sync allocator AllocFn(bytes)");
  static_assert(
    std::is_invocable_v<FreeFn, void*>,
    "synchronous_resource requires a 1-arg sync deallocator FreeFn(ptr)");

 public:
  /// Advertised accessibility — read by has_property_v / resource_has_all_v.
  using properties = TypeSet<Properties...>;

  constexpr synchronous_resource() = default;
  constexpr explicit synchronous_resource(AllocFn alloc, FreeFn free_fn)
      : alloc_(alloc), free_(free_fn) {}

  GCXX_FH auto allocate(std::size_t num_bytes, gcxx::StreamView sv) -> void* {
    (void)sv;  // sync alloc: stream intentionally ignored
    return alloc_(num_bytes);
  }

  GCXX_FH auto deallocate(void* ptr, gcxx::StreamView sv) -> void {
    // Ensure pending stream work completes before the stream-agnostic free.
    sv.Synchronize();
    free_(ptr);
  }

  friend bool operator==(const synchronous_resource&,
                         const synchronous_resource&) noexcept {
    return true;
  }
  friend bool operator!=(const synchronous_resource& lhs,
                         const synchronous_resource& rhs) noexcept {
    return !(lhs == rhs);
  }

 private:
  AllocFn alloc_{};
  FreeFn free_{};
};

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif
