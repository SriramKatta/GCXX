// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_MEMORY_RESOURCE_BASIC_RESOURCE_HPP_
#define GCXX_RUNTIME_MEMORY_MEMORY_RESOURCE_BASIC_RESOURCE_HPP_

#include <cstddef>
#include <type_traits>
#include <utility>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/memory/device_memory_helper.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_MEMORY_BEGIN()

// ─────────────────────────────────────────────────────────────────────────────
// gcxx::memory::basic_resource<AllocFn, FreeFn>
//
// Resource template that composes two function objects (from
// device_memory_helper.hpp) into the buffer<VT, Resource> concept shape:
//   void* allocate(std::size_t num_bytes, gcxx::StreamView)
//   void  deallocate(void* ptr, gcxx::StreamView)
//
// Sync vs async dispatch is compile-time, driven by the function object's
// arity:
//   * 1-arg AllocFn / FreeFn (e.g. device_malloc_t, device_free_t) — sync.
//     Stream is IGNORED on allocate, and Synchronize()'d BEFORE the sync free.
//     The sync-before-free is required for correctness when the buffer was
//     used on a stream but the underlying allocator is stream-agnostic
//     (cudaMalloc, ncclMemAlloc, nvshmem_malloc). Skipping it = use-after-free.
//   * 2-arg AllocFn / FreeFn taking (size, StreamView) / (ptr, StreamView) —
//     async. Stream is threaded through to the underlying driver call, which
//     handles ordering natively (no extra sync needed).
//
// The 3 forwarder resource classes this replaces (sync_device_resource,
// sync_host_resource, async_device_resource) were 1:1 wrappers over these
// same function objects; this template deletes ~90 LOC of boilerplate and
// makes adding NCCL / NVSHMEM resources a 10-line job (see plan T2).
//
// ponytail: sync-before-free adds a stream sync that the old per-resource
// implementations did not have. Correct-but-slower for sync_device_resource
// and sync_host_resource users; required for nccl/nvshmem safety. If profiling
// shows the extra sync hurts, the upgrade path is an opt-out trait — but
// don't add it preemptively.
// ─────────────────────────────────────────────────────────────────────────────
template <typename AllocFn, typename FreeFn>
class basic_resource {
 public:
  constexpr basic_resource() = default;
  constexpr explicit basic_resource(AllocFn alloc, FreeFn free_fn)
      : alloc_(alloc), free_(free_fn) {}

  GCXX_FH auto allocate(std::size_t num_bytes, gcxx::StreamView sv) -> void* {
    if constexpr (std::is_invocable_v<AllocFn, std::size_t>) {
      (void)sv;  // sync alloc: stream intentionally ignored
      return alloc_(num_bytes);
    } else {
      return alloc_(num_bytes, sv);
    }
  }

  GCXX_FH auto deallocate(void* ptr, gcxx::StreamView sv) -> void {
    if constexpr (std::is_invocable_v<FreeFn, void*>) {
      // Sync free: ensure pending stream work completes before the
      // stream-agnostic allocator returns the memory.
      sv.Synchronize();
      free_(ptr);
    } else {
      free_(ptr, sv);
    }
  }

  friend bool operator==(const basic_resource&, const basic_resource&) noexcept {
    return true;
  }
  friend bool operator!=(const basic_resource& lhs,
                         const basic_resource& rhs) noexcept {
    return !(lhs == rhs);
  }

 private:
  AllocFn alloc_{};
  FreeFn free_{};
};

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif
