// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
//
// Phase 2 coverage for synchronous_resource and the resource aliases.
//
//   * Compile-time: static_asserts confirm each alias resolves to the right
//     synchronous_resource instantiation, and the allocator function objects
//     are 1-arg sync (is_invocable_v). synchronous_resource itself
//     static_asserts 1-arg AllocFn/FreeFn, so a 2-arg async allocator is
//     rejected at compile time (async/stream-ordered allocation goes through
//     pooled_device_resource).
//   * Runtime: a mock 1-arg alloc function object lets us count calls and
//     exercise allocate() without a GPU. deallocate()'s sync-before-free calls
//     sv.Synchronize(), which needs a real CUDA context, so it is not
//     runtime-tested here.
//   * Equality: stateless resources compare equal.
#include "tests_common.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

#include <gcxx/api.hpp>

// =============================================================================
// Mocks: count calls so we can assert dispatch behavior without a GPU.
// =============================================================================
namespace {

  // Sentinel "allocation" returned by the mock — non-null so callers see a
  // valid pointer; never dereferenced.
  inline void* mock_alloc_sentinel() {
    // NOLINTNEXTLINE: reinterpret an integer to a pointer to avoid real memory.
    return reinterpret_cast<void*>(0xDEADBEEF);
  }

  // 1-arg sync alloc — what synchronous_resource expects.
  struct mock_sync_alloc_t {
    int* call_count_;
    explicit mock_sync_alloc_t(int* c = nullptr) : call_count_(c) {}
    void* operator()(std::size_t) const {
      if (call_count_)
        ++*call_count_;
      return mock_alloc_sentinel();
    }
  };

}  // namespace

// =============================================================================
// Compile-time: each alias resolves to synchronous_resource<...>.
// =============================================================================
TEST(SynchronousResourceAliasTest,
     AliasesAreSynchronousResourceInstantiations) {
  using gcxx::memory::device_accessible;
  using gcxx::memory::host_accessible;
  using gcxx::memory::managed_device_resource;
  using gcxx::memory::sync_device_resource;
  using gcxx::memory::sync_host_resource;
  using gcxx::memory::synchronous_resource;

  static_assert(
    std::is_same_v<
      sync_device_resource,
      synchronous_resource<gcxx::details_::device_malloc_t,
                           gcxx::details_::device_free_t, device_accessible>>);
  static_assert(
    std::is_same_v<
      sync_host_resource,
      synchronous_resource<gcxx::details_::host_malloc_t,
                           gcxx::details_::host_free_t, host_accessible>>);
  static_assert(
    std::is_same_v<managed_device_resource,
                   synchronous_resource<gcxx::details_::device_managed_malloc_t,
                                        gcxx::details_::device_free_t,
                                        device_accessible, host_accessible>>);
}

// =============================================================================
// Compile-time: the sync allocator function objects are 1-arg (synchronous).
// async_device_resource (cudaMallocAsync-direct) is intentionally dropped —
// async/stream-ordered allocation uses pooled_device_resource.
// =============================================================================
TEST(SynchronousResourceDispatchTest, SyncFunctionObjectsAreOneArg) {
  using namespace gcxx::details_;
  static_assert(std::is_invocable_v<device_malloc_t, std::size_t>);
  static_assert(std::is_invocable_v<device_free_t, void*>);
  static_assert(std::is_invocable_v<host_malloc_t, std::size_t>);
  static_assert(std::is_invocable_v<host_free_t, void*>);
  static_assert(std::is_invocable_v<device_managed_malloc_t, std::size_t>);
}

// =============================================================================
// Equality: stateless resources compare equal.
// =============================================================================
TEST(SynchronousResourceEqualityTest, StatelessResourcesAreEqual) {
  using gcxx::memory::sync_device_resource;
  sync_device_resource a{};
  sync_device_resource b{};
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a != b);
}

// =============================================================================
// Runtime: allocate() dispatches to the 1-arg sync alloc function object.
// (deallocate's sync-before-free calls sv.Synchronize() — needs a CUDA context,
// so not runtime-tested here.)
// =============================================================================
TEST(SynchronousResourceDispatchTest, AllocateCallsSyncAllocator) {
  using mock_resource =
    gcxx::memory::synchronous_resource<mock_sync_alloc_t,
                                       gcxx::details_::device_free_t>;

  int alloc_calls = 0;
  mock_resource res{mock_sync_alloc_t{&alloc_calls},
                    gcxx::details_::device_free_t{}};

  void* p = res.allocate(std::size_t{16}, gcxx::StreamView::Null());
  EXPECT_EQ(alloc_calls, 1);
  EXPECT_NE(p, nullptr);
}

// =============================================================================
// Sanity: the buffer aliases still resolve through resources.hpp.
// =============================================================================
TEST(SynchronousResourceAliasTest, DeviceBufferAliasResolves) {
  using gcxx::memory::device_accessible;
  using gcxx::memory::device_buffer;
  static_assert(std::is_same_v<device_buffer<int>,
                               gcxx::memory::buffer<int, device_accessible>>);
}

TEST(SynchronousResourceAliasTest, HostBufferAliasResolves) {
  using gcxx::memory::host_accessible;
  using gcxx::memory::host_buffer;
  static_assert(std::is_same_v<host_buffer<int>,
                               gcxx::memory::buffer<int, host_accessible>>);
}
