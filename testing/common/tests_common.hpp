// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_TESTING_COMMON_TESTING_COMMON_HPP
#define GCXX_TESTING_COMMON_TESTING_COMMON_HPP

#include <gtest/gtest.h>

#include <cstdlib>
#include <type_traits>

#include <gcxx/api.hpp>
#include <gcxx/macros/template_helper_macros.hpp>

// SFINAE detector (n4502); param pack kept LAST for NVCC's EDG frontend.
#define GCXX_DEFINE_IS_CALLABLE(Name, ...)                               \
  template <typename... Args>                                            \
  using Name##_detail = decltype(__VA_ARGS__);                           \
  template <template <typename...> class Op, typename, typename... Args> \
  struct Name##_detector : std::false_type {};                           \
  template <template <typename...> class Op, typename... Args>           \
  struct Name##_detector<Op, std::void_t<Op<Args...>>, Args...>          \
      : std::true_type {};                                               \
  template <typename... VT>                                              \
  struct Name : Name##_detector<Name##_detail, void, VT...> {};          \
  template <typename... VT>                                              \
  static constexpr bool Name##_v = Name<VT...>::value;

// True iff T exposes a public nested `raw_handle_type` typedef.
template <class T>
GCXX_CONCEPT has_raw_handle_type_v =
  GCXX_REQUIRES_EXPR((T))(sizeof(typename T::raw_handle_type));

#define GCXX_ASSERT_RAW_HANDLE(WRAPPER, EXPECTED)                         \
  static_assert(has_raw_handle_type_v<gcxx::WRAPPER>,                     \
                #WRAPPER " must expose ::raw_handle_type");               \
  static_assert(std::is_same_v<gcxx::WRAPPER::raw_handle_type, EXPECTED>, \
                #WRAPPER "::raw_handle_type must be " #EXPECTED)

// Skips the enclosing test when no GPU device is visible (e.g. CI runners
// without a GPU). Device::available() is a non-throwing probe; Device::count()
// would abort in this situation.
#define GCXX_SKIP_WITHOUT_DEVICE()               \
  do {                                           \
    if (!gcxx::Device::available()) {            \
      GTEST_SKIP() << "No GPU device available"; \
    }                                            \
  } while (false)


// Default device pool for staging allocations in GPU-gated tests (one
// definition shared across test TUs; never instantiated unless called).
inline auto dev_pool() {
  return gcxx::device_default_memory_pool(gcxx::DeviceHandle{0});
}

// Malloc-backed host resource: the pageable leg of the allocation-kind
// bijection (host_accessible + host_allocated). Shared by buffer/scalar
// tests; the kind tag is a superset of the old host-only mock's properties,
// so buffer<VT, host_accessible> construction checks still pass.
struct host_mock_resource {
  void* allocate(gcxx::StreamView, std::size_t num_bytes) {
    return std::malloc(num_bytes);
  }

  void deallocate(gcxx::StreamView, void* ptr) { std::free(ptr); }

  using properties = gcxx::TypeSet<gcxx::host_accessible, gcxx::host_allocated>;
};

#endif