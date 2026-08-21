// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_TESTING_COMMON_TESTING_COMMON_HPP
#define GCXX_TESTING_COMMON_TESTING_COMMON_HPP

#include <gtest/gtest.h>

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

namespace gcxx::testing {

  // Raw backend probe; Device::count() can abort when exceptions are off.
  inline auto haveCudaDevice() -> bool {
    int count      = 0;
    const auto err = ::GCXX_RUNTIME_BACKEND(GetDeviceCount)(&count);
    return err == gcxx::driver::deviceErrSuccess && count > 0;
  }

}  // namespace gcxx::testing

#endif