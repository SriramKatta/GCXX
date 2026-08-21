// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_HELPER_FUNCTION_HPP
#define GCXX_RUNTIME_DETAILS_HELPER_FUNCTION_HPP

#include <type_traits>


#include <gcxx/internal/prologue.hpp>


GCXX_NAMESPACE_MAIN_DETAILS_BEGIN()

// TODO: Sensible implementations for both host and device compatibility

template <class C>
GCXX_FHDC auto data(C& c) {
#if GCXX_DEVICE_COMPILE
  using T = decltype(c.data());
  return static_cast<T>(nullptr);
#else
  return c.data();
#endif
}

template <class C>
GCXX_FHDC auto data(const C& c) {
#if GCXX_DEVICE_COMPILE
  using T = decltype(c.data());
  return static_cast<T>(nullptr);
#else
  return c.data();
#endif
}

template <class T, std::size_t N>
GCXX_FHDC T* data(T (&array)[N]) noexcept {  // NOLINT
  return array;
}

template <class E>
GCXX_FHDC const E* data(std::initializer_list<E> il) noexcept {
  return il.begin();
}

template <class C>
GCXX_FHC auto size(const C& c) {
  return c.size();
}

template <class T, std::size_t N>
GCXX_FHDC std::size_t size(const T (&)[N]) noexcept {  // NOLINT
  return N;
}

// TODO: C++20 has pointer_traits<T>::to_address detection built in.
template <class T>
GCXX_CONCEPT has_ptr_traits_to_address_v =
  GCXX_REQUIRES_EXPR((T), const T& p)(std::pointer_traits<T>::to_address(p));

template <class T>
GCXX_FHDC T* to_address(T* p) noexcept {
  static_assert(!std::is_function_v<T>);
  return p;
}

template <class T>
GCXX_FHDC auto to_address(const T& p) noexcept {
  if constexpr (has_ptr_traits_to_address_v<T>)
    return std::pointer_traits<T>::to_address(p);
  else {
#if GCXX_DEVICE_COMPILE  // TODO: Implement properly.
    return nullptr;
#else
    return to_address(p.operator->());  // recurse, not std::
#endif
  }
}

GCXX_NAMESPACE_MAIN_DETAILS_END()


#endif