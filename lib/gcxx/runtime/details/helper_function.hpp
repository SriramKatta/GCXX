#pragma once
#ifndef GCXX_RUNTIME_DETAILS_HELPER_FUNCTION_HPP
#define GCXX_RUNTIME_DETAILS_HELPER_FUNCTION_HPP

#include <type_traits>


#include <gcxx/internal/prologue.hpp>


GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █            Impl of std::size and std::data             █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
// TODO : make a snesible implementation for both host and device compatibility

template <class C>
GCXX_FHDC auto data(C& c) -> decltype(c.data()) {
#ifdef GCXX_DEVICE_COMPILE
  return nullptr;
#else
  return c.data();
#endif
}

template <class C>
GCXX_FHDC auto data(const C& c) -> decltype(c.data()) {
#ifdef GCXX_DEVICE_COMPILE
  return nullptr;
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
GCXX_FHDC auto size(const C& c) -> decltype(c.size()) {
#ifdef GCXX_DEVICE_COMPILE
  return 0;
#else
  return c.size();
#endif
}

template <class T, std::size_t N>
GCXX_FHDC std::size_t size(const T (&)[N]) noexcept {  // NOLINT
  return N;
}

// TODO : C++ 20 has this implemented so have the conditional compilation
//  ---- detection for pointer_traits<T>::to_address ----
template <class T, class = void>
struct has_ptr_traits_to_address : std::false_type {};

template <class T>
struct has_ptr_traits_to_address<
  T, std::void_t<decltype(std::pointer_traits<T>::to_address(
       std::declval<const T&>()))>> : std::true_type {};

// ---- overload 1: raw pointers ----
template <class T>
GCXX_FHDC T* to_address(T* p) noexcept {
  static_assert(!std::is_function_v<T>);
  return p;
}

// ---- overload 2: fancy pointers ----
template <class T>
GCXX_FHDC auto to_address(const T& p) noexcept {
  if constexpr (has_ptr_traits_to_address<T>::value)
    return std::pointer_traits<T>::to_address(p);
  else
    return to_address(p.operator->());  // recurse, not std::
}

GCXX_NAMESPACE_MAIN_DETAILS_END


#endif