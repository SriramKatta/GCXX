#pragma once
#ifndef GCXX_RUNTIME_DETAILS_HELPER_FUNCTION_HPP
#define GCXX_RUNTIME_DETAILS_HELPER_FUNCTION_HPP

#include <type_traits>


#include <gcxx/macros/define_macros.hpp>


GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █            Impl of std::size and std::data             █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
template <class C>
GCXX_FHDC auto data(C& c) -> decltype(c.data()) {
  return c.data();
}

template <class C>
GCXX_FHDC auto data(const C& c) -> decltype(c.data()) {
  return c.data();
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
  return c.size();
}

template <class T, std::size_t N>
GCXX_FHDC std::size_t size(const T (&)[N]) noexcept {  // NOLINT
  return N;
}

GCXX_NAMESPACE_MAIN_DETAILS_END


#endif