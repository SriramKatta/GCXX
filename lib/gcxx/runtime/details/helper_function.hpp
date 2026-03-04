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
GCXX_FHC auto data(C& c) -> decltype(c.data()) {
  return c.data();
}

template <class C>
GCXX_FHC auto data(const C& c) -> decltype(c.data()) {
  return c.data();
}

template <class T, std::size_t N>
GCXX_FHC T* data(T (&array)[N]) noexcept {  // NOLINT
  return array;
}

template <class E>
GCXX_FHC const E* data(std::initializer_list<E> il) noexcept {
  return il.begin();
}

template <class C>
GCXX_FHC auto size(const C& c) -> decltype(c.size()) {
  return c.size();
}

template <class T, std::size_t N>
GCXX_FHC std::size_t size(const T (&)[N]) noexcept {  // NOLINT
  return N;
}

GCXX_NAMESPACE_MAIN_DETAILS_END


#endif