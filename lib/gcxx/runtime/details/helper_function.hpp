// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_HELPER_FUNCTION_HPP
#define GCXX_RUNTIME_DETAILS_HELPER_FUNCTION_HPP

#include <type_traits>


#include <gcxx/internal/prologue.hpp>


GCXX_NAMESPACE_MAIN_DETAILS_BEGIN()

// One real HD implementation each; the suppression lets host-only containers
// instantiate on the device pass (cf. libcudacxx).

// Detection concepts: a missing member SFINAEs out instead of hard-erroring
// in the deduced bodies.
template <class C>
GCXX_CONCEPT has_data_v = GCXX_REQUIRES_EXPR((C), C& c)(c.data());

template <class C>
GCXX_CONCEPT has_const_data_v = GCXX_REQUIRES_EXPR((C), const C& c)(c.data());

template <class C>
GCXX_CONCEPT has_size_v = GCXX_REQUIRES_EXPR((C), const C& c)(c.size());

GCXX_TEMPLATE(typename C)
GCXX_REQUIRES(has_data_v<C>)
GCXX_FHDC auto data(C& c) {
  GCXX_DIAG_SUPPRESS_EXEC_CHECK_(20011)
  GCXX_DIAG_SUPPRESS_EXEC_CHECK_(20013) GCXX_DIAG_SUPPRESS_EXEC_CHECK_(20014)
    GCXX_DIAG_SUPPRESS_EXEC_CHECK_(20015) return c.data();
  GCXX_DIAG_RESTORE_EXEC_CHECK_(20011)
  GCXX_DIAG_RESTORE_EXEC_CHECK_(20013) GCXX_DIAG_RESTORE_EXEC_CHECK_(20014)
    GCXX_DIAG_RESTORE_EXEC_CHECK_(20015)
}

GCXX_TEMPLATE(typename C)
GCXX_REQUIRES(has_const_data_v<C>)
GCXX_FHDC auto data(const C& c) {
  GCXX_DIAG_SUPPRESS_EXEC_CHECK_(20011)
  GCXX_DIAG_SUPPRESS_EXEC_CHECK_(20013) GCXX_DIAG_SUPPRESS_EXEC_CHECK_(20014)
    GCXX_DIAG_SUPPRESS_EXEC_CHECK_(20015) return c.data();
  GCXX_DIAG_RESTORE_EXEC_CHECK_(20011)
  GCXX_DIAG_RESTORE_EXEC_CHECK_(20013) GCXX_DIAG_RESTORE_EXEC_CHECK_(20014)
    GCXX_DIAG_RESTORE_EXEC_CHECK_(20015)
}

template <class T, std::size_t N>
GCXX_FHDC T* data(T (&array)[N]) noexcept {  // NOLINT
  return array;
}

template <class E>
GCXX_FHDC const E* data(std::initializer_list<E> il) noexcept {
  return il.begin();
}

GCXX_TEMPLATE(typename C)
GCXX_REQUIRES(has_size_v<C>)
GCXX_FHDC auto size(const C& c) {
  GCXX_DIAG_SUPPRESS_EXEC_CHECK_(20011)
  GCXX_DIAG_SUPPRESS_EXEC_CHECK_(20013) GCXX_DIAG_SUPPRESS_EXEC_CHECK_(20014)
    GCXX_DIAG_SUPPRESS_EXEC_CHECK_(20015) return c.size();
  GCXX_DIAG_RESTORE_EXEC_CHECK_(20011)
  GCXX_DIAG_RESTORE_EXEC_CHECK_(20013) GCXX_DIAG_RESTORE_EXEC_CHECK_(20014)
    GCXX_DIAG_RESTORE_EXEC_CHECK_(20015)
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
  GCXX_DIAG_SUPPRESS_EXEC_CHECK_(20011)
  GCXX_DIAG_SUPPRESS_EXEC_CHECK_(20013) GCXX_DIAG_SUPPRESS_EXEC_CHECK_(20014)
    GCXX_DIAG_SUPPRESS_EXEC_CHECK_(
      20015) if constexpr (has_ptr_traits_to_address_v<T>) return std::
      pointer_traits<T>::to_address(p);
  else {
    return to_address(p.operator->());  // recurse, not std::
  }
  GCXX_DIAG_RESTORE_EXEC_CHECK_(20011)
  GCXX_DIAG_RESTORE_EXEC_CHECK_(20013) GCXX_DIAG_RESTORE_EXEC_CHECK_(20014)
    GCXX_DIAG_RESTORE_EXEC_CHECK_(20015)
}

GCXX_NAMESPACE_MAIN_DETAILS_END()


#endif