// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_DETAILS_TYPE_TRAITS_HPP
#define GCXX_RUNTIME_DETAILS_TYPE_TRAITS_HPP

#include <array>
#include <initializer_list>
#include <memory>
#include <type_traits>


#include <gcxx/internal/prologue.hpp>


GCXX_NAMESPACE_MAIN_DETAILS_BEGIN()

template <class _Tp>
struct type_identity {
  using type = _Tp;
};

template <class _Tp>
using type_identity_t = typename type_identity<_Tp>::type;

// For use in false static asserts to always require and specilization.
template <class... VT>
inline constexpr bool is_always_false_v = false;

// True when every passed type is the same (empty/single-type pack is true).
template <typename T, typename... Ts>
struct all_same : std::bool_constant<(std::is_same_v<T, Ts> && ...)> {};

template <typename T, typename... Ts>
GCXX_CXPR inline bool all_same_v = all_same<T, Ts...>::value;

// True when T matches any of Ts (empty pack is false).
template <typename T, typename... Ts>
inline constexpr bool contains_v = (std::is_same_v<T, Ts> || ...);

// Primary template: not a void function pointer.
template <typename VT>
struct is_void_function_pointer : std::false_type {};

// Specialization for function pointers returning void.
template <typename... Args>
struct is_void_function_pointer<void (*)(Args...)> : std::true_type {};

template <typename VT>
GCXX_CXPR inline bool is_void_function_pointer_v =
  is_void_function_pointer<VT>::value;


template <typename T>
GCXX_CONCEPT is_complete_v = GCXX_REQUIRES_EXPR((T))(sizeof(T));

// std::lib has this from c++20.
template <typename VT>
using remove_cvref_t =
  typename std::remove_cv_t<typename std::remove_reference_t<VT>>;

template <typename>
struct is_std_array : std::false_type {};

template <typename VT, std::size_t N>
struct is_std_array<std::array<VT, N>> : std::true_type {};

template <typename VT>
GCXX_CXPR inline bool is_std_array_v = is_std_array<VT>::value;

// Detection concepts: a missing member SFINAEs out instead of hard-erroring
// in the deduced bodies.

template <class C>
GCXX_CONCEPT has_data_v = GCXX_REQUIRES_EXPR((C), C& c)(c.data());

template <class C>
GCXX_CONCEPT has_const_data_v = GCXX_REQUIRES_EXPR((C), const C& c)(c.data());

template <class C>
GCXX_CONCEPT has_size_v = GCXX_REQUIRES_EXPR((C), const C& c)(c.size());

// One real HD implementation each; the suppression lets host-only containers
// instantiate on the device pass (cf. libcudacxx).

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

template <typename VT>
GCXX_CONCEPT has_size_and_data_v = GCXX_REQUIRES_EXPR((VT), VT& v)(
  gcxx::details_::size(v), gcxx::details_::data(v));

template <typename T>
using remove_pointer_t = std::remove_pointer_t<T>;


template <typename T>
GCXX_CONCEPT has_get_to_pointer_v =
  GCXX_REQUIRES_EXPR((T), T t)(requires(std::is_pointer_v<decltype(t.get())>));


template <typename T>
GCXX_CONCEPT is_pointer_or_has_get_v = std::is_pointer_v<remove_cvref_t<T>> ||
                                       has_get_to_pointer_v<remove_cvref_t<T>>;

// Primary template: not a pointer and no .get().
template <typename T, typename = void>
struct pointed_to_type;

// Specialization for raw pointers.
template <typename T>
struct pointed_to_type<T*> {
  using type = T;
};

// Specialization for types with .get() that returns a pointer.
template <typename T>
struct pointed_to_type<T, std::void_t<decltype(std::declval<T>().get())>> {
  using type = std::remove_pointer_t<decltype(std::declval<T>().get())>;
};

template <typename T>
using pointed_to_type_t = typename pointed_to_type<remove_cvref_t<T>>::type;

template <typename VT>
GCXX_FH auto get_raw_pointer(VT* ptr) {
  return ptr;
}

template <typename VT>
GCXX_FH auto get_raw_pointer(VT&& ptr) -> decltype(ptr.get()) {
  return ptr.get();
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


// Helper to check if T& is a valid type (T is not void, basically).
template <class T>
using with_reference = T&;

// In C++17 we use void_t + SFINAE instead of concepts.
template <class T, class = void>
struct can_reference_impl : std::false_type {};

template <class T>
struct can_reference_impl<T, std::void_t<with_reference<T>>> : std::true_type {
};

template <class T>
constexpr bool can_reference = can_reference_impl<T>::value;

// Check that *t yields a referenceable type.
template <class T, class = void>
struct dereferenceable_impl : std::false_type {};

template <class T>
struct dereferenceable_impl<T, std::void_t<decltype(*std::declval<T&>())>>
    : std::bool_constant<can_reference<decltype(*std::declval<T&>())>> {};

// iter_reference_t — only participates if T is dereferenceable.
GCXX_TEMPLATE(class T)
GCXX_REQUIRES(dereferenceable_impl<T>::value)
using iter_reference_t = decltype(*std::declval<T&>());

// Shared guard question: does U(*)[] convert to ET(*)[]?
template <typename U, typename ET>
struct is_ptr_array_convertible
    : std::bool_constant<std::is_convertible_v<U (*)[], ET (*)[]>> {};

template <typename U, typename ET>
GCXX_CXPR inline bool is_ptr_array_convertible_v =
  is_ptr_array_convertible<U, ET>::value;

// Mirrors std::span's SFINAE guards; each trait deduces U from its source.

// ① Constructors 2 & 3 — iterator pair / iterator + size.
template <typename It, typename ET, typename = void>
struct is_iter_ptr_convertible : std::false_type {};

template <typename It, typename ET>
struct is_iter_ptr_convertible<
  It, ET, std::void_t<std::remove_reference_t<iter_reference_t<It>>>>
    : is_ptr_array_convertible<
        std::remove_reference_t<std::remove_reference_t<iter_reference_t<It>>>,
        ET> {};

template <typename It, typename ET>
GCXX_CXPR inline bool is_iter_ptr_convertible_v =
  is_iter_ptr_convertible<It, ET>::value;

// ② Constructors 4, 5 & 6 — C-array / std::array.
template <typename Arr, typename ET, typename = void>
struct is_data_ptr_convertible : std::false_type {};

template <typename Arr, typename ET>
struct is_data_ptr_convertible<
  Arr, ET, std::void_t<decltype(gcxx::details_::data(std::declval<Arr&>()))>>
    : is_ptr_array_convertible<
        std::remove_pointer_t<decltype(gcxx::details_::data(
          std::declval<Arr&>()))>,
        ET> {};

template <typename Arr, typename ET>
GCXX_CXPR inline bool is_data_ptr_convertible_v =
  is_data_ptr_convertible<Arr, ET>::value;

// ③ Constructor 7 — range.
template <typename R, typename ET, typename = void>
struct is_range_ptr_convertible : std::false_type {};

template <typename R, typename ET>
struct is_range_ptr_convertible<
  R, ET, std::void_t<decltype(*std::begin(std::declval<R&>()))>>
    : is_ptr_array_convertible<
        std::remove_reference_t<decltype(*std::begin(std::declval<R&>()))>,
        ET> {};

template <typename R, typename ET>
GCXX_CXPR inline bool is_range_ptr_convertible_v =
  is_range_ptr_convertible<R, ET>::value;

// ④ Constructor 9 — converting span constructor (U passed directly).
template <typename U, typename ET>
using is_type_ptr_convertible = is_ptr_array_convertible<U, ET>;

template <typename U, typename ET>
GCXX_CXPR inline bool is_type_ptr_convertible_v =
  is_type_ptr_convertible<U, ET>::value;


GCXX_NAMESPACE_MAIN_DETAILS_END()


#endif