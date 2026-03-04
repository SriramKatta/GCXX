#pragma once
#ifndef GCXX_RUNTIME_DETAILS_TYPE_TRAITS_HPP
#define GCXX_RUNTIME_DETAILS_TYPE_TRAITS_HPP

#include <array>
#include <type_traits>


#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/helper_function.hpp>


GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

template <class _Tp>
struct type_identity {
  using type = _Tp;
};

template <class _Tp>
using type_identity_t = typename type_identity<_Tp>::type;

template <class VT>
inline constexpr bool is_always_false_v = false;

// Primary template: not a void function pointer
template <typename VT>
struct is_void_function_pointer : std::false_type {};

// Specialization for function pointers returning void
template <typename... Args>
struct is_void_function_pointer<void (*)(Args...)> : std::true_type {};

template <typename VT>
GCXX_CXPR inline bool is_void_function_pointer_v =
  is_void_function_pointer<VT>::value;

template <typename, typename = size_t>
struct is_complete : std::false_type {};

template <typename T>
struct is_complete<T, decltype(sizeof(T))> : std::true_type {};

template <typename T>
inline constexpr bool is_complete_v = is_complete<T>::value;

template <typename VT>
using uncvref_t =
  typename std::remove_cv_t<typename std::remove_reference_t<VT>>;

template <typename>
struct is_std_array : std::false_type {};

template <typename VT, std::size_t N>
struct is_std_array<std::array<VT, N>> : std::true_type {};

template <typename VT>
GCXX_CXPR inline bool is_std_array_v = is_std_array<VT>::value;

template <typename, typename = void>
struct has_size_and_data : std::false_type {};

template <typename VT>
struct has_size_and_data<
  VT, std::void_t<decltype(gcxx::details_::size(std::declval<VT&>())),
                  decltype(gcxx::details_::data(std::declval<VT&>()))>>
    : std::true_type {};

template <typename VT>
GCXX_CXPR inline bool has_size_and_data_v = has_size_and_data<VT>::value;

template <typename T>
using remove_pointer_t = typename std::remove_pointer<T>::type;

template <typename, typename, typename = void>
struct is_container_element_type_compatible : std::false_type {};

template <typename T, typename E>
struct is_container_element_type_compatible<
  T, E,
  typename std::enable_if<
    !std::is_same_v<
      typename std::remove_cv_t<decltype(data(std::declval<T>()))>::type,
      void> &&
    std::is_convertible_v<
      remove_pointer_t<decltype(data(std::declval<T>()))> (*)[],  // NOLINT
      E (*)[]>>>                                                  // NOLINT
    : std::true_type {};

template <typename VT, typename ET>
GCXX_CXPR inline bool is_container_element_type_compatible_v =
  is_container_element_type_compatible<VT, ET>::value;

// Checks whether the element type of an iterator It is compatible with ET.
// Unlike is_container_element_type_compatible, this works on raw pointers and
// any dereferenceable type (not just containers with data()).
template <typename It, typename ET, typename = void>
struct is_iter_element_type_compatible : std::false_type {};

template <typename It, typename ET>
struct is_iter_element_type_compatible<
  It, ET, std::void_t<decltype(*std::declval<It&>())>>
    : std::bool_constant<std::is_convertible_v<
        std::remove_reference_t<decltype(*std::declval<It&>())> (*)[],
        ET (*)[]>> {};

template <typename It, typename ET>
GCXX_CXPR inline bool is_iter_element_type_compatible_v =
  is_iter_element_type_compatible<It, ET>::value;


// TODO : add a condition compilation since this is avialble in c++20
// Helper to check if T& is a valid type (T is not void, basically)
template <class T>
using with_reference = T&;

// In C++17 we use void_t + SFINAE instead of concepts
template <class T, class = void>
struct can_reference_impl : std::false_type {};

template <class T>
struct can_reference_impl<T, std::void_t<with_reference<T>>> : std::true_type {
};

template <class T>
constexpr bool can_reference = can_reference_impl<T>::value;

// Check that *t yields a referenceable type
template <class T, class = void>
struct dereferenceable_impl : std::false_type {};

template <class T>
struct dereferenceable_impl<T, std::void_t<decltype(*std::declval<T&>())>>
    : std::bool_constant<can_reference<decltype(*std::declval<T&>())>> {};

// iter_reference_t — only participates if T is dereferenceable
GCXX_TEMPLATE(class T)
GCXX_REQUIRES(dereferenceable_impl<T>::value)
using iter_reference_t = decltype(*std::declval<T&>());


GCXX_NAMESPACE_MAIN_DETAILS_END


#endif