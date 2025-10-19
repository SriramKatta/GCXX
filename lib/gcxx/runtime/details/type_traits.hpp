#pragma once
#ifndef GCXX_RUNTIME_DETAILS_TYPE_TRAITS_HPP
#define GCXX_RUNTIME_DETAILS_TYPE_TRAITS_HPP

#include <type_traits>


#include <gcxx/macros/define_macros.hpp>


GCXX_NAMESPACE_MAIN_DETAILS_BEGIN

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █                   Useful Type Traits                   █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
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
struct has_size_and_data<VT, std::void_t<decltype(size(std::declval<VT>())),
                                         decltype(data(std::declval<VT>()))>>
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
      remove_pointer_t<decltype(data(std::declval<T>()))> (*)[], E (*)[]>>>
    : std::true_type {};

template <typename VT, typename ET>
GCXX_CXPR inline bool is_container_element_type_compatible_v =
  is_container_element_type_compatible<VT, ET>::value;


GCXX_NAMESPACE_MAIN_DETAILS_END


#endif