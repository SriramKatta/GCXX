#pragma once
#ifndef GCXX_RUNTIME_SPAN_SPAN_HPP
#define GCXX_RUNTIME_SPAN_SPAN_HPP


#include <iterator>
#include <limits>
#include <type_traits>

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>


GCXX_BEGIN_NAMESPACE

inline constexpr std::size_t dynamic_extent =
  std::numeric_limits<std::size_t>::max();

template <class T, std::size_t Extent = gcxx::dynamic_extent>
class span {
  using element_type    = T;
  using value_type      = std::remove_cv_t<T>;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer         = T*;
  using const_pointer   = const T*;
  using reference       = T&;
  using const_reference = const T&;
  using iterator        = pointer;  // dont assume this to be T* in your code
                                    // maybe changed to a bidirectional iterator
  using reverse_iterator = std::reverse_iterator<iterator>;
};

GCXX_END_NAMESPACE


#endif