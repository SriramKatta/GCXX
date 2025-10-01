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

GCXX_DETAILS_BEGIN_NAMESPACE

template <typename VT, std::size_t S>
struct span_storage {
  using pointer                     = VT*;
  constexpr span_storage() noexcept = default;

  constexpr span_storage(VT* v_ptr, std::size_t) noexcept : ptr(v_ptr) {}

  VT* ptr{nullptr};
  static constexpr std::size_t size = S;
};

template <typename VT>
struct span_storage<VT, dynamic_extent> {
  constexpr span_storage() noexcept = default;

  constexpr span_storage(VT* v_ptr, std::size_t v_size) noexcept
      : ptr(v_ptr), size(v_size) {}

  VT* ptr{nullptr};
  std::size_t size{0};
};

GCXX_DETAILS_END_NAMESPACE

template <class VT, std::size_t Extent = dynamic_extent>
class span {
 private:
  // █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
  // █                     Static Asserts                     █
  // █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
  static_assert(std::is_object<VT>::value,
                "An refrence is not supported"
                " need an fully declared type");
  static_assert(detail::is_complete<VT>::value,
                "A forward declaration is not supported"
                " need an fully declared type");
  static_assert(!std::is_abstract<VT>::value,
                "An abstract class type is not supported");
  using storage_type = details_::span_storage<VT, Extent>;

 public:
  // █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
  // █                      Nested Types                      █
  // █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
  using element_type    = VT;
  using value_type      = std::remove_cv_t<element_type>;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer         = element_type*;
  using const_pointer   = const pointer;
  using reference       = element_type&;
  using const_reference = const element_type&;
  using iterator        = pointer;  // dont assume this to be T* in your code
                                    // maybe changed to a bidirectional iterator
  using reverse_iterator = std::reverse_iterator<iterator>;
  // █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
  // █                    Memeber Function                    █
  // █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
  // ==========================================================
  //                        Constructors
  // ==========================================================
  GCXX_CXPR span() GCXX_NOEXCEPT = default;

  // ==========================================================
  //                         operator =
  // ==========================================================

  // ==========================================================
  //                         Iterators
  // ==========================================================

  // ==========================================================
  //                       Element access
  // ==========================================================

  // ==========================================================
  //                         Observers
  // ==========================================================

  // ==========================================================
  //                          subviews
  // ==========================================================


  // █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
  // █                      Data members                      █
  // █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
 public:
  static constexpr size_type extent = Extent;

 private:
  storage_type storage{};
};

GCXX_END_NAMESPACE


#endif