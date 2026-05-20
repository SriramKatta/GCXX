// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_SPANS_SPAN_SPAN_HPP
#define GCXX_RUNTIME_MEMORY_SPANS_SPAN_SPAN_HPP


#include <array>
#include <iterator>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/helper_function.hpp>
#include <gcxx/runtime/details/type_traits.hpp>

/*
https://godbolt.org/z/qb5vq7sh9
TODO : use this as refrence to extend and support restrict span keeping most of
the code same
*/

GCXX_NAMESPACE_MAIN_BEGIN()

template <class VT, std::size_t Extent>
class span;

template <class VT, std::size_t Extent>
class restrict_span;

// since mdspan from kokkos has this already we dont need it here
// GCXX_CXPR inline auto dynamic_extent =
//   std::numeric_limits<std::size_t>::max();

GCXX_NAMESPACE_DETAILS_BEGIN()

// ╔════════════════════════════════════════════════════════╗
// ║               SpanLike detection traits                ║
// ╚════════════════════════════════════════════════════════╝

// Primary: not span-like.
template <typename T, typename = void>
struct is_span_like_impl : std::false_type {};

template <typename T>
struct is_gcxx_span_specialization : std::false_type {};

template <typename T, std::size_t Extent>
struct is_gcxx_span_specialization<gcxx::span<T, Extent>> : std::true_type {};

template <typename T, std::size_t Extent>
struct is_gcxx_span_specialization<gcxx::restrict_span<T, Extent>>
    : std::true_type {};

template <typename T>
GCXX_CXPR inline bool is_gcxx_span_specialization_v =
  is_gcxx_span_specialization<uncvref_t<T>>::value;

// Specialisation: T has .data() → pointer and .size() → integral.
template <typename T>
struct is_span_like_impl<
  T, std::void_t<decltype(gcxx::details_::data(std::declval<T&>())),
                 decltype(gcxx::details_::size(std::declval<T&>()))>>
    : std::bool_constant<
        std::is_pointer_v<decltype(gcxx::details_::data(std::declval<T&>()))> &&
        std::is_integral_v<decltype(gcxx::details_::size(
          std::declval<T&>()))>> {};

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █                      Span Storage                      █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█

template <std::size_t Extent>
struct size_holder {
  GCXX_FHDC size_holder() noexcept = default;

  GCXX_FHDC explicit size_holder(std::size_t) noexcept {}

  static GCXX_FHDC std::size_t size() noexcept { return Extent; }
};

template <>
struct size_holder<gcxx::dynamic_extent> {
  std::size_t m_size{0};

  size_holder() noexcept = default;

  GCXX_FHDC explicit size_holder(std::size_t n) noexcept : m_size(n) {}

  GCXX_FHDC std::size_t size() const noexcept { return m_size; }
};

template <typename VT, std::size_t Extent>
struct span_storage : size_holder<Extent> {
  using data_handle_type = VT*;
  using size_holder<Extent>::size;

  span_storage() noexcept = default;

  GCXX_FHDC span_storage(VT* v_ptr, std::size_t n) noexcept
      : size_holder<Extent>(n), start(v_ptr) {}

  GCXX_FHDC data_handle_type data() const {
    data_handle_type ptr = start;
    return ptr;
  }

  GCXX_FHDC VT& operator[](std::size_t idx) const {
    data_handle_type ptr = start;
    return ptr[idx];
  }

 private:
  data_handle_type start{nullptr};
};

// CAN'T US CRTP since it would discard the RESTRIC_KEYWORD
template <typename VT, std::size_t Extent>
struct restrict_span_storage : size_holder<Extent> {
  using data_handle_type = VT* GCXX_RESTRICT_KEYWORD;
  using size_holder<Extent>::size;

  restrict_span_storage() noexcept = default;

  GCXX_FHDC restrict_span_storage(VT* v_ptr, std::size_t n) noexcept
      : size_holder<Extent>(n), start(v_ptr) {}

  GCXX_FHDC data_handle_type data() const {
    data_handle_type ptr = start;
    return ptr;
  }

  GCXX_FHDC VT& operator[](std::size_t idx) const {
    // just doing this(below line) without accessing it works but unused
    // variable warning so just using it the compiler desugars it out in device
    // code even at -O0 and at -O3 the host code is similar to raw restrict use
    data_handle_type ptr = start;
    return ptr[idx];
  }

 private:
  data_handle_type start{nullptr};
};

template <class VT, std::size_t Extent,
          template <typename, std::size_t> class span_storage_base,
          template <typename, std::size_t> class span_view_base>
class span_base {
 private:
  // █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
  // █                     Static Asserts                     █
  // █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
  GCXX_STATIC_EXPECT(std::is_object_v<VT>,
                     "A reference is not supported,"
                     " need a fully declared type");
  GCXX_STATIC_EXPECT(!std::is_abstract_v<VT>,
                     "An abstract class type is not supported");
  GCXX_STATIC_EXPECT(details_::is_complete_v<VT>,
                     "A forward declaration is not supported,"
                     " need a fully declared type");

  using Self         = span_base<VT, Extent, span_storage_base, span_view_base>;
  using storage_type = span_storage_base<VT, Extent>;
  using data_handle_type = typename storage_type::data_handle_type;

 public:
  // █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
  // █                      Nested Types                      █
  // █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
  using element_type    = VT;
  using value_type      = std::remove_cv_t<element_type>;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer         = element_type*;
  using const_pointer   = const element_type*;
  using reference       = element_type&;
  using const_reference = const element_type&;

  // TODO : change to a  iterator that can be used on device
  using iterator = pointer;  // dont assume this to be T* in your code

  // TODO : change to a  iterator that can be used on device
  using reverse_iterator            = std::reverse_iterator<iterator>;
  static constexpr size_type extent = Extent;

  // █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
  // █                    Member Function                     █
  // █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
  // ==========================================================
  //                        Constructors
  // ==========================================================
  GCXX_TEMPLATE(std::size_t E = Extent)
  GCXX_REQUIRES(E == 0 || E == gcxx::dynamic_extent)
  GCXX_FHDC span_base() GCXX_NOEXCEPT {};  // NOLINT

  GCXX_TEMPLATE(typename It, std::size_t E = Extent)
  GCXX_REQUIRES(gcxx::details_::is_iter_ptr_convertible_v<It, element_type>
                  GCXX_AND(E == gcxx::dynamic_extent))

  GCXX_FHDC span_base(It first, size_type count)
      : m_storage(gcxx::details_::to_address(first), count) {
    GCXX_RUNTIME_EXPECT(extent == gcxx::dynamic_extent || extent == count,
                        "span.ctor from start and count failed");
  }

  GCXX_TEMPLATE(typename It, std::size_t E = Extent)
  GCXX_REQUIRES(
    gcxx::details_::is_iter_ptr_convertible_v<It, element_type> GCXX_AND E !=
    gcxx::dynamic_extent)

  GCXX_FHDC explicit span_base(It first, size_type count)
      : m_storage(gcxx::details_::to_address(first), count) {
    GCXX_RUNTIME_EXPECT(extent == gcxx::dynamic_extent || extent == count,
                        "span.ctor from start and count failed");
  }

  GCXX_TEMPLATE(typename It, typename End, std::size_t E = Extent)
  GCXX_REQUIRES(
    gcxx::details_::is_iter_ptr_convertible_v<It, element_type> GCXX_AND
      gcxx::details_::is_iter_ptr_convertible_v<End, element_type>
        GCXX_AND !std::is_convertible_v<End, std::size_t>
          GCXX_AND(E == gcxx::dynamic_extent))

  GCXX_FHDC span_base(It first, End last)
      : m_storage(gcxx::details_::to_address(first), last - first) {
    GCXX_RUNTIME_EXPECT(
      extent == gcxx::dynamic_extent || extent == (last - first),
      "span.ctor from start and end iterator failed");
  }

  GCXX_TEMPLATE(typename It, typename End, std::size_t E = Extent)
  GCXX_REQUIRES(
    gcxx::details_::is_iter_ptr_convertible_v<It, element_type> GCXX_AND
      gcxx::details_::is_iter_ptr_convertible_v<End, element_type>
        GCXX_AND !std::is_convertible_v<End, std::size_t>
          GCXX_AND(E != gcxx::dynamic_extent))

  GCXX_FHDC explicit span_base(It first, End last)
      : m_storage(gcxx::details_::to_address(first), last - first) {
    GCXX_RUNTIME_EXPECT(
      extent == gcxx::dynamic_extent || extent == (last - first),
      "span.ctor from start and end iterator failed");
  }

  GCXX_TEMPLATE(std::size_t N, std::size_t E = Extent)
  GCXX_REQUIRES(
    (E == gcxx::dynamic_extent || E == N) GCXX_AND
      details_::is_data_ptr_convertible_v<element_type (&)[N], element_type>)

  GCXX_FHDC span_base(
    gcxx::details_::type_identity_t<element_type> (&arr)[N]) noexcept
      : m_storage(arr, N) {}

  GCXX_TEMPLATE(typename U, std::size_t N, std::size_t E = Extent)
  GCXX_REQUIRES((E == gcxx::dynamic_extent || E == N)
                  GCXX_AND details_::is_data_ptr_convertible_v<std::array<U, N>,
                                                               element_type>)

  GCXX_FHC span_base(std::array<U, N>& arr) noexcept
      : m_storage(arr.data(), N) {}

  GCXX_TEMPLATE(typename U, std::size_t N, std::size_t E = Extent)
  GCXX_REQUIRES(
    (E == gcxx::dynamic_extent || E == N) GCXX_AND
      details_::is_data_ptr_convertible_v<const std::array<U, N>, element_type>)

  GCXX_FHC span_base(const std::array<U, N>& arr) noexcept
      : m_storage(arr.data(), N) {}

  GCXX_TEMPLATE(typename R, std::size_t E = Extent)
  GCXX_REQUIRES(
    details_::has_size_and_data_v<R> GCXX_AND
      is_range_ptr_convertible_v<R, element_type>
        GCXX_AND !std::is_array_v<details_::uncvref_t<R>>
          GCXX_AND !gcxx::details_::is_std_array_v<details_::uncvref_t<R>>
            GCXX_AND !details_::is_gcxx_span_specialization_v<R>
              GCXX_AND(E == gcxx::dynamic_extent))

  GCXX_FHC span_base(R&& r)
      : m_storage(gcxx::details_::data(r), gcxx::details_::size(r)) {
    GCXX_RUNTIME_EXPECT(
      extent == gcxx::dynamic_extent || extent == gcxx::details_::size(r),
      "span.ctor from a range failed");
  }

  GCXX_TEMPLATE(typename R, std::size_t E = Extent)
  GCXX_REQUIRES(
    details_::has_size_and_data_v<R> GCXX_AND
      is_range_ptr_convertible_v<R, element_type>
        GCXX_AND !std::is_array_v<details_::uncvref_t<R>>
          GCXX_AND !gcxx::details_::is_std_array_v<details_::uncvref_t<R>>
            GCXX_AND !details_::is_gcxx_span_specialization_v<R>
              GCXX_AND(E != gcxx::dynamic_extent))

  GCXX_FHC explicit span_base(R&& r)
      : m_storage(gcxx::details_::data(r), gcxx::details_::size(r)) {
    GCXX_RUNTIME_EXPECT(
      extent == gcxx::dynamic_extent || extent == gcxx::details_::size(r),
      "span.ctor from a range failed");
  }

  GCXX_TEMPLATE(typename U, std::size_t N, std::size_t E = Extent)
  GCXX_REQUIRES(
    (E == gcxx::dynamic_extent || N == gcxx::dynamic_extent || E == N)
      GCXX_AND details_::is_type_ptr_convertible_v<U, element_type>
        GCXX_AND(E != gcxx::dynamic_extent && N == gcxx::dynamic_extent))

  GCXX_FHC explicit span_base(
    const span_base<U, N, span_storage_base, span_view_base>& source) noexcept
      : m_storage(source.data(), source.size()) {
    GCXX_RUNTIME_EXPECT(
      extent == gcxx::dynamic_extent || extent == source.size(),
      "span.ctor from a source span of diffrent type failed");
  }

  GCXX_TEMPLATE(typename U, std::size_t N, std::size_t E = Extent)
  GCXX_REQUIRES(
    (E == gcxx::dynamic_extent || N == gcxx::dynamic_extent || E == N)
      GCXX_AND details_::is_type_ptr_convertible_v<U, element_type>
        GCXX_AND !(E != gcxx::dynamic_extent && N == gcxx::dynamic_extent))

  GCXX_FHC span_base(
    const span_base<U, N, span_storage_base, span_view_base>& source) noexcept
      : m_storage(source.data(), source.size()) {
    GCXX_RUNTIME_EXPECT(
      extent == gcxx::dynamic_extent || extent == source.size(),
      "span.ctor from a source span of diffrent type failed");
  }

  GCXX_FHDC span_base(const span_base& other) noexcept = default;

  // ==========================================================
  //                         destructor
  // ==========================================================

  GCXX_FHD ~span_base() = default;

  // ==========================================================
  //                         operator =
  // ==========================================================

  GCXX_CXPR auto operator=(const Self&) GCXX_NOEXCEPT->Self& = default;

  // ==========================================================
  //                         Iterators
  // ==========================================================

  GCXX_FHDC auto begin() GCXX_CONST_NOEXCEPT -> iterator { return data(); }

  GCXX_FHDC auto end() GCXX_CONST_NOEXCEPT -> iterator {
    return data() + size();
  }

  GCXX_FH GCXX_CXPR auto rbegin() GCXX_CONST_NOEXCEPT -> reverse_iterator {
    return reverse_iterator(end());
  }

  GCXX_FH GCXX_CXPR auto rend() GCXX_CONST_NOEXCEPT -> reverse_iterator {
    return reverse_iterator(begin());
  }

  // ==========================================================
  //                       Element access
  // ==========================================================

  GCXX_FHDC auto front() const -> reference { return m_storage[0]; }

  GCXX_FHDC auto back() const -> reference {
    return m_storage[size() - 1];
  }  //*rbegin() cant be used since operator * in host only

  GCXX_FHDC auto operator[](size_type idx) const -> reference {
    GCXX_RUNTIME_EXPECT(idx < size(), "Out of bounds access");
    return m_storage[idx];
  }

  GCXX_FHDC auto data() GCXX_CONST_NOEXCEPT -> data_handle_type {
    return m_storage.data();
  }

  // ==========================================================
  //                         Observers
  // ==========================================================

  GCXX_FHDC auto size() GCXX_CONST_NOEXCEPT -> size_type {
    return m_storage.size();
  }

  GCXX_FHDC auto size_bytes() GCXX_CONST_NOEXCEPT -> size_type {
    return size() * sizeof(element_type);
  }

  [[nodiscard]] GCXX_CXPR auto empty() GCXX_CONST_NOEXCEPT -> bool {
    return size() == 0;
  }

  // ==========================================================
  //                          subviews
  // ==========================================================

  template <std::size_t Count>
  GCXX_FHDC auto first() const -> span_view_base<element_type, Count> {
    GCXX_STATIC_EXPECT(Extent == gcxx::dynamic_extent || Count <= Extent,
                       "Span.first count greater than size");
    GCXX_RUNTIME_EXPECT(Count <= size(), "Span.first count greater than size");
    return span_view_base<element_type, Count>(data(), Count);
  }

  GCXX_FHDC auto first(size_type count) const
    -> span_view_base<element_type, gcxx::dynamic_extent> {
    GCXX_RUNTIME_EXPECT(count <= size(), "Span.first count greater thansize");
    return {data(), count};
  }

  template <std::size_t Count>
  GCXX_FHDC auto last() const -> span_view_base<element_type, Count> {
    GCXX_STATIC_EXPECT(Extent == gcxx::dynamic_extent || Count <= Extent,
                       "Span.last count greater than size");
    GCXX_RUNTIME_EXPECT(Count <= size(), "Span.last count greater than size");
    return span_view_base<element_type, Count>(data() + (size() - Count),
                                               Count);
  }

  GCXX_FHDC auto last(size_type count) const
    -> span_view_base<element_type, gcxx::dynamic_extent> {
    GCXX_RUNTIME_EXPECT(count <= size(), "Span.last count greater than size");
    return {data() + (size() - count), count};
  }

  template <std::size_t Offset, std::size_t Count = gcxx::dynamic_extent>
  using subspan_ret_t =
    span_view_base<element_type, Count != gcxx::dynamic_extent
                                   ? Count
                                   : (Extent != Count ? Extent - Offset
                                                      : gcxx::dynamic_extent)>;

  template <std::size_t Offset, std::size_t Count = gcxx::dynamic_extent>
  GCXX_FHDC auto subspan() const -> subspan_ret_t<Offset, Count> {
    GCXX_STATIC_EXPECT(Extent == gcxx::dynamic_extent ||
                         (Offset <= Extent && (Count == gcxx::dynamic_extent ||
                                               Offset + Count <= Extent)),
                       "Span.subspan contract violated");
    GCXX_RUNTIME_EXPECT(Offset <= size() && (Count == gcxx::dynamic_extent ||
                                             Offset + Count <= size()),
                        "Span.subspan contract violated");
    return subspan_ret_t<Offset, Count>(
      data() + Offset, Count != gcxx::dynamic_extent ? Count : size() - Offset);
  }

  GCXX_FHDC auto subspan(size_type offset,
                         size_type count = gcxx::dynamic_extent) const
    -> span_view_base<element_type, gcxx::dynamic_extent> {
    GCXX_RUNTIME_EXPECT(offset <= size() && (count == gcxx::dynamic_extent ||
                                             offset + count <= size()),
                        "Span.subspan contract violated");
    return {data() + offset,
            count == gcxx::dynamic_extent ? size() - offset : count};
  }

 private:
  storage_type m_storage{};
};

GCXX_NAMESPACE_DETAILS_END()

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █                          Span                          █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
template <class VT, std::size_t Extent = gcxx::dynamic_extent>
class span
    : public details_::span_base<VT, Extent, details_::span_storage, span> {
 private:
  using Base = details_::span_base<VT, Extent, details_::span_storage, span>;

 public:
  using Base::Base;
};
// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █                 Span Deduction guides                  █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
template <class It, class EndOrSize>
span(It, EndOrSize)
  -> span<std::remove_reference_t<gcxx::details_::iter_reference_t<It>>>;

template <class T, std::size_t N>
span(T (&)[N]) -> span<T, N>;

template <class T, std::size_t N>
span(std::array<T, N>&) -> span<T, N>;

template <class T, std::size_t N>
span(const std::array<T, N>&) -> span<const T, N>;

template <class R>
span(R&&)
  -> span<
    std::remove_pointer_t<decltype(gcxx::details_::data(std::declval<R&>()))>>;

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █                     Restrcit Span                      █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
template <class VT, std::size_t Extent = gcxx::dynamic_extent>
class restrict_span
    : public details_::span_base<VT, Extent, details_::restrict_span_storage,
                                 restrict_span> {
 private:
  using Base = details_::span_base<VT, Extent, details_::restrict_span_storage,
                                   restrict_span>;

 public:
  using Base::Base;
};
// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █             Restrcit Span Deduction guides             █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
template <class It, class EndOrSize>
restrict_span(It, EndOrSize)
  -> restrict_span<
    std::remove_reference_t<gcxx::details_::iter_reference_t<It>>>;

template <class T, std::size_t N>
restrict_span(T (&)[N]) -> restrict_span<T, N>;

template <class T, std::size_t N>
restrict_span(std::array<T, N>&) -> restrict_span<T, N>;

template <class T, std::size_t N>
restrict_span(const std::array<T, N>&) -> restrict_span<const T, N>;

template <class R>
restrict_span(R&&)
  -> restrict_span<
    std::remove_pointer_t<decltype(gcxx::details_::data(std::declval<R&>()))>>;

// ╔════════════════════════════════════════════════════════╗
// ║           Public span-like concept and helpers         ║
// ╚════════════════════════════════════════════════════════╝

/// Element type of any span-like type: the pointee of .data().
/// Works with references and cv-qualified types.
template <typename T>
using span_element_t = std::remove_pointer_t<decltype(details_::data(
  std::declval<details_::uncvref_t<T>&>()))>;

/// True for any T that exposes .data() → pointer and .size() → integral.
/// Satisfied by gcxx::span, gcxx::restrict_span, std::span, std::vector,
/// std::array, and any user type with those two members.
template <typename T>
GCXX_CONCEPT is_span_like_v =
  details_::is_span_like_impl<details_::uncvref_t<T>>::value;

/// True when T is span-like and its element type is array-convertible to ET.
/// Use this to constrain Copy, kernel launch, or other generic range APIs.
template <typename T, typename ET>
GCXX_CONCEPT is_compatible_span_v =
  is_span_like_v<T> &&
  details_::is_ptr_array_convertible_v<span_element_t<T>, ET>;

GCXX_NAMESPACE_MAIN_END()


#endif