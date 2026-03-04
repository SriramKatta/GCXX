#pragma once
#ifndef GCXX_RUNTIME_MEMORY_SPANS_SPAN_SPAN_HPP
#define GCXX_RUNTIME_MEMORY_SPANS_SPAN_SPAN_HPP


#include <array>
#include <iterator>
#include <limits>
#include <memory>
#include <vector>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/details/helper_function.hpp>
#include <gcxx/runtime/details/type_traits.hpp>

/*
https://godbolt.org/z/qb5vq7sh9
TODO : use this as refrence to extend and support restrict span keeping most of
the code same
*/

GCXX_NAMESPACE_MAIN_BEGIN

// since mdspan from kokkos has this already we dont need it here
// GCXX_CXPR inline auto dynamic_extent =
//   std::numeric_limits<std::size_t>::max();

GCXX_NAMESPACE_DETAILS_BEGIN

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █                      Span Storage                      █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█

template <std::size_t Extent>
struct size_holder {
  size_holder(std::size_t) noexcept {}

  static GCXX_FHDC std::size_t size() noexcept { return Extent; }
};

template <>
struct size_holder<gcxx::dynamic_extent> {
  std::size_t m_size{0};

  size_holder() noexcept = default;

  GCXX_FHDC size_holder(std::size_t n) noexcept : m_size(n) {}

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
          template <typename, std::size_t> class span_storage_base>
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

  using Self             = span_base<VT, Extent, span_storage_base>;
  using storage_type     = span_storage_base<VT, Extent>;
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

  GCXX_TEMPLATE(typename It)
  GCXX_REQUIRES(gcxx::details_::is_container_element_type_compatible_v<
                std::remove_reference_t<gcxx::details_::iter_reference_t<It>>,
                element_type>)

  GCXX_FHDC span_base(It first, size_type count)
      : m_storage(gcxx::details_::to_address(first), count) {
    GCXX_RUNTIME_EXPECT(extent == gcxx::dynamic_extent || extent == count,
                        "span.ctor from start and count failed");
  }


  GCXX_TEMPLATE(typename It, class End)
  GCXX_REQUIRES(
    gcxx::details_::is_container_element_type_compatible_v<
      std::remove_reference_t<gcxx::details_::iter_reference_t<It>>,
      element_type>
      GCXX_AND gcxx::details_::is_container_element_type_compatible_v<
        std::remove_reference_t<gcxx::details_::iter_reference_t<End>>,
        element_type>)

  GCXX_FHDC span_base(It first, End last)
      : m_storage(gcxx::details_::to_address(first), last - first) {
    GCXX_RUNTIME_EXPECT(
      extent == gcxx::dynamic_extent || extent == (last - first),
      "span.ctor from start and end iterator failed");
  }

  // ==========================================================
  //                         destructor
  // ==========================================================

  GCXX_FHD ~span_base() = default;

  // ==========================================================
  //                         operator =
  // ==========================================================

  GCXX_CXPR auto operator=(const Self&) GCXX_NOEXCEPT->Self& = default;

  GCXX_CXPR auto operator=(Self&&) GCXX_NOEXCEPT->Self& = default;

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
  GCXX_FHDC auto first() const
    -> span_base<element_type, Count, span_storage_base> {
    GCXX_STATIC_EXPECT(Count <= size(), "Span.first count greater than size");
    return {data(), Count};
  }

  GCXX_FHDC auto first(size_type count) const
    -> span_base<element_type, gcxx::dynamic_extent, span_storage_base> {
    GCXX_RUNTIME_EXPECT(count <= size(), "Span.first count greater thansize");
    return {data(), count};
  }

  template <std::size_t Count>
  GCXX_FHDC auto last() const
    -> span_base<element_type, Count, span_storage_base> {
    GCXX_STATIC_EXPECT(Count <= size(), "Span.last count greater than size");
    return {data() + (size() - Count), Count};
  }

  GCXX_FHDC auto last(size_type count) const
    -> span_base<element_type, gcxx::dynamic_extent, span_storage_base> {
    GCXX_RUNTIME_EXPECT(count <= size(), "Span.last count greater than size");
    return {data() + (size() - count), count};
  }

  template <std::size_t Offset, std::size_t Count = gcxx::dynamic_extent>
  using subspan_ret_t =
    span_base<element_type,
              Count != gcxx::dynamic_extent
                ? Count
                : (Extent != Count ? Extent - Offset : gcxx::dynamic_extent),
              span_storage_base>;

  template <std::size_t Offset, std::size_t Count = gcxx::dynamic_extent>
  GCXX_FHDC auto subspan() const -> subspan_ret_t<Offset, Count> {
    GCXX_STATIC_EXPECT(Offset <= size() && Count == gcxx::dynamic_extent ||
                         Offset + Count < size(),
                       "Span.subspan contract violated");
    return {data() + Offset,
            Count != gcxx::dynamic_extent ? Count : size() - Offset};
  }

  // GCXX_FHDC auto subspan(size_type offset,
  //                        size_type count = gcxx::dynamic_extent) const
  //   -> span_base<element_type> {
  //   GCXX_RUNTIME_EXPECT(offset <= size() && count == gcxx::dynamic_extent ||
  //                         offset + count < size(),
  //                       "Span.subspan contract violated");
  //   return {data() + offset,
  //           count == gcxx::dynamic_extent ? size() - count : count};
  // }

 private:
  storage_type m_storage{};
};

GCXX_NAMESPACE_DETAILS_END

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █                          Span                          █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
template <class VT, std::size_t Extent = gcxx::dynamic_extent>
class span : public details_::span_base<VT, Extent, details_::span_storage> {
 private:
  using Base = details_::span_base<VT, Extent, details_::span_storage>;

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

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █                     Restrcit Span                      █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
template <class VT, std::size_t Extent = gcxx::dynamic_extent>
class restrict_span
    : public details_::span_base<VT, Extent, details_::restrict_span_storage> {
 private:
  using Base = details_::span_base<VT, Extent, details_::restrict_span_storage>;

 public:
  using Base::Base;
};
// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █             Restrcit Span Deduction guides             █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
template <class It, class EndOrSize>
restrict_span(It, EndOrSize) -> restrict_span<
  std::remove_reference_t<gcxx::details_::iter_reference_t<It>>>;

template <class T, std::size_t N>
restrict_span(T (&)[N]) -> restrict_span<T, N>;

template <class T, std::size_t N>
restrict_span(std::array<T, N>&) -> restrict_span<T, N>;

template <class T, std::size_t N>
restrict_span(const std::array<T, N>&) -> restrict_span<const T, N>;


GCXX_NAMESPACE_MAIN_END


#endif