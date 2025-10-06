#pragma once
#ifndef GCXX_RUNTIME_SPAN_SPAN_HPP
#define GCXX_RUNTIME_SPAN_SPAN_HPP


#include <array>
#include <iterator>
#include <limits>
#include <type_traits>

#include <gcxx/backend/backend.hpp>
#include <gcxx/macros/define_macros.hpp>


GCXX_NAMESPACE_MAIN_BEGIN


GCXX_CXPR inline std::size_t dynamic_extent =
  std::numeric_limits<std::size_t>::max();

template <class VT, std::size_t Extent = gcxx::dynamic_extent>
class span;

GCXX_NAMESPACE_DETAILS_BEGIN

template <typename VT, std::size_t S>
struct span_storage {
  GCXX_FHDC span_storage() noexcept = default;

  GCXX_FHDC span_storage(VT* v_ptr, std::size_t) noexcept : start(v_ptr) {}

  VT* start{nullptr};
  static constexpr std::size_t size = S;
};

template <typename VT>
struct span_storage<VT, dynamic_extent> {
  GCXX_FHDC span_storage() noexcept = default;

  GCXX_FHDC span_storage(VT* v_ptr, std::size_t v_size) noexcept
      : start(v_ptr), size(v_size) {}

  VT* start{nullptr};
  std::size_t size{0};
};

template <typename, typename = size_t>
struct is_complete : std::false_type {};

template <typename T>
struct is_complete<T, decltype(sizeof(T))> : std::true_type {};

template <typename T>
inline constexpr bool is_complete_v = is_complete<T>::value;

template <class C>
GCXX_FHDC auto data(C& c) -> decltype(c.data()) {
  return c.data();
}

template <class C>
GCXX_FHDC auto data(const C& c) -> decltype(c.data()) {
  return c.data();
}

template <class T, std::size_t N>
GCXX_FHDC T* data(T (&array)[N]) noexcept {
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
GCXX_FHDC std::size_t size(const T (&)[N]) noexcept {
  return N;
}

GCXX_NAMESPACE_DETAILS_END

template <class VT, std::size_t Extent>
class span {
 private:
  // █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
  // █                     Static Asserts                     █
  // █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
  static_assert(std::is_object_v<VT>,
                "An refrence is not supported"
                " need an fully declared type");
  static_assert(details_::is_complete_v<VT>,
                "A forward declaration is not supported"
                " need an fully declared type");
  static_assert(!std::is_abstract_v<VT>,
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

  // TODO : change to a  iterator that can be used on device
  using iterator = pointer;  // dont assume this to be T* in your code

  // TODO : change to a  iterator that can be used on device
  using reverse_iterator = std::reverse_iterator<iterator>;

  // █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
  // █                    Memeber Function                    █
  // █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
  // ==========================================================
  //                        Constructors
  // ==========================================================
  template <std::size_t E = Extent,
            typename std::enable_if_t<(E <= 0 || E == dynamic_extent), int> = 0>
  GCXX_CXPR GCXX_FHD span() GCXX_NOEXCEPT {}

  GCXX_CXPR GCXX_FHD span(pointer ptr, size_type count)
      : storage_(ptr, count) {}

  GCXX_CXPR GCXX_FHD span(pointer first_elem, pointer last_elem)
      : storage_(first_elem, last_elem - first_elem) {}

  template <std::size_t N, std::size_t E = Extent,
            typename std::enable_if_t<(E == dynamic_extent || E == N), int> = 0>
  GCXX_CXPR GCXX_FHD span(element_type (&arr)[N]) GCXX_NOEXCEPT
      : storage_(arr, N) {}

  template <typename UVT, std::size_t N, std::size_t E = Extent,
            typename std::enable_if_t<(E == dynamic_extent || E == N), int> = 0>
  GCXX_CXPR GCXX_FHD span(std::array<UVT, N>& arr) GCXX_NOEXCEPT
      : storage_(arr.data(), N) {}

  template <typename UVT, std::size_t N, std::size_t E = Extent,
            typename std::enable_if_t<(E == dynamic_extent || E == N), int> = 0>
  GCXX_CXPR GCXX_FHD span(const std::array<UVT, N>& arr) GCXX_NOEXCEPT
      : storage_(arr.data(), N) {}

  template <typename container, std::size_t E = Extent,
            typename std::enable_if_t<(E == dynamic_extent), int> = 0>
  GCXX_CXPR GCXX_FHD span(const container& arr) GCXX_NOEXCEPT
      : storage_(details_::data(arr), details_::size(arr)) {}

  template <typename container, std::size_t E = Extent,
            typename std::enable_if_t<(E == dynamic_extent), int> = 0>
  GCXX_CXPR GCXX_FHD span(container& arr) GCXX_NOEXCEPT
      : storage_(details_::data(arr), details_::size(arr)) {}

  template <typename OtherElementType, std::size_t OtherExtent,
            typename std::enable_if_t<(Extent == dynamic_extent ||
                                       OtherExtent == dynamic_extent ||
                                       Extent == OtherExtent),
                                      int> = 0>
  GCXX_CXPR GCXX_FHD span(const span<OtherElementType, OtherExtent>& other)
    GCXX_NOEXCEPT : storage_(other.data(), other.size()) {}

  GCXX_CXPR GCXX_FHD span(const span& other) GCXX_NOEXCEPT = default;

  // ==========================================================
  //                         destructor
  // ==========================================================

  GCXX_CXPR GCXX_FHD span() = default;

  // ==========================================================
  //                         operator =
  // ==========================================================

  GCXX_CXPR auto operator=(const span& other) GCXX_NOEXCEPT->span& = default;

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

  GCXX_FHDC auto front() const -> reference { return *begin(); }

  GCXX_FHDC auto back() const -> reference {
    return *(end() - 1);
  }  //*rbegin() cant be used since operator * in host only

  GCXX_FHDC auto operator[](size_type idx) const -> reference {
    return *(data() + idx);
  }

  GCXX_FHDC auto data() GCXX_CONST_NOEXCEPT -> pointer {
    return storage_.start;
  }

  // ==========================================================
  //                         Observers
  // ==========================================================

  GCXX_FHDC auto size() GCXX_CONST_NOEXCEPT -> size_type {
    return storage_.size;
  }

  GCXX_FHDC auto size_bytes() GCXX_CONST_NOEXCEPT -> size_type {
    return size() * sizeof(element_type);
  }

  GCXX_CXPR auto empty() GCXX_CONST_NOEXCEPT -> bool { return size() == 0; }

  // ==========================================================
  //                          subviews
  // ==========================================================

  template <std::size_t Count>
  GCXX_FHDC auto first() const -> span<element_type, Count> {
    return {data(), Count};
  }

  GCXX_FHDC auto first(size_type count) const
    -> span<element_type, dynamic_extent> {
    return {data(), count};
  }

  template <std::size_t Count>
  GCXX_FHDC auto last() const -> span<element_type, Count> {
    return {data() + (size() - Count), Count};
  }

  GCXX_FHDC auto last(size_type count) const
    -> span<element_type, dynamic_extent> {
    return {data() + (size() - count), count};
  }

  template <std::size_t Offset, std::size_t Count = dynamic_extent>
  using subspan_return_t =
    span<element_type,
         Count != dynamic_extent
           ? Count
           : (Extent != Count ? Extent - Offset : dynamic_extent)>;

  template <std::size_t Offset, std::size_t Count = dynamic_extent>
  GCXX_FHDC auto subspan() const -> subspan_return_t<Offset, Count> {
    return {data() + Offset, Count != dynamic_extent ? Count : size() - Offset};
  }

  GCXX_FHDC auto subspan(size_type offset, size_type count = dynamic_extent)
    const -> span<element_type> {
    return {data() + offset, count == dynamic_extent ? size() - count : count};
  }

  // █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
  // █                      Data members                      █
  // █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
 public:
  static constexpr size_type extent = Extent;

 private:
  storage_type storage_{};
};

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █                    Deduction guides                    █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
template <class VT, size_t N>
span(VT (&)[N]) -> span<VT, N>;

template <class VT, size_t N>
span(std::array<VT, N>&) -> span<VT, N>;

template <class VT, size_t N>
span(const std::array<VT, N>&) -> span<const VT, N>;

template <class Container>
span(Container&) -> span<typename std::remove_reference_t<
                   decltype(*std::data(std::declval<Container&>()))>>;

template <class Container>
span(const Container&) -> span<const typename Container::value_type>;


GCXX_NAMESPACE_MAIN_END


#endif