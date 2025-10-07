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

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █                      Span Storage                      █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
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

// █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
// █            Impl of std::size and std::data             █
// █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
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
struct is_span : std::false_type {};

template <typename VT, std::size_t S>
struct is_span<span<VT, S>> : std::true_type {};

template <typename VT>
GCXX_CXPR inline bool is_span_v = is_span<VT>::value;

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

template <typename C, typename U = uncvref_t<C>>
struct is_container {
  static constexpr bool value = !is_span_v<U> && !is_std_array_v<U> &&
                                !std::is_array_v<U> && has_size_and_data_v<C>;
};

template <typename C>
GCXX_CXPR inline bool is_container_v = is_container<C>::value;

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
  static_assert(!std::is_abstract_v<VT>,
                "An abstract class type is not supported");
  static_assert(details_::is_complete_v<VT>,
                "A forward declaration is not supported"
                " need an fully declared type");

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
            typename std::enable_if_t<(E == 0 || E == dynamic_extent), int> = 0>
  GCXX_CXPR GCXX_FHD span() GCXX_NOEXCEPT {}

  GCXX_CXPR GCXX_FHD span(pointer first, size_type count)
      : storage_(first, count) {
    GCXX_DYNAMIC_EXPECT(extent == dynamic_extent || count == dynamic_extent,
                        "Span (ptr,count) contract violation");
  }

  GCXX_CXPR GCXX_FHD span(pointer first, pointer last)
      : storage_(first, last - first) {
    GCXX_DYNAMIC_EXPECT(
      extent == dynamic_extent ||
        (last - first) == static_cast<difference_type>(dynamic_extent),
      "Span (ptr, ptr) contract violation");
  }

  template <std::size_t N, std::size_t E = Extent,
            typename std::enable_if_t<
              (E == dynamic_extent || E == N) &&
                details_::is_container_element_type_compatible_v<
                  element_type (&)[N], element_type>,
              int> = 0>
  GCXX_CXPR GCXX_FHD span(element_type (&arr)[N]) GCXX_NOEXCEPT
      : storage_(arr, N) {}

  template <typename OVT, std::size_t N, std::size_t E = Extent,
            typename std::enable_if_t<
              (E == dynamic_extent || E == N) &&
                details_::is_container_element_type_compatible_v<
                  std::array<OVT, N>&, element_type>,
              int> = 0>
  GCXX_CXPR GCXX_FHD span(std::array<OVT, N>& arr) GCXX_NOEXCEPT
      : storage_(arr.data(), N) {}

  template <typename OVT, std::size_t N, std::size_t E = Extent,
            typename std::enable_if_t<
              (E == dynamic_extent || E == N) &&
                details_::is_container_element_type_compatible_v<
                  const std::array<OVT, N>&, element_type>,
              int> = 0>
  GCXX_CXPR GCXX_FHD span(const std::array<OVT, N>& arr) GCXX_NOEXCEPT
      : storage_(arr.data(), N) {}

  template <typename container, std::size_t E = Extent,
            typename std::enable_if_t<
              E == dynamic_extent && details_::is_container_v<container> &&
                details_::is_container_element_type_compatible_v<
                  const container&, element_type>,
              int> = 0>
  GCXX_CXPR GCXX_FHD span(const container& arr) GCXX_NOEXCEPT
      : storage_(details_::data(arr), details_::size(arr)) {}

  template <typename container, std::size_t E = Extent,
            typename std::enable_if_t<
              E == dynamic_extent && details_::is_container_v<container> &&
                details_::is_container_element_type_compatible_v<container&,
                                                                 element_type>,
              int> = 0>
  GCXX_CXPR GCXX_FHD span(container& arr) GCXX_NOEXCEPT
      : storage_(details_::data(arr), details_::size(arr)) {}

  template <typename OVT, std::size_t OtherExtent,
            typename std::enable_if_t<
              (Extent == dynamic_extent || OtherExtent == dynamic_extent ||
               Extent == OtherExtent) &&
                std::is_convertible_v<OVT (*)[], VT (*)[]>,
              int> = 0>
  GCXX_CXPR GCXX_FHD span(const span<OVT, OtherExtent>& other) GCXX_NOEXCEPT
      : storage_(other.data(), other.size()) {}

  GCXX_CXPR GCXX_FHD span(const span& other) GCXX_NOEXCEPT = default;

  // ==========================================================
  //                         destructor
  // ==========================================================

  GCXX_CXPR GCXX_FHD span() = default;

  // ==========================================================
  //                         operator =
  // ==========================================================

  GCXX_CXPR auto operator=(const span&) GCXX_NOEXCEPT->span& = default;

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
    GCXX_DYNAMIC_EXPECT(idx < size(), "Out of bounds access");
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
    GCXX_STATIC_EXPECT(Count <= size(), "Span.first count greater than size");
    return {data(), Count};
  }

  GCXX_FHDC auto first(size_type count) const
    -> span<element_type, dynamic_extent> {
    GCXX_DYNAMIC_EXPECT(count <= size(), "Span.first count greater than size");
    return {data(), count};
  }

  template <std::size_t Count>
  GCXX_FHDC auto last() const -> span<element_type, Count> {
    GCXX_STATIC_EXPECT(Count <= size(), "Span.last count greater than size");
    return {data() + (size() - Count), Count};
  }

  GCXX_FHDC auto last(size_type count) const
    -> span<element_type, dynamic_extent> {
    GCXX_DYNAMIC_EXPECT(count <= size(), "Span.last count greater than size");
    return {data() + (size() - count), count};
  }

  template <std::size_t Offset, std::size_t Count = dynamic_extent>
  using subspan_ret_t =
    span<element_type,
         Count != dynamic_extent
           ? Count
           : (Extent != Count ? Extent - Offset : dynamic_extent)>;

  template <std::size_t Offset, std::size_t Count = dynamic_extent>
  GCXX_FHDC auto subspan() const -> subspan_ret_t<Offset, Count> {
    GCXX_STATIC_EXPECT(
      Offset <= size() && Count == dynamic_extent || Offset + Count < size(),
      "Span.subspan contract failure");
    return {data() + Offset, Count != dynamic_extent ? Count : size() - Offset};
  }

  GCXX_FHDC auto subspan(size_type offset, size_type count = dynamic_extent)
    const -> span<element_type> {
    GCXX_EXPECT(
      offset <= size() && count == dynamic_extent || offset + count < size(),
      "Span.subspan contract failure");
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