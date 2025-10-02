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


inline constexpr std::size_t dynamic_extent =
  std::numeric_limits<std::size_t>::max();

template <class VT, std::size_t Extent>
class span;

GCXX_NAMESPACE_DETAILS_BEGIN

template <typename VT, std::size_t S>
struct span_storage {
  using pointer                     = VT*;
  constexpr span_storage() noexcept = default;

  constexpr span_storage(VT* v_ptr, std::size_t) noexcept : start(v_ptr) {}

  VT* start{nullptr};
  static constexpr std::size_t size = S;
};

template <typename VT>
struct span_storage<VT, dynamic_extent> {
  constexpr span_storage() noexcept = default;

  constexpr span_storage(VT* v_ptr, std::size_t v_size) noexcept
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

GCXX_NAMESPACE_DETAILS_END

template <class VT, std::size_t Extent = dynamic_extent>
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
  using iterator        = pointer;  // dont assume this to be T* in your code
                                    // maybe changed to a bidirectional iterator
  using reverse_iterator = std::reverse_iterator<iterator>;

  // █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
  // █                    Memeber Function                    █
  // █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█
  // ==========================================================
  //                        Constructors
  // ==========================================================
  template <std::size_t E = Extent,
            typename std::enable_if_t<(E <= 0 || E == dynamic_extent), int> = 0>
  GCXX_CXPR span() GCXX_NOEXCEPT {}

  GCXX_CXPR span(pointer ptr, size_type count) : storage_(ptr, count) {}

  GCXX_CXPR span(pointer first_elem, pointer last_elem)
      : storage_(first_elem, last_elem - first_elem) {}

  template <std::size_t N, std::size_t E = Extent,
            typename std::enable_if_t<(E == dynamic_extent || E == N), int> = 0>
  GCXX_CXPR span(element_type (&arr)[N]) GCXX_NOEXCEPT : storage_(arr, N) {}

  template <typename UVT, std::size_t N, std::size_t E = Extent,
            typename std::enable_if_t<(E == dynamic_extent || E == N), int> = 0>
  GCXX_CXPR span(std::array<UVT, N>& arr) GCXX_NOEXCEPT
      : storage_(arr.data(), N) {}

  template <typename UVT, std::size_t N, std::size_t E = Extent,
            typename std::enable_if_t<(E == dynamic_extent || E == N), int> = 0>
  GCXX_CXPR span(const std::array<UVT, N>& arr) GCXX_NOEXCEPT
      : storage_(arr.data(), N) {}

  template <typename container, std::size_t E = Extent,
            typename std::enable_if_t<(E == dynamic_extent), int> = 0>
  GCXX_CXPR span(const container& arr) GCXX_NOEXCEPT
      : storage_(std::data(arr), std::size(arr)) {}

  template <typename container, std::size_t E = Extent,
            typename std::enable_if_t<(E == dynamic_extent), int> = 0>
  GCXX_CXPR span(container& arr) GCXX_NOEXCEPT
      : storage_(std::data(arr), std::size(arr)) {}

  template <typename OtherElementType, std::size_t OtherExtent,
            typename std::enable_if_t<(Extent == dynamic_extent ||
                                       OtherExtent == dynamic_extent ||
                                       Extent == OtherExtent),
                                      int> = 0>
  GCXX_CXPR span(const span<OtherElementType, OtherExtent>& other) GCXX_NOEXCEPT
      : storage_(other.data(), other.size()) {}

  GCXX_CXPR span(const span& other) GCXX_NOEXCEPT = default;

  // ==========================================================
  //                         destructor
  // ==========================================================

  ~span() = default;

  // ==========================================================
  //                         operator =
  // ==========================================================

  GCXX_CXPR auto operator=(const span& other) GCXX_NOEXCEPT->span& = default;

  // ==========================================================
  //                         Iterators
  // ==========================================================

  GCXX_CXPR auto begin() GCXX_CONST_NOEXCEPT -> iterator { return data(); }

  GCXX_CXPR auto end() GCXX_CONST_NOEXCEPT -> iterator {
    return data() + size();
  }

  GCXX_CXPR auto rbegin() GCXX_CONST_NOEXCEPT -> reverse_iterator {
    return reverse_iterator(end());
  }

  GCXX_CXPR auto rend() GCXX_CONST_NOEXCEPT -> reverse_iterator {
    return reverse_iterator(begin());
  }

  // ==========================================================
  //                       Element access
  // ==========================================================

  GCXX_CXPR auto front() const -> reference { return *begin(); }

  GCXX_CXPR auto back() const -> reference { return *rbegin(); }

  GCXX_CXPR auto operator[](size_type idx) const -> reference {
    return *(data() + idx);
  }

  GCXX_CXPR auto data() GCXX_CONST_NOEXCEPT -> pointer {
    return storage_.start;
  }

  // ==========================================================
  //                         Observers
  // ==========================================================

  GCXX_CXPR auto size() GCXX_CONST_NOEXCEPT -> size_type {
    return storage_.size;
  }

  GCXX_CXPR auto size_bytes() GCXX_CONST_NOEXCEPT -> size_type {
    return size() * sizeof(element_type);
  }

  GCXX_CXPR auto empty() GCXX_CONST_NOEXCEPT -> bool { return size() == 0; }

  // ==========================================================
  //                          subviews
  // ==========================================================

  template <std::size_t Count>
  GCXX_CXPR auto first() const -> span<element_type, Count> {
    return {data(), Count};
  }

  GCXX_CXPR auto first(size_type count) const
    -> span<element_type, dynamic_extent> {
    return {data(), count};
  }

  template <std::size_t Count>
  GCXX_CXPR auto last() const -> span<element_type, Count> {
    return {data() + (size() - Count), Count};
  }

  GCXX_CXPR auto last(size_type count) const
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
  constexpr auto subspan() const -> subspan_return_t<Offset, Count> {
    return {data() + Offset, Count != dynamic_extent ? Count : size() - Offset};
  }

  constexpr auto subspan(size_type offset,
                         size_type count = dynamic_extent) const
    -> span<element_type> {
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

GCXX_NAMESPACE_MAIN_END


#endif