// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_ITERATORS_STRIDE_ITERATOR_HPP_
#define GCXX_ITERATORS_STRIDE_ITERATOR_HPP_

#include <cstddef>
#include <iterator>
#include <type_traits>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/iterators/iterator_traits.hpp>
#include <gcxx/types/size_holder.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

// The delegating dereference members (GCXX_FHDC bodies calling the wrapped
// iterator's space-tagged dereference) trip nvcc/NVHPC's deferred-device-pass
// notes #20013-D/#20015-D whenever Iterator's deref is single-space. That
// call is the intended restriction mechanism, and genuine cross-space misuse
// is still a hard error at the actual call site, so the notes are silenced
// for the class body only (same treatment as reverse_iterator).
#if defined(__NVCOMPILER)
#pragma diag_suppress 20013
#pragma diag_suppress 20015
#elif defined(__CUDACC__)
#pragma nv_diag_suppress 20013
#pragma nv_diag_suppress 20015
#endif

// ─────────────────────────────────────────────────────────────────────────────
// gcxx::stride_iterator<Iterator, Stride>
//
// Wraps ANY randomly-traversable iterator — raw pointers,
// heterogeneous_iterator, other adapters — and visits every Stride-th element.
//
// Every operation delegates to the wrapped iterator, so its memory-access
// restrictions survive: wrapping a heterogeneous_iterator keeps dereference
// space-tagged exactly as the forward iterator's.
//
// Stride is Also compile-time settable
// ─────────────────────────────────────────────────────────────────────────────
template <typename Iterator_t, std::size_t Stride = details_::dynamic_size>
class stride_iterator {
  using traits_t = iterator_traits<Iterator_t>;

  static_assert(
    std::is_convertible_v<typename traits_t::iterator_category,
                          gcxx::random_access_iterator_tag>,
    "stride_iterator requires a randomly-traversable (random-access) "
    "iterator");

 public:
  using iterator_type     = Iterator_t;
  using iterator_category = typename traits_t::iterator_category;
  using value_type        = typename traits_t::value_type;
  using difference_type   = typename traits_t::difference_type;
  using pointer           = typename traits_t::pointer;
  using reference         = typename traits_t::reference;

  /// The stride template parameter (details_::dynamic_size when runtime).
  static constexpr std::size_t stride_extent = Stride;

  stride_iterator() noexcept = default;

  /// Runtime-stride ctor (only when Stride is the dynamic sentinel).
  GCXX_TEMPLATE(std::size_t S = Stride)
  GCXX_REQUIRES(S == details_::dynamic_size)
  GCXX_FHDC stride_iterator(Iterator_t it, difference_type stride) noexcept
      : current_(it), m_stride(static_cast<std::size_t>(stride)) {}

  /// Fixed-stride ctor: the stride comes from the template parameter.
  GCXX_TEMPLATE(std::size_t S = Stride)
  GCXX_REQUIRES(S != details_::dynamic_size)
  GCXX_FHDC explicit stride_iterator(Iterator_t it) noexcept : current_(it) {}

  /// Converting ctor (e.g. iterator → const_iterator rewrapee).
  GCXX_TEMPLATE(typename Other)
  GCXX_REQUIRES(!std::is_same_v<Other, Iterator_t> GCXX_AND
                  std::is_convertible_v<const Other&, Iterator_t>)
  GCXX_FHDC stride_iterator(const stride_iterator<Other, Stride>& u) noexcept
      : current_(u.base()), m_stride(static_cast<std::size_t>(u.stride())) {}

  /// The wrapped iterator at the same position.
  GCXX_FHDC auto base() const noexcept -> Iterator_t { return current_; }

  // ╔════════════════════════════════════════════════════════╗
  // ║   dereference — delegated to the wrapped iterator       ║
  // ║   (this is what preserves its space restrictions)       ║
  // ╚════════════════════════════════════════════════════════╝
  GCXX_FHDC auto operator*() const noexcept -> reference { return *current_; }
  GCXX_FHDC auto operator->() const noexcept -> pointer {
    if constexpr (std::is_pointer_v<Iterator_t>) {
      return current_;
    } else {
      return current_.operator->();
    }
  }
  GCXX_FHDC auto operator[](difference_type n) const noexcept -> reference {
    return *(current_ + n * stride());
  }

  // ╔════════════════════════════════════════════════════════╗
  // ║        stride-step movement (all delegated)            ║
  // ╚════════════════════════════════════════════════════════╝
  GCXX_FHDC auto operator++() noexcept -> stride_iterator& {
    current_ += stride();
    return *this;
  }
  GCXX_FHDC auto operator++(int) noexcept -> stride_iterator {
    stride_iterator tmp = *this;
    current_ += stride();
    return tmp;
  }
  GCXX_FHDC auto operator--() noexcept -> stride_iterator& {
    current_ -= stride();
    return *this;
  }
  GCXX_FHDC auto operator--(int) noexcept -> stride_iterator {
    stride_iterator tmp = *this;
    current_ -= stride();
    return tmp;
  }
  GCXX_FHDC auto operator+=(difference_type n) noexcept -> stride_iterator& {
    current_ += n * stride();
    return *this;
  }
  GCXX_FHDC auto operator-=(difference_type n) noexcept -> stride_iterator& {
    current_ -= n * stride();
    return *this;
  }

  // ╔════════════════════════════════════════════════════════╗
  // ║                       observers                        ║
  // ╚════════════════════════════════════════════════════════╝
  /// The stride (in elements) between successive positions.
  GCXX_FHDC auto stride() const noexcept -> difference_type {
    return static_cast<difference_type>(m_stride.size());
  }

 private:
  Iterator_t current_{};
  [[no_unique_address]] details_::size_holder<Stride> m_stride{};
};

#if defined(__NVCOMPILER)
#pragma diag_warning 20013
#pragma diag_warning 20015
#elif defined(__CUDACC__)
#pragma nv_diag_default 20013
#pragma nv_diag_default 20015
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Arithmetic + comparisons. Difference is measured in STRIDE STEPS so that
// (end - begin) is the logical length of the strided range.
// ─────────────────────────────────────────────────────────────────────────────
template <typename Iterator_t, std::size_t Stride>
GCXX_FHDC auto operator+(
  stride_iterator<Iterator_t, Stride> it,
  typename stride_iterator<Iterator_t, Stride>::difference_type n) noexcept
  -> stride_iterator<Iterator_t, Stride> {
  return it += n;
}

template <typename Iterator_t, std::size_t Stride>
GCXX_FHDC auto operator+(
  typename stride_iterator<Iterator_t, Stride>::difference_type n,
  stride_iterator<Iterator_t, Stride> it) noexcept
  -> stride_iterator<Iterator_t, Stride> {
  return it += n;
}

template <typename Iterator_t, std::size_t Stride>
GCXX_FHDC auto operator-(
  stride_iterator<Iterator_t, Stride> it,
  typename stride_iterator<Iterator_t, Stride>::difference_type n) noexcept
  -> stride_iterator<Iterator_t, Stride> {
  return it -= n;
}

template <typename Iterator_t, std::size_t Stride>
GCXX_FHDC auto operator-(const stride_iterator<Iterator_t, Stride>& a,
                         const stride_iterator<Iterator_t, Stride>& b) noexcept
  -> typename stride_iterator<Iterator_t, Stride>::difference_type {
  return (a.base() - b.base()) / a.stride();
}

template <typename Iterator_t, std::size_t Stride>
GCXX_FHDC auto operator==(const stride_iterator<Iterator_t, Stride>& a,
                          const stride_iterator<Iterator_t, Stride>& b) noexcept
  -> bool {
  return a.base() == b.base();
}
template <typename Iterator_t, std::size_t Stride>
GCXX_FHDC auto operator!=(const stride_iterator<Iterator_t, Stride>& a,
                          const stride_iterator<Iterator_t, Stride>& b) noexcept
  -> bool {
  return a.base() != b.base();
}
template <typename Iterator_t, std::size_t Stride>
GCXX_FHDC auto operator<(const stride_iterator<Iterator_t, Stride>& a,
                         const stride_iterator<Iterator_t, Stride>& b) noexcept
  -> bool {
  return a.base() < b.base();
}
template <typename Iterator_t, std::size_t Stride>
GCXX_FHDC auto operator>(const stride_iterator<Iterator_t, Stride>& a,
                         const stride_iterator<Iterator_t, Stride>& b) noexcept
  -> bool {
  return a.base() > b.base();
}
template <typename Iterator_t, std::size_t Stride>
GCXX_FHDC auto operator<=(const stride_iterator<Iterator_t, Stride>& a,
                          const stride_iterator<Iterator_t, Stride>& b) noexcept
  -> bool {
  return a.base() <= b.base();
}
template <typename Iterator_t, std::size_t Stride>
GCXX_FHDC auto operator>=(const stride_iterator<Iterator_t, Stride>& a,
                          const stride_iterator<Iterator_t, Stride>& b) noexcept
  -> bool {
  return a.base() >= b.base();
}

// ─────────────────────────────────────────────────────────────────────────────
// make_stride_iterator: factories. E.g. iterate every 3rd element of an array
// a of 9: make_stride_iterator(a, 3) .. make_stride_iterator(a + 9, 3) → 3
// steps.
// For a compile-time stride: make_stride_iterator<4>(p); ctad determines the
// type
// ─────────────────────────────────────────────────────────────────────────────
template <typename Iterator_t>
GCXX_FHDC auto make_stride_iterator(
  Iterator_t it,
  typename stride_iterator<Iterator_t>::difference_type stride) noexcept
  -> stride_iterator<Iterator_t> {
  return stride_iterator<Iterator_t>(it, stride);
}

template <std::size_t Stride, typename Iterator_t>
GCXX_FHDC auto make_stride_iterator(Iterator_t it) noexcept
  -> stride_iterator<Iterator_t, Stride> {
  return stride_iterator<Iterator_t, Stride>(it);
}

GCXX_NAMESPACE_MAIN_END()

#endif
