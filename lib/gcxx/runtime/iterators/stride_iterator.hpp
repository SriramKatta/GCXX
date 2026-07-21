// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_ITERATORS_STRIDE_ITERATOR_HPP_
#define GCXX_RUNTIME_ITERATORS_STRIDE_ITERATOR_HPP_

#include <cstddef>
#include <iterator>
#include <type_traits>

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

// ─────────────────────────────────────────────────────────────────────────────
// gcxx::stride_iterator<T>
//
// A random-access iterator that advances by a fixed STRIDE per step, visiting
// elements [base, base+stride, base+2*stride, …]. Useful for walking strided
// storage (e.g. every Nth element of a buffer: the columns of a row-major
// matrix, interleaved channels, decimated sequences).
//
// Wraps a raw pointer (T* / const T*). All operators are host+device callable
// (GCXX_FHDC); the caller is responsible for only dereferencing from a space
// that can reach the pointed-to memory (deref of a pointer has no compile-time
// space restriction — the safety is the pointer's reachability, same as a raw
// pointer). operator-(a,b) reports distance in STRIDE STEPS, not raw elements.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
class stride_iterator {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type        = std::remove_cv_t<T>;
  using difference_type   = std::ptrdiff_t;
  using pointer           = T*;
  using reference         = T&;

  stride_iterator() noexcept = default;
  GCXX_FHDC stride_iterator(pointer p, difference_type stride) noexcept
      : ptr_(p), stride_(stride) {}

  // ──────────────────────────── dereference ──────────────────────────────────
  GCXX_FHDC auto operator*() const noexcept -> reference { return *ptr_; }
  GCXX_FHDC auto operator->() const noexcept -> pointer { return ptr_; }
  GCXX_FHDC auto operator[](difference_type n) const noexcept -> reference {
    return *(ptr_ + n * stride_);
  }

  // ────────────────────────── stride-step movement ───────────────────────────
  GCXX_FHDC auto operator++() noexcept -> stride_iterator& {
    ptr_ += stride_;
    return *this;
  }
  GCXX_FHDC auto operator++(int) noexcept -> stride_iterator {
    stride_iterator tmp = *this;
    ptr_ += stride_;
    return tmp;
  }
  GCXX_FHDC auto operator--() noexcept -> stride_iterator& {
    ptr_ -= stride_;
    return *this;
  }
  GCXX_FHDC auto operator--(int) noexcept -> stride_iterator {
    stride_iterator tmp = *this;
    ptr_ -= stride_;
    return tmp;
  }
  GCXX_FHDC auto operator+=(difference_type n) noexcept -> stride_iterator& {
    ptr_ += n * stride_;
    return *this;
  }
  GCXX_FHDC auto operator-=(difference_type n) noexcept -> stride_iterator& {
    ptr_ -= n * stride_;
    return *this;
  }

  // ────────────────────────────── observers ──────────────────────────────────
  /// The underlying pointer (the element this iterator currently addresses).
  GCXX_FHDC auto base() const noexcept -> pointer { return ptr_; }
  /// The stride (in elements) between successive positions.
  GCXX_FHDC auto stride() const noexcept -> difference_type { return stride_; }

 private:
  pointer ptr_{nullptr};
  difference_type stride_{1};
};

// ─────────────────────────────────────────────────────────────────────────────
// Arithmetic + comparisons. Difference is measured in STRIDE STEPS so that
// (end - begin) is the logical length of the strided range.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
GCXX_FHDC auto operator+(
  stride_iterator<T> it,
  typename stride_iterator<T>::difference_type n) noexcept
  -> stride_iterator<T> {
  return it += n;
}

template <typename T>
GCXX_FHDC auto operator+(typename stride_iterator<T>::difference_type n,
                         stride_iterator<T> it) noexcept -> stride_iterator<T> {
  return it += n;
}

template <typename T>
GCXX_FHDC auto operator-(
  stride_iterator<T> it,
  typename stride_iterator<T>::difference_type n) noexcept
  -> stride_iterator<T> {
  return it -= n;
}

template <typename T>
GCXX_FHDC auto operator-(const stride_iterator<T>& a,
                         const stride_iterator<T>& b) noexcept ->
  typename stride_iterator<T>::difference_type {
  return (a.base() - b.base()) / a.stride();
}

template <typename T>
GCXX_FHDC auto operator==(const stride_iterator<T>& a,
                          const stride_iterator<T>& b) noexcept -> bool {
  return a.base() == b.base();
}
template <typename T>
GCXX_FHDC auto operator!=(const stride_iterator<T>& a,
                          const stride_iterator<T>& b) noexcept -> bool {
  return a.base() != b.base();
}
template <typename T>
GCXX_FHDC auto operator<(const stride_iterator<T>& a,
                         const stride_iterator<T>& b) noexcept -> bool {
  return a.base() < b.base();
}
template <typename T>
GCXX_FHDC auto operator>(const stride_iterator<T>& a,
                         const stride_iterator<T>& b) noexcept -> bool {
  return a.base() > b.base();
}
template <typename T>
GCXX_FHDC auto operator<=(const stride_iterator<T>& a,
                          const stride_iterator<T>& b) noexcept -> bool {
  return a.base() <= b.base();
}
template <typename T>
GCXX_FHDC auto operator>=(const stride_iterator<T>& a,
                          const stride_iterator<T>& b) noexcept -> bool {
  return a.base() >= b.base();
}

// ─────────────────────────────────────────────────────────────────────────────
// make_stride_iterator: factory. E.g. iterate every 3rd element of an array a
// of 9: make_stride_iterator(a, 3) .. make_stride_iterator(a + 9, 3) → 3 steps.
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
GCXX_FHDC auto make_stride_iterator(
  T* p, typename stride_iterator<T>::difference_type stride) noexcept
  -> stride_iterator<T> {
  return stride_iterator<T>(p, stride);
}

GCXX_NAMESPACE_MAIN_END()

#endif
