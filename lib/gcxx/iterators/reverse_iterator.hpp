// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_ITERATORS_REVERSE_ITERATOR_HPP_
#define GCXX_ITERATORS_REVERSE_ITERATOR_HPP_

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/iterators/iterator_traits.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()


// Notes #20013-D/#20015-D: calling a __host__ fn from __host__ __device__.
#if defined(__NVCOMPILER)
#pragma diag_suppress 20013
#pragma diag_suppress 20015
#elif defined(__CUDACC__)
#pragma nv_diag_suppress 20013
#pragma nv_diag_suppress 20015
#endif

// gcxx::reverse_iterator<I>: adapts a random-access iterator to go backwards.

template <typename Iterator_t>
class reverse_iterator {
  using traits_t = iterator_traits<Iterator_t>;

  static_assert(
    std::is_convertible_v<typename traits_t::iterator_category,
                          gcxx::random_access_iterator_tag>,
    "reverse_iterator requires a randomly-traversable (random-access) "
    "iterator");

 public:
  using iterator_type     = Iterator_t;
  using iterator_category = typename traits_t::iterator_category;
  using value_type        = typename traits_t::value_type;
  using difference_type   = typename traits_t::difference_type;
  using pointer           = typename traits_t::pointer;
  using reference         = typename traits_t::reference;

  GCXX_TEMPLATE(typename It = Iterator_t)
  GCXX_REQUIRES(std::is_default_constructible_v<It>)
  reverse_iterator() noexcept : current_() {}

  explicit GCXX_FHDC reverse_iterator(Iterator_t x) noexcept : current_(x) {}

  GCXX_TEMPLATE(typename Other)
  GCXX_REQUIRES(!std::is_same_v<Other, Iterator_t> GCXX_AND
                  std::is_convertible_v<const Other&, Iterator_t>)
  GCXX_FHDC reverse_iterator(const reverse_iterator<Other>& u) noexcept
      : current_(u.base()) {}

  GCXX_TEMPLATE(typename Other)
  GCXX_REQUIRES(!std::is_same_v<Other, Iterator_t> GCXX_AND
                  std::is_convertible_v<const Other&, Iterator_t>
                    GCXX_AND std::is_assignable_v<Iterator_t&, const Other&>)
  GCXX_FHDC auto operator=(const reverse_iterator<Other>& u) noexcept
    -> reverse_iterator& {
    current_ = u.base();
    return *this;
  }

  GCXX_FHDC auto base() const noexcept -> Iterator_t { return current_; }

  // Dereference (delegated; preserves space restrictions).
  GCXX_FHDC auto operator*() const -> reference {
    Iterator_t tmp = current_;
    --tmp;
    return *tmp;
  }

  GCXX_TEMPLATE(typename It = Iterator_t)
  GCXX_REQUIRES(std::is_pointer_v<It>)
  GCXX_FHDC auto operator->() const -> pointer {
    Iterator_t tmp = current_;
    --tmp;
    return tmp;
  }
  GCXX_TEMPLATE(typename It = Iterator_t)
  GCXX_REQUIRES(!std::is_pointer_v<It>)
  GCXX_FHDC auto operator->() const -> pointer {
    Iterator_t tmp = current_;
    --tmp;
    return tmp.operator->();
  }

  GCXX_FHDC auto operator[](difference_type n) const -> reference {
    return *(*this + n);
  }

  // Random-access mechanics (direction flipped).
  GCXX_FHDC auto operator++() noexcept -> reverse_iterator& {
    --current_;
    return *this;
  }
  GCXX_FHDC auto operator++(int) noexcept -> reverse_iterator {
    reverse_iterator tmp{*this};
    --current_;
    return tmp;
  }
  GCXX_FHDC auto operator--() noexcept -> reverse_iterator& {
    ++current_;
    return *this;
  }
  GCXX_FHDC auto operator--(int) noexcept -> reverse_iterator {
    reverse_iterator tmp{*this};
    ++current_;
    return tmp;
  }
  GCXX_FHDC auto operator+=(difference_type n) noexcept -> reverse_iterator& {
    current_ -= n;
    return *this;
  }
  GCXX_FHDC auto operator-=(difference_type n) noexcept -> reverse_iterator& {
    current_ += n;
    return *this;
  }

  GCXX_FHDC friend auto operator+(const reverse_iterator& it,
                                  difference_type n) noexcept
    -> reverse_iterator {
    return reverse_iterator{it.base() - n};
  }
  GCXX_FHDC friend auto operator+(difference_type n,
                                  const reverse_iterator& it) noexcept
    -> reverse_iterator {
    return reverse_iterator{it.base() - n};
  }
  GCXX_FHDC friend auto operator-(const reverse_iterator& it,
                                  difference_type n) noexcept
    -> reverse_iterator {
    return reverse_iterator{it.base() + n};
  }
  // std::reverse_iterator: x - y == y.base() - x.base().
  GCXX_FHDC friend auto operator-(const reverse_iterator& a,
                                  const reverse_iterator& b) noexcept
    -> difference_type {
    return b.base() - a.base();
  }

  // Relational (delegated, direction flipped).
  GCXX_FHDC friend auto operator==(const reverse_iterator& a,
                                   const reverse_iterator& b) noexcept -> bool {
    return a.base() == b.base();
  }
  GCXX_FHDC friend auto operator!=(const reverse_iterator& a,
                                   const reverse_iterator& b) noexcept -> bool {
    return a.base() != b.base();
  }
  GCXX_FHDC friend auto operator<(const reverse_iterator& a,
                                  const reverse_iterator& b) noexcept -> bool {
    return a.base() > b.base();
  }
  GCXX_FHDC friend auto operator>(const reverse_iterator& a,
                                  const reverse_iterator& b) noexcept -> bool {
    return a.base() < b.base();
  }
  GCXX_FHDC friend auto operator<=(const reverse_iterator& a,
                                   const reverse_iterator& b) noexcept -> bool {
    return a.base() >= b.base();
  }
  GCXX_FHDC friend auto operator>=(const reverse_iterator& a,
                                   const reverse_iterator& b) noexcept -> bool {
    return a.base() <= b.base();
  }

 protected:
  Iterator_t current_{};
};

#if defined(__NVCOMPILER)
#pragma diag_warning 20013
#pragma diag_warning 20015
#elif defined(__CUDACC__)
#pragma nv_diag_default 20013
#pragma nv_diag_default 20015
#endif

template <typename Iterator_t>
GCXX_FH auto make_reverse_iterator(Iterator_t i) noexcept
  -> reverse_iterator<Iterator_t> {
  return reverse_iterator<Iterator_t>{i};
}


GCXX_NAMESPACE_MAIN_END()

#endif
