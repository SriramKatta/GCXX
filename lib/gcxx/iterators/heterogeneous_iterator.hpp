// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_ITERATORS_HETEROGENEOUS_ITERATOR_HPP_
#define GCXX_ITERATORS_HETEROGENEOUS_ITERATOR_HPP_

#include <cstddef>
#include <type_traits>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/iterators/iterator_traits.hpp>
#include <gcxx/runtime/memory/buffers/properties.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

// ─────────────────────────────────────────────────────────────────────────────
// gcxx::heterogeneous_iterator<CvTp, Properties...>
//
// A contiguous (random-access) iterator whose dereference operators are
// API-tagged by the element's memory accessibility (carried as Properties...):
//   * host          → deref callable only from the host (GCXX_FHC)
//   * device        → deref callable only from the device (GCXX_FDC)
//   * host_device   → deref callable from both (GCXX_FHDC)  [managed memory]
// So an iterator over device-only storage cannot be dereferenced from host
// code (compile error), and vice versa. Mirrors CCCL's heterogeneous_iterator.
//
//  Only dereference is space-restricted. Requires at least one execution-space
//  property
// ─────────────────────────────────────────────────────────────────────────────
namespace detail {

  template <bool IsConst, typename T>
  using maybe_const = std::conditional_t<IsConst, const T, T>;

  // Primary (default): both host and device may deref. Reached only for
  // host_device (managed) memory
  template <typename Tp, bool IsConst, memory_accessibility Space>
  class iter_access {
   private:
    using element_ptr_t = maybe_const<IsConst, Tp>*;
    using element_ref_t = maybe_const<IsConst, Tp>&;

   public:
    iter_access() noexcept = default;
    explicit GCXX_FHDC iter_access(element_ptr_t p) noexcept : ptr_(p) {}

    GCXX_FHDC auto operator*() const noexcept -> element_ref_t { return *ptr_; }
    GCXX_FHDC auto operator->() const noexcept -> element_ptr_t { return ptr_; }
    GCXX_FHDC auto operator[](std::ptrdiff_t n) const noexcept
      -> element_ref_t {
      return *(ptr_ + n);
    }

   protected:
    element_ptr_t ptr_{nullptr};
  };


  // ╔════════════════════════════════════════════════════════╗
  // ║                    Host-only deref                     ║
  // ╚════════════════════════════════════════════════════════╝
  template <typename Tp, bool IsConst>
  class iter_access<Tp, IsConst, memory_accessibility::host> {
   private:
    using element_ptr_t = maybe_const<IsConst, Tp>*;
    using element_ref_t = maybe_const<IsConst, Tp>&;

   public:
    iter_access() noexcept = default;
    explicit GCXX_FHDC iter_access(element_ptr_t p) noexcept : ptr_(p) {}

    GCXX_FHC auto operator*() const noexcept -> element_ref_t { return *ptr_; }
    GCXX_FHC auto operator->() const noexcept -> element_ptr_t { return ptr_; }
    GCXX_FHC auto operator[](std::ptrdiff_t n) const noexcept -> element_ref_t {
      return *(ptr_ + n);
    }

   protected:
    element_ptr_t ptr_{nullptr};
  };

  // ╔════════════════════════════════════════════════════════╗
  // ║                   Device-only deref                    ║
  // ╚════════════════════════════════════════════════════════╝
  template <typename Tp, bool IsConst>
  class iter_access<Tp, IsConst, memory_accessibility::device> {
   private:
    using element_ptr_t = maybe_const<IsConst, Tp>*;
    using element_ref_t = maybe_const<IsConst, Tp>&;

   public:
    iter_access() noexcept = default;
    explicit GCXX_FHDC iter_access(element_ptr_t p) noexcept : ptr_(p) {}

    GCXX_FDC auto operator*() const noexcept -> element_ref_t { return *ptr_; }
    GCXX_FDC auto operator->() const noexcept -> element_ptr_t { return ptr_; }
    GCXX_FDC auto operator[](std::ptrdiff_t n) const noexcept -> element_ref_t {
      return *(ptr_ + n);
    }

   protected:
    element_ptr_t ptr_{nullptr};
  };


  template <typename CvTp, typename... Properties>
  using iter_access_for =
    iter_access<std::remove_const_t<CvTp>, std::is_const_v<CvTp>,
                accessibility_from_static_properties<
                  is_host_accessible<Properties...>,
                  is_device_accessible<Properties...>>()>;

}  // namespace detail

template <typename CvTp, typename... Properties>
class heterogeneous_iterator
    : public detail::iter_access_for<CvTp, Properties...> {
  using base = detail::iter_access_for<CvTp, Properties...>;

  static_assert(contains_execution_space_property<Properties...>,
                "heterogeneous_iterator requires host_accessible and/or "
                "device_accessible in its Properties");

 public:
  using iterator_category = random_access_iterator_tag;
  using value_type        = std::remove_const_t<CvTp>;
  using difference_type   = std::ptrdiff_t;
  using pointer           = CvTp*;
  using reference         = CvTp&;

  heterogeneous_iterator() noexcept = default;
  explicit GCXX_FHDC heterogeneous_iterator(pointer p) noexcept : base(p) {}

  // ╔════════════════════════════════════════════════════════╗
  // ║                random-access mechanics                 ║
  // ╚════════════════════════════════════════════════════════╝

  GCXX_FHDC auto operator++() noexcept -> heterogeneous_iterator& {
    ++this->ptr_;
    return *this;
  }
  GCXX_FHDC auto operator++(int) noexcept -> heterogeneous_iterator {
    auto tmp = *this;
    ++this->ptr_;
    return tmp;
  }
  GCXX_FHDC auto operator--() noexcept -> heterogeneous_iterator& {
    --this->ptr_;
    return *this;
  }
  GCXX_FHDC auto operator--(int) noexcept -> heterogeneous_iterator {
    auto tmp = *this;
    --this->ptr_;
    return tmp;
  }
  GCXX_FHDC auto operator+=(difference_type n) noexcept
    -> heterogeneous_iterator& {
    this->ptr_ += n;
    return *this;
  }
  GCXX_FHDC auto operator-=(difference_type n) noexcept
    -> heterogeneous_iterator& {
    this->ptr_ -= n;
    return *this;
  }
  GCXX_FHDC friend auto operator+(heterogeneous_iterator it,
                                  difference_type n) noexcept
    -> heterogeneous_iterator {
    return it += n;
  }
  GCXX_FHDC friend auto operator+(difference_type n,
                                  heterogeneous_iterator it) noexcept
    -> heterogeneous_iterator {
    return it += n;
  }
  GCXX_FHDC friend auto operator-(heterogeneous_iterator it,
                                  difference_type n) noexcept
    -> heterogeneous_iterator {
    return it -= n;
  }
  GCXX_FHDC friend auto operator-(const heterogeneous_iterator& a,
                                  const heterogeneous_iterator& b) noexcept
    -> difference_type {
    return a.ptr_ - b.ptr_;
  }
  GCXX_FHDC friend auto operator==(const heterogeneous_iterator& a,
                                   const heterogeneous_iterator& b) noexcept
    -> bool {
    return a.ptr_ == b.ptr_;
  }
  GCXX_FHDC friend auto operator!=(const heterogeneous_iterator& a,
                                   const heterogeneous_iterator& b) noexcept
    -> bool {
    return a.ptr_ != b.ptr_;
  }
  GCXX_FHDC friend auto operator<(const heterogeneous_iterator& a,
                                  const heterogeneous_iterator& b) noexcept
    -> bool {
    return a.ptr_ < b.ptr_;
  }
  GCXX_FHDC friend auto operator>(const heterogeneous_iterator& a,
                                  const heterogeneous_iterator& b) noexcept
    -> bool {
    return a.ptr_ > b.ptr_;
  }
  GCXX_FHDC friend auto operator<=(const heterogeneous_iterator& a,
                                   const heterogeneous_iterator& b) noexcept
    -> bool {
    return a.ptr_ <= b.ptr_;
  }
  GCXX_FHDC friend auto operator>=(const heterogeneous_iterator& a,
                                   const heterogeneous_iterator& b) noexcept
    -> bool {
    return a.ptr_ >= b.ptr_;
  }
};


GCXX_NAMESPACE_MAIN_END()

#endif
