// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_BUFFERS_HETEROGENEOUS_ITERATOR_HPP_
#define GCXX_RUNTIME_MEMORY_BUFFERS_HETEROGENEOUS_ITERATOR_HPP_

#include <cstddef>
#include <iterator>
#include <type_traits>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/memory/buffers/properties.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_MEMORY_BEGIN()

// ─────────────────────────────────────────────────────────────────────────────
// heterogeneous_iterator<CvTp, Properties...>
//
// A contiguous (random-access) iterator over a buffer's storage whose
// dereference operators are API-tagged by the buffer's memory accessibility:
//   * host          → deref callable only from the host (GCXX_FHC)
//   * device        → deref callable only from the device (GCXX_FDC)
//   * host_device   → deref callable from both (GCXX_FHDC)  [managed memory]
// So a device-only buffer's iterator cannot be dereferenced from host code
// (compile error), and vice versa. Mirrors CCCL's heterogeneous_iterator.
//
// The space-dependent deref tagging lives in detail::iter_access (specialized
// per memory_accessibility); this class adds the random-access iterator
// mechanics and the pointer storage.
// ─────────────────────────────────────────────────────────────────────────────
namespace detail {

  template <bool IsConst, typename T>
  using maybe_const = std::conditional_t<IsConst, const T, T>;

  // Primary (default): both host and device may deref. Used for host_device
  // (managed) and as a permissive fallback for unknown accessibility.
  template <typename Tp, bool IsConst, memory_accessibility Space>
  class iter_access {
   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type        = Tp;
    using difference_type   = std::ptrdiff_t;
    using pointer           = maybe_const<IsConst, Tp>*;
    using reference         = maybe_const<IsConst, Tp>&;

    iter_access() noexcept = default;
    explicit constexpr iter_access(pointer p) noexcept : ptr_(p) {}

    GCXX_FHDC auto operator*() const noexcept -> reference { return *ptr_; }
    GCXX_FHDC auto operator->() const noexcept -> pointer { return ptr_; }
    GCXX_FHDC auto operator[](difference_type n) const noexcept -> reference {
      return *(ptr_ + n);
    }

   protected:
    pointer ptr_{nullptr};
  };

  // Host-only deref.
  template <typename Tp, bool IsConst>
  class iter_access<Tp, IsConst, memory_accessibility::host> {
   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type        = Tp;
    using difference_type   = std::ptrdiff_t;
    using pointer           = maybe_const<IsConst, Tp>*;
    using reference         = maybe_const<IsConst, Tp>&;

    iter_access() noexcept = default;
    explicit constexpr iter_access(pointer p) noexcept : ptr_(p) {}

    GCXX_FHC auto operator*() const noexcept -> reference { return *ptr_; }
    GCXX_FHC auto operator->() const noexcept -> pointer { return ptr_; }
    GCXX_FHC auto operator[](difference_type n) const noexcept -> reference {
      return *(ptr_ + n);
    }

   protected:
    pointer ptr_{nullptr};
  };

  // Device-only deref.
  template <typename Tp, bool IsConst>
  class iter_access<Tp, IsConst, memory_accessibility::device> {
   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type        = Tp;
    using difference_type   = std::ptrdiff_t;
    using pointer           = maybe_const<IsConst, Tp>*;
    using reference         = maybe_const<IsConst, Tp>&;

    iter_access() noexcept = default;
    explicit constexpr iter_access(pointer p) noexcept : ptr_(p) {}

    GCXX_FDC auto operator*() const noexcept -> reference { return *ptr_; }
    GCXX_FDC auto operator->() const noexcept -> pointer { return ptr_; }
    GCXX_FDC auto operator[](difference_type n) const noexcept -> reference {
      return *(ptr_ + n);
    }

   protected:
    pointer ptr_{nullptr};
  };

}  // namespace detail

template <typename CvTp, typename... Properties>
class heterogeneous_iterator
    : public detail::iter_access<std::remove_const_t<CvTp>,
                                 std::is_const_v<CvTp>,
                                 accessibility_from_static_properties<
                                   is_host_accessible<Properties...>,
                                   is_device_accessible<Properties...>>()> {
  using base =
    detail::iter_access<std::remove_const_t<CvTp>, std::is_const_v<CvTp>,
                        accessibility_from_static_properties<
                          is_host_accessible<Properties...>,
                          is_device_accessible<Properties...>>()>;

 public:
  using iterator_category = typename base::iterator_category;
  using value_type        = typename base::value_type;
  using difference_type   = typename base::difference_type;
  using pointer           = typename base::pointer;
  using reference         = typename base::reference;

  heterogeneous_iterator() noexcept = default;
  explicit constexpr heterogeneous_iterator(pointer p) noexcept : base(p) {}

  // ─────────────────────────── random-access mechanics ───────────────────────
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

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif
