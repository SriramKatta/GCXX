// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_BUFFERS_BUFFER_HPP_
#define GCXX_RUNTIME_MEMORY_BUFFERS_BUFFER_HPP_

#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/memory/buffers/buffer_storage.hpp>
#include <gcxx/runtime/memory/buffers/heterogeneous_iterator.hpp>
#include <gcxx/runtime/memory/buffers/no_init.hpp>
#include <gcxx/runtime/memory/buffers/properties.hpp>
#include <gcxx/runtime/memory/copy.hpp>
#include <gcxx/runtime/memory/fill.hpp>
#include <gcxx/runtime/memory/spans/spans.hpp>
#include <gcxx/runtime/runtime_error.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_NAMESPACE_MEMORY_BEGIN()

// ─────────────────────────────────────────────────────────────────────────────
// gcxx::memory::buffer<VT, Properties...>
//
// A typed raw-storage owner allocated from a memory resource in stream order,
// modelled on cuda::buffer (CCCL/libcudacxx). Properties... are the buffer's
// accessibility contract (device_accessible / host_accessible); the resource —
// which decides the allocation strategy — is passed at construction and
// type-erased into a buffer_storage<VT>. device_buffer<T> == buffer<T,
// device_accessible> is ONE type regardless of which allocator backed it.
//
// The resource must advertise (via `using properties`) every one of this
// buffer's Properties; the ctor static_asserts that (resource_has_all_v), so
// passing a host-only resource to a device_accessible buffer is a compile
// error. Elements are NOT constructed or destroyed — VT must be trivially
// copyable.
//
// Responsibilities:
//   * buffer_storage<VT> — owns the raw byte block + the type-erased
//                          any_resource (allocation/deallocation); RAII.
//   * buffer<VT, Properties...> — typed data()/iterator access, ctor
//                          validation, accessor gating on Properties,
//                          copy/cross-ctors.
//
// Cross-instantiations are friends so the cross-properties ctor/move can reach
// into another buffer's storage.
// ─────────────────────────────────────────────────────────────────────────────
template <typename VT, typename... Properties>
class buffer {
  static_assert(std::is_trivially_copyable_v<VT>,
                "buffer requires trivially copyable VT");
  static_assert(contains_execution_space_property<Properties...>,
                "buffer requires device_accessible or host_accessible");

  template <typename, typename...>
  friend class buffer;

 public:
  /// The buffer's accessibility contract (uniform with resources).
  using properties       = TypeSet<Properties...>;
  using buffer_t         = buffer_storage<VT>;
  using value_type       = VT;
  using reference        = value_type&;
  using const_reference  = const value_type&;
  using size_type        = std::size_t;
  using pointer          = value_type*;
  using const_pointer    = const value_type*;
  using iterator         = heterogeneous_iterator<VT, Properties...>;
  using const_iterator   = heterogeneous_iterator<const VT, Properties...>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  // ─────────────────────── resource-taking ctors ─────────────────────────────
  // Each accepts any resource whose advertised properties ⊇ this buffer's
  // Properties, validates that at compile time (validate_resource), and
  // type-erases it.

  /// Empty buffer bound to a stream + resource (no allocation).
  GCXX_TEMPLATE(typename Resource)
  GCXX_REQUIRES(!std::is_same_v<std::decay_t<Resource>, buffer>)
  buffer(gcxx::StreamView stream, Resource&& resource)
      : m_storage(stream, any_resource(std::forward<Resource>(resource))) {
    validate_resource<Resource>();
  }

  /// Allocate n elements (uninitialized).
  GCXX_TEMPLATE(typename Resource)
  GCXX_REQUIRES(!std::is_same_v<std::decay_t<Resource>, buffer>)
  buffer(gcxx::StreamView stream, Resource&& resource, size_type n)
      : m_storage(stream, any_resource(std::forward<Resource>(resource)), n) {
    validate_resource<Resource>();
  }

  /// Allocate n elements; storage left uninitialized (explicit no-init tag).
  GCXX_TEMPLATE(typename Resource)
  GCXX_REQUIRES(!std::is_same_v<std::decay_t<Resource>, buffer>)
  buffer(gcxx::StreamView stream, Resource&& resource, size_type n, no_init_t)
      : m_storage(stream, any_resource(std::forward<Resource>(resource)), n) {
    validate_resource<Resource>();
  }

  /// Allocate n elements and initialize every element to value.
  GCXX_TEMPLATE(typename Resource)
  GCXX_REQUIRES(!std::is_same_v<std::decay_t<Resource>, buffer>)
  buffer(gcxx::StreamView stream, Resource&& resource, size_type n,
         const value_type& value)
      : m_storage(stream, any_resource(std::forward<Resource>(resource)), n) {
    validate_resource<Resource>();
    if (n != 0) {
      pointer p = data();  // Fill takes Ptr& (lvalue) — bind to a local.
      Fill(stream, p, value, n);
    }
  }

  /// Allocate il.size() elements and copy from the initializer list.
  GCXX_TEMPLATE(typename Resource)
  GCXX_REQUIRES(!std::is_same_v<std::decay_t<Resource>, buffer>)
  buffer(gcxx::StreamView stream, Resource&& resource,
         std::initializer_list<value_type> il)
      : m_storage(stream, any_resource(std::forward<Resource>(resource)),
                  il.size()) {
    validate_resource<Resource>();
    if (il.size() != 0) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): Copy requires
      // same-cv pointers; il.begin() is const. Destination is freshly
      // allocated.
      Copy(stream, data(), const_cast<value_type*>(il.begin()), il.size());
    }
  }

  /// Allocate rng.size() elements and copy from a sized range. SFINAE on
  /// non-integral Range so it does not shadow the size ctor.
  GCXX_TEMPLATE(typename Resource, typename Range)
  GCXX_REQUIRES(!std::is_same_v<std::decay_t<Resource>, buffer>
                  GCXX_AND !std::is_integral_v<std::decay_t<Range>>)
  buffer(gcxx::StreamView stream, Resource&& resource, Range&& rng)
      : m_storage(stream, any_resource(std::forward<Resource>(resource)),
                  static_cast<size_type>(rng.size())) {
    validate_resource<Resource>();
    if (rng.size() != 0) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): see ilist ctor.
      Copy(stream, data(), const_cast<value_type*>(std::data(rng)), rng.size());
    }
  }

  /// Allocate distance(first,last) elements and copy from [first,last).
  /// Requires a contiguous iterator. SFINAE on non-integral Iter.
  GCXX_TEMPLATE(typename Resource, typename Iter)
  GCXX_REQUIRES(!std::is_same_v<std::decay_t<Resource>, buffer>
                  GCXX_AND !std::is_integral_v<std::decay_t<Iter>>)
  buffer(gcxx::StreamView stream, Resource&& resource, Iter first, Iter last)
      : m_storage(stream, any_resource(std::forward<Resource>(resource)),
                  static_cast<size_type>(std::distance(first, last))) {
    validate_resource<Resource>();
    auto count = std::distance(first, last);
    if (count != 0) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): see ilist ctor.
      Copy(stream, data(), const_cast<value_type*>(std::addressof(*first)),
           static_cast<size_type>(count));
    }
  }

  // ───────────────────────── copy / move / cross ─────────────────────────────
  /// Deep copy (same Properties). Allocates a new block and copies.
  buffer(const buffer& other)
      : m_storage(other.stream(), other.m_storage.borrow_resource(),
                  other.size()) {
    if (other.size() != 0) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): see ilist ctor.
      Copy(other.stream(), data(), const_cast<value_type*>(other.data()),
           other.size());
    }
  }

  buffer(buffer&&) noexcept = default;

  buffer& operator=(const buffer& other) {
    if (this != &other) {
      buffer tmp(other);
      *this = std::move(tmp);
    }
    return *this;
  }
  buffer& operator=(buffer&&) noexcept = default;

  /// Cross-properties copy ctor (CCCL __properties_match: other's Properties ⊇
  /// this buffer's). Borrows the source's resource + stream and deep-copies.
  /// A narrowing copy (e.g. managed host+device → host-only), NOT host↔device.
  GCXX_TEMPLATE(typename... OtherProperties)
  GCXX_REQUIRES((TypeSet<OtherProperties...>::template contains<Properties> &&
                 ...))
  explicit buffer(const buffer<VT, OtherProperties...>& other)
      : m_storage(other.stream(), other.m_storage.borrow_resource(),
                  other.size()) {
    if (other.size() != 0) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): see ilist ctor.
      Copy(other.stream(), data(), const_cast<value_type*>(other.data()),
           other.size());
    }
  }

  /// Cross-properties move ctor: steal the source's storage.
  GCXX_TEMPLATE(typename... OtherProperties)
  GCXX_REQUIRES((TypeSet<OtherProperties...>::template contains<Properties> &&
                 ...))
  buffer(buffer<VT, OtherProperties...>&& other) noexcept
      : m_storage(std::move(other.m_storage)) {}

  ~buffer() = default;

  // ───────────────────────────── element access ──────────────────────────────
  GCXX_FHDC auto data() noexcept -> pointer {
    return static_cast<pointer>(m_storage.get());
  }
  GCXX_FHDC auto data() const noexcept -> const_pointer {
    return static_cast<const_pointer>(m_storage.get());
  }

  GCXX_FHDC auto begin() noexcept -> iterator { return iterator(data()); }
  GCXX_FHDC auto begin() const noexcept -> const_iterator {
    return const_iterator(data());
  }
  GCXX_FHDC auto end() noexcept -> iterator {
    return iterator(data() + size());
  }
  GCXX_FHDC auto end() const noexcept -> const_iterator {
    return const_iterator(data() + size());
  }
  GCXX_FHDC auto cbegin() const noexcept -> const_iterator {
    return const_iterator(data());
  }
  GCXX_FHDC auto cend() const noexcept -> const_iterator {
    return const_iterator(data() + size());
  }
  GCXX_FHDC auto rbegin() const noexcept -> const_reverse_iterator {
    return const_reverse_iterator(end());
  }
  GCXX_FHDC auto rend() const noexcept -> const_reverse_iterator {
    return const_reverse_iterator(begin());
  }

  // ─────────────────────────── element access (gated) ────────────────────────
  // Gated on host_accessible: dereferencing device-only memory from the host is
  // UB, so the accessors are SFINAE-removed unless the buffer's Properties
  // include host_accessible. first/last/subspan below stay un-gated — they
  // return spans (views), safe to construct without touching memory.
  GCXX_TEMPLATE(bool H = is_host_accessible<Properties...>)
  GCXX_REQUIRES(H)
  GCXX_FHDC auto operator[](size_type i) noexcept -> reference {
    GCXX_RUNTIME_EXPECT(i < size(), "buffer::operator[] index out of range");
    return data()[i];
  }
  GCXX_TEMPLATE(bool H = is_host_accessible<Properties...>)
  GCXX_REQUIRES(H)
  GCXX_FHDC auto operator[](size_type i) const noexcept -> const_reference {
    GCXX_RUNTIME_EXPECT(i < size(), "buffer::operator[] index out of range");
    return data()[i];
  }

  GCXX_TEMPLATE(bool H = is_host_accessible<Properties...>)
  GCXX_REQUIRES(H)
  GCXX_FH auto at(size_type i) -> reference {
    if (i >= size())
      throw std::out_of_range{"buffer::at"};
    return data()[i];
  }
  GCXX_TEMPLATE(bool H = is_host_accessible<Properties...>)
  GCXX_REQUIRES(H)
  GCXX_FH auto at(size_type i) const -> const_reference {
    if (i >= size())
      throw std::out_of_range{"buffer::at"};
    return data()[i];
  }

  GCXX_TEMPLATE(bool H = is_host_accessible<Properties...>)
  GCXX_REQUIRES(H)
  GCXX_FHDC auto front() noexcept -> reference {
    GCXX_RUNTIME_EXPECT(!empty(), "buffer::front on empty buffer");
    return data()[0];
  }
  GCXX_TEMPLATE(bool H = is_host_accessible<Properties...>)
  GCXX_REQUIRES(H)
  GCXX_FHDC auto front() const noexcept -> const_reference {
    GCXX_RUNTIME_EXPECT(!empty(), "buffer::front on empty buffer");
    return data()[0];
  }

  GCXX_TEMPLATE(bool H = is_host_accessible<Properties...>)
  GCXX_REQUIRES(H)
  GCXX_FHDC auto back() noexcept -> reference {
    GCXX_RUNTIME_EXPECT(!empty(), "buffer::back on empty buffer");
    return data()[size() - 1];
  }
  GCXX_TEMPLATE(bool H = is_host_accessible<Properties...>)
  GCXX_REQUIRES(H)
  GCXX_FHDC auto back() const noexcept -> const_reference {
    GCXX_RUNTIME_EXPECT(!empty(), "buffer::back on empty buffer");
    return data()[size() - 1];
  }

  // ─────────────────────────────── slicing ───────────────────────────────────
  GCXX_FHDC auto first(size_type n) noexcept -> gcxx::span<value_type> {
    GCXX_RUNTIME_EXPECT(n <= size(), "buffer::first count out of range");
    return {data(), n};
  }
  GCXX_FHDC auto first(size_type n) const noexcept
    -> gcxx::span<const value_type> {
    GCXX_RUNTIME_EXPECT(n <= size(), "buffer::first count out of range");
    return {data(), n};
  }
  GCXX_FHDC auto last(size_type n) noexcept -> gcxx::span<value_type> {
    GCXX_RUNTIME_EXPECT(n <= size(), "buffer::last count out of range");
    return {data() + size() - n, n};
  }
  GCXX_FHDC auto last(size_type n) const noexcept
    -> gcxx::span<const value_type> {
    GCXX_RUNTIME_EXPECT(n <= size(), "buffer::last count out of range");
    return {data() + size() - n, n};
  }
  GCXX_FHDC auto subspan(size_type offset,
                         size_type count = gcxx::dynamic_extent) noexcept
    -> gcxx::span<value_type> {
    GCXX_RUNTIME_EXPECT(offset <= size(),
                        "buffer::subspan offset out of range");
    return count == gcxx::dynamic_extent
             ? gcxx::span<value_type>{data() + offset, size() - offset}
             : gcxx::span<value_type>{data() + offset, count};
  }
  GCXX_FHDC auto subspan(size_type offset,
                         size_type count = gcxx::dynamic_extent) const noexcept
    -> gcxx::span<const value_type> {
    GCXX_RUNTIME_EXPECT(offset <= size(),
                        "buffer::subspan offset out of range");
    return count == gcxx::dynamic_extent
             ? gcxx::span<const value_type>{data() + offset, size() - offset}
             : gcxx::span<const value_type>{data() + offset, count};
  }

  // ─────────────────────────────── observers ─────────────────────────────────
  GCXX_FHDC auto size() const noexcept -> size_type {
    return m_storage.size_bytes() / sizeof(VT);
  }
  GCXX_FHDC auto size_bytes() const noexcept -> size_type {
    return m_storage.size_bytes();
  }
  GCXX_FHDC auto empty() const noexcept -> bool { return m_storage.empty(); }

  GCXX_FH auto memory_resource() const noexcept -> const any_resource& {
    return m_storage.memory_resource();
  }
  GCXX_FHDC auto stream() const noexcept -> gcxx::StreamView {
    return m_storage.stream();
  }
  GCXX_FH auto set_stream(gcxx::StreamView new_stream) -> void {
    m_storage.set_stream(new_stream);
  }

  // ─────────────────────────────── operations ────────────────────────────────
  GCXX_FH auto destroy() -> void { m_storage.destroy(); }
  GCXX_FH auto destroy(gcxx::StreamView s) -> void { m_storage.destroy(s); }

  /// Reallocate to n elements (discards contents) reusing the stored resource.
  GCXX_FH auto resize(size_type n) -> void {
    *this = buffer(stream(), m_storage.borrow_resource(), n, no_init);
  }

  GCXX_FH auto storage() noexcept -> buffer_t& { return m_storage; }
  GCXX_FH auto storage() const noexcept -> const buffer_t& { return m_storage; }

  // ──────────────────── launch integration (device-gated) ────────────────────
  // Lets a device_accessible buffer be treated as a span when passed to a
  // launch customization point (CCCL cuda::launch parity).
  GCXX_TEMPLATE(bool D = is_device_accessible<Properties...>)
  GCXX_REQUIRES(D)
  GCXX_FH friend auto transform_launch_argument(gcxx::StreamView,
                                                buffer& self) noexcept
    -> gcxx::span<value_type> {
    return {self.data(), self.size()};
  }
  GCXX_TEMPLATE(bool D = is_device_accessible<Properties...>)
  GCXX_REQUIRES(D)
  GCXX_FH friend auto transform_launch_argument(gcxx::StreamView,
                                                const buffer& self) noexcept
    -> gcxx::span<const value_type> {
    return {self.data(), self.size()};
  }

 private:
  /// Compile-time gate for resource-taking ctors: the resource must be copy
  /// constructible (buffer owns a copy) and advertise ⊇ this buffer's
  /// Properties. Centralized here so every resource ctor gets the same clear
  /// static_assert.
  template <typename Resource>
  static constexpr auto validate_resource() -> void {
    static_assert(
      std::is_copy_constructible_v<std::decay_t<Resource>>,
      "buffer owns a copy of the resource; it must be copy constructible");
    static_assert(resource_has_all_v<std::decay_t<Resource>, Properties...>,
                  "resource properties do not satisfy this buffer's Properties "
                  "(e.g. a host_accessible resource cannot back a "
                  "device_accessible buffer)");
  }

  buffer_t m_storage{};
};

// ─────────────────────────────────────────────────────────────────────────────
// make_buffer: CTAD-friendly factory. Explicit Properties template args
// (the buffer's accessibility is the user's claim, not inferable from the
// resource): make_buffer<int, device_accessible>(stream, resource, …).
// ─────────────────────────────────────────────────────────────────────────────
template <typename VT, typename... Properties, typename Resource,
          typename... Args>
GCXX_FH auto make_buffer(gcxx::StreamView stream, Resource&& resource,
                         Args&&... args) -> buffer<VT, Properties...> {
  return buffer<VT, Properties...>{stream, std::forward<Resource>(resource),
                                   std::forward<Args>(args)...};
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience aliases.
// ─────────────────────────────────────────────────────────────────────────────
template <typename VT>
using device_buffer = buffer<VT, device_accessible>;

template <typename VT>
using host_buffer = buffer<VT, host_accessible>;

GCXX_NAMESPACE_MEMORY_END()

GCXX_NAMESPACE_MAIN_END()

#endif
