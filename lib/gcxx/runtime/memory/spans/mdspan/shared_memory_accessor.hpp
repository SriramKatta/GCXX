// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_MEMORY_SPANS_MDSPAN_SHARED_MEMORY_ACCESSOR_HPP_
#define GCXX_RUNTIME_MEMORY_SPANS_MDSPAN_SHARED_MEMORY_ACCESSOR_HPP_

#include <cassert>
#include <type_traits>

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

// ─────────────────────────────────────────────────────────────────────────────
// shared_memory_accessor
//
// Port of CCCL libcudacxx's cuda/__mdspan/shared_memory_accessor.h: an
// accessor policy marking the viewed data as CUDA/HIP *shared* memory. It is
// device-only — every member asserts when reached from host code — and in
// device builds access/offset additionally verify the data handle really
// points into shared memory (via nvcc's __isShared intrinsic; on HIP this
// cannot be verified and is skipped). data_handle_type is normalized to a
// plain element_type* (stripping e.g. the restrict qualifier), exactly like
// CCCL's wrapper.
//
// Shared memory is NOT a device view for the BLAS layer (gcxx::blas): shared
// addresses only exist inside a running kernel, so shared views do not
// satisfy gcxx::is_device_view_v and are rejected by the BLAS operations.
//
// Deviations from CCCL, both noted for parity reviews:
//   - the host-usage check uses assert() (compiled out with NDEBUG) instead
//     of CCCL's always-on _CCCL_VERIFY
//   - the max-shared-memory-allocation bounds check (PTX sreg based) is not
//     ported
// ─────────────────────────────────────────────────────────────────────────────

// Fire on host usage; a no-op in device compilation passes.
#if GCXX_DEVICE_COMPILE
#define GCXX_SHARED_MEM_VERIFY_DEVICE_ONLY() (void)0
#else
#define GCXX_SHARED_MEM_VERIFY_DEVICE_ONLY()                      \
  assert(false && "gcxx::shared_memory_accessor cannot be used "  \
                  "in HOST code")
#endif

template <class Accessor>
class shared_memory_accessor : public Accessor {
  static_assert(std::is_pointer_v<typename Accessor::data_handle_type>,
                "Accessor::data_handle_type must be a raw pointer");

 public:
  using offset_policy =
    shared_memory_accessor<typename Accessor::offset_policy>;
  using data_handle_type = typename Accessor::element_type*;
  using reference        = typename Accessor::reference;
  using element_type     = typename Accessor::element_type;

  GCXX_TEMPLATE(class Base = Accessor)
  GCXX_REQUIRES(std::is_default_constructible_v<Base>)
  constexpr shared_memory_accessor() noexcept(
    std::is_nothrow_default_constructible_v<Base>)
      : Accessor{} {
    GCXX_SHARED_MEM_VERIFY_DEVICE_ONLY();
  }

  constexpr shared_memory_accessor(const Accessor& acc) noexcept(
    std::is_nothrow_copy_constructible_v<Accessor>)
      : Accessor{acc} {
    GCXX_SHARED_MEM_VERIFY_DEVICE_ONLY();
  }

  // Same-space conversion; implicit iff the base conversion is implicit.
  GCXX_TEMPLATE(class OtherAccessor)
  GCXX_REQUIRES(std::is_constructible_v<Accessor, const OtherAccessor&> GCXX_AND
                std::is_convertible_v<const OtherAccessor&, Accessor>)
  constexpr shared_memory_accessor(
    const shared_memory_accessor<OtherAccessor>& acc) noexcept(
    std::is_nothrow_constructible_v<Accessor, const OtherAccessor&>)
      : Accessor{acc} {
    GCXX_SHARED_MEM_VERIFY_DEVICE_ONLY();
  }

  GCXX_TEMPLATE(class OtherAccessor)
  GCXX_REQUIRES(std::is_constructible_v<Accessor, const OtherAccessor&> GCXX_AND
                !std::is_convertible_v<const OtherAccessor&, Accessor>)
  explicit constexpr shared_memory_accessor(
    const shared_memory_accessor<OtherAccessor>& acc) noexcept(
    std::is_nothrow_constructible_v<Accessor, const OtherAccessor&>)
      : Accessor{acc} {
    GCXX_SHARED_MEM_VERIFY_DEVICE_ONLY();
  }

  // Verifies (where the backend can) that the handle points into shared
  // memory before forwarding to the base accessor.
  constexpr reference access(data_handle_type p, std::size_t i) const {
#if GCXX_CUDA_MODE() && GCXX_DEVICE_COMPILE
    assert(__isShared(p) && "data handle is not a shared memory pointer");
#endif
    GCXX_SHARED_MEM_VERIFY_DEVICE_ONLY();
    return Accessor::access(p, i);
  }

  constexpr data_handle_type offset(data_handle_type p,
                                    std::size_t i) const {
#if GCXX_CUDA_MODE() && GCXX_DEVICE_COMPILE
    assert(__isShared(p) && "data handle is not a shared memory pointer");
#endif
    GCXX_SHARED_MEM_VERIFY_DEVICE_ONLY();
    return Accessor::offset(p, i);
  }
};

// Identity trait, mirroring is_device_accessor_v / is_restrict_accessor_v.
template <class>
GCXX_CXPR inline bool is_shared_memory_accessor_v = false;

template <class Accessor>
GCXX_CXPR inline bool
  is_shared_memory_accessor_v<shared_memory_accessor<Accessor>> = true;

GCXX_NAMESPACE_MAIN_END()

#endif
