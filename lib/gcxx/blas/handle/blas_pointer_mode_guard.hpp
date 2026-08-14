// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_HANDLE_BLAS_POINTER_MODE_GUARD_HPP_
#define GCXX_BLAS_HANDLE_BLAS_POINTER_MODE_GUARD_HPP_

#include <gcxx/blas/handle/blas_handle_view.hpp>
#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_BEGIN()

// RAII guard for a BLAS handle's pointer mode.
class [[maybe_unused]] BlasPointerModeGuard {
 public:
  // Saves the handle's current pointer mode, then switches it to `mode`. When
  // the handle already is in `mode` no backend call is made
  GCXX_FH BlasPointerModeGuard(BlasHandleView h,
                               driver::deviceBlasPointerMode_t mode);

  // Convenience overload selecting host/device mode from whether the scalar
  // argument was a device_scalar.
  GCXX_FH BlasPointerModeGuard(BlasHandleView h, bool device_mode);

  // Restores the pointer mode saved at construction.
  GCXX_FH ~BlasPointerModeGuard();

  BlasPointerModeGuard(const BlasPointerModeGuard&)            = delete;
  BlasPointerModeGuard& operator=(const BlasPointerModeGuard&) = delete;
  BlasPointerModeGuard(BlasPointerModeGuard&&)                 = delete;
  BlasPointerModeGuard& operator=(BlasPointerModeGuard&&)      = delete;

 private:
  BlasHandleView m_handle;
  driver::deviceBlasPointerMode_t m_saved{};
  bool m_changed{false};
};

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_END()


#include <gcxx/blas/details/handle/blas_pointer_mode_guard.inl>


#endif
