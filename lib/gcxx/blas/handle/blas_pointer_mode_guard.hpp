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
<<<<<<< HEAD
  GCXX_FH BlasPointerModeGuard(BlasHandleView h,
                               driver::deviceBlasPointerMode_t mode);


  GCXX_FH BlasPointerModeGuard(BlasHandleView h, bool device_mode);

=======
  // Saves the handle's current pointer mode, then switches it to `mode`.
  GCXX_FH BlasPointerModeGuard(BlasHandleView h,
                               driver::deviceBlasPointerMode_t mode);

  // Restores the pointer mode saved at construction.
>>>>>>> f6989c9 (Amending to new examples)
  GCXX_FH ~BlasPointerModeGuard();

  BlasPointerModeGuard(const BlasPointerModeGuard&)            = delete;
  BlasPointerModeGuard& operator=(const BlasPointerModeGuard&) = delete;
  BlasPointerModeGuard(BlasPointerModeGuard&&)                 = delete;
  BlasPointerModeGuard& operator=(BlasPointerModeGuard&&)      = delete;

 private:
  BlasHandleView m_handle;
  driver::deviceBlasPointerMode_t m_saved{};
<<<<<<< HEAD
  bool m_changed{false};
=======
>>>>>>> f6989c9 (Amending to new examples)
};

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_END()


#include <gcxx/blas/details/handle/blas_pointer_mode_guard.inl>


#endif
