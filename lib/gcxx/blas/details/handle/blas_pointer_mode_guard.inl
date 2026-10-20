// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_DETAILS_HANDLE_BLAS_POINTER_MODE_GUARD_INL_
#define GCXX_BLAS_DETAILS_HANDLE_BLAS_POINTER_MODE_GUARD_INL_

#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_BEGIN()

GCXX_FH BlasPointerModeGuard::BlasPointerModeGuard(
  BlasHandleView h, driver::deviceBlasPointerMode_t mode)
    : m_handle(h),
      m_saved(h.getPointerMode()),  // use h, not m_handle: members initialize
                                    // in declaration order, so a declaration
                                    // reorder would read m_handle before its
                                    // lifetime began (UB); h is always valid
      m_changed(m_saved != mode) {
  if (m_changed) {
    m_handle.setPointerMode(mode);
  }
}

GCXX_FH BlasPointerModeGuard::BlasPointerModeGuard(BlasHandleView h,
                                                   bool device_mode)
    : BlasPointerModeGuard(h, device_mode ? driver::deviceBlasPointerModeDevice
                                          : driver::deviceBlasPointerModeHost) {
}

GCXX_FH BlasPointerModeGuard::~BlasPointerModeGuard() {
  if (m_changed) {
    m_handle.setPointerMode(m_saved);
  }
}

GCXX_NAMESPACE_MAIN_BLAS_DETAILS_END()

#endif
