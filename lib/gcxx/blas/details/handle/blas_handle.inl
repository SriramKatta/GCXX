// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_DETAILS_HANDLE_BLAS_HANDLE_INL_
#define GCXX_BLAS_DETAILS_HANDLE_BLAS_HANDLE_INL_

#include <utility>

#include <gcxx/internal/prologue.hpp>

#include <gcxx/blas/handle/blas_handle_view.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()


BlasHandle::BlasHandle() : BlasHandleView(driver::blasCreate()) {}

GCXX_FH BlasHandle::~BlasHandle() noexcept {
  if (m_handle != nullptr) {
    driver::blasDestroy(m_handle);
  }
}

GCXX_FH BlasHandle::BlasHandle(BlasHandle&& other) noexcept
    : BlasHandleView(std::exchange(other.m_handle, nullptr)) {}

GCXX_FH auto BlasHandle::operator=(BlasHandle&& other) noexcept
    -> BlasHandle& {
  if (this != &other) {
    if (m_handle != nullptr) {
      driver::blasDestroy(m_handle);
    }
    m_handle = std::exchange(other.m_handle, nullptr);
  }
  return *this;
}


GCXX_FH auto BlasHandle::from_native_handle(
  driver::deviceBlasHandle_t handle) noexcept -> BlasHandle {
  return BlasHandle(handle);
}

GCXX_FH auto BlasHandle::release() noexcept -> BlasHandleView {
  auto h   = m_handle;
  m_handle = nullptr;
  return BlasHandleView{h};
}

GCXX_FH BlasHandle::BlasHandle(driver::deviceBlasHandle_t handle) noexcept
    : BlasHandleView(handle) {}

GCXX_NAMESPACE_MAIN_END()

#endif
