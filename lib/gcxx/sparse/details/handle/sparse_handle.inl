// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_SPARSE_DETAILS_HANDLE_SPARSE_HANDLE_INL_
#define GCXX_SPARSE_DETAILS_HANDLE_SPARSE_HANDLE_INL_

#include <utility>

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_sparse.hpp>
#include <gcxx/sparse/handle/sparse_handle_view.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()


inline SparseHandle::SparseHandle()
    : SparseHandleView(driver::sparseCreate()) {}

GCXX_FH SparseHandle::~SparseHandle() noexcept {
  destroy();
}

GCXX_FH SparseHandle::SparseHandle(SparseHandle&& other) noexcept
    : SparseHandleView(std::exchange(other.m_handle, nullptr)) {}

GCXX_FH auto SparseHandle::operator=(SparseHandle&& other) noexcept
  -> SparseHandle& {
  if (this != &other) {
    destroy();
    m_handle = std::exchange(other.m_handle, nullptr);
  }
  return *this;
}


GCXX_FH auto SparseHandle::from_native_handle(
  driver::deviceSparseHandle_t handle) noexcept -> SparseHandle {
  return SparseHandle(handle);
}

GCXX_FH auto SparseHandle::release() noexcept -> SparseHandleView {
  auto h   = m_handle;
  m_handle = nullptr;
  return SparseHandleView{h};
}

GCXX_FH auto SparseHandle::destroy() noexcept -> void {
  if (m_handle != nullptr) {
    driver::sparseDestroy(m_handle);
    m_handle = nullptr;
  }
}

GCXX_FH SparseHandle::SparseHandle(driver::deviceSparseHandle_t handle) noexcept
    : SparseHandleView(handle) {}

GCXX_NAMESPACE_MAIN_END()

#endif
