// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_SPARSE_DETAILS_HANDLE_SPARSE_HANDLE_VIEW_INL_
#define GCXX_SPARSE_DETAILS_HANDLE_SPARSE_HANDLE_VIEW_INL_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime/stream/stream_view.hpp>
#include <gcxx/runtime_backend/backend_sparse.hpp>
#include <gcxx/runtime_backend/backend_sparse_handles.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FHC SparseHandleView::SparseHandleView(
  deviceSparseHandle_t handle) noexcept
    : m_handle(handle) {}

GCXX_FH auto SparseHandleView::setStream(gcxx::StreamView stream) -> void {
  driver::sparseSetStream(m_handle, stream.getRawStream());
}

GCXX_FH auto SparseHandleView::getStream() const -> gcxx::StreamView {
  return gcxx::StreamView{driver::sparseGetStream(m_handle)};
}

GCXX_FH auto SparseHandleView::getVersion() const -> int {
  return driver::sparseGetVersion(m_handle);
}

GCXX_FHC auto SparseHandleView::getHandle() const noexcept
  -> deviceSparseHandle_t {
  return m_handle;
}

GCXX_FHC auto SparseHandleView::operator==(
  const SparseHandleView& rhs) const noexcept -> bool {
  return m_handle == rhs.m_handle;
}

GCXX_FHC auto SparseHandleView::operator!=(
  const SparseHandleView& rhs) const noexcept -> bool {
  return m_handle != rhs.m_handle;
}

GCXX_NAMESPACE_MAIN_END()

#endif
