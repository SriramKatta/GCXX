// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_BLAS_DETAILS_HANDLE_BLAS_HANDLE_VIEW_INL_
#define GCXX_BLAS_DETAILS_HANDLE_BLAS_HANDLE_VIEW_INL_

#include <gcxx/internal/prologue.hpp>

#include <gcxx/runtime/stream/stream_view.hpp>
#include <gcxx/runtime_backend/backend_blas.hpp>
#include <gcxx/runtime_backend/backend_blas_handles.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FHC BlasHandleView::BlasHandleView(deviceBlasHandle_t handle) noexcept
    : m_handle(handle) {}

GCXX_FH auto BlasHandleView::setStream(gcxx::StreamView stream) -> void {
  driver::blasSetStream(m_handle, stream.getRawHandle());
}

GCXX_FH auto BlasHandleView::getStream() const -> gcxx::StreamView {
  return gcxx::StreamView{driver::blasGetStream(m_handle)};
}

GCXX_FH auto BlasHandleView::getVersion() const -> int {
  return driver::blasGetVersion(m_handle);
}

GCXX_FHC auto BlasHandleView::getRawHandle() const noexcept
  -> deviceBlasHandle_t {
  return m_handle;
}

GCXX_FHC auto BlasHandleView::operator==(
  const BlasHandleView& rhs) const noexcept -> bool {
  return m_handle == rhs.m_handle;
}

GCXX_FHC auto BlasHandleView::operator!=(
  const BlasHandleView& rhs) const noexcept -> bool {
  return m_handle != rhs.m_handle;
}

GCXX_NAMESPACE_MAIN_END()

#endif
