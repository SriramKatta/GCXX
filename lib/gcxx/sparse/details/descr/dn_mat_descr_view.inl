// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_SPARSE_DETAILS_DESCR_DN_MAT_DESCR_VIEW_INL_
#define GCXX_SPARSE_DETAILS_DESCR_DN_MAT_DESCR_VIEW_INL_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_sparse.hpp>
#include <gcxx/runtime_backend/backend_sparse_handles.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FHC DnMatDescrView::DnMatDescrView(deviceDnMatDescr_t descr) noexcept
    : m_descr(descr) {}

GCXX_FH auto DnMatDescrView::getValues() const -> void* {
  return driver::sparseDnMatGetValues(m_descr);
}

GCXX_FH auto DnMatDescrView::setValues(void* values) -> void {
  driver::sparseDnMatSetValues(m_descr, values);
}

GCXX_FHC auto DnMatDescrView::getDescriptor() const noexcept
  -> deviceDnMatDescr_t {
  return m_descr;
}

GCXX_FHC auto DnMatDescrView::operator==(
  const DnMatDescrView& rhs) const noexcept -> bool {
  return m_descr == rhs.m_descr;
}

GCXX_FHC auto DnMatDescrView::operator!=(
  const DnMatDescrView& rhs) const noexcept -> bool {
  return m_descr != rhs.m_descr;
}

GCXX_NAMESPACE_MAIN_END()

#endif
