// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_SPARSE_DETAILS_DESCR_DN_VEC_DESCR_VIEW_INL_
#define GCXX_SPARSE_DETAILS_DESCR_DN_VEC_DESCR_VIEW_INL_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_sparse.hpp>
#include <gcxx/runtime_backend/backend_sparse_handles.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FHC DnVecDescrView::DnVecDescrView(deviceDnVecDescr_t descr) noexcept
    : m_descr(descr) {}

GCXX_FH auto DnVecDescrView::getValues() const -> void* {
  return driver::sparseDnVecGetValues(m_descr);
}

GCXX_FH auto DnVecDescrView::setValues(void* values) -> void {
  driver::sparseDnVecSetValues(m_descr, values);
}

GCXX_FHC auto DnVecDescrView::getRawHandle() const noexcept
  -> deviceDnVecDescr_t {
  return m_descr;
}

GCXX_FHC auto DnVecDescrView::operator==(
  const DnVecDescrView& rhs) const noexcept -> bool {
  return m_descr == rhs.m_descr;
}

GCXX_FHC auto DnVecDescrView::operator!=(
  const DnVecDescrView& rhs) const noexcept -> bool {
  return m_descr != rhs.m_descr;
}

GCXX_NAMESPACE_MAIN_END()

#endif
