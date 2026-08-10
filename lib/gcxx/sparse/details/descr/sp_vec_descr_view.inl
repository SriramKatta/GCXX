// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_SPARSE_DETAILS_DESCR_SP_VEC_DESCR_VIEW_INL_
#define GCXX_SPARSE_DETAILS_DESCR_SP_VEC_DESCR_VIEW_INL_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_sparse.hpp>
#include <gcxx/runtime_backend/backend_sparse_handles.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FHC SpVecDescrView::SpVecDescrView(deviceSpVecDescr_t descr) noexcept
    : m_descr(descr) {}

GCXX_FH auto SpVecDescrView::getValues() const -> void* {
  return driver::sparseSpVecGetValues(m_descr);
}

GCXX_FH auto SpVecDescrView::setValues(void* values) -> void {
  driver::sparseSpVecSetValues(m_descr, values);
}

GCXX_FHC auto SpVecDescrView::getDescriptor() const noexcept
  -> deviceSpVecDescr_t {
  return m_descr;
}

GCXX_FHC auto SpVecDescrView::operator==(
  const SpVecDescrView& rhs) const noexcept -> bool {
  return m_descr == rhs.m_descr;
}

GCXX_FHC auto SpVecDescrView::operator!=(
  const SpVecDescrView& rhs) const noexcept -> bool {
  return m_descr != rhs.m_descr;
}

GCXX_NAMESPACE_MAIN_END()

#endif
