// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_SPARSE_DETAILS_DESCR_SP_MAT_DESCR_VIEW_INL_
#define GCXX_SPARSE_DETAILS_DESCR_SP_MAT_DESCR_VIEW_INL_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_sparse.hpp>
#include <gcxx/runtime_backend/backend_sparse_handles.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

GCXX_FHC SpMatDescrView::SpMatDescrView(deviceSpMatDescr_t descr) noexcept
    : m_descr(descr) {}

GCXX_FH auto SpMatDescrView::getFormat() const -> driver::deviceSparseFormat_t {
  return driver::sparseSpMatGetFormat(m_descr);
}

GCXX_FH auto SpMatDescrView::getIndexBase() const
  -> driver::deviceSparseIndexBase_t {
  return driver::sparseSpMatGetIndexBase(m_descr);
}

GCXX_FH auto SpMatDescrView::getValues() const -> void* {
  return driver::sparseSpMatGetValues(m_descr);
}

GCXX_FH auto SpMatDescrView::setValues(void* values) -> void {
  driver::sparseSpMatSetValues(m_descr, values);
}

GCXX_FHC auto SpMatDescrView::getDescriptor() const noexcept
  -> deviceSpMatDescr_t {
  return m_descr;
}

GCXX_FHC auto SpMatDescrView::operator==(
  const SpMatDescrView& rhs) const noexcept -> bool {
  return m_descr == rhs.m_descr;
}

GCXX_FHC auto SpMatDescrView::operator!=(
  const SpMatDescrView& rhs) const noexcept -> bool {
  return m_descr != rhs.m_descr;
}

GCXX_NAMESPACE_MAIN_END()

#endif
