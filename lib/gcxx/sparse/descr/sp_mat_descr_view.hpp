// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_SPARSE_DESCR_SP_MAT_DESCR_VIEW_HPP_
#define GCXX_SPARSE_DESCR_SP_MAT_DESCR_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_sparse.hpp>
#include <gcxx/runtime_backend/backend_sparse_handles.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class SpMatDescrView {
 protected:
  using deviceSpMatDescr_t = driver::deviceSpMatDescr_t;

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  deviceSpMatDescr_t m_descr{nullptr};

 public:
  using raw_handle_type = driver::deviceSpMatDescr_t;

  SpMatDescrView()               = delete;
  SpMatDescrView(int)            = delete;
  SpMatDescrView(std::nullptr_t) = delete;

  GCXX_FHC explicit SpMatDescrView(deviceSpMatDescr_t descr) noexcept;

  // Introspection.

  GCXX_FH auto getFormat() const -> driver::deviceSparseFormat_t;

  GCXX_FH auto getIndexBase() const -> driver::deviceSparseIndexBase_t;

  GCXX_FH auto getValues() const -> void*;

  GCXX_FH auto setValues(void* values) -> void;

  GCXX_FHC auto getRawHandle() const noexcept -> deviceSpMatDescr_t;

  // Comparison.
  GCXX_FHC auto operator==(const SpMatDescrView& rhs) const noexcept -> bool;
  GCXX_FHC auto operator!=(const SpMatDescrView& rhs) const noexcept -> bool;
};

GCXX_NAMESPACE_MAIN_END()


#include <gcxx/sparse/details/descr/sp_mat_descr_view.inl>


#endif
