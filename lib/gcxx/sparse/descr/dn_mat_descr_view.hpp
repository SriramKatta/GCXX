// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_SPARSE_DESCR_DN_MAT_DESCR_VIEW_HPP_
#define GCXX_SPARSE_DESCR_DN_MAT_DESCR_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_sparse.hpp>
#include <gcxx/runtime_backend/backend_sparse_handles.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()


class DnMatDescrView {
 protected:
  using deviceDnMatDescr_t = driver::deviceDnMatDescr_t;

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  deviceDnMatDescr_t m_descr{nullptr};

 public:
  using raw_handle_type = driver::deviceDnMatDescr_t;

  DnMatDescrView()               = delete;
  DnMatDescrView(int)            = delete;
  DnMatDescrView(std::nullptr_t) = delete;

  GCXX_FHC explicit DnMatDescrView(deviceDnMatDescr_t descr) noexcept;

  // Introspection.

  GCXX_FH auto getValues() const -> void*;

  GCXX_FH auto setValues(void* values) -> void;

  GCXX_FHC auto getRawHandle() const noexcept -> deviceDnMatDescr_t;

  // Comparison.
  GCXX_FHC auto operator==(const DnMatDescrView& rhs) const noexcept -> bool;
  GCXX_FHC auto operator!=(const DnMatDescrView& rhs) const noexcept -> bool;
};

GCXX_NAMESPACE_MAIN_END()


#include <gcxx/sparse/details/descr/dn_mat_descr_view.inl>


#endif
