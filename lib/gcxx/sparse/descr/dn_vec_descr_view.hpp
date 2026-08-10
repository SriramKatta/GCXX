// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_SPARSE_DESCR_DN_VEC_DESCR_VIEW_HPP_
#define GCXX_SPARSE_DESCR_DN_VEC_DESCR_VIEW_HPP_

#include <gcxx/internal/prologue.hpp>
#include <gcxx/runtime_backend/backend_sparse.hpp>
#include <gcxx/runtime_backend/backend_sparse_handles.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class DnVecDescrView {
 protected:
  using deviceDnVecDescr_t = driver::deviceDnVecDescr_t;

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  deviceDnVecDescr_t m_descr{nullptr};

 public:
  DnVecDescrView()               = delete;
  DnVecDescrView(int)            = delete;
  DnVecDescrView(std::nullptr_t) = delete;

  GCXX_FHC explicit DnVecDescrView(deviceDnVecDescr_t descr) noexcept;

  // ╔════════════════════════════════════════════════════════╗
  // ║                    Introspection                      ║
  // ╚════════════════════════════════════════════════════════╝

  GCXX_FH auto getValues() const -> void*;

  GCXX_FH auto setValues(void* values) -> void;

  GCXX_FHC auto getDescriptor() const noexcept -> deviceDnVecDescr_t;

  // ╔════════════════════════════════════════════════════════╗
  // ║                      Comparison                        ║
  // ╚════════════════════════════════════════════════════════╝
  GCXX_FHC auto operator==(const DnVecDescrView& rhs) const noexcept -> bool;
  GCXX_FHC auto operator!=(const DnVecDescrView& rhs) const noexcept -> bool;
};

GCXX_NAMESPACE_MAIN_END()


#include <gcxx/sparse/details/descr/dn_vec_descr_view.inl>


#endif
